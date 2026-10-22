// AF2's triangle multiplication (fused projection weights), as a reusable
// module.  Follows the block interface set in msacol.cuh: bf16 activations,
// float32 gradients across the boundary, forward() writes its DELTA.
//
// The contraction is why this is laid out the way it is: 128 independent N x N
// matrix products, one per channel, so each channel's plane has to be
// contiguous -- `a` and `b` are held channel-major and the batched GEMM walks
// them with stride M.  A row-major [M, I] array cannot be fed to cuBLAS for it
// at all: within one channel, stepping along k moves by I and along i by N*I,
// and neither is the unit stride a column-major matrix needs.
// NO `#pragma once`: this body is included ONCE PER WIDTH by trimul.cuh --
// the Evoformer's C=I=128 and the template pair stack's C=I=64 -- each into
// its own namespace.  TM_NS / TM_C / TM_I select which pass this is.  The
// includes below carry their own guards, so re-entering them is a no-op.
#include <mma.h>

#include "common.cuh"
#include <type_traits>

#include "lt.cuh"

namespace TM_NS {

static const int C = TM_C;
static const int I = TM_I;
static const int PRE = 2 * I + 2 * I + C;
static const float LN_EPS = 1e-5f;
static const int TILE = 32;

__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + __expf(-v)); }

__device__ __forceinline__ float blockSum(float v, float* scratch, int tid) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((tid & 31) == 0) scratch[tid >> 5] = v;
  __syncthreads();
  v = scratch[0] + scratch[1] + scratch[2] + scratch[3];
  __syncthreads();
  return v;
}

// ---------------------------------------------------------------- LayerNorms
// ONE WARP A ROW, NOT ONE BLOCK.  C is 128, so a lane can hold four channels
// and both reductions become shuffles -- no shared memory and no
// __syncthreads, where a block-per-row paid two block-wide barriers for every
// one of the 50176 rows.  That was the whole reason these kernels sat 2.3x off
// bandwidth while doing nothing but a mean and a variance.
//
// The four channels a lane owns are strided by 32, not adjacent, so each of
// the four loads is one fully coalesced 32-lane access across the row.
__global__ void lnFwdK(const __nv_bfloat16* __restrict__ pair,
                       const float* __restrict__ scale,
                       const float* __restrict__ offset, __nv_bfloat16* __restrict__ x,
                       float* __restrict__ mu, float* __restrict__ rstd, int M) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  int lane = threadIdx.x & 31;
  const int K = C / 32;
  const size_t base = (size_t)row * C;
  float v[K], s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    v[k] = __bfloat162float(pair[base + lane + k * 32]);
    s += v[k];
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  float m = s / C, s2 = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { v[k] -= m; s2 += v[k] * v[k]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
  float r = rsqrtf(s2 / C + LN_EPS);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = lane + k * 32;
    x[base + c] = __float2bfloat16(v[k] * r * scale[c] + offset[c]);
  }
  if (lane == 0) { mu[row] = m; rstd[row] = r; }
}

// ACC adds into `dpair` instead of overwriting it, so the block can hand its
// running gradient straight in.  The alternative -- write a scratch, then
// acc[i] += d[i] -- costs an extra 33 MB write and 33 MB read a call, and the
// stack makes 288 of them: 57.6 ms, measured with the NO_ACC probe, and accK
// is already at ~83% of bandwidth, so the only way to get it is to not do it.
// Aliasing with the incoming cotangent is safe: every backward in this
// directory reads `dout` once, at the top, into a bf16 copy.
template <bool ACC = false, class DT = float>
__global__ void lnBwdK(const __nv_bfloat16* __restrict__ dx,
                       const __nv_bfloat16* __restrict__ pair,
                       const float* __restrict__ scale, const float* __restrict__ mu,
                       const float* __restrict__ rstd, DT* __restrict__ dpair, int M) {
  // one warp a row, like lnFwdK above
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  int lane = threadIdx.x & 31;
  const int K = C / 32;
  const size_t base = (size_t)row * C;
  float r = rstd[row], mu_ = mu[row];
  float xh[K], dxh[K], s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = lane + k * 32;
    xh[k] = (__bfloat162float(pair[base + c]) - mu_) * r;
    dxh[k] = __bfloat162float(dx[base + c]) * scale[c];
    s1 += dxh[k];
    s2 += dxh[k] * xh[k];
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s1 += __shfl_xor_sync(0xffffffff, s1, o);
    s2 += __shfl_xor_sync(0xffffffff, s2, o);
  }
  float mean_d = s1 / C, mean_dx = s2 / C;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    size_t o = base + lane + k * 32;
    float v = r * (dxh[k] - mean_d - xh[k] * mean_dx);
    // DT is bf16 on the pair track, so the view the next module reads IS the
    // accumulator -- no second buffer and no second pass
    if (ACC) v += toF(dpair[o]);
    stF(dpair + o, v);
  }
}

// The centre norm straddles the two layouts: the contraction's output is
// channel-major and the output projection wants row-major.  Doing both in one
// kernel means one side is uncoalesced, and it was the write -- each thread
// wrote its position's 128 contiguous channels while its warp neighbours were
// 256 B away, so a warp issued 32 transactions for 64 useful bytes and the
// kernel ran at 123 GB/s.  Splitting it costs one extra pass over a 16.8 MB
// tensor and makes every access coalesced.
__global__ void centerStatsK(const __nv_bfloat16* __restrict__ t, float* __restrict__ cmu,
                             float* __restrict__ crstd, int M, int Mt) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)M) return;
  // TWO PASSES: mean((x-mu)^2), which is what AF2 computes
  // (use_fast_variance=False), NOT E[x^2] - E[x]^2.
  //
  // THE ONE-PASS FORM WAS HERE AND IT WAS THE BUG.  Algebraically identical,
  // catastrophically worse conditioned: it subtracts two large nearly-equal
  // numbers, so it keeps only the digits by which E[x^2] exceeds E[x]^2.  `t`
  // is a sum over N terms and on the model's real activations its channel mean
  // is large beside its channel variance, which is exactly the regime that
  // destroys.  The comment that used to sit here said "gated -- it moves
  // nothing", and that was true of the gate it was run against: N(0,1) inputs
  // give `t` a near-zero mean, where the two forms agree to the last bit.  The
  // same module measured 6.85x the bf16 floor on real activations.
  //
  // WELFORD WAS TRIED AND IS WORSE ON BOTH COUNTS: the stack reads 9.04e-03
  // forward and 1.47e-01 on grad/pair against two passes' 6.94e-03 and
  // 1.03e-01, and it is not faster (1059.7 against 1059.9 ms) because `t` is
  // still in L2 for the second pass.  Its sequential d/(c+1) buys nothing.
  float s = 0.f;
#pragma unroll 8
  for (int c = 0; c < I; ++c) s += __bfloat162float(t[(size_t)c * Mt + ij]);
  float m = s / I;
  float s2 = 0.f;
#pragma unroll 8
  for (int c = 0; c < I; ++c) {
    float d = __bfloat162float(t[(size_t)c * Mt + ij]) - m;
    s2 += d * d;
  }
  cmu[ij] = m;
  crstd[ij] = rsqrtf(s2 / I + LN_EPS);
}

__global__ void centerWriteK(const __nv_bfloat16* __restrict__ t,
                             const float* __restrict__ scale, const float* __restrict__ offset,
                             const float* __restrict__ cmu, const float* __restrict__ crstd,
                             __nv_bfloat16* __restrict__ cn, int M, int Mt) {
  __shared__ float tile[TILE][TILE + 1];
  int ij0 = blockIdx.x * TILE, c0 = blockIdx.y * TILE;
  for (int j = 0; j < TILE; j += 8) {
    int c = c0 + threadIdx.y + j, ij = ij0 + threadIdx.x;
    tile[threadIdx.y + j][threadIdx.x] =
        ij < M ? __bfloat162float(t[(size_t)c * Mt + ij]) : 0.f;
  }
  __syncthreads();
  for (int j = 0; j < TILE; j += 8) {
    int ij = ij0 + threadIdx.y + j, c = c0 + threadIdx.x;
    if (ij < M)
      cn[(size_t)ij * I + c] = __float2bfloat16(
          (tile[threadIdx.x][threadIdx.y + j] - cmu[ij]) * crstd[ij] * scale[c] + offset[c]);
  }
}

// ------------------------------- the centre LayerNorm, statistics and write
// in one pass
//
// `centerStatsK` reads all of `t` TWICE (AF2's variance is mean((x-mu)^2),
// not E[x^2]-E[x]^2, and the one-pass form was a real bug -- see above) and
// `centerWriteK` reads it a third time, so the step moves 67.2 MB where the
// data is 16.8.  Staging a tile of `t` in shared makes all three passes read
// it once: 34.1 MB, and the two passes over c become shared traffic.
//
// It also makes the KEEP_STATS tier pointless -- the replay recomputes the
// statistics out of a tile it has to load anyway -- but cmu/crstd are still
// WRITTEN, because the backward reads them.
//
// `t` is channel-major and `cn` is row-major, so this is the transpose too.
#ifndef TM_BLKC
#define TM_BLKC 64
#endif
static const int BLKC = TM_BLKC;      // positions a block
static const int LDC = BLKC + 2;      // pad: p is the contiguous index
static const int PARTS = 256 / BLKC;  // threads sharing a position

__global__ void __launch_bounds__(256) centerFusedK(
    const __nv_bfloat16* __restrict__ t, const float* __restrict__ scale,
    const float* __restrict__ offset, float* __restrict__ cmu,
    float* __restrict__ crstd, __nv_bfloat16* __restrict__ cn, int M) {
  __shared__ __nv_bfloat16 sh[I * LDC];
  __shared__ float red[PARTS][BLKC], mu[BLKC], rs[BLKC];
  const int tid = threadIdx.x;
  const size_t ij0 = (size_t)blockIdx.x * BLKC;
  const int nrow = (int)min((size_t)BLKC, (size_t)M - ij0);

  for (int i = tid; i < I * BLKC; i += 256) {
    int c = i / BLKC, p = i % BLKC;
    sh[c * LDC + p] = p < nrow ? t[(size_t)c * M + ij0 + p] : __float2bfloat16(0.f);
  }
  __syncthreads();

  // 4 threads a position, 32 channels each; consecutive threads take
  // consecutive positions, which is the contiguous index of the tile
  const int part = tid / BLKC, p = tid % BLKC;
  static_assert(256 % BLKC == 0, "the block splits evenly over positions");
  static_assert(I % PARTS == 0, "and the channels over the threads sharing one");
  float s = 0.f;
  for (int c = part * (I / PARTS); c < (part + 1) * (I / PARTS); ++c)
    s += __bfloat162float(sh[c * LDC + p]);
  red[part][p] = s;
  __syncthreads();
  if (part == 0) {
    float a = 0.f;
#pragma unroll
    for (int u = 0; u < PARTS; ++u) a += red[u][p];
    mu[p] = a / I;
  }
  __syncthreads();
  float m = mu[p], s2 = 0.f;
  for (int c = part * (I / PARTS); c < (part + 1) * (I / PARTS); ++c) {
    float d = __bfloat162float(sh[c * LDC + p]) - m;
    s2 += d * d;
  }
  red[part][p] = s2;
  __syncthreads();
  if (part == 0) {
    float a = 0.f;
#pragma unroll
    for (int u = 0; u < PARTS; ++u) a += red[u][p];
    rs[p] = rsqrtf(a / I + LN_EPS);
    if (p < nrow) { cmu[ij0 + p] = mu[p]; crstd[ij0 + p] = rs[p]; }
  }
  __syncthreads();

  for (int i = tid; i < I * BLKC; i += 256) {
    int c = i / BLKC, q = i % BLKC;
    if (q >= nrow) continue;
    cn[(ij0 + q) * I + c] = __float2bfloat16(
        (__bfloat162float(sh[c * LDC + q]) - mu[q]) * rs[q] * scale[c] + offset[c]);
  }
}

// Its backward goes the other way, so the uncoalesced side is the READ of dcn.
// Transpose it first, then both the statistics and the write are coalesced.
__global__ void transposeRowToChanK(const __nv_bfloat16* __restrict__ src,
                                    __nv_bfloat16* __restrict__ dst, int M, int W) {
  __shared__ float tile[TILE][TILE + 1];
  int ij0 = blockIdx.x * TILE, c0 = blockIdx.y * TILE;
  for (int j = 0; j < TILE; j += 8) {
    int ij = ij0 + threadIdx.y + j, c = c0 + threadIdx.x;
    tile[threadIdx.y + j][threadIdx.x] =
        ij < M ? __bfloat162float(src[(size_t)ij * W + c]) : 0.f;
  }
  __syncthreads();
  for (int j = 0; j < TILE; j += 8) {
    int c = c0 + threadIdx.y + j, ij = ij0 + threadIdx.x;
    if (ij < M) dst[(size_t)c * M + ij] = __float2bfloat16(tile[threadIdx.x][threadIdx.y + j]);
  }
}

// `ROWMAJ` reads the cotangent in the layout `gemm dcn` produced it, [M][I],
// instead of the channel-major copy `transposeRowToChanK` used to make.  A
// thread owns one position, so it walks I contiguous values and the warp's
// 32 loads are 256 bytes apart -- but every byte of each 256-byte run is used
// within sixteen iterations and the block's working set is 64 KB, so L1 is
// meant to absorb it.  Measured, not assumed: see the README.
template <bool ROWMAJ = false>
__global__ void centerLnBwdK(const __nv_bfloat16* __restrict__ dcnT,
                             const __nv_bfloat16* __restrict__ t,
                             const float* __restrict__ scale, const float* __restrict__ cmu,
                             const float* __restrict__ crstd,
                             __nv_bfloat16* __restrict__ dt, int M, int Mt,
                             int Mdc) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)M) return;
  float m = cmu[ij], r = crstd[ij];
  float sum_d = 0.f, sum_dx = 0.f;
  const size_t drow = ROWMAJ ? ij * I : ij;      // and stride 1 against M
#pragma unroll 8
  for (int c = 0; c < I; ++c) {
    size_t o = (size_t)c * Mt + ij;
    size_t d = ROWMAJ ? drow + c : (size_t)c * Mdc + ij;
    float xhat = (__bfloat162float(t[o]) - m) * r;
    float dxhat = __bfloat162float(dcnT[d]) * scale[c];
    sum_d += dxhat; sum_dx += dxhat * xhat;
  }
  float mean_d = sum_d / I, mean_dx = sum_dx / I;
  // PROBE_ONEPASS makes the SECOND pass read a small, cache-resident slice
  // instead of the real one.  The arithmetic is wrong; the write traffic and
  // the instruction mix are right, so the difference is what the second
  // HBM read costs -- which is exactly the prize a shared-staged version
  // would collect, and worth knowing before writing one.
  //
  // MEASURED at N=256: module backward 1.348 -> 1.261 ms, so the second pass
  // is 0.087 ms a call = 8.4 ms of the stack's backward, 1.2%.  The byte
  // ratio says build it (84 -> 50.4 MB, 1.67x, above the ~1.55x break-even,
  // and still bandwidth-bound after) -- but a staged version collects less
  // than the probe's upper bound, the realisation rate here is ~60%, and the
  // closest precedent is the fused centre-LayerNorm FORWARD, which made the
  // same trade and LOST.  ~5 ms expected for a kernel with a cross-thread
  // reduction and a losing precedent.  NOT BUILT.  The probe stays so the
  // next person can re-price it in thirty seconds instead of an hour.
#ifndef PROBE_ONEPASS
#define PROBE_ONEPASS 0
#endif
#pragma unroll 8
  for (int c = 0; c < I; ++c) {
    size_t o = (size_t)c * Mt + ij;
    size_t d = ROWMAJ ? drow + c : (size_t)c * Mdc + ij;
    if (PROBE_ONEPASS) { o &= 4095; d &= 4095; }
    float xhat = (__bfloat162float(t[o]) - m) * r;
    float dxhat = __bfloat162float(dcnT[d]) * scale[c];
    dt[o] = __float2bfloat16(r * (dxhat - mean_d - xhat * mean_dx));
  }
}

// ------------------------------------------------- gating, and the transpose
//
// `gateTransK` USED TO LIVE HERE: it read the fused projection row-major and
// wrote the gated halves channel-major through a 32x32 shared tile.  It is
// gone because `projGateK` below does the projection and the gating in one
// pass and the row-major `pre` it read never exists any more.  `git log` has
// it if the split path is ever wanted back for the forward; TM_SPLIT only
// restores the BACKWARD pair, which is the half that was worth keeping an
// A/B for.
// ------------------------------------------------ the fused projection+gate
// `gemm pre` and `gateTrans` are ONE memory pass pretending to be two.  The
// GEMM is not compute-bound -- doubling its K costs 4% (0.256 -> 0.267 ms at
// N=256) -- it is bound by writing a 640-channel intermediate that gateTrans
// then reads all of back to produce 384.  This does the projection and the
// gating in one kernel, so `pre` never reaches HBM.
//
// WHAT IT WRITES INSTEAD OF `pre`.  The backward needs the two sigmoids, and
// it can have them directly: with a = mask * p * sigmoid(g) in hand,
//
//     dpre[gate] = dleft * p * mask * gl * (1 - gl) = dleft * a * (1 - gl)
//
// so storing the GATE and the GATED PRODUCT is as good as storing the raw
// projection and the raw gate, with no division and no extra bytes -- and it
// retires the 640-channel buffer, 84 MB a module at N=256 against 67 for the
// two gates.  `KEEPG` is false on the stack's own forward, where nothing
// survives the next block, and true in the backward's replay.
//
// MEASURED at N=256 against the pair it replaces (0.603 ms):
//   0.423 ms  KEEPG=false   1.43x
//   0.482 ms  KEEPG=true    1.25x
// The floor for this traffic is 0.249 ms (a pure-copy kernel moving the same
// bytes in the same layout), so it is not at the roof -- but the gap is not
// the layout, the write granularity or the occupancy.  Those were measured and
// ruled out one at a time; see the README.
// TM_MINB asks ptxas for this many blocks an SM, capping registers to reach
// it.  The BMF/BMG sweeps moved occupancy indirectly, through the tile size,
// and that conflates it with the work per block; this moves it on its own.
// Measured at N=256, module forward+backward in ms:
//
//   TM_MINB      0 (no hint)      2            3
//   fwd              0.868      0.885      0.873
//   bwd              1.432      1.349      1.388    <- 2 is 5.8% off the bwd
//
// 2 it is: the backward is twice the forward and the gradient is unchanged
// (cos 0.999963581 either way).  3 spills 16 bytes and gives it back.
#ifndef TM_MINB
#define TM_MINB 2
#endif
#if TM_MINB
#define TM_LB(T) __launch_bounds__(T, TM_MINB)
#else
#define TM_LB(T) __launch_bounds__(T)
#endif
#ifndef TM_BMF
#define TM_BMF 32
#endif
static const int BMF = TM_BMF;      // rows a block
static const int WF = I / 16;       // warps, one per 16-column slice of I
static const int RTF = BMF / 16;    // row tiles a warp accumulates
static const int LDXF = C + 8;

// AN OCCUPANCY HINT, AND THE OLD ONE ASKED FOR THE WRONG NUMBER.  The note
// that used to sit here said "TM_MINB=2 on both kernels moved bwd 721.0 ->
// 718.1 and fwd 286.2 -> 288.8, a wash" and concluded projGateK does not want
// the hint.  But `cudaOccupancyMaxActiveBlocksPerMultiprocessor` on the real
// kernel says it ALREADY gets 3 blocks an SM, and that its 74 registers are
// what holds it there while its 25216 bytes of shared memory would allow 4:
//
//   projGateK   regs 74  thr 256  smem 25216  ->  3   (registers 3, shared 4)
//
// So minBlocks=2 was a floor it already cleared -- the experiment could only
// ever have measured noise.  4 is the number that matches the shared memory,
// and it needs 65536/(4*256) = 64 registers.
//
// This is localfold opus-55-opt's finding, in its own words: "it had grown to
// 160 registers, so the registers held it to ONE... __launch_bounds__(256, 2)
// holds it to 128 with LOCAL 0: -11% to -13%, byte-identical".
#ifndef TM_PROJ_MINB
#define TM_PROJ_MINB 4
#endif
template <bool KEEPG>
__global__ void __launch_bounds__(32 * WF, TM_PROJ_MINB) projGateK(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ wcat,
    const __nv_bfloat16* __restrict__ bcat, const float* __restrict__ mask,
    __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ bb,
    gate_t* __restrict__ gl, gate_t* __restrict__ gr,
    gt2_t* __restrict__ t2, int M, int Mt) {
  extern __shared__ char smem[];
  __nv_bfloat16* xs = (__nv_bfloat16*)smem;                  // [BMF][LDXF]
  // Two 16x16 tiles a warp, not the whole strip: a product and its gate have
  // to be live together, and staging more row tiles to widen the write was
  // MEASURED SLOWER (0.478 / 0.503 / 0.656 ms at 1 / 2 / 4 tiles) -- the
  // shared it costs buys back less than the occupancy it spends.
  float* stage = (float*)(xs + (size_t)BMF * LDXF);          // [WF][2][16][16]
  float* mk = stage + (size_t)WF * 2 * 256;                  // [BMF]
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const size_t r0 = (size_t)blockIdx.x * BMF;
  float* st = stage + (size_t)warp * 2 * 256;

  // N is a multiple of 32 so M = N*N is a multiple of 1024, but the standalone
  // gates run whatever a dump holds, so the tail is guarded rather than assumed.
  const int nrow = (int)min((size_t)BMF, (size_t)M - r0);
  for (int i = tid; i < BMF; i += 32 * WF) mk[i] = i < nrow ? mask[r0 + i] : 0.f;
  for (int i = tid; i < BMF * C; i += 32 * WF)
    xs[(i / C) * LDXF + (i % C)] =
        (i / C) < nrow ? x[(r0 + i / C) * C + (i % C)] : __float2bfloat16(0.f);
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[RTF], gat[RTF];

  // one group's 16 columns for this warp.  The weight fragment is loaded
  // straight from global with ldm = PRE: it is the whole 164 KB weight for
  // every block, so it lives in L2, and staging it in shared was worth 12%
  // (0.473 -> 0.418 with the stride removed) against an occupancy halving.
#define TM_PROJECT(G, DST)                                                     \
  do {                                                                         \
    _Pragma("unroll") for (int r = 0; r < RTF; ++r)                            \
        nvcuda::wmma::fill_fragment(DST[r], 0.f);                              \
    for (int kk = 0; kk < C / 16; ++kk) {                                      \
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, \
                             nvcuda::wmma::row_major> fb;                      \
      nvcuda::wmma::load_matrix_sync(                                          \
          fb, TM_FRAGPERM                                                      \
                  ? wcat + ((size_t)((G) * (I / 16) + warp) * C + kk * 16) * 16 \
                  : wcat + (size_t)(kk * 16) * PRE + (G) * I + warp * 16,      \
          TM_FRAGPERM ? 16 : PRE);                                             \
      _Pragma("unroll") for (int r = 0; r < RTF; ++r) {                        \
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,             \
                               __nv_bfloat16, nvcuda::wmma::row_major> fa;     \
        nvcuda::wmma::load_matrix_sync(fa, xs + (size_t)(r * 16) * LDXF + kk * 16, LDXF); \
        nvcuda::wmma::mma_sync(DST[r], fa, fb, DST[r]);                        \
      }                                                                        \
    }                                                                          \
  } while (0)

  const int c0 = warp * 16;

  // ONE K-LOOP FOR ALL FIVE PROJECTIONS.  Five separate TM_PROJECT passes
  // reload the same x fragment five times and give the scheduler one
  // dependent MMA chain at a time; this keeps all five accumulators live, so
  // the x fragment is loaded once a k-step and five independent chains
  // overlap.  That is the ILP a latency-bound kernel wants, and it is why
  // BMF = 32 (RTF = 2, ten accumulator fragments) rather than 64.
  // TM_ALLG=0 restores the five passes.
// MEASURED AND REJECTED: 0.911 against 0.885 ms.  The extra ILP is not what
// this kernel is short of.
#ifndef TM_ALLG
#define TM_ALLG 0
#endif
#if TM_ALLG
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> all[5][RTF];
#pragma unroll
  for (int g = 0; g < 5; ++g)
#pragma unroll
    for (int r = 0; r < RTF; ++r) nvcuda::wmma::fill_fragment(all[g][r], 0.f);
  for (int kk = 0; kk < C / 16; ++kk) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                           nvcuda::wmma::row_major> fa[RTF];
#pragma unroll
    for (int r = 0; r < RTF; ++r)
      nvcuda::wmma::load_matrix_sync(fa[r], xs + (size_t)(r * 16) * LDXF + kk * 16, LDXF);
#pragma unroll
    for (int g = 0; g < 5; ++g) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                             nvcuda::wmma::row_major> fb;
      nvcuda::wmma::load_matrix_sync(
          fb, TM_FRAGPERM ? wcat + ((size_t)(g * (I / 16) + warp) * C + kk * 16) * 16
                          : wcat + (size_t)(kk * 16) * PRE + g * I + warp * 16,
          TM_FRAGPERM ? 16 : PRE);
#pragma unroll
      for (int r = 0; r < RTF; ++r)
        nvcuda::wmma::mma_sync(all[g][r], fa[r], fb, all[g][r]);
    }
  }
  // gproj/ggate are (0,2) -> a,gl and (1,3) -> bb,gr; group 4 is t2
  auto shipPair = [&](int gp, int gg, __nv_bfloat16* out, gate_t* gout) {
#pragma unroll
    for (int r = 0; r < RTF; ++r) {
      nvcuda::wmma::store_matrix_sync(st, all[gp][r], 16, nvcuda::wmma::mem_row_major);
      nvcuda::wmma::store_matrix_sync(st + 256, all[gg][r], 16,
                                      nvcuda::wmma::mem_row_major);
      __syncwarp();
      for (int i = lane; i < 256; i += 32) {
        int c = i / 16, row = r * 16 + i % 16, e = (i % 16) * 16 + c;
        if (row >= nrow) continue;
        float g = sigmoidf(st[256 + e] + __bfloat162float(bcat[gg * I + c0 + c]));
        float pv = st[e] + __bfloat162float(bcat[gp * I + c0 + c]);
        size_t ij = r0 + row;
        out[(size_t)(c0 + c) * Mt + ij] = __float2bfloat16(mk[row] * pv * g);
        if (KEEPG) gout[(size_t)(c0 + c) * Mt + ij] = gateStore(g);
      }
      __syncwarp();
    }
  };
  shipPair(0, 2, a, gl);
  shipPair(1, 3, bb, gr);
#pragma unroll
  for (int r = 0; r < RTF; ++r) {
    nvcuda::wmma::store_matrix_sync(st, all[4][r], 16, nvcuda::wmma::mem_row_major);
    __syncwarp();
    // t2 is ROW-major, so here the pair is two channels, not two rows
    for (int i = lane; i < 64; i += 32) {   // four channels a lane
      int row = r * 16 + i / 4, c = (i % 4) * 4, e = (i / 4) * 16 + c;
      if (row >= nrow) continue;
      const float* b4 = nullptr; (void)b4;
      float q[4];
#pragma unroll
      for (int u = 0; u < 4; ++u)
        q[u] = sigmoidf(st[e + u] + __bfloat162float(bcat[4 * I + c0 + c + u]));
      gt2_t* d = t2 + (r0 + row) * C + c0 + c;
#if TM_GATE8T
      *(uchar4*)d = make_uchar4((gt2_t)__float2uint_rn(__saturatef(q[0]) * 255.f),
                                (gt2_t)__float2uint_rn(__saturatef(q[1]) * 255.f),
                                (gt2_t)__float2uint_rn(__saturatef(q[2]) * 255.f),
                                (gt2_t)__float2uint_rn(__saturatef(q[3]) * 255.f));
#else
      *(__nv_bfloat162*)(d) = __floats2bfloat162_rn(q[0], q[1]);
      *(__nv_bfloat162*)(d + 2) = __floats2bfloat162_rn(q[2], q[3]);
#endif
    }
    __syncwarp();
  }
#else
  auto emit = [&](int gproj, int ggate, __nv_bfloat16* out, gate_t* gout) {
    TM_PROJECT(ggate, gat);
    TM_PROJECT(gproj, acc);
#pragma unroll
    for (int r = 0; r < RTF; ++r) {
      nvcuda::wmma::store_matrix_sync(st, acc[r], 16, nvcuda::wmma::mem_row_major);
      nvcuda::wmma::store_matrix_sync(st + 256, gat[r], 16, nvcuda::wmma::mem_row_major);
      __syncwarp();
      // A LANE WALKS ij, NOT THE CHANNEL.  `a` is channel-major, so a lane
      // stepping c jumps M*2 = 128 KB and the warp's store becomes 32 scalar
      // transactions.  Getting this backwards cost 2.1x and left the output
      // bit-identical, which is why it is spelled out here.
      // TWO ROWS A LANE, ONE 4-BYTE STORE.  The warp was issuing a 2-byte
      // scalar store per element -- 25 M of them a call -- and this kernel is
      // not bandwidth-bound (its 67 MB forward is SLOWER than its 101 MB
      // replay), so the store ISSUE rate is the suspect.  `a` is
      // channel-major, so two consecutive ij of one channel are adjacent and
      // pack into a __nv_bfloat162; eight lanes then cover a channel's 16
      // rows as 32 contiguous bytes, exactly as before.
      // FOUR consecutive ij a lane, one 8-byte store.  Two was worth 4.2%;
      // `a` is channel-major so four consecutive ij of one channel are
      // adjacent, and the stage holds them 16 floats apart.
      for (int i = lane; i < 64; i += 32) {
        int c = i / 4, row = r * 16 + (i % 4) * 4;
        if (row >= nrow) continue;
        int e0 = ((i % 4) * 4) * 16 + c;
        float bg = __bfloat162float(bcat[ggate * I + c0 + c]);
        float bp = __bfloat162float(bcat[gproj * I + c0 + c]);
        float gq[4], pq[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          gq[u] = sigmoidf(st[256 + e0 + u * 16] + bg);
          pq[u] = st[e0 + u * 16] + bp;
        }
        size_t o = (size_t)(c0 + c) * Mt + r0 + row;
        __nv_bfloat162 w0 = __floats2bfloat162_rn(mk[row] * pq[0] * gq[0],
                                                  mk[row + 1] * pq[1] * gq[1]);
        __nv_bfloat162 w1 = __floats2bfloat162_rn(mk[row + 2] * pq[2] * gq[2],
                                                  mk[row + 3] * pq[3] * gq[3]);
        *(__nv_bfloat162*)(out + o) = w0;
        *(__nv_bfloat162*)(out + o + 2) = w1;
        if (KEEPG) {
#if TM_GATE8
          *(uchar4*)(gout + o) = make_uchar4(gateStore(gq[0]), gateStore(gq[1]),
                                             gateStore(gq[2]), gateStore(gq[3]));
#else
          *(__nv_bfloat162*)(gout + o) = __floats2bfloat162_rn(gq[0], gq[1]);
          *(__nv_bfloat162*)(gout + o + 2) = __floats2bfloat162_rn(gq[2], gq[3]);
#endif
        }
      }
      __syncwarp();
    }
  };
  auto emitT2 = [&] {
    TM_PROJECT(4, acc);
#pragma unroll
    for (int r = 0; r < RTF; ++r) {
      nvcuda::wmma::store_matrix_sync(st, acc[r], 16, nvcuda::wmma::mem_row_major);
      __syncwarp();
      for (int i = lane; i < 256; i += 32) {
        int row = r * 16 + i / 16, c = i % 16;
        if (row >= nrow) continue;
        {
          float q = sigmoidf(st[i] + __bfloat162float(bcat[4 * I + c0 + c]));
#if TM_GATE8T
          t2[(r0 + row) * C + c0 + c] = (gt2_t)__float2uint_rn(__saturatef(q) * 255.f);
#else
          t2[(r0 + row) * C + c0 + c] = __float2bfloat16(q);
#endif
        }
      }
      __syncwarp();
    }
  };

  // THE WARPS ARE IN LOCKSTEP AND THAT IS THE SUSPECT.  Each of the three
  // units below is an MMA phase followed by a write phase with nothing to
  // overlap it, and every warp enters them together: the tensor cores are
  // saturated while the memory pipe is idle, then the reverse.  The MMA is
  // ~85 us of a 473 us kernel and the traffic ~250, so perfect overlap would
  // be ~280.  Rotating the order by warp index de-phases them for free.
  // TM_ROTATE=0 at compile time restores lockstep; measured both ways.
// MEASURED AND REJECTED: de-phasing the warps' MMA and write phases by
// rotating the group order costs 1.130 -> 1.228 ms on the forward.  The
// lockstep was not the problem.
#ifndef TM_ROTATE
#define TM_ROTATE 0
#endif
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int g = TM_ROTATE ? (u + warp) % 3 : u;
    if (g == 0) emit(0, 2, a, gl);
    else if (g == 1) emit(1, 3, bb, gr);
    else emitT2();
  }
#endif
#undef TM_PROJECT
}

inline size_t projGateSmem() {
  return (size_t)BMF * LDXF * 2 + (size_t)WF * 2 * 256 * 4 + (size_t)BMF * 4;
}

// The mirror image: channel-major da/db in, row-major dpre out.
//
// IT READS THE GATES, NOT `pre`.  projGateK writes sigmoid(gate) and the gated
// product rather than the raw projection, and that is enough:
//
//   a    = mask * p * gl                     so  p * mask * gl * (1 - gl)
//   dpre[gate] = dleft * p * mask * gl * (1 - gl)  =  dleft * a * (1 - gl)
//
// -- no division, the same four reads of [I][M] it used to make into `pre`,
// and the 640-channel buffer is gone.
__global__ void combineTransK(const __nv_bfloat16* __restrict__ da,
                              const __nv_bfloat16* __restrict__ db,
                              const __nv_bfloat16* __restrict__ a,
                              const __nv_bfloat16* __restrict__ bb,
                              const gate_t* __restrict__ gl,
                              const gate_t* __restrict__ gr,
                              const __nv_bfloat16* __restrict__ dt2pre,
                              const float* __restrict__ mask,
                              __nv_bfloat16* __restrict__ dpreBlk, int M, int r0,
                              int rows) {
  // six channel-major reads to transpose, where there used to be two plus a
  // row-major read of `pre`.  Staged in bf16, not float: at float the six
  // tiles are 25 KB and cost more occupancy than the conversion saves.
  __shared__ __nv_bfloat16 sh[6][TILE][TILE + 1];
  int loc0 = blockIdx.x * TILE, c0 = blockIdx.y * TILE;

  for (int j = 0; j < TILE; j += 8) {
    int c = c0 + threadIdx.y + j, loc = loc0 + threadIdx.x;
    bool ok = loc < rows;
    size_t o = (size_t)c * M + r0 + loc;
    const __nv_bfloat16* src[4] = {da, db, a, bb};
#pragma unroll
    for (int u = 0; u < 4; ++u)
      sh[u][threadIdx.y + j][threadIdx.x] = ok ? src[u][o] : __float2bfloat16(0.f);
    // the two gates are a narrower type, so they cannot ride the uniform loop
    const gate_t* gsrc[2] = {gl, gr};
#pragma unroll
    for (int u = 0; u < 2; ++u)
      sh[4 + u][threadIdx.y + j][threadIdx.x] =
          ok ? __float2bfloat16(gateLoad(gsrc[u][o])) : __float2bfloat16(0.f);
  }
  __syncthreads();
  for (int j = 0; j < TILE; j += 8) {
    int loc = loc0 + threadIdx.y + j, c = c0 + threadIdx.x;
    if (loc >= rows) continue;
    size_t ij = r0 + loc;
    float mk = mask[ij];
    __nv_bfloat16* drow = dpreBlk + (size_t)loc * PRE;

    float dleft = __bfloat162float(sh[0][threadIdx.x][threadIdx.y + j]);
    float dright = __bfloat162float(sh[1][threadIdx.x][threadIdx.y + j]);
    float av = __bfloat162float(sh[2][threadIdx.x][threadIdx.y + j]);
    float bv = __bfloat162float(sh[3][threadIdx.x][threadIdx.y + j]);
    float glv = __bfloat162float(sh[4][threadIdx.x][threadIdx.y + j]);
    float grv = __bfloat162float(sh[5][threadIdx.x][threadIdx.y + j]);

    drow[c] = __float2bfloat16(dleft * glv * mk);
    drow[I + c] = __float2bfloat16(dright * grv * mk);
    drow[2 * I + c] = __float2bfloat16(dleft * av * (1.f - glv));
    drow[2 * I + I + c] = __float2bfloat16(dright * bv * (1.f - grv));
  }
  if (blockIdx.y == 0) {
    for (int j = 0; j < TILE; j += 8) {
      int loc = loc0 + threadIdx.y + j;
      if (loc < rows)
        for (int c = threadIdx.x; c < C; c += TILE)
          dpreBlk[(size_t)loc * PRE + 4 * I + c] = dt2pre[(size_t)(r0 + loc) * C + c];
    }
  }
}

// ------------------------------------------- the fused combine + dX contract
// `combineTrans` and `gemm dx` are the backward's mirror of the same waste:
// combine writes a 640-channel dpre (84 MB at N=256) and the GEMM reads all of
// it back, 302 MB of traffic for a contraction whose inputs are 117 MB and
// whose output is 17.  Measured together they are 0.784 ms a call at N=256,
// which is 385 GB/s -- at the roof, for bytes that need not move.
//
// This builds dpre a GROUP at a time in shared and contracts it immediately,
// so dpre never reaches HBM:
//
//     dX = sum over the five groups  dpre_g [BMG][I] @ W_g [I][C]
//
// W_g is rows g*I.. of `wcatT`, which is already [PRE][C] row-major -- the
// layout a wmma matrix_b wants, contiguous, no transpose.  The accumulator for
// dX lives in registers across all five groups, so the only shared is two
// dpre tiles, reused.
//
// The gradient of the gating, written out once so the next reader does not
// have to re-derive it (p is the raw projection, a = mask * p * gl):
//
//     dpre[left ]      = dleft  * gl * mask
//     dpre[left gate]  = dleft  * p * mask * gl * (1 - gl) = dleft * a * (1 - gl)
//
#ifndef TM_BMG
#define TM_BMG 64
#endif
// MEASURED AND REJECTED: four tiles is 78 KB, one block an SM, and 1.864 ms
// against two tiles' 1.559.  It looked like a 1.028 ms win while the launch
// was silently failing for want of a shared-memory attribute.  TM_CDX4=1 to
// re-run it.
#ifndef TM_CDX4
#define TM_CDX4 0
#endif
// PROBE: make the matrix_a address kk-invariant so the compiler hoists all
// RTG loads out of the kk loop -- wrong arithmetic, IDENTICAL mma_sync count,
// one eighth the LDS traffic.  Tells apart an LDS-path bottleneck (worth an
// mma.sync + ldmatrix rewrite) from true tensor-core issue (not worth it).
#ifndef PROBE_LDS
#define PROBE_LDS 0
#endif
// The contraction's inner step is `load fb; for r: {load fa[r]; mma}` -- each
// mma waits on the shared load immediately before it.  TM_LOADALL issues all
// RTG activation fragments first so the four loads overlap and the four mma
// then issue back to back.  Costs RTG x 4 registers.  PROBE_LDS said the
// loads are not the bottleneck; this asks whether their LATENCY is.
//
// **+6.4% WORSE: 1.389 ms against 1.305, three reps each.**  The compiler was
// already scheduling these better than the explicit hoist, and holding four
// fragments costs more than the overlap returns.
// combineDx reads its six activation planes ONCE each -- 100.8 MB a call,
// pure streaming, no reuse -- while its weight fragments come straight from
// GLOBAL and are re-read every tile: 164 KB a block over 1024 blocks is
// 168 MB if they do not stay in L2, and L2 is 6 MB.  So the streaming planes
// may be evicting the one thing that wants to be cached.  __ldcs marks the
// plane loads as streaming (evict-first), which is free if the hypothesis is
// wrong and worth 168 MB if it is right.  Nobody had priced this.
__device__ __forceinline__ __nv_bfloat162 ldcs2(const __nv_bfloat16* p) {
  float f = __ldcs((const float*)p);
  __nv_bfloat162 r; memcpy(&r, &f, 4); return r;
}
__device__ __forceinline__ uchar4 ldcs4(const unsigned char* p) {
  float f = __ldcs((const float*)p);
  uchar4 r; memcpy(&r, &f, 4); return r;
}
// **NO DIFFERENCE: 1.306 ms against 1.307 over three reps.**  The weights are
// already staying in L2; the streaming planes are not evicting them.
#ifndef TM_LDCS
#define TM_LDCS 0
#endif
// PROBE: keep every load, store and conversion in the build and drop only its
// ARITHMETIC (five multiplies and a subtract an element, over 16384 elements
// a block).  Wrong math, identical traffic, so the delta is what the build's
// ALU work costs -- the other half of this kernel, against the MMA.
#ifndef PROBE_BUILD
#define PROBE_BUILD 0
#endif

#ifndef TM_LOADALL
#define TM_LOADALL 0
#endif
static const int BMG = TM_BMG;
static const int WG = C / 16;       // warps, one per 16-column slice of C
static const int RTG = BMG / 16;
static const int LDG = I + 8;

__global__ void TM_LB(32 * WG) combineDxK(
    const __nv_bfloat16* __restrict__ da, const __nv_bfloat16* __restrict__ db,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ bb,
    const gate_t* __restrict__ gl, const gate_t* __restrict__ gr,
    const __nv_bfloat16* __restrict__ dt2pre, const float* __restrict__ mask,
    const __nv_bfloat16* __restrict__ wcatT, __nv_bfloat16* __restrict__ dx, int M,
    int Mt) {
  extern __shared__ char smem[];
  // TM_CDX4 gives the two builds their own tiles, so both issue their six
  // plane reads BEFORE any contraction instead of alternating build/contract
  // behind six block barriers.  More shared, fewer barriers, better memory
  // level parallelism -- measured both ways.
  __nv_bfloat16* dp0 = (__nv_bfloat16*)smem;                    // [BMG][LDG]
  __nv_bfloat16* dp1 = dp0 + (size_t)BMG * LDG;
  __nv_bfloat16* dq0 = dp1 + (size_t)BMG * LDG;                 // TM_CDX4 only
  __nv_bfloat16* dq1 = dq0 + (size_t)(TM_CDX4 ? BMG * LDG : 0);
  float* stage = (float*)(dq1 + (size_t)(TM_CDX4 ? BMG * LDG : 0));
  float* mk = stage + (size_t)WG * 256;                         // [BMG]
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const size_t r0 = (size_t)blockIdx.x * BMG;
  const int nrow = (int)min((size_t)BMG, (size_t)M - r0);
  float* st = stage + (size_t)warp * 256;
  // the mask is per ROW; reading it inside the element loop was I = 128
  // lookups a row through L1 for one value
  for (int i = tid; i < BMG; i += 32 * WG) mk[i] = i < nrow ? mask[r0 + i] : 0.f;
  // THE MASK IS SHARED AND EVERY THREAD READS ALL OF IT.  The build used to
  // open with a __syncthreads of its own, which covered this by accident;
  // TM_CDX4 drops that barrier (nothing is reading the tiles yet) and the
  // cover went with it -- grad/pair cos 0.2099 with the forward still
  // perfect, because only the `dleft * gl * mask` half is corrupted.
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[RTG];
#pragma unroll
  for (int r = 0; r < RTG; ++r) nvcuda::wmma::fill_fragment(acc[r], 0.f);

  // dx += dp[.][.] @ wcatT[g*I ..][.], the 16 columns this warp owns
#define TM_CONTRACT(SRC, G)                                                    \
  do {                                                                         \
    __syncthreads();                                                           \
    for (int kk = 0; kk < I / 16; ++kk) {                                      \
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, \
                             nvcuda::wmma::row_major> fb;                      \
      nvcuda::wmma::load_matrix_sync(                                          \
          fb, TM_FRAGPERM                                                      \
                  ? wcatT + ((size_t)warp * PRE + (G) * I + kk * 16) * 16      \
                  : wcatT + (size_t)((G) * I + kk * 16) * C + warp * 16,       \
          TM_FRAGPERM ? 16 : C);                                               \
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,               \
                             __nv_bfloat16, nvcuda::wmma::row_major> fa[RTG];  \
      _Pragma("unroll") for (int r = 0; r < RTG; ++r)                          \
        nvcuda::wmma::load_matrix_sync(fa[TM_LOADALL ? r : 0],                 \
            (SRC) + (size_t)(r * 16) * LDG + (PROBE_LDS ? 0 : kk) * 16, LDG);  \
      _Pragma("unroll") for (int r = 0; r < RTG; ++r) {                        \
        if (!TM_LOADALL)                                                       \
          nvcuda::wmma::load_matrix_sync(fa[0],                                \
              (SRC) + (size_t)(r * 16) * LDG + (PROBE_LDS ? 0 : kk) * 16, LDG);\
        nvcuda::wmma::mma_sync(acc[r], fa[TM_LOADALL ? r : 0], fb, acc[r]);    \
      }                                                                        \
    }                                                                          \
  } while (0)

  // A LANE WALKS ij: these six inputs are channel-major, so consecutive
  // threads must take consecutive rows or every read is a scalar transaction.
  auto build = [&](const __nv_bfloat16* d, const __nv_bfloat16* pg,
                   const gate_t* g, __nv_bfloat16* dp0, __nv_bfloat16* dp1,
                   bool wait) {
    // the contraction that came before is still reading these tiles
    if (wait) __syncthreads();
    // two rows a lane: these six planes are channel-major, so a pair of
    // consecutive ij is one 4-byte load instead of two 2-byte ones
    for (int i = tid; i < BMG * I / 4; i += 32 * WG) {
      int c = i / (BMG / 4), row = (i % (BMG / 4)) * 4;
      size_t o = (size_t)c * Mt + r0 + row;
      __nv_bfloat162 dv2[2], pv2[2];
      float gq[4];
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        dv2[w] = TM_LDCS ? ldcs2(d + o + 2 * w)
                         : *(const __nv_bfloat162*)(d + o + 2 * w);
        pv2[w] = TM_LDCS ? ldcs2(pg + o + 2 * w)
                         : *(const __nv_bfloat162*)(pg + o + 2 * w);
      }
#if TM_GATE8
      // four gates in one 4-byte load, where bf16 needed two
      uchar4 g4 = TM_LDCS ? ldcs4(g + o) : *(const uchar4*)(g + o);
      gq[0] = gateLoad(g4.x); gq[1] = gateLoad(g4.y);
      gq[2] = gateLoad(g4.z); gq[3] = gateLoad(g4.w);
#else
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        __nv_bfloat162 G2 = *(const __nv_bfloat162*)(g + o + 2 * w);
        gq[2 * w] = __bfloat162float(G2.x); gq[2 * w + 1] = __bfloat162float(G2.y);
      }
#endif
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float v0 = 0.f, v1 = 0.f;
        if (row + u < nrow) {              // a tail row contributes nothing,
          const __nv_bfloat162 &D = dv2[u >> 1], &P = pv2[u >> 1];  // but the
          float dv = __bfloat162float((u & 1) ? D.y : D.x);          // MMA reads
          float gv = gq[u];
          float pv = __bfloat162float((u & 1) ? P.y : P.x);
          if (PROBE_BUILD) { v0 = dv + gv; v1 = pv; }
          else { v0 = dv * gv * mk[row + u]; v1 = dv * pv * (1.f - gv); }
        }
        dp0[(size_t)(row + u) * LDG + c] = __float2bfloat16(v0);
        dp1[(size_t)(row + u) * LDG + c] = __float2bfloat16(v1);
      }
    }
  };

  if (TM_CDX4) {
    build(da, a, gl, dp0, dp1, false);
    build(db, bb, gr, dq0, dq1, false);
    TM_CONTRACT(dp0, 0); TM_CONTRACT(dp1, 2);
    TM_CONTRACT(dq0, 1); TM_CONTRACT(dq1, 3);
  } else {
    build(da, a, gl, dp0, dp1, true);  TM_CONTRACT(dp0, 0);  TM_CONTRACT(dp1, 2);
    build(db, bb, gr, dp0, dp1, true); TM_CONTRACT(dp0, 1);  TM_CONTRACT(dp1, 3);
  }
  // group 4 is the output gate, already row-major and already a gradient
  __syncthreads();   // the group-1/3 contraction is still reading dp0
  for (int i = tid; i < BMG * C; i += 32 * WG) {
    int row = i / C, c = i % C;
    dp0[(size_t)row * LDG + c] =
        row < nrow ? dt2pre[(r0 + row) * C + c] : __float2bfloat16(0.f);
  }
  TM_CONTRACT(dp0, 4);
#undef TM_CONTRACT

  const int c0 = warp * 16;
#pragma unroll
  for (int r = 0; r < RTG; ++r) {
    nvcuda::wmma::store_matrix_sync(st, acc[r], 16, nvcuda::wmma::mem_row_major);
    __syncwarp();
    // dx is row-major [M][C], so here a lane walks the CHANNEL
    for (int i = lane; i < 64; i += 32) {     // dx row-major: four channels
      int row = r * 16 + i / 4, c = (i % 4) * 4, e = (i / 4) * 16 + c;
      if (row >= nrow) continue;
      __nv_bfloat16* d = dx + (size_t)(r0 + row) * C + c0 + c;
      *(__nv_bfloat162*)(d) = __floats2bfloat162_rn(st[e], st[e + 1]);
      *(__nv_bfloat162*)(d + 2) = __floats2bfloat162_rn(st[e + 2], st[e + 3]);
    }
    __syncwarp();
  }
}

inline size_t combineDxSmem() {
  return (TM_CDX4 ? 4 : 2) * (size_t)BMG * LDG * 2 + (size_t)WG * 256 * 4 +
         (size_t)BMG * 4;
}

// ---------------------------------------------------------------- output gate
__global__ void outGateK(const __nv_bfloat16* __restrict__ o,
                         const gt2_t* __restrict__ t2,
                         __nv_bfloat16* __restrict__ out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(__bfloat162float(o[i]) * gt2Load(t2[i]));
}

__global__ void dOutGateK(const __nv_bfloat16* __restrict__ dout,
                          const __nv_bfloat16* __restrict__ o,
                          const gt2_t* __restrict__ t2,
                          __nv_bfloat16* __restrict__ dO,
                          __nv_bfloat16* __restrict__ dt2pre, size_t n,
                          const uint32_t* __restrict__ seedp = nullptr,
                          uint32_t depth = 0, uint32_t mod = 0,
                          float rate = 0.f, int N = 0, size_t p0 = 0) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = __bfloat162float(dout[i]);
  if (seedp) {
    size_t gi = p0 * C + i;
    int c = (int)(gi % C);
    int col = (int)((gi / C) % N);
    uint32_t idx = (uint32_t)(col * C + c);
    g = __bfloat162float(__float2bfloat16(g * dropScale(seedp, depth, mod, idx, rate)));
  }
  float s = gt2Load(t2[i]);
  dO[i] = __float2bfloat16(g * s);
  // d/d(pre_gl) of out = o * sigmoid(pre_gl)
  dt2pre[i] = __float2bfloat16(g * __bfloat162float(o[i]) * s * (1.f - s));
}

__global__ void f32ToBf16K(const float* __restrict__ a, __nv_bfloat16* __restrict__ b,
                           size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __float2bfloat16(a[i]);
}

__global__ void transposeWK(const float* __restrict__ a, __nv_bfloat16* __restrict__ b,
                            int R, int Cc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R * Cc) b[(size_t)(i % Cc) * R + i / Cc] = __float2bfloat16(a[i]);
}

// ------------------------------------------------------- the batched contraction
// Row-major C[Mo, No] = opA(A) @ opB(B), `batch` planes of each, stride M apart.
static void rbmm(cublasHandle_t h, bool ta, bool tb, int Mo, int No, int K,
                 const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* Cc,
                 int batch, long long stride) {
  float alpha = 1.f, beta = 0.f;
  CUB(cublasGemmStridedBatchedEx(
      h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, No, Mo, K,
      &alpha, B, CUDA_R_16BF, tb ? K : No, stride, A, CUDA_R_16BF, ta ? Mo : K,
      stride, &beta, Cc, CUDA_R_16BF, No, stride, batch, CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
}

// ---------------------------------------------------------------- the module
struct TriMul {
  Lt lt;
  cublasHandle_t cub;
  cudaStream_t st = nullptr;
  cudaGraphExec_t graph = nullptr;
  int N, M;
  bool outgoing;

  const float *ln_scale, *ln_offset, *cn_scale, *cn_offset;
  __nv_bfloat16 *x, *a, *bb, *t, *cn, *o, *out;
  gt2_t* t2;
  gate_t *gl, *gr;
  __nv_bfloat16 *dpre, *da, *db, *dt, *dcn, *dcnT, *dO, *dt2pre, *dx;
  __nv_bfloat16 *wcat, *wcatT, *wcatG, *wcatD, *bcat, *wo, *woT, *bo;
  float *mu, *rstd, *cmu, *crstd;

  // Every pointer setWeights() assigns, and nothing else.  A 48-deep stack
  // needs 48 sets of WEIGHTS but only one set of scratch, so it calls
  // setWeights() once a depth, snapshots these slots, and re-points them per
  // block -- which is why the split between the two groups above is load-bearing
  // rather than tidiness.
  // THE SCRATCH, in a fixed order, with the size each one was allocated at.
  //
  // This file is included TWICE -- tmul at C=I=128 for the Evoformer and the
  // extra stack, tmul64 at 64 for the template pair stack -- so the two
  // instantiations have the SAME member list in the SAME order, and every
  // entry in the wide one is at least as large as the narrow one's.  That is
  // what lets the template run out of the Evoformer's bytes: not an
  // allocator deciding which buffers look safe to lend, but this list,
  // element by element, with the size checked at the point of adoption.
  //
  // WEIGHTS ARE NOT HERE.  weightSlots() is the other list and the two must
  // stay disjoint: a weight is live for the whole process, scratch is live
  // for one module's call.
  std::vector<std::pair<void**, size_t>> scratchBuffers() {
    const size_t mc = (size_t)M * C, mi = (size_t)M * I, mp = (size_t)M * PRE;
    const size_t rc = tmRows * C, ri = tmRows * I;
    return {
      {(void**)&x, rc * 2}, {(void**)&a, mi * 2}, {(void**)&bb, mi * 2},
      {(void**)&t, mi * 2}, {(void**)&cn, ri * 2}, {(void**)&o, mc * 2},
      {(void**)&gl, mi * sizeof(gate_t)}, {(void**)&gr, mi * sizeof(gate_t)},
      {(void**)&t2, mc * sizeof(gt2_t)},
      {(void**)&dpre, split_cdx ? mp * 2 : 0},
      {(void**)&da, mi * 2}, {(void**)&db, mi * 2}, {(void**)&dt, mi * 2},
      {(void**)&dcn, transpose_dcn ? ri * 2 : 0}, {(void**)&dcnT, ri * 2},
      {(void**)&dO, rc * 2}, {(void**)&dt2pre, mc * 2}, {(void**)&dx, rc * 2},
      {(void**)&mu, (size_t)M * 4}, {(void**)&rstd, (size_t)M * 4},
      {(void**)&cmu, (size_t)M * 4}, {(void**)&crstd, (size_t)M * 4},
      {(void**)&out, need_out ? mc * 2 : 0},
    };
  }

  // Take the donor's bytes for every buffer in that list.  The donor is the
  // WIDE instantiation and this may be the narrow one, so the check is
  // `theirs >= mine` rather than equality -- and it aborts rather than
  // truncating, because a buffer that is too small by one element is a wrong
  // gradient that every shape check and every finiteness check passes.
  template <class Donor>
  void adoptScratchFrom(Donor& donor, const char* who) {
    auto mine = scratchBuffers();
    auto theirs = donor.scratchBuffers();
    if (mine.size() != theirs.size()) {
      fprintf(stderr, "%s: scratch lists differ in length, %zu vs %zu -- the "
              "two instantiations of trimul_body.cuh have diverged\n",
              who, mine.size(), theirs.size());
      exit(1);
    }
    for (size_t i = 0; i < mine.size(); ++i) {
      if (!mine[i].second) continue;          // this build does not use it
      // THE DONOR MAY NOT HAVE IT AT ALL, and that is not an error: the
      // Evoformer fuses the output gate into its residual add and so sets
      // need_out = false, while the template pair block reads `out`.  A
      // buffer the donor does not own is simply kept, and the saving is the
      // rest.  A donor buffer that exists but is TOO SMALL is a different
      // thing entirely and still aborts.
      if (!theirs[i].second) continue;
      if (theirs[i].second < mine[i].second) {
        fprintf(stderr, "%s: scratch[%zu] donor has %zu bytes, needs %zu\n",
                who, i, theirs[i].second, mine[i].second);
        exit(1);
      }
      if (*mine[i].first) CU(cudaFree(*mine[i].first));
      *mine[i].first = *theirs[i].first;
    }
  }

  std::vector<void**> weightSlots() {
    return {(void**)&ln_scale, (void**)&ln_offset, (void**)&cn_scale, (void**)&cn_offset, (void**)&wcat, (void**)&wcatT, (void**)&wcatG, (void**)&wcatD, (void**)&bcat, (void**)&wo, (void**)&woT, (void**)&bo};
  }

  bool split_cdx = envOn("TM_SPLIT");
  // MEASURED AND REJECTED: reading the cotangent row-major, as `gemm dcn`
  // produced it, deletes a 33.6 MB transpose through HBM and is still SLOWER
  // -- 1.562 against 1.503 ms on the module backward.  L1 does not absorb the
  // 256-byte-strided read the way the comment on centerLnBwdK hoped.  Same
  // shape of result as the fused centre LayerNorm above: `t` and `dcn` are
  // 16.8 MB each and live in L2, so the passes this was saving were cheap and
  // the access pattern it spent to save them was not.  TM_ROWMAJ=1 to re-run.
  bool transpose_dcn = !envOn("TM_ROWMAJ");
  // MEASURED AND REJECTED: one shared-staged pass moves 34.1 MB against the
  // three-pass path's 67.2 and is still SLOWER -- 1.228 against 1.047 ms on
  // the module forward.  Half the traffic and 17% worse: `t` is 16.8 MB and
  // stays in L2 between the three passes, so they were never paying for it,
  // while the staged version spends 18.4 KB of shared and a block-wide
  // reduction to save reads that were already free.  TM_CENTERF=1 to re-run.
  bool three_pass = !envOn("TM_CENTERF");

  // TM_ROWS: how many pair positions a chunk.  EVERYTHING EXCEPT THE
  // CONTRACTION IS ROW-WISE over M -- the two LayerNorms, the fused
  // projection, the centre norm, both output GEMMs and the combine all touch
  // one position at a time -- so a chunk of positions is exact and needs no
  // recompute.  The contraction is not: `t_c = A_c B_c^T` reads every row of
  // a channel's plane, so `a`, `bb`, `t`, `dt`, `da` and `db` STAY WHOLE and
  // the module runs as two chunk loops with the batched GEMM between them.
  // That is the whole reason this is safe where chunking the contraction
  // itself would need a cross-chunk accumulator.
  //
  // What it frees: the six buffers that live inside ONE loop -- `x` and `cn`
  // on the forward, `dO`, `dcnT` (and `dcn`) and `dx` on the backward -- 530
  // MB across the block's two instances at N=544.  `dt2pre` is produced in
  // the backward's first loop and read in its second, so it stays whole:
  // lifetime, not size, is what decides.
  //
  // THE UNIT IS A POSITION, NOT A ROW OF N -- M is N*N, so a sane chunk is a
  // FRACTION of M (M/4, M/8), not a small number.  TM_ROWS=128 at N=256 is
  // 512 chunks and runs 22x slower; the value is tens of thousands.
  //
  // AND IT IS ROUNDED TO 1024, which is TWO separate requirements and both
  // are hard faults rather than slowdowns:
  //
  //   * projGateK stores four consecutive positions of one channel as an
  //     8-byte pair and combineDxK reads them back the same way, so a chunk
  //     base that is not 4-aligned makes every one of those misaligned.
  //     TM_ROWS=97 dies with "misaligned address" inside whichever GEMM runs
  //     next.
  //   * combineDxK's vector loads are guarded only on the WRITE -- a partial
  //     final block reads BMG rows and keeps the ones it wants.  Unchunked
  //     that is safe because M = N*N with N a multiple of 32 is a multiple of
  //     1024 and every block is full.  A chunk whose length is not also a
  //     multiple of 1024 breaks that and reads past `a` by up to 63
  //     positions: TM_ROWS=12345 is an illegal access, not a wrong answer.
  //
  // Rounding to 1024 gives both, and leaves the tail chunk a multiple of 1024
  // as well.
  // Unset, it follows nbChunkOn(): M/16 above the threshold, whole below.
  // TM_ROWS overrides at any length; TM_ROWS=0 forces it off.
  static size_t tmChunk(size_t m, size_t n) {
    const char* e = getenv("TM_ROWS");
    size_t r = e ? (size_t)atoll(e) : (nbChunkOn(n) ? m / 16 : 0);
    r &= ~(size_t)1023;
    return (r > 0 && r < m) ? r : m;
  }
  size_t tmRows = 0;   // positions a chunk; M when unchunked

  // Set before alloc(): does this caller read `out`, or does it fuse the
  // output gate into its own residual add?  block.cuh fuses, tplblock.cuh
  // does not.  See the comment at the allocation.
  bool need_out = true;

  void alloc(int N_, bool out_) {
    N = N_; M = N * N; outgoing = out_;
    lt.init();
    CUB(cublasCreate(&cub));
    CU(cudaStreamCreate(&st));
    CUB(cublasSetStream(cub, st));
    size_t mc = (size_t)M * C, mi = (size_t)M * I;
    auto B = [&](__nv_bfloat16** p, size_t n) { CU(cudaMalloc(p, n * 2)); };
    // THE TWO NON-DEFAULT PATHS DO NOT CHUNK.  `TM_SPLIT` materialises the
    // 640-channel `dpre`, and `TM_CENTERF`'s fused centre norm indexes `t`
    // with the grid's own M; both are A/B fallbacks, and teaching them the
    // two-stride convention buys nothing.  Asking for both is not an error,
    // it just runs one chunk.
    tmRows = (split_cdx || !three_pass) ? (size_t)M : tmChunk((size_t)M, (size_t)N_);
    const size_t rc = tmRows * C, ri = tmRows * I;
    // MEASURED, NOT ASSUMED: computing `pre` in row blocks small enough to keep
    // the GEMM's output in L2 (so the gate kernel reads it back from cache
    // rather than HBM) is MONOTONICALLY WORSE -- 4.39 / 3.37 / 3.16 / 3.02 /
    // 3.00 / 2.94 / 2.91 ms at 1024 ... 65536 rows.  A GEMM over 2048 rows is
    // far enough off peak to swamp the traffic saved, and the gate kernel's own
    // streaming writes evict the block before it is read back.  One call it is.
    size_t mp = (size_t)M * PRE;
    // `pre` is GONE: projGateK never writes the 640-channel intermediate, and
    // the backward reads the two gates instead.  84 MB a module at N=256
    // against 34 for gl+gr.
    // `x` and `cn` live inside one chunk loop apiece -- see TM_ROWS
    B(&x, rc); B(&a, mi); B(&bb, mi); B(&t, mi); B(&cn, ri);
    CU(cudaMalloc(&gl, mi * sizeof(gate_t)));
    CU(cudaMalloc(&gr, mi * sizeof(gate_t)));
    B(&o, mc);
    // `out` is written only by outGateK, which runs only when the CALLER does
    // not fuse the output gate into its residual add -- and block.cuh fuses by
    // default, so the Evoformer never wants it.  tplblock.cuh does: 52.4 MB an
    // instance at N=640, 105 for its tmo+tmi pair.
    //
    // It used to be allocated on first use, to keep those megabytes off the
    // path that never touches them.  That lost the 617-640 band: by the time
    // the first forward runs, prealloc has handed the rest of the card to
    // XLA (14.93 GB taken, 3.19 left, XLA takes it) and a 105 MB cudaMalloc
    // in the middle of a step has nowhere to go -- `trimul_body.cuh:1407 cuda:
    // out of memory`, which is an ORDERING failure, not a size one.  Asking
    // at alloc() time, where there is room, is all it takes.
    //
    // `need_out` is the caller's promise about which it is.  The lazy branch
    // in forward() stays as the fallback for a caller that promised wrong
    // (NO_FUSE_OUTGATE flips block.cuh's answer after alloc is long gone).
    if (need_out) B(&out, mc); else out = nullptr;
    CU(cudaMalloc(&t2, mc * sizeof(gt2_t)));
    // THE COTANGENTS, and NOT when nothing will back-propagate.  This is
    // exactly the set adoptBackwardScratch() lends between the two
    // instances, which is the list of what only a backward reads -- the
    // forward names none of them.  ~450 MB of a tri-mul's 971 at N=608.
    if (nblean::on()) {
      dpre = nullptr; da = db = dt = nullptr; dcn = dcnT = nullptr;
      dO = dt2pre = dx = nullptr;
    } else {
    // `dpre` exists only for the split path: combineDxK contracts each group
    // out of shared and never materialises it.
    if (split_cdx) B(&dpre, mp); else dpre = nullptr;
    B(&da, mi); B(&db, mi); B(&dt, mi);
    // dcn (row-major) is the transpose path's staging and nothing else
    if (transpose_dcn) B(&dcn, ri); else dcn = nullptr;
    // dcnT is the channel-major copy of dcn; only the transpose path needs it
    B(&dcnT, ri);   // the channel-major cotangent, either way
    // dO and dx are chunk-local; dt2pre crosses from the first loop to the
    // second and cannot shrink
    B(&dO, rc); B(&dt2pre, mc); B(&dx, rc);
    }
    CU(cudaMalloc(&mu, M * 4)); CU(cudaMalloc(&rstd, M * 4));
    CU(cudaMalloc(&cmu, M * 4)); CU(cudaMalloc(&crstd, M * 4));

  }



  void setWeights(Oracle& o, const std::string& P) {
    auto dev = [&](const std::string& k) { o[k].to_device(); return o[k].dev; };
    ln_scale = dev(P + "left_norm_input/scale");
    ln_offset = dev(P + "left_norm_input/offset");
    cn_scale = dev(P + "center_norm/scale");
    cn_offset = dev(P + "center_norm/offset");
    // [Wp | Wg | Wgl] as one [C, 640] matrix, plus its transpose for the forward
    {
      float *wp = dev(P + "projection/weights"), *wg = dev(P + "gate/weights"),
            *wgl = dev(P + "gating_linear/weights");
      float* cat;
      CU(cudaMalloc(&cat, (size_t)C * PRE * 4));
      for (int r = 0; r < C; ++r) {
        CU(cudaMemcpy(cat + (size_t)r * PRE, wp + (size_t)r * 2 * I, 2 * I * 4,
                      cudaMemcpyDeviceToDevice));
        CU(cudaMemcpy(cat + (size_t)r * PRE + 2 * I, wg + (size_t)r * 2 * I, 2 * I * 4,
                      cudaMemcpyDeviceToDevice));
        CU(cudaMemcpy(cat + (size_t)r * PRE + 4 * I, wgl + (size_t)r * C, C * 4,
                      cudaMemcpyDeviceToDevice));
      }
      CU(cudaMalloc(&wcat, (size_t)C * PRE * 2));
      recordWeightPack(wcat);
      CU(cudaMalloc(&wcatT, (size_t)C * PRE * 2));
      recordWeightPack(wcatT);
      f32ToBf16K<<<((size_t)C * PRE + 255) / 256, 256>>>(cat, wcat, (size_t)C * PRE);
      transposeWK<<<((size_t)C * PRE + 255) / 256, 256>>>(cat, wcatT, C, PRE);
      CU(cudaFree(cat));
      // the fragment-contiguous copies: wcatG for projGateK (out of the
      // [C][PRE] form), wcatD for combineDxK (out of the [PRE][C] one)
      CU(cudaMalloc(&wcatG, (size_t)C * PRE * 2));
      recordWeightPack(wcatG);
      CU(cudaMalloc(&wcatD, (size_t)C * PRE * 2));
      recordWeightPack(wcatD);
      permuteFragK<<<((size_t)C * PRE + 255) / 256, 256>>>(wcat, wcatG, C, PRE, 0, PRE);
      permuteFragK<<<((size_t)C * PRE + 255) / 256, 256>>>(wcatT, wcatD, PRE, C, 0, C);

      float *bp = dev(P + "projection/bias"), *bg = dev(P + "gate/bias"),
            *bgl = dev(P + "gating_linear/bias");
      float* bcatf;
      CU(cudaMalloc(&bcatf, PRE * 4));
      CU(cudaMemcpy(bcatf, bp, 2 * I * 4, cudaMemcpyDeviceToDevice));
      CU(cudaMemcpy(bcatf + 2 * I, bg, 2 * I * 4, cudaMemcpyDeviceToDevice));
      CU(cudaMemcpy(bcatf + 4 * I, bgl, C * 4, cudaMemcpyDeviceToDevice));
      CU(cudaMalloc(&bcat, PRE * 2));
      recordWeightPack(bcat);
      f32ToBf16K<<<1, PRE>>>(bcatf, bcat, PRE);
      CU(cudaFree(bcatf));
    }
    {
      float* wof = dev(P + "output_projection/weights");
      CU(cudaMalloc(&wo, (size_t)I * C * 2));
      recordWeightPack(wo);
      CU(cudaMalloc(&woT, (size_t)I * C * 2));
      recordWeightPack(woT);
      f32ToBf16K<<<((size_t)I * C + 255) / 256, 256>>>(wof, wo, (size_t)I * C);
      transposeWK<<<((size_t)I * C + 255) / 256, 256>>>(wof, woT, I, C);
      CU(cudaMalloc(&bo, C * 2));
      recordWeightPack(bo);
      f32ToBf16K<<<1, C>>>(dev(P + "output_projection/bias"), bo, C);
    }
  }

  // What a caller can KEEP per depth: the centre LayerNorm's statistics.  They
  // are 0.4 MB a call and centerStatsK reads all 25.7 MB of `t` twice to
  // produce them, so keeping them is the cheapest 5.7 ms in the stack.
  std::vector<KeepSlot> keepSlots() {
    size_t mi = (size_t)M * I * 2, mc = (size_t)M * C * 2;
    // KEEP_ACTS: everything lnFwd + projGate would have to rebuild.  `mu`
    // and `rstd` come along because the input LayerNorm's backward reads
    // them, so without them lnFwd still has to run and the big five buy
    // nothing.
    return {{(void**)&cmu, (size_t)M * 4, KEEP_STATS},
            {(void**)&crstd, (size_t)M * 4, KEEP_STATS},
            {(void**)&a, mi, KEEP_ACTS},  {(void**)&bb, mi, KEEP_ACTS},
            {(void**)&gl, mi / 2 * sizeof(gate_t), KEEP_ACTS},
            {(void**)&gr, mi / 2 * sizeof(gate_t), KEEP_ACTS},
            {(void**)&t2, mc / 2 * sizeof(gt2_t), KEEP_ACTS},
            // `t` is the contraction's output; keeping it retires the replay's
            // batched GEMM.  It USED to ride along in KEEP_ACTS as "slightly
            // better value than the tier it joins" -- which stopped being true
            // when the three sigmoids went to uint8 and made the rest of the
            // set cheaper.  It is now the worst member, so it gets its own
            // tier and is dropped first.  See KeepTier.
            {(void**)&t, mi, KEEP_TMUL_T},
            {(void**)&mu, (size_t)M * 4, KEEP_ACTS},
            {(void**)&rstd, (size_t)M * 4, KEEP_ACTS}};
  }

  // THE TWO TRI-MULS CAN SHARE THEIR BACKWARD SCRATCH.  Every d* buffer here
  // is internal to backward(): its only output is the caller's `dpair`, and
  // nothing reads da/db/dt/dcnT/dO/dt2pre/dx after it returns.  The block runs
  // d:trimul_in to completion before d:trimul_out starts, on one stream, so
  // the later instance can hand its own back and use the earlier one's.  The
  // FORWARD intermediates cannot be shared this way and are not touched here:
  // the replay writes all nine modules' before the first backward reads any.
  // 7 x N^2 x I x 2 = 118 MB at N=256.
  void adoptBackwardScratch(const TriMul& o) {
    for (void* p : {(void*)da, (void*)db, (void*)dt, (void*)dcnT, (void*)dO,
                    (void*)dt2pre, (void*)dx, (void*)dcn, (void*)dpre})
      if (p) CU(cudaFree(p));
    da = o.da; db = o.db; dt = o.dt; dcnT = o.dcnT;
    dO = o.dO; dt2pre = o.dt2pre; dx = o.dx; dcn = o.dcn; dpre = o.dpre;
  }

  // RANGE PROBE (TM_RANGE=1): what the keep tier's bf16 tensors span, which
  // is what decides whether an 8-bit format needs a per-channel scale.
  // Measured over 96 calls (48 depths x two tri-muls): |a| median 14.4 max
  // 37, |bb| median 14.5 max 33.5, |t| ~4400 -- so a FIXED scale of 48 clears
  // a/bb with margin and needs no reduction pass.
  void reportRange(const char* tag) {
    if (!envOn("TM_RANGE")) return;
    static int seen = 0; if (seen++ >= 96) return;
    size_t n = (size_t)M * I;
    auto mx = [&](const __nv_bfloat16* p) {
      std::vector<__nv_bfloat16> h(n);
      CU(cudaMemcpy(h.data(), p, n * 2, cudaMemcpyDeviceToHost));
      double m = 0, s = 0;
      for (size_t i = 0; i < n; ++i) {
        double v = __bfloat162float(h[i]); m = fmax(m, fabs(v)); s += v * v;
      }
      return std::make_pair(m, sqrt(s / n));
    };
    auto A = mx(a), B = mx(bb), T = mx(t);
    fprintf(stderr, "TM_RANGE %s  a |max| %.4g rms %.4g | bb %.4g %.4g | "
            "t %.4g %.4g\n", tag, A.first, A.second, B.first, B.second,
            T.first, T.second);
  }

  // PROBE (TM_Q8, a BITMASK: 1 = a, 2 = bb, 4 = t): round the keep tier's
  // tensors to 8-bit in place, BEFORE the centre-LN so the statistics are
  // computed from the quantised values -- a first version quantised after and
  // the stack went NaN, which was the probe and not the idea.
  void quant8Probe() {
    const char* e = getenv("TM_Q8");
    int m = e ? atoi(e) : 0;
    if (!m) return;
    size_t n = (size_t)M * I;
    const char* sc = getenv("TM_Q8_SCALE");
    float hi = sc ? atof(sc) : 48.f;
    auto q = [&](__nv_bfloat16* p, float h) {
      quant8InPlaceK<<<(n + 255) / 256, 256, 0, st>>>(p, h / 127.f, n);
    };
    if (m & 1) q(a, hi);
    if (m & 2) q(bb, hi);
    if (m & 4) q(t, 8192.f);
  }

  // `haveStats` says cmu/crstd already hold this depth's values
  // `fusedOut` means the caller will apply the output gate itself, out of `o`
  // and `t2`, as part of its residual add -- so `out` is never written.
  // `keepGates` asks for the two sigmoids to be written as well, which only
  // the backward's replay needs -- the stack's own forward passes false and
  // saves 0.06 ms a call (0.423 against 0.482 at N=256).
  void forward(const __nv_bfloat16* pair, const float* mask,
               bool haveStats = false, bool fusedOut = false,
               bool keepGates = true, bool haveActs = false, bool haveT = false) {
    size_t mc = (size_t)M * C;
    // haveActs: a, bb, gl, gr, t2, mu and rstd are already this depth's, so
    // the input LayerNorm and the fused projection have nothing to produce.
    // haveT says the same of `t`, which is a SEPARATE tier -- the contraction
    // is skipped on its own, so a budget that can afford the projection set
    // but not the contraction's output still gets most of the saving.
    dim3 bt(TILE, 8);
    {
      static bool once = false;
      if (!once) {
        CU(cudaFuncSetAttribute((const void*)projGateK<true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)projGateSmem()));
        CU(cudaFuncSetAttribute((const void*)projGateK<false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)projGateSmem()));
        once = true;
      }
    }
    // LOOP ONE: the input LayerNorm and the fused projection, a chunk of
    // positions at a time.  `x` is chunk-sized and indexed from its base; the
    // channel-major outputs a/bb/gl/gr are offset by the chunk and keep the
    // GLOBAL stride M, which is what the extra `Mt` argument carries.  mask,
    // t2, mu and rstd are row-wise and simply offset.
    if (!haveActs)
    for (size_t p0 = 0; p0 < (size_t)M; p0 += tmRows) {
      const int pm = (int)(((size_t)M - p0 < tmRows) ? (size_t)M - p0 : tmRows);
      PROF_STEP("tm.f lnFwd",
        lnFwdK<<<(pm + 7) / 8, 256, 0, st>>>(pair + p0 * C, ln_scale, ln_offset, x,
                                             mu + p0, rstd + p0, pm));
      PROF_STEP("tm.f projGate",
        if (keepGates)
          projGateK<true><<<(pm + BMF - 1) / BMF, 32 * WF, projGateSmem(), st>>>(
              x, TM_FRAGPERM ? wcatG : wcat, bcat, mask + p0, a + p0, bb + p0,
              gl + p0, gr + p0, t2 + p0 * C, pm, M);
        else
          projGateK<false><<<(pm + BMF - 1) / BMF, 32 * WF, projGateSmem(), st>>>(
              x, TM_FRAGPERM ? wcatG : wcat, bcat, mask + p0, a + p0, bb + p0,
              nullptr, nullptr, t2 + p0 * C, pm, M);
        CU(cudaGetLastError()));
    }
    // outgoing  T_c = A_c B_c^T ;  incoming  T_c = B_c^T A_c
    if (!haveT)
    PROF_STEP("tm.f bmm",
      if (outgoing) rbmm(cub, false, true, N, N, N, a, bb, t, I, (long long)M);
      else          rbmm(cub, true, false, N, N, N, bb, a, t, I, (long long)M));
    quant8Probe();
    reportRange(outgoing ? "outgoing" : "incoming");
    // THE THREE-PASS PATH IS THE DEFAULT AND IT IS THE RIGHT ONE.  This
    // comment used to name a flag `TM_CENTER3` that does not exist and read
    // as though the FUSED kernel were the default; the flag is `TM_CENTERF`
    // and it opts IN to the fused one.  Measured, two alternating pairs at
    // N=256 depth 48:
    //
    //   three-pass     fwd+bwd 842.28 / 849.15 ms
    //   TM_CENTERF=1           903.46 / 910.68      +7%
    //
    // and the kernel table says why: `centerFusedK` is 85.34 ms over 192
    // calls against the three-pass's 23.06 (centerStats 7.24 + centerWrite
    // 15.82).  0.44 ms a call for a 33.6 MB pass is 76 GB/s, 16% of roof --
    // it loads a tile, reduces over it, and then cannot do anything else
    // with the SM while it does.  Fewer passes over the data is not the same
    // as less time.
    //
    // `haveStats` only means anything on this path: the fused kernel
    // recomputes the statistics out of a tile it has to load regardless,
    // which also makes KEEP_STATS -- the best tier on the board at 136 ms/GB
    // -- worth nothing there.
    // LOOP TWO: the centre LayerNorm and the output projection.  Same shape
    // as loop one -- `cn` is chunk-sized, `t` is offset and keeps stride M.
    if (three_pass) {
      for (size_t p0 = 0; p0 < (size_t)M; p0 += tmRows) {
        const int pm = (int)(((size_t)M - p0 < tmRows) ? (size_t)M - p0 : tmRows);
        dim3 gt((pm + TILE - 1) / TILE, I / TILE);
        PROF_STEP("tm.f centerLn",
          if (!haveStats)
            centerStatsK<<<(pm + 255) / 256, 256, 0, st>>>(t + p0, cmu + p0,
                                                           crstd + p0, pm, M);
          centerWriteK<<<gt, bt, 0, st>>>(t + p0, cn_scale, cn_offset, cmu + p0,
                                          crstd + p0, cn, pm, M));
        PROF_STEP("tm.f gemm out",
          lt.matmul(pm, C, I, cn, woT, o + p0 * C, CUDA_R_16BF,
                    CUBLASLT_EPILOGUE_BIAS, bo, nullptr, st));
      }
    } else {
      PROF_STEP("tm.f centerLn",
        centerFusedK<<<(M + BLKC - 1) / BLKC, 256, 0, st>>>(t, cn_scale, cn_offset,
                                                            cmu, crstd, cn, M));
      PROF_STEP("tm.f gemm out",
        lt.matmul(M, C, I, cn, woT, o, CUDA_R_16BF, CUBLASLT_EPILOGUE_BIAS, bo,
                  nullptr, st));
    }
    if (!fusedOut) {
      if (!out) CU(cudaMalloc(&out, mc * 2));
      PROF_STEP("tm.f outGate", outGateK<<<(mc + 255) / 256, 256, 0, st>>>(o, t2, out, mc));
    }
  }

  // dout f32 [N][N][C] -> dpair f32, both without the residual
  // `doutB` is the cotangent already narrowed by whoever produced it, so this
  // module skips its own f32ToBf16; `dpairB` asks it to narrow its own result
  // for the next one.  Both optional -- a standalone gate passes neither.
  template <class DT>
  void backward(const DT* dout, const __nv_bfloat16* pair, const float* mask,
                DT* dpair, bool acc = false,
                const uint32_t* drop_seed = nullptr, uint32_t drop_depth = 0,
                uint32_t drop_mod = 0, float drop_rate = 0.f) {
    dim3 bt(TILE, 8);
    // LOOP ONE, mirroring the forward's: the output gate's backward, the
    // output projection's, and the centre LayerNorm's.  `dO`, `dcnT`/`dcn`
    // and (on the f32 path) `dx` are chunk-sized and used at their base;
    // `dt2pre` is read by loop TWO and stays whole; `t` and `dt` are
    // channel-major at the global stride M.
    for (size_t p0 = 0; p0 < (size_t)M; p0 += tmRows) {
      const int pm = (int)(((size_t)M - p0 < tmRows) ? (size_t)M - p0 : tmRows);
      const size_t pc = (size_t)pm * C;
      dim3 gt((pm + TILE - 1) / TILE, I / TILE);
    const __nv_bfloat16* dxin;
    if constexpr (std::is_same<DT, __nv_bfloat16>::value) {
      dxin = dout + p0 * C;              // already the GEMM's width
    } else {
      PROF_STEP("tm.b f32ToBf16",
        f32ToBf16K<<<(pc + 255) / 256, 256, 0, st>>>(dout + p0 * C, dx, pc));
      dxin = dx;
    }
    PROF_STEP("tm.b dOutGate",
      dOutGateK<<<(pc + 255) / 256, 256, 0, st>>>(dxin, o + p0 * C, t2 + p0 * C,
                                                  dO, dt2pre + p0 * C, pc,
                                                  drop_seed, drop_depth,
                                                  drop_mod, drop_rate, N, p0));
    // CHANNEL-MAJOR STRAIGHT OUT OF THE GEMM.  centerLnBwdK reads its
    // cotangent as [I][M], to match `t`, and that used to be arranged by
    // writing dcn [M][I] and transposing it through HBM -- 33.6 MB each way,
    // which the byte-ratio rule said was 1.4x and therefore marginal.
    //
    // IT IS A WASH, MEASURED: module backward 1.349 ms with the transposed
    // GEMM against 1.350 with the explicit transpose.  **cuBLASLt's
    // transposed output is not free** -- writing D as M x Nn column-major
    // means each output element's k-loop neighbours are M elements apart, and
    // that scattered write costs about what the coalesced transpose kernel
    // cost. Kept because equal-and-simpler beats equal: one kernel and one
    // 16.8 MB buffer fewer. TM_TRANSPOSE=1 restores the round trip.
    PROF_STEP("tm.b gemm dcn",
      lt.matmul(pm, I, C, dO, wo, transpose_dcn ? (void*)dcn : (void*)dcnT,
              CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT, nullptr, nullptr, st,
              /*tout*/!transpose_dcn));
    PROF_STEP("tm.b centerLnBwd",
      // TM_TRANSPOSE restores the channel-major round trip, which is the A/B
      // this was decided on: the transpose is 33.6 MB of the step's 117.6.
      if (transpose_dcn)
        transposeRowToChanK<<<gt, bt, 0, st>>>(dcn, dcnT, pm, I);
      // `dcnT` is the chunk's own buffer, so its channel stride is pm, while
      // `t` and `dt` keep M.  Two strides, which is why centerLnBwdK takes
      // both: handing it one is a silently wrong read of 128 planes.
      centerLnBwdK<false><<<(pm + 255) / 256, 256, 0, st>>>(dcnT, t + p0, cn_scale,
                                                            cmu + p0, crstd + p0,
                                                            dt + p0, pm, M, pm));
    }
    PROF_STEP("tm.b bmm",
      if (outgoing) {
        // T = A B^T  ->  dA = dT B,  dB = dT^T A
        rbmm(cub, false, false, N, N, N, dt, bb, da, I, (long long)M);
        rbmm(cub, true, false, N, N, N, dt, a, db, I, (long long)M);
      } else {
        // T = B^T A  ->  dA = B dT,  dB = A dT^T
        rbmm(cub, false, false, N, N, N, bb, dt, da, I, (long long)M);
        rbmm(cub, false, true, N, N, N, a, dt, db, I, (long long)M);
      });
    // TM_SPLIT runs the old two-kernel path instead, which is the A/B this was
    // decided on and the fallback if a card ever disagrees.
    if (split_cdx) {
      dim3 gtM((M + TILE - 1) / TILE, I / TILE);
      PROF_STEP("tm.b combineTrans",
        combineTransK<<<gtM, bt, 0, st>>>(da, db, a, bb, gl, gr, dt2pre, mask, dpre,
                                         M, 0, M));
      PROF_STEP("tm.b gemm dx",
        lt.matmul(M, C, PRE, dpre, wcat, dx, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
                nullptr, nullptr, st));
    } else {
      // OVER 48 KB NEEDS ASKING.  TM_CDX4's four tiles are 78 KB a block, and
      // a launch that exceeds the default dynamic-shared limit FAILS SILENTLY
      // here -- the gradient came back cos 0.0727 with a plausible timing,
      // because a kernel that never runs is fast.  Nothing in this directory
      // was checking a launch's error code; the CU below is the fix for the
      // class, not just the case.
      static bool once = false;
      if (!once) {
        CU(cudaFuncSetAttribute((const void*)combineDxK,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)combineDxSmem()));
        once = true;
      }
      // LOOP TWO: the combine and the input LayerNorm's backward.  `dx` is
      // chunk-sized again -- loop one is finished with it.
      for (size_t p0 = 0; p0 < (size_t)M; p0 += tmRows) {
        const int pm = (int)(((size_t)M - p0 < tmRows) ? (size_t)M - p0 : tmRows);
        PROF_STEP("tm.b combineDx",
          combineDxK<<<(pm + BMG - 1) / BMG, 32 * WG, combineDxSmem(), st>>>(
              da + p0, db + p0, a + p0, bb + p0, gl + p0, gr + p0,
              dt2pre + p0 * C, mask + p0, TM_FRAGPERM ? wcatD : wcatT,
              dx, pm, M);
          CU(cudaGetLastError()));
        PROF_STEP("tm.b lnBwd",
          if (acc) lnBwdK<true, DT><<<(pm + 7) / 8, 256, 0, st>>>(
              dx, pair + p0 * C, ln_scale, mu + p0, rstd + p0, dpair + p0 * C, pm);
          else     lnBwdK<false, DT><<<(pm + 7) / 8, 256, 0, st>>>(
              dx, pair + p0 * C, ln_scale, mu + p0, rstd + p0, dpair + p0 * C, pm));
      }
      return;
    }
    // the TM_SPLIT path above falls through to here with whole buffers
    PROF_STEP("tm.b lnBwd",
      if (acc) lnBwdK<true, DT><<<(M + 7) / 8, 256, 0, st>>>(dx, pair, ln_scale, mu, rstd, dpair, M);
      else     lnBwdK<false, DT><<<(M + 7) / 8, 256, 0, st>>>(dx, pair, ln_scale, mu, rstd, dpair, M));
  }
};

}  // namespace TM_NS
