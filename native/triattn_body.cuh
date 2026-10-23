// AF2's triangle attention, as a reusable module -- the flash path only.
//
// Follows the block interface set in msacol.cuh: bf16 activations, float32
// gradients across the boundary, forward() writes its DELTA.
//
// The forward is localfold's tuned FlashAttention-2 (lf_flash.cuh) and the
// backward is flash_bwd.cuh.  The materialised path and the hand-rolled WMMA
// forward this directory used to carry are gone: they existed to gate the flash
// path, and that job is done.  dnbb is accumulated by atomics rather than
// reduced from a materialised score gradient -- 1.946 ms against 2.356 -- so
// nothing here writes an N x N tensor.
// NO `#pragma once`: included once per WIDTH by triattn.cuh.  TA_NS / TA_C
// pick the pass, and TA_FA / TA_FB name the flash build of the matching head
// dim -- see the note there.
#include "common.cuh"
#include "lt.cuh"
#include <type_traits>

#include "lf_flash.cuh"
#include "flash_bwd.cuh"

namespace TA_NS {

static const int C = TA_C;
static const int H = 4;
static const int D = C / H;
static const int QKVG = 4 * C;
static const int TOT = QKVG + H;
static const float LN_EPS = 1e-5f;
static const int TILE = 32;
// keys per LANE in the softmax kernels: one warp handles one row, so MAXJ * 32
// covers N <= 512
static const int MAXJ = 16;
static const int ROWS_PER_BLOCK = 4;   // 4 warps

__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + __expf(-v)); }

__device__ __forceinline__ float blockSum(float v, float* s, int tid, int nw) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((tid & 31) == 0) s[tid >> 5] = v;
  __syncthreads();
  v = 0.f;
  for (int i = 0; i < nw; ++i) v += s[i];
  __syncthreads();
  return v;
}

// Butterfly reductions: every lane ends with the result, so no broadcast and no
// shared memory.  The softmax rows are short and there are hundreds of
// thousands of them, so a whole block per row spent its time in __syncthreads.
__device__ __forceinline__ float warpSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warpMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ float blockMax(float v, float* s, int tid, int nw) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
  if ((tid & 31) == 0) s[tid >> 5] = v;
  __syncthreads();
  float m = -1e30f;
  for (int i = 0; i < nw; ++i) m = fmaxf(m, s[i]);
  __syncthreads();
  return m;
}

// ---------------------------------------------------------------- LayerNorm
__global__ void lnFwdK(const __nv_bfloat16* __restrict__ pair,
                       const float* __restrict__ scale,
                       const float* __restrict__ offset, __nv_bfloat16* __restrict__ x,
                       float* __restrict__ mu, float* __restrict__ rstd, int M) {
  int row = blockIdx.x, c = threadIdx.x;
  float v = __bfloat162float(pair[(size_t)row * C + c]);
  __shared__ float red[4];
  float m = blockSum(v, red, c, 4) / C;
  float d = v - m;
  float r = rsqrtf(blockSum(d * d, red, c, 4) / C + LN_EPS);
  x[(size_t)row * C + c] = __float2bfloat16(d * r * scale[c] + offset[c]);
  if (c == 0) { mu[row] = m; rstd[row] = r; }
}

// THE PROJECTION RUNS IN F16, NOT BF16, and this is why.  localfold's flash
// kernel takes half, so every value the attention touches was converted right
// after the bf16 GEMM that produced it -- `combToHalfK`, which moved 134 MB a
// call to change a dtype and nothing else, and measured 31.3 ms of the stack's
// forward (21% of the whole triangle attention, against the flash kernel's own
// 45.8) plus the same again inside the backward's replay.  Asking cuBLASLt for
// an f16 matmul instead costs nothing -- the accumulator is f32 either way --
// and lets D point straight at qkvgH, so the copy disappears rather than
// getting faster.  f16 over bf16 here is also the more accurate of the two: 10
// mantissa bits against 7, on values that were being rounded to f16 anyway one
// kernel later.  Its narrower exponent is not a risk for a LayerNorm output
// times a frozen AF2 weight.
// ONE WARP A ROW: C is 128, so a lane holds four channels and both reductions
// are shuffles -- no shared memory, no __syncthreads, where a block-per-row
// paid two block-wide barriers for each of 50176 rows.
template <bool TRANSPOSE = false>
__global__ void lnFwdHalfK(const __nv_bfloat16* __restrict__ pair,
                           const float* __restrict__ scale,
                           const float* __restrict__ offset,
                           half* __restrict__ x, float* __restrict__ mu,
                           float* __restrict__ rstd, int M, int N = 0, int cr0 = 0) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  int lane = threadIdx.x & 31;
  const int K = C / 32;
  const size_t base = (size_t)row * C;
  const size_t pbase = TRANSPOSE
      ? ((size_t)((uint32_t)row % (uint32_t)N) * (uint32_t)N + ((uint32_t)cr0 + (uint32_t)row / (uint32_t)N)) * C
      : base;
  float v[K], s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    v[k] = __bfloat162float(pair[pbase + lane + k * 32]);
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
    x[base + c] = __float2half(v[k] * r * scale[c] + offset[c]);
  }
  if (lane == 0) { mu[row] = m; rstd[row] = r; }
}

// The pair bias, [M][H] as the small GEMM leaves it -> [H][M] as flash wants
// it, AND MULTIPLIED BY log2(e): the kernel keeps its scores in the log2
// domain, so a bias handed to it in the natural one is wrong by that factor
// everywhere it is not zero.  Dropping it here (it used to live in
// nbbToHalfK) cost cos 0.9701 against 0.99998 -- the same failure that
// function's own comment records, re-earned.
__global__ void nbbRowToChanHalfK(const half* __restrict__ src,
                                  half* __restrict__ dst, size_t M) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= M) return;
#pragma unroll
  for (int h = 0; h < H; ++h)
    dst[(size_t)h * M + ij] =
        __float2half(fmaxf(__half2float(src[ij * H + h]) * LOG2E, -6e4f));
}

// [rows][cols] f32 -> [cols0][rows] f16: a weight in the [Nn, K] form the
// matmul wants, taking only the `cols0` columns starting at `c0`.
__global__ void transposeWHalfK(const float* __restrict__ src,
                                half* __restrict__ dst, int rows, int cols,
                                int c0, int cols0) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)rows * cols0) return;
  int r = i / cols0, c = i % cols0;
  dst[(size_t)c * rows + r] = __float2half(src[(size_t)r * cols + c0 + c]);
}

__global__ void f32ToHalfK(const float* __restrict__ src, half* __restrict__ dst,
                           size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2half(src[i]);
}

// ACC adds into `dpair` instead of overwriting it, so the block can hand its
// running gradient straight in.  The alternative -- write a scratch, then
// acc[i] += d[i] -- costs an extra 33 MB write and 33 MB read a call, and the
// stack makes 288 of them: 57.6 ms, measured with the NO_ACC probe, and accK
// is already at ~83% of bandwidth, so the only way to get it is to not do it.
// Aliasing with the incoming cotangent is safe: every backward in this
// directory reads `dout` once, at the top, into a bf16 copy.
template <bool ACC = false, class DT = float, bool TRANSPOSE = false>
__global__ void lnBwdK(const __nv_bfloat16* __restrict__ dx,
                       const __nv_bfloat16* __restrict__ pair,
                       const float* __restrict__ scale, const float* __restrict__ mu,
                       const float* __restrict__ rstd, DT* __restrict__ dpair,
                       int M, int N = 0) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  int lane = threadIdx.x & 31;
  const int K = C / 32;
  const size_t base = (size_t)row * C;
  const size_t pbase = TRANSPOSE
      ? ((size_t)((uint32_t)row % (uint32_t)N) * (uint32_t)N + ((uint32_t)row / (uint32_t)N)) * C
      : base;
  float r = rstd[row], mu_ = mu[row];
  float xh[K], dxh[K], s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = lane + k * 32;
    xh[k] = (__bfloat162float(pair[pbase + c]) - mu_) * r;
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
    size_t o = pbase + lane + k * 32;
    float v = r * (dxh[k] - mean_d - xh[k] * mean_dx);
    if (ACC) v += toF(dpair[o]);
    stF(dpair + o, v);
  }
}






// d(av) and d(gate pre-activation); the latter goes straight into the combined
// gradient's gate columns.
// IT READS AND WRITES HALF, which is what spared two whole kernels.  The gate
// sits between the flash forward's output (avH, half) and the flash backward's
// input (davH, half), so a bf16 version of it needs avH converted down on the
// way in and dav converted back up on the way out -- two passes over 8.4M
// elements whose only purpose was the dtype, 23 of the 34.2 ms this step cost.
// The gate pre-activation comes from qkvgH for the same reason.
// AND IT COMPUTES THE ROW DOT, which used to be its own kernel.
// TA_FB::rowDotK produced D[b,h,i] = sum_d dav[i,d] * av[i,d] by reading davH and
// avH again -- 25.7 MB a call that this kernel has just written and read.  The
// sum is over the 32 channels of ONE HEAD, which is exactly one warp's lanes
// here, so a warp reduction gets it for free: one block an (i,j), 128 threads,
// warp h owns head h.
__global__ void dGateK(const __nv_bfloat16* __restrict__ dag,
                       const half* __restrict__ avH,
                       const half* __restrict__ qkvg,
                       half* __restrict__ davH,
                       __nv_bfloat16* __restrict__ dcomb,
                       float* __restrict__ Dv, int N_, size_t M) {
  constexpr int C2 = C / 2, D2 = D / 2, RPB = 256 / C2;
  size_t ij = (size_t)blockIdx.x * RPB + threadIdx.x / C2;
  if (ij >= M) return;
  // ONE HEAD PER WARP WAS AN ASSUMPTION, NOT A FACT.  `h = c >> 5` and the
  // full-warp reduce below are both the statement D == 32.  At D = 16 two
  // heads share a warp: h collapsed them to one and the reduction summed
  // across the pair.  Segmenting by D is identical at D = 32 (D >> 1 == 16,
  // c % D == 0 <-> lane == 0) and correct for any D that divides the warp.
  int c2 = threadIdx.x % C2;
  int c = c2 * 2, h = c2 / D2, sub = c2 % D2;
  size_t i = ij * C + c;
  half2 q2 = *(const half2*)(qkvg + ij * QKVG + 3 * C + c);
  float g0 = sigmoidf(__half2float(q2.x));
  float g1 = sigmoidf(__half2float(q2.y));
  __nv_bfloat162 d2 = *(const __nv_bfloat162*)(dag + i);
  float d0 = __bfloat162float(d2.x);
  float d1 = __bfloat162float(d2.y);
  float dav0 = d0 * g0, dav1 = d1 * g1;
  *(half2*)(davH + i) = __floats2half2_rn(dav0, dav1);
  half2 av2 = *(const half2*)(avH + i);
  float av0 = __half2float(av2.x);
  float av1 = __half2float(av2.y);
  *(__nv_bfloat162*)(dcomb + ij * TOT + 3 * C + c) =
      __floats2bfloat162_rn(d0 * av0 * g0 * (1.f - g0),
                            d1 * av1 * g1 * (1.f - g1));
  // D = sum over this head's D channels of dav * av (preserving exact tree order)
  float s0 = dav0 * av0, s1 = dav1 * av1;
#pragma unroll
  for (int o = D2 >> 1; o; o >>= 1) {
    s0 += __shfl_down_sync(0xffffffff, s0, o);
    s1 += __shfl_down_sync(0xffffffff, s1, o);
  }
  if (sub == 0) {
    int b = (int)(ij / N_), r = (int)(ij % N_);
    Dv[((size_t)b * H + h) * N_ + r] = s0 + s1;
  }
}

// ag = sigmoid(gate) * av: the flash kernel's GATED output rebuilt from the
// UNGATED one it also returns.  That is what lets the backward's replay skip
// the flash forward entirely -- see `haveFlash` in forward().
__global__ void gateFromAvK(const half* __restrict__ avH,
                            const half* __restrict__ qkvg,
                            __nv_bfloat16* __restrict__ ag, size_t M) {
  size_t i2 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 >= M * (C / 2)) return;
  size_t ij = i2 / (C / 2);
  int c = (int)(i2 % (C / 2)) * 2;
  size_t i = i2 * 2;
  half2 g2 = *(const half2*)(qkvg + ij * QKVG + 3 * C + c);
  half2 av2 = *(const half2*)(avH + i);
  float g0 = sigmoidf(__half2float(g2.x));
  float g1 = sigmoidf(__half2float(g2.y));
  *(__nv_bfloat162*)(ag + i) =
      __floats2bfloat162_rn(__half2float(av2.x) * g0,
                            __half2float(av2.y) * g1);
}

// ------------------------------------------------- the ending node's transpose
// Swap the first two axes of [N][N][Cc], with the CHANNEL axis contiguous.
//
// THE OBVIOUS KERNEL IS UNCOALESCED ON BOTH SIDES.  One thread an (i,j),
// looping over c, puts consecutive threads a stride of N*Cc apart on the read
// and Cc apart on the write, so each of the Cc elements is its own memory
// transaction -- 128 of them a thread, none of them merged.  It is not a
// bandwidth problem that a faster card would fix: the traffic here is 33 MB and
// should take 0.08 ms, and it was taking 0.96 ms.  That alone made triangle
// attention's ENDING node cost 1.59x its starting node for identical work
// (123.5 ms against 77.6 ms over a 48-block stack), because the ending node is
// the starting node run on a transposed pair.
//
// Here a thread owns one 16-byte chunk of ONE channel row, and the threads of a
// row are consecutive, so each (i,j) row moves as a contiguous run on both
// sides.  The transpose is in the (i,j) INDEXING, which costs nothing, instead
// of in the access pattern, which costs everything.
__global__ void transposePairVecK(const uint4* __restrict__ src,
                                  uint4* __restrict__ dst, int N, int V) {
  int v = threadIdx.x;
  int j = blockIdx.x * blockDim.y + threadIdx.y;
  int i = blockIdx.y;
  if (j >= N || v >= V) return;
  dst[((size_t)i * N + j) * V + v] = src[((size_t)j * N + i) * V + v];
}

// The ending node's way back out: un-transpose the f32 gradient it computed
// internally into the caller's accumulator, at the caller's width.  DT is bf16
// on the pair track, so this is also where the gradient is narrowed -- there is
// no separate pass for it.
template <class DT>
__global__ void transposePairOutK(const float4* __restrict__ src,
                                  DT* __restrict__ dst, int N, int V, bool acc) {
  int v = threadIdx.x;
  int j = blockIdx.x * blockDim.y + threadIdx.y;
  int i = blockIdx.y;
  if (j >= N || v >= V) return;
  float4 a = src[((size_t)j * N + i) * V + v];
  size_t o = (((size_t)i * N + j) * V + v) * 4;   // elements, not float4s
  float x[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float val = x[k];
    if (acc) val += toF(dst[o + k]);
    stF(dst + o + k, val);
  }
}

template <class DT>
static inline void transposePairOut(const float* src, DT* dst, int N, bool acc,
                                    cudaStream_t st) {
  int V = C * 4 / 16;                        // float4s in one channel row
  int ty = 256 / V;
  dim3 blk(V, ty), grd((N + ty - 1) / ty, N);
  transposePairOutK<DT><<<grd, blk, 0, st>>>(
      reinterpret_cast<const float4*>(src), dst, N, V, acc);
}

// `rowBytes` is the size of one channel row and must be a multiple of 16: 256
// for bf16 x 128 channels, 512 for float x 128.
static inline void transposePairVec(const void* src, void* dst, int N,
                                    int rowBytes, cudaStream_t st) {
  int V = rowBytes / 16;
  int ty = V >= 256 ? 1 : 256 / V;          // ~256 threads a block
  dim3 blk(V, ty), grd((N + ty - 1) / ty, N);
  transposePairVecK<<<grd, blk, 0, st>>>(
      reinterpret_cast<const uint4*>(src), reinterpret_cast<uint4*>(dst), N, V);
}

// The [N][N] mask is only 256 KB at N=256, but the same uncoalesced read costs
// it the same order of magnitude it costs the pair, so it gets the standard
// shared-memory tile: read a 32x32 tile by rows, write it back by rows from the
// other axis.  +1 on the tile's second extent is the usual bank-conflict pad.
__global__ void transposeMaskK(const float* __restrict__ src,
                               float* __restrict__ dst, int N) {
  __shared__ float t[32][33];
  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  if (x < N && y < N) t[threadIdx.y][threadIdx.x] = src[(size_t)y * N + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x; y = blockIdx.x * 32 + threadIdx.y;
  if (x < N && y < N) dst[(size_t)y * N + x] = t[threadIdx.x][threadIdx.y];
}

// f32 gradients (dQ written, dK/dV accumulated by atomics) into the combined
// gradient's q/k/v columns.
__global__ void scatterQKVK(const float* __restrict__ dQ, const float* __restrict__ dK,
                            const float* __restrict__ dV, __nv_bfloat16* __restrict__ dcomb,
                            size_t M, int tot, int C_) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * (size_t)C_) return;
  size_t r = i / C_; int c = i % C_;
  // dQ is null when flashBwdK wrote it straight into dcomb as bf16
  if (dQ) dcomb[r * tot + c] = __float2bfloat16(dQ[i]);
  dcomb[r * tot + C_ + c] = __float2bfloat16(dK[i]);
  dcomb[r * tot + 2 * C_ + c] = __float2bfloat16(dV[i]);
}
// dnbb arrives as f32 from the kernel's atomics; land it in the combined
// gradient's bias columns.
__global__ void nbbScatterK(const float* __restrict__ dnbb,
                            __nv_bfloat16* __restrict__ dcomb, size_t M, int tot,
                            int off, int H_) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * (size_t)H_) return;
  size_t ij = i / H_; int h = i % H_;
  dcomb[ij * tot + off + h] = __float2bfloat16(dnbb[(size_t)h * M + ij]);
}


// -------------------------------------- the fused scatter + dX contraction
// The same shape of waste as tri-mul's `combineTrans` + `gemm dx`, and found
// the same way.  The two scatters land dK/dV/dnbb in the 516-wide combined
// gradient and the GEMM reads all of it back: 291 MB of traffic at N=256,
// measured at 0.528 ms a call -- 384 GB/s, at the roof -- for a contraction
// whose inputs are 102 MB and whose output is 17.
//
// This builds each of the five column groups in shared and contracts it
// straight away, so the K and V slices of `dcomb` are never written:
//
//     dX = sum over g  dcomb_g [BMS][K_g] @ wcatT[g's rows] [K_g][C]
//
// dQ and the gate gradient are already IN dcomb (flashBwdK and dGateK put
// them there) and are read from it; dK and dV come from the f32 atomic
// buffers directly.  `wcatT` is [TOT][C] row-major already -- a wmma
// matrix_b with ldm = C -- but TOT = 516 is not a multiple of 16, so it is
// allocated with TOTP = 528 rows and the tail zeroed; see setWeights.
static const int TOTP = ((TOT + 15) / 16) * 16;
#ifndef TA_BMS
#define TA_BMS 64
#endif
// MEASURED AND REJECTED with tri-mul's four-tile variant, and for the same
// reason: 78 KB is one block an SM, 3.260 ms against one tile's 3.073.
// THE PERMUTATION HELPS TRI-MUL AND HURTS HERE, measured: 3.147 against
// 3.034 ms on the module backward, the reverse of trimul's 1.562 against
// 1.592.  Both kernels read the same 16 weight rows from all eight warps, so
// in the ORIGINAL layout one 256-byte line serves every warp; the permuted
// layout gives each warp its own 512-byte run, fully used but shared with
// nobody.  scatterDxK contracts five groups and reuses those lines more than
// projGateK's five separate projections do, which is the only difference
// that survives.  So the flag is per module, not global.
#ifndef TA_FRAGPERM
#define TA_FRAGPERM 0
#endif

#ifndef TA_SDX4
#define TA_SDX4 0
#endif
static const int BMS = TA_BMS;
static const int WS = C / 16;       // warps, one per 16-column slice of C
static const int RTS = BMS / 16;
static const int LDS = C + 8;

// see TM_MINB in trimul.cuh: asking for blocks an SM directly was worth 5.8%
// of the tri-mul backward
// Measured at N=256, module forward+backward in ms:
//   TA_MINB   0 (no hint)      2          3          4
//   bwd            3.166      3.152      2.973      3.035
// 3 here against tri-mul's 2, because scatterDxK carries one tile where
// combineDxK carries two and has the registers to spare.
#ifndef TA_MINB
#define TA_MINB 3
#endif
#if TA_MINB
#define TA_LB(T) __launch_bounds__(T, TA_MINB)
#else
#define TA_LB(T) __launch_bounds__(T)
#endif
__global__ void TA_LB(32 * WS) scatterDxK(
    const __nv_bfloat16* __restrict__ dcomb, const float* __restrict__ dK,
    const float* __restrict__ dV, const float* __restrict__ dnbb,
    const long long* __restrict__ dnbbI,
    const __nv_bfloat16* __restrict__ wcatT, __nv_bfloat16* __restrict__ dx,
    size_t M) {
  extern __shared__ char smem[];
  // TA_SDX4: four tiles, so all four C-wide groups are BUILT before any is
  // contracted.  The same change in tri-mul's combineDxK took its module
  // backward 1.563 -> 1.028 ms AT ONE BLOCK AN SM, which says the barriers
  // and the memory level parallelism matter here and the occupancy does not.
  __nv_bfloat16* tile = (__nv_bfloat16*)smem;             // [TA_SDX4?4:1][BMS][LDS]
  float* stage = (float*)(tile + (size_t)(TA_SDX4 ? 4 : 1) * BMS * LDS);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const size_t r0 = (size_t)blockIdx.x * BMS;
  const int nrow = (int)min((size_t)BMS, M - r0);
  float* st = stage + (size_t)warp * 256;

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[RTS];
#pragma unroll
  for (int r = 0; r < RTS; ++r) nvcuda::wmma::fill_fragment(acc[r], 0.f);

  // KW is the group's width in W rows, always a multiple of 16
#define TA_CONTRACT(SRC, ROW0, KW)                                                  \
  do {                                                                         \
    __syncthreads();                                                           \
    for (int kk = 0; kk < (KW) / 16; ++kk) {                                   \
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16, \
                             nvcuda::wmma::row_major> fb;                      \
      nvcuda::wmma::load_matrix_sync(                                          \
          fb, TA_FRAGPERM                                                      \
                  ? wcatT + ((size_t)warp * TOTP + (ROW0) + kk * 16) * 16      \
                  : wcatT + (size_t)((ROW0) + kk * 16) * C + warp * 16,        \
          TA_FRAGPERM ? 16 : C);                                               \
      _Pragma("unroll") for (int r = 0; r < RTS; ++r) {                        \
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,             \
                               __nv_bfloat16, nvcuda::wmma::row_major> fa;     \
        nvcuda::wmma::load_matrix_sync(fa, (SRC) + (size_t)(r * 16) * LDS + kk * 16, LDS); \
        nvcuda::wmma::mma_sync(acc[r], fa, fb, acc[r]);                        \
      }                                                                        \
    }                                                                          \
  } while (0)

  // a C-wide group out of dcomb, at column offset `off`
  // two channels a lane: both sources are row-major, so the pair is one
  // 4-byte (bf16x2) or 8-byte (float2) access instead of two
  auto fromComb = [&](int off, __nv_bfloat16* tile, bool wait) {
    if (wait) __syncthreads();           // the last contraction still reads it
    for (int i = tid; i < BMS * C / 4; i += 32 * WS) {   // four channels a lane
      int row = i / (C / 4), c = (i % (C / 4)) * 4;
      const __nv_bfloat16* src = dcomb + (size_t)(r0 + row) * TOT + off + c;
      __nv_bfloat16* dst = tile + (size_t)row * LDS + c;
      *(uint2*)dst = row < nrow ? *(const uint2*)src : make_uint2(0u, 0u);
    }
  };
  auto fromF32 = [&](const float* src, __nv_bfloat16* tile, bool wait) {
    if (wait) __syncthreads();
    for (int i = tid; i < BMS * C / 4; i += 32 * WS) {   // four channels a lane
      int row = i / (C / 4), c = (i % (C / 4)) * 4;
      const float* sp = src + (size_t)(r0 + row) * C + c;
      __nv_bfloat16* dst = tile + (size_t)row * LDS + c;
      float4 v = row < nrow ? *(const float4*)sp : make_float4(0.f, 0.f, 0.f, 0.f);
      *(__nv_bfloat162*)(dst) = __floats2bfloat162_rn(v.x, v.y);
      *(__nv_bfloat162*)(dst + 2) = __floats2bfloat162_rn(v.z, v.w);
    }
  };

  __nv_bfloat16* t1 = tile + (size_t)(TA_SDX4 ? 1 : 0) * BMS * LDS;
  __nv_bfloat16* t2 = tile + (size_t)(TA_SDX4 ? 2 : 0) * BMS * LDS;
  __nv_bfloat16* t3 = tile + (size_t)(TA_SDX4 ? 3 : 0) * BMS * LDS;
  if (TA_SDX4) {
    fromComb(0, tile, false);            // dQ
    fromF32(dK, t1, false);              // dK
    fromF32(dV, t2, false);              // dV
    fromComb(3 * C, t3, false);          // the output gate
    TA_CONTRACT(tile, 0, C);  TA_CONTRACT(t1, C, C);
    TA_CONTRACT(t2, 2 * C, C); TA_CONTRACT(t3, 3 * C, C);
  } else {
    fromComb(0, tile, false);    TA_CONTRACT(tile, 0, C);
    fromF32(dK, tile, true);     TA_CONTRACT(tile, C, C);
    fromF32(dV, tile, true);     TA_CONTRACT(tile, 2 * C, C);
    fromComb(3 * C, tile, true); TA_CONTRACT(tile, 3 * C, C);
  }
  // dnbb is H = 4 wide and channel-major; the tile is zeroed out to 16 so the
  // padded rows of wcatT (which are zero) contract to nothing.
  __syncthreads();
  for (int i = tid; i < BMS * 16; i += 32 * WS) {
    int row = i / 16, c = i % 16;
    float v = 0.f;
    if (row < nrow && c < H) {
      size_t idx = (size_t)c * M + r0 + row;
      v = dnbbI ? (float)dnbbI[idx] * (1.0f / 1099511627776.0f) : dnbb[idx];
    }
    tile[(size_t)row * LDS + c] = __float2bfloat16(v);
  }
  TA_CONTRACT(tile, QKVG, 16);
#undef TA_CONTRACT

  const int c0 = warp * 16;
#pragma unroll
  for (int r = 0; r < RTS; ++r) {
    nvcuda::wmma::store_matrix_sync(st, acc[r], 16, nvcuda::wmma::mem_row_major);
    __syncwarp();
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

inline size_t scatterDxSmem() {
  return (size_t)(TA_SDX4 ? 4 : 1) * BMS * LDS * 2 + (size_t)WS * 256 * 4;
}

__global__ void halfToBf16K(const half* __restrict__ a, __nv_bfloat16* __restrict__ b,
                            size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __float2bfloat16(__half2float(a[i]));
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

static void gbmm(cublasHandle_t h, bool ta, bool tb, int Mo, int No, int K,
                 const __nv_bfloat16* A, int lda, long long sA,
                 const __nv_bfloat16* B, int ldb, long long sB,
                 __nv_bfloat16* Cc, int ldc, long long sC, int batch) {
  float alpha = 1.f, beta = 0.f;
  CUB(cublasGemmStridedBatchedEx(
      h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, No, Mo, K,
      &alpha, B, CUDA_R_16BF, ldb, sB, A, CUDA_R_16BF, lda, sA, &beta, Cc,
      CUDA_R_16BF, ldc, sC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// ---------------------------------------------------------------- the module
// dnbb's contribution to dx, added in place.  The chunked backward cannot use
// the fused scatterDxK: that one wants dnbb FINISHED while dKf/dVf are still
// the chunk's, and dnbb only finishes when every chunk has run.  Splitting the
// GEMM by columns is exact --
//     dx = dcomb[:, :QKVG] @ wcat[:QKVG]   (per chunk, in the loop)
//        + dcomb[:, QKVG:] @ wcat[QKVG:]   (once, here)
// -- and the second term is a rank-H update with H = 4, so it is cheaper as a
// kernel than as a GEMM into a 51 MB temp that then has to be added.
// dnbbF is [H][M]; wnbb is wcat's last H rows, [H][C].
__global__ void nbbAddDxK(const float* __restrict__ dnbbF,
                          const __nv_bfloat16* __restrict__ wnbb,
                          __nv_bfloat16* __restrict__ dx, size_t M, int Hh, int Cc,
                          int tot, int off) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * (size_t)Cc) return;
  const size_t m = i / Cc;
  const int c = (int)(i % Cc);
  // wcat is [C][TOT], NOT [TOT][C] -- the prep writes the raw param into wcat
  // and the TRANSPOSE into wcatT -- so the nbb weight for (h, c) is at
  // c * TOT + QKVG + h.  Reading it as [TOT][C] gives a gradient that is the
  // right size and the wrong direction: cos 0.8719, caught by the gate.
  float acc = 0.f;
  for (int h = 0; h < Hh; ++h)
    acc += dnbbF[(size_t)h * M + m] *
           __bfloat162float(wnbb[(size_t)c * tot + off + h]);
  dx[i] = __float2bfloat16(__bfloat162float(dx[i]) + acc);
}

struct TriAttn {
  Lt lt;
  cublasHandle_t cub;
  cudaStream_t st = nullptr;
  cudaGraphExec_t graph = nullptr;
  int N, M;
  bool ending;

  const float *ln_scale, *ln_offset;
  // `w`/`dw` were the materialised [N][H][N][N] attention weights of the
  // pre-flash implementation -- 134 MB EACH at N=256, and the flash kernel
  // never materialises a score matrix.  `comb` went when the projection moved
  // to f16 and wrote qkvgH directly; `av`/`dav` went when dGateK started
  // reading and writing half.  776 MB across the two triangle attentions,
  // allocated every run and touched by nothing.
  // `x` WAS HERE AND IT WAS DEAD.  The module allocated M*C bf16 -- 70 MB an
  // instance at N=522, 139 MB across the trunk's two and as much again in the
  // template stack -- and then never read or wrote it: the input LayerNorm
  // writes the HALF `xh`, which is already chunk-sized, and nothing else ever
  // wanted a bf16 copy.  It survived because an unused pointer costs nothing
  // but memory and memory is the thing nobody was counting.  Found while
  // listing what triangle attention holds at the length ceiling.
  __nv_bfloat16 *ag, *out;
  half *xh = nullptr, *wqkvgTh = nullptr, *bqkvgH = nullptr,
       *wnbbTh = nullptr, *nbbMH = nullptr;
  __nv_bfloat16 *dcomb, *dag, *dx;
  half *qkvgH, *biasH, *outH, *avH;   // localfold's flash kernel works in f16
  float* lseF;                        // its row log-sum-exp, in the log2 domain
  half* davH;
  float *dQf, *dKf, *dVf, *lseD, *dnbbF;  // f32: dK, dV and dnbb arrive by atomics
  // dnbb's int64 fixed-point twin; see nbbFixed in flash_bwd_body.cuh.
  // NE_NONDET_NBB=1 goes back to the float atomics, which are 1.4e-02
  // nondeterministic run to run and were the reason a design trajectory
  // could not be reproduced.
  long long* dnbbI = nullptr;
  __nv_bfloat16 *wcat, *wcatT, *wcatS, *bcat, *wo, *woT, *bo;
  float *mu, *rstd;
  // the ENDING node is the same computation on the transposed pair.  The old
  // driver transposed outside the struct; with the activation an argument the
  // module has to own it, or a block would have to know which orientation each
  // instance is -- which is exactly what the interface is meant to hide.
  __nv_bfloat16 *pairT = nullptr, *outT = nullptr;
  float *maskT = nullptr, *doutT = nullptr, *dpairT = nullptr;

  // Every pointer setWeights() assigns, and nothing else.  A 48-deep stack
  // needs 48 sets of WEIGHTS but only one set of scratch, so it calls
  // setWeights() once a depth, snapshots these slots, and re-points them per
  // block -- which is why the split between the two groups above is load-bearing
  // rather than tidiness.
  // EVERY POINTER setWeights ASSIGNS BELONGS HERE.  The stack holds one Block
  // and swaps 48 sets of weight pointers through this list, so a buffer that
  // setWeights allocates but weightSlots omits is silently shared by all 48
  // depths -- every block then runs the LAST block's weights for that one
  // module.  It gates green alone and the stack diverges (pair rms 173 against
  // the oracle's 63), which is how the f16 projection's three new buffers
  // announced themselves, and transition.cuh's six before them.
  // What a caller can KEEP per depth so the replay need not recompute it.
  // avH and lseF are the flash forward's outputs (34 ms over the stack for
  // 1.31 GB); biasH is the pair bias, 0.4 MB a call for the 4.5 ms of the
  // small GEMM and the channel-major scatter that build it -- by far the best
  // ratio in the module.
  std::vector<KeepSlot> keepSlots() {
    return {{(void**)&avH, (size_t)M * C * sizeof(half), KEEP_FLASH},
            {(void**)&lseF, (size_t)N * H * N * sizeof(float), KEEP_FLASH},
            {(void**)&biasH, (size_t)H * M * sizeof(half), KEEP_BIAS}};
  }

  std::vector<void**> weightSlots() {
    // EVERY BUFFER setWeights() ALLOCATES BELONGS HERE.  wcatS was left out
    // when it was added and the whole 48-deep stack ran block 47's attention
    // weights in its backward -- grad/pair cos 0.607 with the forward exact,
    // and both single-module gates green, because one depth cannot see it.
    return {(void**)&ln_scale, (void**)&ln_offset, (void**)&wcat, (void**)&wcatT,
            (void**)&wcatS, (void**)&bcat, (void**)&wo, (void**)&woT, (void**)&bo,
            (void**)&wqkvgTh, (void**)&wnbbTh, (void**)&bqkvgH};
  }

  int taRows = 0;   // rows a chunk; N when unchunked

  // ONE PLACE DECIDES HOW qkvgH IS INDEXED.  It is chunk-sized, so every
  // reader uses its base -- except when taRows == N, where the chunk IS the
  // whole thing and `cr0` is always 0 anyway.  Writing the offset out at each
  // of the five call sites is how the row-chunk bugs in this file happened.
  half* qkvgAt(int cr0) {
    return qkvg_chunked ? qkvgH : qkvgH + (size_t)cr0 * N * QKVG;
  }
  bool qkvg_chunked = false;

  // TA_ROWS=<n> sets it; 0/unset means one chunk of N, i.e. exactly what this
  // module did before.  Measured on the standalone gate at N=128: a chunk of
  // N/2 and N/4 are bit-identical to the unchunked run and cost 0% and 1.9%;
  // N/8 costs 12.7%.
  // Unset, it follows nbChunkOn(): N/4 above the threshold, whole below.
  // TA_ROWS overrides at any length; TA_ROWS=0 forces it off.
  static int taChunkRows(int n) {
    const char* e = getenv("TA_ROWS");
    // nbChunkOnAt, NOT nbChunkOn: the pin is deliberately not applied here.
    //
    // The claim above -- N/2 and N/4 bit-identical to the unchunked run --
    // holds at the Evoformer's width and NOT at the template's.  Gated
    // directly (realloc_gate GATE_TPL, 512 -> 128): with the chunk policy
    // pinned the template stack differs, with TA_ROWS=0 it is bit-identical,
    // and the full-step stage diff named `tpl_out` as the first stage to
    // diverge with the embedder before it matching exactly.  So a short
    // state must chunk as ITS length says, not as the allocation's does.
    //
    // The cost is that the row buffers grow when a short state runs
    // unchunked where the long one chunked -- 224 rows against 608/4 = 152.
    // That is one bounded fresh allocation, and the adopt in this file
    // re-aliases after it, because a reuse pass re-runs alloc() in order.
    const int r = e ? atoi(e) : (nbChunkOnAt((size_t)n) ? n / 4 : 0);
    return (r > 0 && r < n) ? r : n;
  }

  void alloc(int N_, bool ending_) {
    N = N_; M = N * N; ending = ending_;
    lt.init();
    CUB(cublasCreate(&cub));
    CU(cudaStreamCreate(&st));
    CUB(cublasSetStream(cub, st));
    size_t mc = (size_t)M * C, mt = (size_t)M * TOT;
    auto B = [&](__nv_bfloat16** p, size_t n) { CU(cudaMalloc(p, n * 2)); };
    // `ag` is CHUNK-SIZED.  It is the gated attention output on its way into
    // the output projection, and both ends of that journey now happen inside
    // one row chunk -- see flashRows and forward().  70 MB an instance at
    // N=522.  `out` is the module's output and stays whole.
    B(&ag, (size_t)(TA_SWAP ? N_ : taChunkRows(N_)) * N_ * C); B(&out, mc);
    // NOTHING BELOW IS ALLOCATED WHEN NOTHING WILL BACK-PROPAGATE.  The
    // set is adoptBackwardScratch()'s, which is this module's own
    // statement of what only a backward reads, plus lseD (the backward's
    // logsumexp derivative; lseF is the FORWARD's and stays) and dQf,
    // which is already dead.  No forward kernel names any of them.
    const bool lean = nblean::on();
    if (!lean) {
    B(&dcomb, (size_t)(TA_SWAP ? N_ : taChunkRows(N_)) * N_ * TOT); B(&dx, mc);   // dx stays whole: the
    // nbb term has to land in it before lnBwd, and that term needs every chunk
    // dag is chunk-sized: the output GEMM writes it and dGateK reads it, both
    // inside one row chunk.  dKf/dVf/dcomb cannot follow -- scatterDxK and
    // nbbScatterK consume them only once dnbb is complete, and dnbb sums over
    // EVERY row -- so they stay full.
    B(&dag, (size_t)(TA_SWAP ? N_ : taChunkRows(N_)) * N_ * C);
    } else {
      dcomb = dx = dag = nullptr;
    }
    // TA_ROWS: the module's scratch is sized by a ROW CHUNK, not by N.
    // Triangle attention is independent per row, so rows [r0, r0+rows) can be
    // run on their own; `xh` and `outH` are written and read inside the
    // FORWARD and never outlive it, so chunking them needs no replay.
    // (qkvgH, avH, lseF are read again by the backward and stay full.)
    // TA_SWAPPROBE prices a STRIDED read of the whole qkvgH, so it cannot
    // run against a chunk-sized one.  Pin it to a single chunk, the way
    // trimul pins TM_SPLIT and TM_CENTERF, rather than teach a probe the
    // chunked layout.
    taRows = TA_SWAP ? N : taChunkRows(N);
    const size_t mcChunk = (size_t)taRows * N * C;
    CU(cudaMalloc(&xh, mcChunk * sizeof(half)));
    CU(cudaMalloc(&nbbMH, (size_t)M * H * sizeof(half)));
    // `qkvgH` IS CHUNK-SIZED: 279 MB an instance at N=522, the largest thing
    // this module holds and the one the length ceiling runs into.  It can
    // shrink because forward()'s pass two produces and consumes it in one
    // loop, and because the backward REPLAYS it per chunk rather than reading
    // the forward's.  It is NOT a keep slot, which avH and lseF are -- those
    // are declared to the stack at full size and must stay there.
    // TA_REPLAY, AND IT IS OFF BY DEFAULT ON PURPOSE.  Chunking qkvgH means
    // the backward must REPLAY the forward for each chunk, and that costs 7%
    // at the lengths where chunking is on (512: 7269 -> 7776 ms).  It frees
    // 279 MB an instance and MOVES the blocker out of this module -- at 522
    // the OOM goes from triattn:772 to the pair transition -- but it does not
    // yet buy a single residue, and a 7% regression for an unrealised benefit
    // is not something to ship.  Turn it on when something downstream cashes
    // the 558 MB in.
    qkvg_chunked = (taRows < N_) && envOn("TA_REPLAY");
    CU(cudaMalloc(&qkvgH, (size_t)(qkvg_chunked ? taRows : N_) * N_ * 4 * C * sizeof(half)));
    CU(cudaMalloc(&biasH, (size_t)H * M * sizeof(half)));
    CU(cudaMalloc(&outH, mcChunk * sizeof(half)));
    CU(cudaMalloc(&avH, mc * sizeof(half)));
    CU(cudaMalloc(&lseF, (size_t)N * H * N * sizeof(float)));
    // davH likewise: dGateK writes it, the flash backward reads it, one chunk.
    if (!lean)
      CU(cudaMalloc(&davH, (size_t)(TA_SWAP ? N_ : taChunkRows(N_)) * N_ * C * sizeof(half)));
    else
      davH = nullptr;
    // dQf IS DEAD AND WAS STILL ALLOCATED.  flashBwdK writes dQ through the
    // LAST of its three destinations -- `if (dQb) ... else if (dcomb) ... else
    // dQ[...]` -- and every call here passes dcomb as dQb, so the `dQ`
    // parameter is never written.  scatterDxK takes dK, dV and dnbb but not
    // dQ, and the TA_SPLIT path passes nullptr for it too.  M*C*4 = 33.6 MB
    // an instance, 67 MB a block.  nullptr rather than removed, so the
    // parameter still documents what the kernel can do.
    dQf = nullptr;
    // dKf/dVf/dcomb are chunk-sized too.  They are per-row, and with the
    // column-split GEMM below the scatter that consumes them runs INSIDE the
    // chunk, so nothing waits for dnbb.  ~413 MB a module at N=448.
    if (!lean) {
    CU(cudaMalloc(&dKf, (size_t)taRows * N * C * 4));
    CU(cudaMalloc(&dVf, (size_t)taRows * N * C * 4));
    CU(cudaMalloc(&lseD, (size_t)N * H * N * 4));
    CU(cudaMalloc(&dnbbF, (size_t)H * M * 4));
    if (!envOn("NE_NONDET_NBB")) CU(cudaMalloc(&dnbbI, (size_t)H * M * 8));
    } else {
      dKf = dVf = lseD = dnbbF = nullptr; dnbbI = nullptr;
    }
    CU(cudaMalloc(&mu, M * 4)); CU(cudaMalloc(&rstd, M * 4));

  }


  // THE WHOLE STEP AS ONE GRAPH.  A design run replays one shape for thousands
  // of iterations, so every launch configuration is known once and the host
  // never has to issue them again.  This is a property of the problem -- fixed
  // sequence length, frozen weights -- not of the kernels.

  // THE SCRATCH, in a fixed order, with the size each was allocated at.
  // See the long note on scratchBuffers() in trimul_body.cuh: this file is
  // also included twice -- tattn at C=128/D=32, tattn64 at C=64/D=16 -- so
  // the two member lists correspond element by element.  Several entries are
  // the SAME size at both widths rather than larger, because H = 4 either
  // way; adoptScratchFrom asks for >=, not >.
  //
  // Entries that are null on this build report 0 and are skipped: dQf is
  // deliberately dead, dnbbI is behind NE_NONDET_NBB, and the orientation
  // buffers exist only on the `ending` instance.
  std::vector<std::pair<void**, size_t>> scratchBuffers() {
    const size_t mc = (size_t)M * C, mt = (size_t)M * TOT;
    const size_t chunkRows = (size_t)(TA_SWAP ? N : taChunkRows(N));
    const size_t mcChunk = (size_t)taRows * N * C;
    return {
      {(void**)&ag, chunkRows * N * C * 2}, {(void**)&out, mc * 2},
      {(void**)&dcomb, dcomb ? chunkRows * N * TOT * 2 : 0}, {(void**)&dx, dx ? mc * 2 : 0},
      {(void**)&dag, dag ? chunkRows * N * C * 2 : 0},
      {(void**)&xh, mcChunk * sizeof(half)},
      {(void**)&nbbMH, (size_t)M * H * sizeof(half)},
      {(void**)&qkvgH, (size_t)(qkvg_chunked ? taRows : N) * N * 4 * C * sizeof(half)},
      {(void**)&biasH, (size_t)H * M * sizeof(half)},
      {(void**)&outH, mcChunk * sizeof(half)},
      {(void**)&avH, mc * sizeof(half)},
      {(void**)&lseF, (size_t)N * H * N * sizeof(float)},
      {(void**)&davH, davH ? chunkRows * N * C * sizeof(half) : 0},
      {(void**)&dKf, dKf ? (size_t)taRows * N * C * 4 : 0},
      {(void**)&dVf, dVf ? (size_t)taRows * N * C * 4 : 0},
      {(void**)&lseD, lseD ? (size_t)N * H * N * 4 : 0},
      {(void**)&dnbbF, dnbbF ? (size_t)H * M * 4 : 0},
      {(void**)&dnbbI, dnbbI ? (size_t)H * M * 8 : 0},
      {(void**)&mu, (size_t)M * 4}, {(void**)&rstd, (size_t)M * 4},
      {(void**)&pairT, 0},
      {(void**)&outT, 0},
      {(void**)&maskT, (ending && maskT) ? (size_t)M * 4 : 0},
    };
  }

  template <class Donor>
  void adoptScratchFrom(Donor& donor, const char* who) {
    auto mine = scratchBuffers();
    auto theirs = donor.scratchBuffers();
    if (mine.size() != theirs.size()) {
      fprintf(stderr, "%s: scratch lists differ in length, %zu vs %zu -- the "
              "two instantiations of triattn_body.cuh have diverged\n",
              who, mine.size(), theirs.size());
      exit(1);
    }
    for (size_t i = 0; i < mine.size(); ++i) {
      if (!mine[i].second || !theirs[i].second) continue;
      if (theirs[i].second < mine[i].second) continue;
      if (*mine[i].first) CU(cudaFree(*mine[i].first));
      *mine[i].first = *theirs[i].first;
    }
  }

  void allocOrientation() {
    if (!ending) return;
    CU(cudaMalloc(&maskT, (size_t)M * 4));
  }

  // Same argument as tri-mul's: every d* here is internal to backward(),
  // whose only output is the caller's dpair, and d:triattn_end finishes before
  // d:triattn_start begins.  dKf/dVf/dnbbF are memset at the top of each
  // backward and accumulated by atomics within it, so sharing them is safe in
  // the same way.  The transposed pair (doutT/dpairT) is NOT shared: only the
  // ending instance has it.  186 MB at N=256.
  void adoptBackwardScratch(const TriAttn& o) {
    for (void* p : {(void*)dcomb, (void*)dag, (void*)dx, (void*)davH,
                    (void*)dKf, (void*)dVf, (void*)dnbbF, (void*)dnbbI})
      if (p) CU(cudaFree(p));
    dcomb = o.dcomb; dag = o.dag; dx = o.dx; davH = o.davH;
    dKf = o.dKf; dVf = o.dVf; dnbbF = o.dnbbF; dnbbI = o.dnbbI;
  }


  void setWeights(Oracle& o, const std::string& P) {
    auto dev = [&](const std::string& k) { o[k].to_device(); return o[k].dev; };
    ln_scale = dev(P + "query_norm/scale");
    ln_offset = dev(P + "query_norm/offset");
    // [Wq | Wk | Wv | Wg | W2d] as one [C, 4C+H] matrix.  query_w is pre-scaled by
    // key_dim^-0.5, which AF2 applies to q after the projection.
    {
      float *qw = dev(P + "attention/query_w"), *kw = dev(P + "attention/key_w"),
            *vw = dev(P + "attention/value_w"), *gw = dev(P + "attention/gating_w"),
            *w2 = dev(P + "feat_2d_weights");
      std::vector<float> host((size_t)C * TOT, 0.f);
      std::vector<float> h_q(o[P + "attention/query_w"].host);
      auto& t_k = o[P + "attention/key_w"]; auto& t_v = o[P + "attention/value_w"];
      auto& t_g = o[P + "attention/gating_w"]; auto& t_2 = o[P + "feat_2d_weights"];
      float sc = 1.f / sqrtf((float)D);
      for (int r = 0; r < C; ++r) {
        for (int c = 0; c < C; ++c) {
          host[(size_t)r * TOT + c] = h_q[(size_t)r * C + c] * sc;
          host[(size_t)r * TOT + C + c] = t_k.host[(size_t)r * C + c];
          host[(size_t)r * TOT + 2 * C + c] = t_v.host[(size_t)r * C + c];
          host[(size_t)r * TOT + 3 * C + c] = t_g.host[(size_t)r * C + c];
        }
        for (int h = 0; h < H; ++h)
          host[(size_t)r * TOT + QKVG + h] = t_2.host[(size_t)r * H + h];
      }
      (void)qw; (void)kw; (void)vw; (void)gw; (void)w2;
      float* d;
      CU(cudaMalloc(&d, host.size() * 4));
      CU(cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
      CU(cudaMalloc(&wcat, host.size() * 2));
      recordWeightPack(wcat);
      // wcatT IS PADDED TO TOTP = 528 ROWS AND ZEROED.  scatterDxK contracts
      // the H = 4 bias columns as a 16-wide wmma tile, so it reads W rows
      // 512..527; the 12 that do not exist must be zero, not whatever the
      // allocator left there.
      CU(cudaMalloc(&wcatT, (size_t)TOTP * C * 2));
      recordWeightPack(wcatT);
      CU(cudaMemset(wcatT, 0, (size_t)TOTP * C * 2));
      f32ToBf16K<<<(host.size() + 255) / 256, 256>>>(d, wcat, host.size());
      transposeWK<<<(host.size() + 255) / 256, 256>>>(d, wcatT, C, TOT);
      // the fragment-contiguous copy scatterDxK reads: ldm = 16 instead of
      // C, so a weight fragment is 512 contiguous bytes and not 16 sectors
      // 256 bytes apart.  Built off the PADDED wcatT, so the tail stays zero.
      if (TA_FRAGPERM) {
        CU(cudaMalloc(&wcatS, (size_t)TOTP * C * 2));
      recordWeightPack(wcatS);
        permuteFragK<<<((size_t)TOTP * C + 255) / 256, 256>>>(wcatT, wcatS, TOTP,
                                                              C, 0, C);
      } else {
        wcatS = nullptr;   // still a slot, so the per-depth swap stays honest
      }
      // the same weight split in two and in f16, for the forward's two GEMMs.
      // `wcat`/`wcatT` stay because the BACKWARD's dX = dY W reads the combined
      // bf16 form, where the gradient arrives as one [M, TOT] block.
      CU(cudaMalloc(&wqkvgTh, (size_t)QKVG * C * sizeof(half)));
      recordWeightPack(wqkvgTh);
      CU(cudaMalloc(&wnbbTh, (size_t)H * C * sizeof(half)));
      recordWeightPack(wnbbTh);
      transposeWHalfK<<<((size_t)C * QKVG + 255) / 256, 256>>>(d, wqkvgTh, C, TOT, 0, QKVG);
      transposeWHalfK<<<((size_t)C * H + 255) / 256, 256>>>(d, wnbbTh, C, TOT, QKVG, H);
      CU(cudaFree(d));

      // only the gate carries a bias
      std::vector<float> hb(TOT, 0.f);
      auto& gb = o[P + "attention/gating_b"];
      for (int c = 0; c < C; ++c) hb[3 * C + c] = gb.host[c];
      float* db;
      CU(cudaMalloc(&db, TOT * 4));
      CU(cudaMemcpy(db, hb.data(), TOT * 4, cudaMemcpyHostToDevice));
      CU(cudaMalloc(&bcat, TOT * 2));
      recordWeightPack(bcat);
      f32ToBf16K<<<1, TOT>>>(db, bcat, TOT);
      CU(cudaMalloc(&bqkvgH, QKVG * sizeof(half)));
      recordWeightPack(bqkvgH);
      f32ToHalfK<<<(QKVG + 255) / 256, 256>>>(db, bqkvgH, QKVG);
      CU(cudaFree(db));
    }
    {
      float* wof = dev(P + "attention/output_w");   // [H, D, C] == [C, C]
      CU(cudaMalloc(&wo, (size_t)C * C * 2));
      recordWeightPack(wo);
      CU(cudaMalloc(&woT, (size_t)C * C * 2));
      recordWeightPack(woT);
      f32ToBf16K<<<((size_t)C * C + 255) / 256, 256>>>(wof, wo, (size_t)C * C);
      transposeWK<<<((size_t)C * C + 255) / 256, 256>>>(wof, woT, C, C);
      CU(cudaMalloc(&bo, C * 2));
      recordWeightPack(bo);
      f32ToBf16K<<<1, C>>>(dev(P + "attention/output_b"), bo, C);
    }
  }

  // pair [N][N][C] bf16 -> out [N][N][C] bf16, the delta
  // `haveFlash` says avH and lseF ALREADY hold this depth's values, so the
  // flash forward can be skipped.  Only the stack's backward passes it: the
  // replay exists to rebuild what the backward reads, and avH/lseF are the one
  // part of that which is expensive to recompute (34 ms over the stack) and
  // cheap to keep (13.6 MB a call at N=224).  The stack owns a per-depth slot
  // for each and swaps them in use(), exactly as it does the weights.
  static inline const bool TA_SWAP = envOn("TA_SWAPPROBE");
  // `flashRows` WAS HERE.  It ran the flash over row chunks and widened
  // outH into `ag`, with the qkvg GEMM still outside it in a pass of its own.
  // forward()'s pass two now does both, plus the GEMM, so that `qkvgH` can be
  // chunk-sized -- which it could not be while one loop produced it and
  // another consumed it.  `git log` has the original.


  void forward(const __nv_bfloat16* pairIn, const float* maskIn,
               bool haveFlash = false, bool haveBias = false,
               bool keepFlash = true, bool fusedTransposeOut = false,
               bool haveMaskT = false) {
    const __nv_bfloat16* pair = pairIn;
    const float* mask = maskIn;
    if (ending) {
      if (!fusedTransposeOut && !outT) CU(cudaMalloc(&outT, (size_t)M * C * 2));
      if (!haveMaskT) {
        dim3 mg((N + 31) / 32, (N + 31) / 32);
        transposeMaskK<<<mg, dim3(32, 32), 0, st>>>(maskIn, maskT, N);
      }
      mask = maskT;
    }
    size_t mc = (size_t)M * C;
    LF_STREAM = st;
    // ROW CHUNKS.  xh is chunk-sized, so the LN and the two GEMMs that read it
    // run per chunk; mu/rstd outlive the forward (lnBwd reads them) and are
    // offset instead.  With TA_ROWS unset this is one pass over all N rows.
    // PASS ONE: THE BIAS, AND ONLY THE BIAS.  `biasH` is [H][N][N] and the
    // flash reads all of it for any row, so every row's nbbMH has to exist
    // before the first flash can run.  That is the one thing in this forward
    // that cannot be done a chunk at a time, so it gets its own pass and
    // everything else gets to share one -- which is what lets `qkvgH`, the
    // 279 MB an instance that used to span the two, be chunk-sized.
    //
    // It costs a second LayerNorm over the pair when the bias is not kept.
    // When it IS kept (KEEP_BIAS) this pass does not run at all.
    if (!haveBias) {
      for (int cr0 = 0; cr0 < N; cr0 += taRows) {
        const int crows = (N - cr0 < taRows) ? (N - cr0) : taRows;
        const size_t Mc = (size_t)crows * N;
        PROF_STEP("ta.f lnFwd",
          if (ending)
            lnFwdHalfK<true><<<(Mc + 7) / 8, 256, 0, st>>>(pair,
              ln_scale, ln_offset, xh, mu + (size_t)cr0 * N, rstd + (size_t)cr0 * N, Mc, N, cr0);
          else
            lnFwdHalfK<false><<<(Mc + 7) / 8, 256, 0, st>>>(pair + (size_t)cr0 * N * C,
              ln_scale, ln_offset, xh, mu + (size_t)cr0 * N, rstd + (size_t)cr0 * N, Mc));
        PROF_STEP("ta.f gemm nbb",
          lt.matmul((int)Mc, H, C, xh, wnbbTh, nbbMH + (size_t)cr0 * N * H,
                    CUDA_R_16F, CUBLASLT_EPILOGUE_DEFAULT, nullptr, nullptr, st));
      }
    }
    // (Nn = QKVG makes D a row-major [M, 4C] f16 block, which IS qkvgH's
    // layout, so the flash kernel reads the GEMM's output with no copy; the
    // GEMM itself now runs inside the chunk loop above.)
    // the pair bias is its own GEMM only because it would otherwise widen the
    // one above past the 4C stride flash assumes.  H = 4 columns: 33 MFLOP.
    // THE BIAS IS ITS OWN TIER.  It used to ride on haveFlash, which was right
    // only because the keep was all-or-nothing.  It is 0.52 MB a call against
    // avH's 16.8 and worth 104 ms/GB against 22, so the stack can now afford
    // it at lengths where it cannot afford the flash outputs -- and the
    // BACKWARD reads biasH either way, so it is rebuilt here whenever it is
    // not held, flash kept or not.
    if (!haveBias) {
      PROF_STEP("ta.f nbbToChan",
        nbbRowToChanHalfK<<<(M + 255) / 256, 256, 0, st>>>(nbbMH, biasH, M));
      if (envOn("NO_NBB"))
        CU(cudaMemsetAsync(biasH, 0, (size_t)H * M * sizeof(half), st));
    }
    // q is already scaled by key_dim^-0.5 inside Wq, so the kernel's scale is 1.
    // Calling flashGridHalfAt directly is how the two extra outputs (avOut,
    // lseOut) are reached without patching every dispatch layer -- and it is
    // also how the warp count gets chosen.
    //
    // EIGHT WARPS, NOT THE FOUR THE HEURISTIC PICKS.  flashGridHalfLaunch sets
    // warps=4 whenever rows >= 32, which is right for the shapes it was tuned
    // on; on this one it is not.  Measured over three runs of 30 reps, module
    // forward 0.883 +- 0.002 ms against 0.918 +- 0.002 -- 3.8%, and reproducible
    // (a single run of this gate has ~7% noise, so it takes repeats to see).
    // BK 32 and 64 were both worse than 48, and localfold's register-staged
    // form (LOCALFOLD_FLASH_REG=1, the pre-Ampere path) is worse than the
    // cp.async one it picks by default here: 0.991 against 0.883 ms.
    // PASS TWO: qkvg, the attention, the widen and the projection, all on
    // the same rows.  THE CALLER OFFSETS THE DATA POINTERS: inside the flash
    // the row is `rl = blockIdx.y % rowsHere` and outAt()/base index with rl,
    // NOT with the absolute `r = cr0 + rl` -- cr0 reaches only the mask and
    // the bias.  Passing base pointers for those made every chunk write from
    // row 0 and the last one win, which the gate caught as cos 0.717.
    //
    // TA_SWAPPROBE is not reachable from here: it prices a strided read of
    // the WHOLE qkvgH, so alloc() pins taRows to N when it is on.
    for (int cr0 = 0; cr0 < N; cr0 += taRows) {
      const int crows = (N - cr0 < taRows) ? (N - cr0) : taRows;
      const size_t Mc = (size_t)crows * N, mcc = Mc * C;
      if (haveBias || taRows < N) {
        PROF_STEP("ta.f lnFwd",
          if (ending)
            lnFwdHalfK<true><<<(Mc + 7) / 8, 256, 0, st>>>(pair,
              ln_scale, ln_offset, xh, mu + (size_t)cr0 * N, rstd + (size_t)cr0 * N, Mc, N, cr0);
          else
            lnFwdHalfK<false><<<(Mc + 7) / 8, 256, 0, st>>>(pair + (size_t)cr0 * N * C,
              ln_scale, ln_offset, xh, mu + (size_t)cr0 * N, rstd + (size_t)cr0 * N, Mc));
      }
      PROF_STEP("ta.f gemm qkvg",
        lt.matmul((int)Mc, QKVG, C, xh, wqkvgTh, qkvgAt(cr0),
                  CUDA_R_16F, CUBLASLT_EPILOGUE_BIAS, bqkvgH, nullptr, st));
      if (!haveFlash) {
        half* av_dst = keepFlash ? (avH + (size_t)cr0 * N * C) : nullptr;
        float* lse_dst = keepFlash ? (lseF + (size_t)cr0 * H * N) : nullptr;
        PROF_STEP("ta.f flash",
          TA_SWAP
            ? flashGridHalfAt<D, 8, 48>(qkvgAt(cr0), biasH, N, mask, /*out*/nullptr, N, H,
                                   cr0, crows, false, 1.f, nullptr,
                                   av_dst, lse_dst,
                                   /*rs*/ (size_t)4 * H * 32, /*ps*/ (size_t)N * 4 * H * 32,
                                   /*ors*/ (size_t)H * 32, /*ops*/ (size_t)N * H * 32, ag)
            : flashGridHalfAt<D, 8, 48>(qkvgAt(cr0), biasH, N, mask, /*out*/nullptr, N, H,
                                   cr0, crows, false, 1.f, nullptr,
                                   av_dst, lse_dst, 0, 0, 0, 0, ag));
      } else {
        // outH was never written this pass, so rebuild the gated output from avH
        PROF_STEP("ta.f gateFromAv",
          gateFromAvK<<<(mcc / 2 + 255) / 256, 256, 0, st>>>(
              avH + (size_t)cr0 * N * C, qkvgAt(cr0), ag, Mc));
      }
      PROF_STEP("ta.f gemm out",
        lt.matmul((int)Mc, C, C, ag, woT,
                  ((ending && !fusedTransposeOut) ? outT : out) + (size_t)cr0 * N * C, CUDA_R_16BF,
                  CUBLASLT_EPILOGUE_BIAS, bo, nullptr, st));
    }
    if (ending && !fusedTransposeOut) transposePairVec(outT, out, N, C * 2, st);
  }

  // The backward against the flash forward: nothing but lse was kept, so P is
  // recomputed inside the kernel.  dS is materialised for one reason only --
  // dnbb reduces over the batch axis the grid is parallel in.
  // dout f32 -> dpair f32, both without the residual
  // `doutB` is the cotangent already narrowed by its producer and `dpairB`
  // asks for the same service for the next module.  For the ENDING node doutB
  // is worth more than the narrowing it saves: the orientation transpose can
  // then move 25.7 MB of bf16 instead of 51.4 MB of f32.
  // DT is the cotangent's width.  On the pair track it is bf16, which the
  // ending node turns to advantage twice: its orientation transpose moves half
  // the bytes, and the narrowing on the way out happens inside the
  // un-transpose rather than in a pass of its own.
  template <class DT>
  void backward(const DT* doutIn, const __nv_bfloat16* pairIn, const float* maskIn,
                DT* dpairOut, bool acc = false, bool alreadyTransposed = false) {
    const __nv_bfloat16* pair = pairIn;
    const float* mask = maskIn;
    const void* dout = doutIn;
    constexpr bool narrow = std::is_same<DT, __nv_bfloat16>::value;
    const __nv_bfloat16* dxin = nullptr;
    if (ending) {
      if constexpr (narrow) {
        if (alreadyTransposed) {
          dxin = reinterpret_cast<const __nv_bfloat16*>(doutIn);
        } else {
          // transpose the bf16 cotangent straight into dx: half the bytes of the
          // f32 transpose, and no narrowing pass at all
          PROF_STEP("ta.b transpose", transposePairVec(doutIn, dx, N, C * 2, st));
          dxin = dx;
        }
      } else {
        if (!doutT) CU(cudaMalloc(&doutT, (size_t)M * C * 4));
        transposePairVec(doutIn, doutT, N, C * 4, st);
        dout = doutT;
      }
      mask = maskT;
    } else if constexpr (narrow) {
      dxin = reinterpret_cast<const __nv_bfloat16*>(doutIn);
    }
    if (!dxin) {
      PROF_STEP("ta.b f32ToBf16",
        f32ToBf16K<<<((size_t)M * C + 255) / 256, 256, 0, st>>>(
            reinterpret_cast<const float*>(dout), dx, (size_t)M * C));
      dxin = dx;
    }
    size_t mc = (size_t)M * C;
    LF_STREAM = st;
    // (the output GEMM runs inside the row-chunk loop below)
    // (dGateK runs inside the row-chunk loop below)
    // cudaMemsetAsync, and NOT the vectorised kernel beside it.  These three
    // are 6.54 GB of zeroing over a fwd+bwd and PROBE_NOZERO prices them at
    // 26.3 ms.  A float4 zeroing kernel (TA_ZEROK=1, zeroFast in common.cuh)
    // is 3.6 ms FASTER standalone and 3.4 ms SLOWER in the model -- see the
    // README.  The model is what ships.
    const bool probe = envOn("NO_DKV") || envOn("NO_DNBB") || envOn("TA_ZEROK");
    if (!envOn("PROBE_NOZERO")) {
      if (!envOn("TA_ZEROK")) {
        if (dnbbI) CU(cudaMemsetAsync(dnbbI, 0, (size_t)H * M * 8, st));
        else       CU(cudaMemsetAsync(dnbbF, 0, (size_t)H * M * 4, st));
      } else {
        zeroFast(dnbbF, (size_t)H * M * 4, st);
        if (dnbbI) CU(cudaMemsetAsync(dnbbI, 0, (size_t)H * M * 8, st));
      }
    }
    {
      static bool once = false;
      if (!once) {
        // THE INSTANTIATION HAS TO MATCH THE ONE LAUNCHED.  This used to name
        // flashBwdK<true,true,true> while the launch below is
        // <true,true,false,true> -- a different function, so the attribute
        // landed on a kernel nothing runs.  Harmless only because bwdSmem() is
        // 30.5 KB and fits under the 48 KB default.
        CU(cudaFuncSetAttribute((const void*)TA_FB::flashBwdK<true, true, false, true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)TA_FB::bwdSmem()));
        // ASKING FOR MAX-SHARED CARVEOUT IS SLOWER, measured: 3.450 against
        // 3.330 ms on the module backward.  At 30.5 KB a block the driver's
        // default already fits three blocks an SM, and the extra L1 is worth
        // more than the shared memory would be.
        once = true;
      }
      dim3 g((N + TA_FB::BM - 1) / TA_FB::BM, H, N);
      // dS is NOT materialised: dnbb comes straight out of the tile by atomic
      // accumulation.  Measured at N=256: 1.946 ms against 2.356 for writing the
      // 134 MB score gradient and reducing it in a second pass, even though the
      // contention is N_b deep.  The earlier reasoning that dnbb forces dS into
      // memory was wrong, and nothing else reads dS.
      // MEASURED: writing dQ and atomically accumulating dK/dV straight into
      // dcomb as bf16x2 -- which would retire three f32 buffers and the scatter
      // below -- costs the kernel 1.946 -> 3.527 ms and the whole module 4.93 ->
      // 6.22, slower than the materialised path it replaces.  The gradient was
      // unchanged (cos 0.99993), so this is throughput, not correctness: the
      // f32 atomic into a packed [M][128] buffer beats a bf16x2 atomic into a
      // 516-wide one, and only half the lanes carry a pair.
      // NO_DNBB is a PROBE, not an option: it drops the bias gradient, so the
      // gradient it produces is wrong.  It exists to price the atomic, whose
      // contention is N deep -- every one of the N pair rows accumulates into
      // the same [h][i][j].  Measured at N=256 depth 48: the table below.
      // NO_DKV is a PROBE like NO_DNBB: it drops dK and dV, so the gradient is
      // wrong.  It prices the dK/dV atomics, which the roofline says are 57%
      // of this kernel's memory traffic (268 MB of 470) because BM=64 splits
      // each (b,h) into four query tiles that all accumulate into the same
      // keys.
      // THE TWO PROBES BELOW STILL LAUNCH THE WHOLE ROW GRID, and davH is
      // chunk-sized now, so they would read past it.  They produce a wrong
      // gradient by design and exist only to price the atomics; refuse the
      // combination rather than let it read out of bounds.
      if (taRows < N && (envOn("NO_DKV") || envOn("NO_DNBB"))) {
        fprintf(stderr, "triattn: NO_DKV/NO_DNBB price the atomics on the "
                        "whole row grid and cannot run with TA_ROWS; unset "
                        "one of them.\n");
        abort();
      }
      if (envOn("NO_DKV")) {
        PROF_STEP("ta.b flashBwd",
          TA_FB::flashBwdK<false, false, false, true><<<g, 32 * TA_FB::WARPS, TA_FB::bwdSmem(), st>>>(
            qkvgH, 4 * C, biasH, davH, lseF, lseD, mask, nullptr, dQf, dKf, dVf, N, H,
            dnbbF, dnbbI, nullptr, 0, dcomb, TOT));
      } else if (envOn("NO_DNBB")) {
        PROF_STEP("ta.b flashBwd",
          TA_FB::flashBwdK<true, true, false, false><<<g, 32 * TA_FB::WARPS, TA_FB::bwdSmem(), st>>>(
            qkvgH, 4 * C, biasH, davH, lseF, lseD, mask, nullptr, dQf, dKf, dVf, N, H,
            nullptr, nullptr, nullptr, 0, dcomb, TOT));
      } else
      // ONE ROW CHUNK AT A TIME.  dag and davH are chunk-sized, so the GEMM
      // that fills dag, the gate that turns it into davH, and the flash that
      // reads davH all run together on the same rows.  The kernel takes the
      // row from blockIdx.z and indexes qkvg/mask/lse/dK/dV/dcomb by it, so
      // the pointers are offset and gridDim.z is the chunk; `bias` is
      // [H][N][N] and shared across rows, and dnbb ACCUMULATES over rows, so
      // both are left whole.  dnbb's int64 fixed point (9c986ad) is what makes
      // that accumulation independent of the order the chunks arrive in.
      for (int cr0 = 0; cr0 < N; cr0 += taRows) {
        const int crows = (N - cr0 < taRows) ? (N - cr0) : taRows;
        const size_t Mc = (size_t)crows * N;
        // REPLAY THE FORWARD FOR THESE ROWS.  qkvgH, avH and lseF are the
        // three the backward still reads whole, and they are 353 MB an
        // instance at N=522 -- the largest thing left in this module and the
        // reason 522 does not fit.  Recomputing them a chunk at a time is
        // what lets them shrink: LN -> qkvg GEMM -> flash forward, exactly
        // the three steps forward() runs, on exactly these rows.
        //
        // `biasH` is NOT replayed: it is [H][N][N], shared across every row,
        // and only 2.2 MB at N=522, so it stays whole and the forward's copy
        // is still good.  mu/rstd are rewritten with the values they already
        // hold, which lnBwd reads after the loop.
        //
        // Only when chunking: unchunked there is one chunk and the replay
        // would be a whole second forward for nothing.
        if (qkvg_chunked) {
          PROF_STEP("ta.b replay lnFwd",
            if (ending)
              lnFwdHalfK<true><<<(Mc + 7) / 8, 256, 0, st>>>(
                  pair, ln_scale, ln_offset, xh,
                  mu + (size_t)cr0 * N, rstd + (size_t)cr0 * N, Mc, N, cr0);
            else
              lnFwdHalfK<false><<<(Mc + 7) / 8, 256, 0, st>>>(
                  pair + (size_t)cr0 * N * C, ln_scale, ln_offset, xh,
                  mu + (size_t)cr0 * N, rstd + (size_t)cr0 * N, Mc));
          PROF_STEP("ta.b replay qkvg",
            lt.matmul((int)Mc, QKVG, C, xh, wqkvgTh,
                      qkvgAt(cr0), CUDA_R_16F,
                      CUBLASLT_EPILOGUE_BIAS, bqkvgH, nullptr, st));
          PROF_STEP("ta.b replay flash",
            flashGridHalfAt<D, 8, 48>(
                qkvgAt(cr0), biasH, N, mask, /*out*/nullptr, N, H,
                cr0, crows, false, 1.f, nullptr,
                avH + (size_t)cr0 * N * C, lseF + (size_t)cr0 * H * N));
        }
        PROF_STEP("ta.b gemm dag",
          lt.matmul((int)Mc, C, C, dxin + (size_t)cr0 * N * C, wo, dag, CUDA_R_16BF,
                    CUBLASLT_EPILOGUE_DEFAULT, nullptr, nullptr, st));
        constexpr int DG_RPB = 256 / (C / 2);
        PROF_STEP("ta.b gate",
          dGateK<<<(Mc + DG_RPB - 1) / DG_RPB, 256, 0, st>>>(
            dag, avH + (size_t)cr0 * N * C,
            qkvgAt(cr0), davH, dcomb,
            lseD + (size_t)cr0 * H * N, N, Mc));
        if (!envOn("PROBE_NOZERO")) {
          CU(cudaMemsetAsync(dKf, 0, (size_t)Mc * C * 4, st));
          CU(cudaMemsetAsync(dVf, 0, (size_t)Mc * C * 4, st));
        }
        dim3 gc((N + TA_FB::BM - 1) / TA_FB::BM, H, crows);
        PROF_STEP("ta.b flashBwd",
          TA_FB::flashBwdK<true, true, false, true><<<gc, 32 * TA_FB::WARPS, TA_FB::bwdSmem(), st>>>(
            qkvgAt(cr0), 4 * C, biasH, davH,
            lseF + (size_t)cr0 * H * N, lseD + (size_t)cr0 * H * N,
            mask + (size_t)cr0 * N, nullptr, dQf, dKf, dVf, N, H,
            dnbbF, dnbbI, nullptr, 0, dcomb, TOT));
        // THE SCATTER AND ITS GEMM COME INSIDE THE CHUNK.  dcomb/dKf/dVf hold
        // only these rows, so they must be consumed before the next chunk --
        // which is exactly what lets them be chunk-sized.  The nbb columns of
        // dcomb are still zero here; their contribution is added to dx after
        // the loop, once dnbb is complete.
        if (taRows < N) {
          PROF_STEP("ta.b scatterQKV",
            scatterQKVK<<<((size_t)Mc * C + 255) / 256, 256, 0, st>>>(
                nullptr, dKf, dVf, dcomb, Mc, TOT, C));
          PROF_STEP("ta.b gemm dx",
            lt.matmul((int)Mc, C, TOT, dcomb, wcat, dx + (size_t)cr0 * N * C,
                      CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT, nullptr, nullptr, st));
        }
      }
    // fixed point -> float, once, in a kernel nothing else races with
    // (when scatterDxK runs below, it reads and zeros dnbbI directly)
    if (dnbbI && (taRows < N || envOn("TA_SPLIT") || probe))
      TA_FB::nbbFromFixedK<<<(int)(((size_t)H * M + 255) / 256), 256, 0, st>>>(
          dnbbI, dnbbF, (size_t)H * M);
    }
    // CHUNKED: the scatter and its GEMM already ran, inside each chunk, on the
    // QKVG columns alone.  All that is left is dnbb's term, which could not be
    // added until every chunk had contributed to it.
    if (taRows < N) {
      PROF_STEP("ta.b nbbAddDx",
        nbbAddDxK<<<(size_t)(mc + 255) / 256, 256, 0, st>>>(
            dnbbF, wcat, dx, M, H, C, TOT, QKVG));
      if (!dnbbI && !envOn("PROBE_NOZERO"))
        CU(cudaMemsetAsync(dnbbF, 0, (size_t)H * M * 4, st));
    } else
    // TA_SPLIT runs the old scatter-then-GEMM pair, which is the A/B this was
    // decided on and the fallback if a card ever disagrees.
    if (envOn("TA_SPLIT")) {
      PROF_STEP("ta.b scatter",
        nbbScatterK<<<((size_t)M * H + 255) / 256, 256, 0, st>>>(dnbbF, dcomb, M, TOT,
                                                              QKVG, H);
        scatterQKVK<<<(mc + 255) / 256, 256, 0, st>>>(nullptr, dKf, dVf, dcomb, M, TOT, C));
      PROF_STEP("ta.b gemm dx",
        lt.matmul(M, C, TOT, dcomb, wcat, dx, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
                nullptr, nullptr, st));
    } else {
      // see the note in trimul.cuh: four tiles is 78 KB, over the default
      // dynamic-shared limit, and the failed launch was silent
      static bool once2 = false;
      if (!once2) {
        CU(cudaFuncSetAttribute((const void*)scatterDxK,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)scatterDxSmem()));
        once2 = true;
      }
      PROF_STEP("ta.b scatterDx",
        scatterDxK<<<(size_t)(M + BMS - 1) / BMS, 32 * WS, scatterDxSmem(), st>>>(
            dcomb, dKf, dVf, dnbbF, probe ? nullptr : dnbbI,
            TA_FRAGPERM ? wcatS : wcatT, dx, M);
        CU(cudaGetLastError()));
    }
    PROF_STEP("ta.b lnBwd",
      if (ending) {
        if (acc)
          lnBwdK<true, DT, true><<<(M + 7) / 8, 256, 0, st>>>(dx, pair, ln_scale, mu, rstd, dpairOut, M, N);
        else
          lnBwdK<false, DT, true><<<(M + 7) / 8, 256, 0, st>>>(dx, pair, ln_scale, mu, rstd, dpairOut, M, N);
      } else if (acc)
        lnBwdK<true, DT, false><<<(M + 7) / 8, 256, 0, st>>>(dx, pair, ln_scale, mu, rstd, dpairOut, M);
      else
        lnBwdK<false, DT, false><<<(M + 7) / 8, 256, 0, st>>>(dx, pair, ln_scale, mu, rstd, dpairOut, M));
  }

};
}  // namespace TA_NS
