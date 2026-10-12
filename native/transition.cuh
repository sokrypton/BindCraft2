// AF2's transition as a reusable module -- the seventh and last Evoformer
// module type, and the one that appears in both tracks: the pair transition is
// [128 -> 512 -> 128] over N*N rows, the MSA transition the same module at
// [256 -> 1024 -> 256] over S*N rows.  Width is a template parameter only
// because it is the LayerNorm block's thread count; everything else is runtime.
//
//   out = Linear_4C_to_C( relu( Linear_C_to_4C( LayerNorm(x) ) ) )
//
// The block interface is msacol.cuh's: bf16 activations, float32 gradients
// across the boundary, forward() writes its DELTA and the caller keeps the
// residual, and the LayerNorm backward recomputes from the bf16 input plus the
// saved mean/inverse rather than keeping a float32 copy.
//
// WHAT THE BACKWARD DOES NOT DO.  Design freezes the weights, so every backward
// GEMM is dX = dY W^T alone -- no X^T dY.  That is half a trainer's backward
// FLOPs, and it means the 4C-wide relu output never has to be kept for dW.  It
// is kept anyway, but only as the relu's own output, which the mask pass reads
// in one streaming sweep; asking cuBLASLt for the sign bitmask instead costs it
// a 5.6x slower algorithm (0.199 ms with RELU_BIAS against 1.098 with
// RELU_AUX_BIAS on the same GEMM).
#pragma once
#include <mma.h>

#include "common.cuh"
#include <type_traits>

#include "lt.cuh"

namespace trans {

static const float LN_EPS = 1e-5f;  // common_modules.LayerNorm default

template <int NW>
__device__ __forceinline__ float blockSum(float v, float* s, int tid) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((tid & 31) == 0) s[tid >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) t += s[i];
  __syncthreads();
  return t;
}

// LayerNorm over the last axis, one row a block, reading bf16 and writing bf16
// for the tensor-core GEMM that follows.  The statistics stay float32: a bf16
// mean over C channels loses more than the GEMM does.
// ONE WARP A ROW.  C is 128 (pair) or 256 (msa), so a lane holds 4 or 8
// channels and both reductions are shuffles -- no shared memory and no
// __syncthreads, where a block-per-row paid two block-wide barriers a row.
template <int C>
__global__ void lnFwdK(const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ scale,
                       const float* __restrict__ offset,
                       __nv_bfloat16* __restrict__ xn, float* __restrict__ mu,
                       float* __restrict__ rstd, int M) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  int lane = threadIdx.x & 31;
  const int K = C / 32;
  const size_t base = (size_t)row * C;
  float v[K], s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    v[k] = __bfloat162float(x[base + lane + k * 32]);
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
    xn[base + c] = __float2bfloat16(v[k] * r * scale[c] + offset[c]);
  }
  if (lane == 0) { mu[row] = m; rstd[row] = r; }
}

template <int C, bool ACC = false, class DT = float>
__global__ void lnBwdK(const __nv_bfloat16* __restrict__ dxn,
                       const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ scale,
                       const float* __restrict__ mu, const float* __restrict__ rstd,
                       DT* __restrict__ dx, int M) {
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
    xh[k] = (__bfloat162float(x[base + c]) - mu_) * r;
    dxh[k] = __bfloat162float(dxn[base + c]) * scale[c];
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
    if (ACC) v += toF(dx[o]);
    stF(dx + o, v);
  }
}

// ONE BALLOT A WARP.  Each lane reads one element -- fully coalesced -- and
// __ballot_sync turns the warp's 32 signs into the 32-bit word lane 0 writes.
// RELU_AUX_BIAS would have had cuBLASLt produce this for free in the GEMM's
// epilogue, and it is not free: it forces a 5.6x slower algorithm (0.199 ms
// -> 1.098 on the same GEMM, measured at the top of this file).
__global__ void packReluMaskK(const __nv_bfloat16* __restrict__ r,
                              unsigned* __restrict__ bits, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  bool pos = i < n && __bfloat162float(r[i]) > 0.f;
  unsigned m = __ballot_sync(0xffffffffu, pos);
  if ((threadIdx.x & 31) == 0 && i < n) bits[i >> 5] = m;
}

// dr *= (r > 0), reading the relu OUTPUT the forward's RELU_BIAS epilogue wrote.
__global__ void maskByReluK(__nv_bfloat16* __restrict__ dr,
                            const __nv_bfloat16* __restrict__ r, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && __bfloat162float(r[i]) <= 0.f) dr[i] = __float2bfloat16(0.f);
}

// ------------------------------- the fused relu mask + dX contraction
// The byte-ratio rule picked this one out of every remaining adjacent pair in
// the backward.  `maskByReluK` reads dr and h (134 MB at N=256) and writes
// back the zeroed entries, then `gemm dx` reads dr again (67) and writes dxn
// (17): 251 MB where the contraction's inputs are 134 and its output 17, a
// ratio of 1.66x.  Everything else adjacent scored 1.1-1.4x, below the ~1.55x
// break-even a hand-written kernel at ~65% of cuBLAS streaming implies.
//
//     dX[m][c] = sum over i  dr[m][i] * (h[m][i] > 0) * w1t[i][c]
//
// `w1t` is the forward's own transposed weight, [I][C] row-major, which is
// exactly the [K][N] a wmma matrix_b wants.  The masked dr tile is built a
// 128-column chunk at a time in shared and contracted immediately, so the
// masked dr never exists in HBM.
#ifndef TR_BMD
#define TR_BMD 64
#endif
static const int BMD = TR_BMD;
static const int WD = 8;             // warps, one per 16 output channels
static const int RTD = BMD / 16;
static const int LDD = 128 + 8;      // the chunk width, padded

// BITS is the KEEP_RELU path: `h` is replaced by one bit an element, which is
// 44% of this kernel's traffic replaced by 2.8%.  Templated rather than
// branched so the non-keeping path keeps its exact instruction stream.
// 94 registers hold this to 2 blocks an SM where its 25600 bytes of shared
// memory would allow 3 -- the same shape as projGateK -- and 3 is reachable
// at 80 registers with NO spills.  **AND IT LOSES**: fwd+bwd 858.25 ->
// 862.89 ms at N=256 depth 48.  So the occupancy is not what this kernel is
// short of either, which is the fourth kernel here to give that answer.
// TR_MASK_MINB=3 to re-measure.
#ifndef TR_MASK_MINB
#define TR_MASK_MINB 2
#endif
template <int CC, bool BITS = false>
__global__ void __launch_bounds__(32 * WD, TR_MASK_MINB) maskDxK(
    const __nv_bfloat16* __restrict__ dr, const __nv_bfloat16* __restrict__ h,
    const unsigned* __restrict__ hbit,
    const __nv_bfloat16* __restrict__ w1t, __nv_bfloat16* __restrict__ dxn,
    int M, int I) {
  extern __shared__ char smem[];
  __nv_bfloat16* tile = (__nv_bfloat16*)smem;              // [BMD][LDD]
  float* stage = (float*)(tile + (size_t)BMD * LDD);       // [WD][16][16]
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const size_t r0 = (size_t)blockIdx.x * BMD;
  const int nrow = (int)min((size_t)BMD, (size_t)M - r0);
  float* st = stage + (size_t)warp * 256;
  static_assert(CC == 128, "one 16-column slice a warp");

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> acc[RTD];
#pragma unroll
  for (int r = 0; r < RTD; ++r) nvcuda::wmma::fill_fragment(acc[r], 0.f);

  for (int ch = 0; ch < I; ch += 128) {
    __syncthreads();                 // the last contraction still reads `tile`
    // two channels a lane: dr and h are row-major, so the pair is one 4-byte
    // access each
    for (int i = tid; i < BMD * 32; i += 32 * WD) {   // four channels a lane
      int row = i / 32, c = (i % 32) * 4;
      size_t o = (size_t)(r0 + row) * I + ch + c;
      __nv_bfloat16* dst = tile + (size_t)row * LDD + c;
#pragma unroll
      for (int w = 0; w < 2; ++w) {
        __nv_bfloat162 v = __floats2bfloat162_rn(0.f, 0.f);
        if (row < nrow) {
          __nv_bfloat162 d = *(const __nv_bfloat162*)(dr + o + 2 * w);
          bool p0, p1;
          if (BITS) {
            size_t b = o + 2 * w;
            p0 = (hbit[b >> 5] >> (b & 31)) & 1u;
            p1 = (hbit[(b + 1) >> 5] >> ((b + 1) & 31)) & 1u;
          } else {
            __nv_bfloat162 rr = *(const __nv_bfloat162*)(h + o + 2 * w);
            p0 = __bfloat162float(rr.x) > 0.f;
            p1 = __bfloat162float(rr.y) > 0.f;
          }
          v = __floats2bfloat162_rn(p0 ? __bfloat162float(d.x) : 0.f,
                                    p1 ? __bfloat162float(d.y) : 0.f);
        }
        *(__nv_bfloat162*)(dst + 2 * w) = v;
      }
    }
    __syncthreads();
    for (int kk = 0; kk < 8; ++kk) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                             nvcuda::wmma::row_major> fb;
      nvcuda::wmma::load_matrix_sync(
          fb, w1t + (size_t)(ch + kk * 16) * CC + warp * 16, CC);
#pragma unroll
      for (int r = 0; r < RTD; ++r) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major> fa;
        nvcuda::wmma::load_matrix_sync(fa, tile + (size_t)(r * 16) * LDD + kk * 16,
                                       LDD);
        nvcuda::wmma::mma_sync(acc[r], fa, fb, acc[r]);
      }
    }
  }

  const int c0 = warp * 16;
#pragma unroll
  for (int r = 0; r < RTD; ++r) {
    nvcuda::wmma::store_matrix_sync(st, acc[r], 16, nvcuda::wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 64; i += 32) {     // dxn row-major: four channels
      int row = r * 16 + i / 4, c = (i % 4) * 4, e = (i / 4) * 16 + c;
      if (row >= nrow) continue;
      __nv_bfloat16* d = dxn + (size_t)(r0 + row) * CC + c0 + c;
      *(__nv_bfloat162*)(d) = __floats2bfloat162_rn(st[e], st[e + 1]);
      *(__nv_bfloat162*)(d + 2) = __floats2bfloat162_rn(st[e + 2], st[e + 3]);
    }
    __syncwarp();
  }
}

inline size_t maskDxSmem() {
  return (size_t)BMD * LDD * 2 + (size_t)WD * 256 * 4;
}

__global__ void f32ToBf16K(const float* __restrict__ a,
                           __nv_bfloat16* __restrict__ b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __float2bfloat16(a[i]);
}

// [R, Cc] row-major f32 -> [Cc, R] row-major bf16, once per weight at setup.
__global__ void transposeBf16K(const float* __restrict__ a,
                               __nv_bfloat16* __restrict__ b, int R, int Cc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= R * Cc) return;
  int r = i / Cc, c = i % Cc;
  b[(size_t)c * R + r] = __float2bfloat16(a[i]);
}

// AF2 has exactly two transition widths, so the LayerNorm's block width is a
// template parameter with two instantiations rather than a kernel argument that
// would make the shared buffer a VLA.
#define TRANS_LN(KERN, Cc, M, ST, ...)                                         \
  do {                                                                         \
    if ((Cc) == 128) KERN<128><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__);     \
    else if ((Cc) == 256) KERN<256><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__); \
    /* 64 is the TEMPLATE pair stack's width.  The kernel is warp-per-row    \
       with K = C/32 registers a lane, so it needs only C % 32 == 0. */      \
    else if ((Cc) == 64) KERN<64><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__);  \
    else { fprintf(stderr, "no LayerNorm for width %d\n", (Cc)); exit(1); }     \
  } while (0)

// the same dispatch for the backward's LayerNorm, which also picks whether to
// add into the caller's accumulator or overwrite it
#define TRANS_LN_ACC(KERN, Cc, ACC, DT, M, ST, ...)                            \
  do {                                                                         \
    if ((Cc) == 128) {                                                         \
      if (ACC) KERN<128, true, DT><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__); \
      else KERN<128, false, DT><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__);    \
    } else if ((Cc) == 64) {                                                   \
      if (ACC) KERN<64, true, DT><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__);  \
      else KERN<64, false, DT><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__);     \
    } else if ((Cc) == 256) {                                                  \
      if (ACC) KERN<256, true, DT><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__); \
      else KERN<256, false, DT><<<((M) + 7) / 8, 256, 0, ST>>>(__VA_ARGS__);    \
    } else { fprintf(stderr, "no LayerNorm for width %d\n", (Cc)); exit(1); }   \
  } while (0)

struct Trans {
  Lt lt;
  cudaStream_t st = nullptr;
  int M = 0, C = 0, I = 0;
  bool split_mask = envOn("TR_SPLIT");
  // weights
  float *ln_scale = nullptr, *ln_offset = nullptr;
  __nv_bfloat16 *w1 = nullptr, *w1t = nullptr, *w2 = nullptr, *w2t = nullptr;
  __nv_bfloat16 *b1 = nullptr, *b2 = nullptr;  // the epilogue's bias must match
  // activations and save-set
  __nv_bfloat16 *xn = nullptr, *h = nullptr, *out = nullptr;
  __nv_bfloat16 *dy = nullptr, *dr = nullptr, *dxn = nullptr;
  float *mu = nullptr, *rstd = nullptr;
  // THE RELU SIGN, ONE BIT AN ELEMENT.  The backward reads `h` for exactly
  // one thing -- `dr *= (h > 0)` -- and `h` is M*4C, 67 MB a depth at N=256
  // and 44% of maskDx's traffic.  A bitmask is 8.4 MB, which is cheap enough
  // to KEEP per depth, and keeping it lets the replay skip this module
  // outright: `mu`/`rstd` are the only other thing its backward needs.
  // Only the PAIR transition uses it; the MSA one keeps `h` itself under
  // KEEP_MSA, where M = S*N makes it free anyway.
  unsigned* hbit = nullptr;
  bool is_pair = false;      // set by Block for `ptr`

  // Every pointer setWeights() assigns, and nothing else.  A 48-deep stack
  // needs 48 sets of WEIGHTS but only one set of scratch, so it calls
  // setWeights() once a depth, snapshots these slots, and re-points them per
  // block -- which is why the split between the two groups above is load-bearing
  // rather than tidiness.
  std::vector<void**> weightSlots() {
    return {(void**)&ln_scale, (void**)&ln_offset, (void**)&w1, (void**)&w1t, (void**)&w2, (void**)&w2t, (void**)&b1, (void**)&b2};
  }

  // ONLY THE MSA TRANSITION OFFERS THESE.  This class serves both transitions,
  // and the PAIR one is M = N^2: its `h` is M*4C, 67 MB a depth at N=256, for
  // 12.8 ms of replay -- 3 ms/GB, the worst candidate measured.  The MSA one
  // is M = S*N.  Block::keepSlots() calls this for `mtr` and not for `ptr`.
  std::vector<KeepSlot> keepSlots() {
    size_t mi = (size_t)M * I;
    // `xn` is forward-only: backward() takes the input as an argument and
    // reads only the relu sign, `mu` and `rstd`.
    if (is_pair)
      return {{(void**)&hbit, mi / 8,          KEEP_RELU},
              {(void**)&mu,   (size_t)M * 4,   KEEP_RELU},
              {(void**)&rstd, (size_t)M * 4,   KEEP_RELU}};
    return {{(void**)&h,    mi * 2,          KEEP_MSA},
            {(void**)&mu,   (size_t)M * 4,   KEEP_MSA},
            {(void**)&rstd, (size_t)M * 4,   KEEP_MSA}};
  }

  // `factor` is the transition's intermediate multiple.  The Evoformer's is
  // 4; the TEMPLATE pair stack's is 2 (num_intermediate_factor in the
  // config), which is why it cannot be assumed.
  // TR_ROWS: how many pair elements a chunk.  The transition is ELEMENTWISE
  // over M, so a chunk is a contiguous run and splitting it is exact.  xn and
  // dy/dr/dxn live inside one phase and become chunk-sized with no replay;
  // `h` crosses into the backward (maskDxK reads it unless KEEP_RELU supplied
  // the sign bits) and `out` is the module's output, so both stay whole.
  // Unset, it follows nbChunkOn(): M/8 above the threshold, whole below --
  // and only for the PAIR transition, whose M is N*N.  The MSA transition's
  // M is S*N, it is 2.2 ms of the stack, and `n` would be meaningless there.
  // TR_ROWS overrides at any length; TR_ROWS=0 forces it off.
  static size_t trChunk(size_t m, bool pair) {
    const char* e = getenv("TR_ROWS");
    size_t r;
    if (e) r = (size_t)atoll(e);
    else if (pair && nbChunkOn((size_t)llround(sqrt((double)m)))) r = m / 8;
    else r = 0;
    return (r > 0 && r < m) ? r : m;
  }

  size_t trRows = 0;   // pair elements a chunk; M when unchunked
  // CHUNKING `h` WAS TRIED AGAIN AND REVERTED AGAIN.  279 MB at N=522 and
  // the biggest thing this module holds, so it keeps being tempting.  The
  // plan was sound on paper -- keepSlots() really does return hbit/mu/rstd
  // and not `h` when is_pair, so the keep-slot objection above is not the
  // blocker -- and the standalone gate was green at four chunk sizes.
  //
  // IT WAS GREEN BECAUSE IT TESTED NOTHING.  `is_pair` is set by Block for
  // `ptr` and the standalone gate never sets it, so the guard
  // (`... && is_pair && ...`) was false in every gate run and `h` stayed
  // whole.  In the model it is true, and |grad| went 7.54635 -> 2.66972 at
  // 487.  This is the SECOND time this exact buffer has been chunked, passed
  // the standalone gate on a branch the gate does not reach, and been caught
  // by the model.  Next time: gate the branch, not the module.
  __nv_bfloat16* hAt(size_t t0) { return h + t0 * I; }   // h is whole

  void alloc(int M_, int C_, cudaStream_t s = nullptr, int factor = 4) {
    M = M_; C = C_; I = factor * C; st = s;
    lt.init();
    size_t mc = (size_t)M * C, mi = (size_t)M * I;
    trRows = trChunk((size_t)M, is_pair);
    const size_t cc = trRows * C, ci = trRows * I;
    CU(cudaMalloc(&xn, cc * 2));
    // h STAYS WHOLE.  Making it chunk-sized and having the backward read the
    // relu sign from hbit instead (maskDxK<128,true>, the KEEP_RELU path)
    // looks right and corrupts memory: keepSlots() declares h at mi*2 and the
    // stack's tier machinery swaps that pointer, so a chunk-sized h is a
    // buffer two parties disagree about the length of.  The symptom was an
    // illegal access surfacing in trimul_body and opm -- somewhere else
    // entirely, asynchronously.  Freeing it needs the keep-slot contract
    // changed too, which is a bigger change than it is worth here.
    CU(cudaMalloc(&h, mi * 2));
    CU(cudaMalloc(&out, mc * 2));
    CU(cudaMalloc(&dy, cc * 2));
    CU(cudaMalloc(&dr, ci * 2));
    CU(cudaMalloc(&dxn, cc * 2));
    if (is_pair) CU(cudaMalloc(&hbit, mi / 8));
    CU(cudaMalloc(&mu, (size_t)M * 4));
    CU(cudaMalloc(&rstd, (size_t)M * 4));
  }

  void setWeights(Oracle& o, const std::string& P_) {
    // haiku names submodules created inside __call__ under "~"
    std::string P = P_;
    if (!o.has(P + "transition1/weights")) P = P_ + "~/";
    auto dev = [&](const std::string& k) { o[k].to_device(); return o[k].dev; };
    // allocated HERE, not in alloc(), so that calling setWeights once a depth
    // leaves 48 independent sets behind -- which is what the stack snapshots.
    // Allocated in alloc() they would all be the same six buffers and every
    // depth would end up running the last block's transition.
    CU(cudaMalloc(&w1, (size_t)C * I * 2));
    CU(cudaMalloc(&w1t, (size_t)C * I * 2));
    CU(cudaMalloc(&w2, (size_t)I * C * 2));
    CU(cudaMalloc(&w2t, (size_t)I * C * 2));
    CU(cudaMalloc(&b1, (size_t)I * 2));
    CU(cudaMalloc(&b2, (size_t)C * 2));
    ln_scale = dev(P + "input_layer_norm/scale");
    ln_offset = dev(P + "input_layer_norm/offset");
    float* w1f = dev(P + "transition1/weights");  // [C, I]
    float* w2f = dev(P + "transition2/weights");  // [I, C]
    int ci = C * I;
    f32ToBf16K<<<(ci + 255) / 256, 256>>>(w1f, w1, ci);
    f32ToBf16K<<<(ci + 255) / 256, 256>>>(w2f, w2, ci);
    transposeBf16K<<<(ci + 255) / 256, 256>>>(w1f, w1t, C, I);  // [C,I]->[I,C]
    transposeBf16K<<<(ci + 255) / 256, 256>>>(w2f, w2t, I, C);  // [I,C]->[C,I]
    f32ToBf16K<<<(I + 255) / 256, 256>>>(dev(P + "transition1/bias"), b1, I);
    f32ToBf16K<<<(C + 255) / 256, 256>>>(dev(P + "transition2/bias"), b2, C);
    CU(cudaDeviceSynchronize());
  }

  // Writes the module's DELTA into `out`; the residual stays with the caller.
  // `tail` false skips the second GEMM, which only produces `out`.  Its own
  // backward reads `h`, the relu output, written above the guard.
  //
  // ONLY THE LAST STEP OF A TRACK CAN USE THIS, and a transition is that step
  // in both (msa_transition closes the MSA track, pair_transition the pair).
  // The tempting generalisation -- "no backward reads any module's out, so the
  // stack's replay can stop short everywhere" -- is false and the stack gate
  // says so at once (grad/pair rms 0.34 against the oracle's 10.8): a module's
  // out is added to the running activation to make the NEXT intermediate, and
  // that intermediate is exactly what the next module's backward reads.  Only
  // the two outputs the replay genuinely throws away are free.
  // `want_mask` is the SAVE pass.  The binder preset runs num_recycle=1, so
  // there are two forwards a step and only the second is differentiated;
  // packing the mask in both costs +14 ms against 15 saved, which is the
  // measurement that kept this idea unbuilt until the gating existed.
  void forward(const __nv_bfloat16* x, bool tail = true, bool want_mask = false) {
    // ELEMENTWISE OVER M, so the whole body runs a chunk at a time.  xn is
    // chunk-sized and used at base; h/out/mu/rstd/hbit outlive the chunk and
    // are offset.  TR_ROWS unset means one chunk of M -- what it did before.
    for (size_t t0 = 0; t0 < (size_t)M; t0 += trRows) {
      const size_t tm = ((size_t)M - t0 < trRows) ? (size_t)M - t0 : trRows;
      const size_t ti = tm * I;
      TRANS_LN(lnFwdK, C, (int)tm, st, x + t0 * C, ln_scale, ln_offset, xn,
               mu + t0, rstd + t0, tm);
      // bias and relu ride in the first GEMM's epilogue
      PROF_STEP("tr.f gemm1+relu",
        lt.matmul((int)tm, I, C, xn, w1t, hAt(t0), CUDA_R_16BF,
                  CUBLASLT_EPILOGUE_RELU_BIAS, b1, nullptr, st));
      if (want_mask && hbit)
        PROF_STEP("tr.f packMask",
          packReluMaskK<<<(ti + 255) / 256, 256, 0, st>>>(hAt(t0),
              hbit + t0 * I / 8, ti));
      if (!tail) continue;
      PROF_STEP("tr.f gemm2",
        lt.matmul((int)tm, C, I, hAt(t0), w2t, out + t0 * C, CUDA_R_16BF,
                  CUBLASLT_EPILOGUE_BIAS, b2, nullptr, st));
    }
  }

  // `dout` is the cotangent on the DELTA; `dx` gets the input's, likewise
  // without the residual's share.
  // `doutB` is the cotangent already narrowed by its producer; `dxB` asks for
  // the same for the next module in the chain.
  // DT is the cotangent's width; `dout` comes in at the same width it goes out
  template <class DT>
  // `use_bits`: the relu sign came from KEEP_RELU and `h` was never rebuilt.
  void backward(const DT* dout, const __nv_bfloat16* x, DT* dx,
                bool acc = false, bool use_bits = false) {
    size_t mc = (size_t)M * C, mi = (size_t)M * I;
    // SAME CHUNKS AS THE FORWARD.  dy/dr/dxn are chunk-sized and used at
    // base; h, hbit, mu, rstd and the caller's dout/x/dx are whole and
    // offset.  dr is 206 MB at N=448 and is one of the two allocations that
    // ran out at 637 residues.
    for (size_t t0 = 0; t0 < (size_t)M; t0 += trRows) {
      const size_t tm = ((size_t)M - t0 < trRows) ? (size_t)M - t0 : trRows;
      const size_t tc = tm * C, ti = tm * I;
    // a bf16 cotangent IS the GEMM's operand; an f32 one has to be narrowed
    const __nv_bfloat16* dyin;
    if constexpr (std::is_same<DT, __nv_bfloat16>::value) {
      dyin = dout + t0 * C;
    } else {
      f32ToBf16K<<<(tc + 255) / 256, 256, 0, st>>>(dout + t0 * C, dy, tc);
      dyin = dy;
    }
    PROF_STEP("tr.b gemm dr",
      lt.matmul((int)tm, I, C, dyin, w2, dr, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st));
    // TR_SPLIT runs the mask and the contraction as two passes, which is the
    // A/B this was decided on.  The fused kernel only exists for C = 128 (the
    // PAIR transition, 44.6 ms of the stack's backward); the MSA transition is
    // 2.2 ms and keeps the split path.
    if (C == 128 && !split_mask) {
      static bool once = false;
      if (!once) {
        CU(cudaFuncSetAttribute((const void*)maskDxK<128, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)maskDxSmem()));
        CU(cudaFuncSetAttribute((const void*)maskDxK<128, true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)maskDxSmem()));
        once = true;
      }
      PROF_STEP("tr.b maskDx",
        if (use_bits)
          maskDxK<128, true><<<(tm + BMD - 1) / BMD, 32 * WD, maskDxSmem(), st>>>(
              dr, hAt(t0), hbit + t0 * I / 8, w1t, dxn, tm, I);
        else
          maskDxK<128, false><<<(tm + BMD - 1) / BMD, 32 * WD, maskDxSmem(), st>>>(
              dr, h + t0 * I, hbit + t0 * I / 8, w1t, dxn, tm, I);
        CU(cudaGetLastError()));
    } else {
      maskByReluK<<<(ti + 255) / 256, 256, 0, st>>>(dr, hAt(t0), ti);
      lt.matmul((int)tm, C, I, dr, w1, dxn, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
                nullptr, nullptr, st);
    }
    TRANS_LN_ACC(lnBwdK, C, acc, DT, (int)tm, st, dxn, x + t0 * C, ln_scale,
                 mu + t0, rstd + t0, dx + t0 * C, tm);
    }
  }
};

}  // namespace trans
