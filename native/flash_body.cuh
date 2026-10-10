// FlashAttention for AF2's triangle attention: the scores never reach memory.
//
// The materialised path in triattn.cu is at its traffic floor and still loses to
// Triton, because that floor is high: the [N_b, H, N, N] score tensor is 134 MB
// at N=256 and it is written and read five times over a forward and backward.
// Flash removes the tensor, not the traffic around it.
//
// THE SHAPE.  batch = N (the pair's first index), N queries over N keys, H = 4
// heads of D = 32.  Each block owns one (b, h, 64-query tile) and streams the
// keys in tiles of 32.  Four warps, sixteen queries each.
//
// THE BIAS IS NOT OPTIONAL.  AF2 adds `nbb[h, i, j]`, computed from the pair
// itself, to every score -- so the forward reads it and the backward owes
// `dnbb = sum over b of dS`.  That reduction runs over the batch axis while
// dQ reduces over keys and dK/dV over queries, which is why the backward is
// three passes and not two: no single loop order serves all three.
//
// WMMA rather than raw mma.sync: at D = 32 the fragment bookkeeping dominates
// the arithmetic either way, and the opaque-accumulator problem is handled by
// keeping the running output in shared memory rather than in fragments.
// NO `#pragma once`: this body is included once per HEAD DIM by flash.cuh --
// FD=32 for the Evoformer's triangle attention and the MSA row attention,
// FD=16 for the template pair stack (pair_channel 64 over 4 heads).  FA_NS /
// FA_FD select the pass.
#include <cuda_bf16.h>
#include <mma.h>

#include "common.cuh"

namespace FA_NS {

static const int BM = 128;    // queries per block
static const int BN = 32;    // keys per step
static const int FD = FA_FD;  // head dim
static const int LDQ = FD + 8;
static const int LDPS = 40;
                             // the kernel takes the projection's leading
                             // dimension as a parameter, and a parameter of the
                             // same name shadows the constant inside the kernel
                             // -- which silently placed the f32 scratch at
                             // Ps + BM*516 instead of Ps + BM*40.
static const int LDS = 40;
                             // rejects it at run time (illegal access), so 40.
static const int WARPS = BM / 16;

__device__ __forceinline__ float wMax(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float wSum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// q, k, v are slices of the projection: element (b, n, h, d) at
// (b*N + n)*LDP + blk*128 + h*FD + d.  Nothing is copied out of it.
__global__ __launch_bounds__(32 * WARPS) void flashFwdK(
    const __nv_bfloat16* __restrict__ proj, int projLD, int qoff, int koff, int voff,
    const __nv_bfloat16* __restrict__ nbb, const float* __restrict__ mask,
    __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int N, int H) {
  extern __shared__ char smem[];
  __nv_bfloat16* Qs = (__nv_bfloat16*)smem;                 // [BM][LDQ]
  __nv_bfloat16* Ks = Qs + BM * LDQ;                        // [BN][LDQ]
  __nv_bfloat16* Vs = Ks + BN * LDQ;                        // [BN][LDQ]
  __nv_bfloat16* Ps = Vs + BN * LDQ;                        // [BM][LDPS]
  float* Ss = (float*)(Ps + BM * LDPS);                      // [BM][LDS]
  float* mrun = Ss + BM * LDS;                              // [BM]
  float* lrun = mrun + BM;                                  // [BM]

  int it = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  int i0 = it * BM;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  size_t M = (size_t)N * N;

  // Q tile, and the running state
  for (int r = tid / FD; r < BM; r += (32 * WARPS) / FD) {
    int d = tid % FD, i = i0 + r;
    Qs[r * LDQ + d] = i < N ? proj[((size_t)b * N + i) * projLD + qoff + h * FD + d]
                            : __float2bfloat16(0.f);
  }
  for (int r = tid; r < BM; r += 32 * WARPS) {
    mrun[r] = -1e30f; lrun[r] = 0.f;
  }
  // The running output lives in REGISTERS, not shared: lane `l` owns channel
  // `l` of all 16 of this warp's rows.  In shared it was 8 KB a block, and
  // shared memory is what caps occupancy here (2 blocks an SM, 17% of the
  // card's threads) -- the kernel is latency-bound, not bandwidth-bound.
  float acc[16];
#pragma unroll
  for (int r = 0; r < 16; ++r) acc[r] = 0.f;
  __syncthreads();

  for (int j0 = 0; j0 < N; j0 += BN) {
    for (int r = tid / FD; r < BN; r += (32 * WARPS) / FD) {
      int d = tid % FD, j = j0 + r;
      bool ok = j < N;
      Ks[r * LDQ + d] = ok ? proj[((size_t)b * N + j) * projLD + koff + h * FD + d]
                           : __float2bfloat16(0.f);
      Vs[r * LDQ + d] = ok ? proj[((size_t)b * N + j) * projLD + voff + h * FD + d]
                           : __float2bfloat16(0.f);
    }
    __syncthreads();

    // S = Q K^T for this warp's 16 queries against the BN keys
    using namespace nvcuda::wmma;
    for (int nt = 0; nt < BN / 16; ++nt) {
      fragment<accumulator, 16, 16, 16, float> cS;
      fill_fragment(cS, 0.f);
#pragma unroll
      for (int kk = 0; kk < FD / 16; ++kk) {
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> aQ;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> bK;
        load_matrix_sync(aQ, Qs + (size_t)warp * 16 * LDQ + kk * 16, LDQ);
        load_matrix_sync(bK, Ks + (size_t)nt * 16 * LDQ + kk * 16, LDQ);
        mma_sync(cS, aQ, bK, cS);
      }
      store_matrix_sync(Ss + (size_t)warp * 16 * LDS + nt * 16, cS, LDS,
                        mem_row_major);
    }
    __syncwarp();

    // online softmax, one row at a time across the warp (BN == 32 == warp width)
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      int row = warp * 16 + r, i = i0 + row;
      int j = j0 + lane;
      float s = -1e30f;
      if (i < N && j < N) {
        s = Ss[(size_t)row * LDS + lane] +
            __bfloat162float(nbb[(size_t)h * M + (size_t)i * N + j]) +
            1e9f * (mask[(size_t)b * N + j] - 1.f);
      }
      float rmax = wMax(s);
      float mold = mrun[row];
      float mnew = fmaxf(mold, rmax);
      float corr = __expf(mold - mnew);
      float p = __expf(s - mnew);
      float psum = wSum(p);
      if (lane == 0) { mrun[row] = mnew; lrun[row] = lrun[row] * corr + psum; }
      Ps[(size_t)row * LDPS + lane] = __float2bfloat16(p);
      acc[r] *= corr;   // lane == channel, so the rescale is register-local
    }
    __syncwarp();

    // acc += P V
    for (int dt = 0; dt < FD / 16; ++dt) {
      fragment<accumulator, 16, 16, 16, float> cO;
      fill_fragment(cO, 0.f);
#pragma unroll
      for (int kk = 0; kk < BN / 16; ++kk) {
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> aP;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> bV;
        load_matrix_sync(aP, Ps + (size_t)warp * 16 * LDPS + kk * 16, LDPS);
        load_matrix_sync(bV, Vs + (size_t)kk * 16 * LDQ + dt * 16, LDQ);
        mma_sync(cO, aP, bV, cO);
      }
      // the accumulator's lane mapping is opaque, so land it in shared and add
      store_matrix_sync(Ss + (size_t)warp * 16 * LDS, cO, LDS, mem_row_major);
      __syncwarp();
      int c = lane - dt * 16;
      if (c >= 0 && c < 16) {
#pragma unroll
        for (int r = 0; r < 16; ++r)
          acc[r] += Ss[(size_t)(warp * 16 + r) * LDS + c];
      }
      __syncwarp();
    }
    __syncthreads();
  }

  // normalise and write
#pragma unroll
  for (int r = 0; r < 16; ++r) {
    int row = warp * 16 + r, i = i0 + row;
    if (i >= N) continue;
    float inv = 1.f / lrun[row];
    if (lane < FD)
      out[((size_t)b * N + i) * (H * FD) + h * FD + lane] =
          __float2bfloat16(acc[r] * inv);
    if (lane == 0) lse[((size_t)b * H + h) * N + i] = mrun[row] + __logf(lrun[row]);
  }
}

inline size_t fwdSmem() {
  return (size_t)(BM * LDQ + BN * LDQ + BN * LDQ + BM * LDPS) * sizeof(__nv_bfloat16) +
         (size_t)(BM * LDS + 2 * BM) * sizeof(float);
}

}  // namespace FA_NS
