// AF2's attention backward as FlashAttention-2 lays it out, on mma.sync: PORTED from sokrypton/localfold,
// branch opus-55-opt-backprop, cuda/grad/flash.cuh (bwdBf), to this directory's contract - fp16 operands, scores in
// the log2 domain, the fused q/k/v/gate projection read in place, the gradients landing in the combined bf16 dcomb.
//
//   "THE BEER-WARE LICENSE" (Revision 42):
//   Sergey Ovchinnikov <https://github.com/sokrypton> wrote this. As long as you
//   retain this notice you can do whatever you want with this stuff. If we meet
//   some day, and you think this stuff is worth it, you can buy me a beer in
//   return.
//
// WHY THE OTHER LOOP ORDER.  flashBwdK owns (b, h, QUERY tile) and so holds dQ but must add dK and dV by atomics
// (22% of it at N 256, by its own NO_DKV probe). This owns (b, h, KEY block): dK and dV are summed over every query
// in registers and written ONCE, as bf16, straight into dcomb - no f32 dK/dV, no memsets, no scatter - and dQ is
// the one added by atomics, into an f32 buffer the last of a (b, h)'s key blocks narrows into dcomb while it is still
// in L2. The bias gradient comes straight from the registers that hold dS.
//
// What it is, per (b, h, 64 keys): 4 warps, 16 keys each, the queries in tiles of 64 double-buffered by cp.async
// (Q, dO, the bias tile, the rows' lse and D); S^T = K Q^T and dP^T = V dO^T from K and V fragments held in
// registers; dV += P^T dO, dK += dS^T Q from the accumulators; dS stored transposed in its own XOR-swizzled tile for
// dQ = dS K (ldmatrix.trans both ways).
//
// TWO THINGS THAT WERE WORTH MOST ON AN A100 (localfold's commit messages have the numbers):
//  - L2 prices an atomic by the 32-byte SECTOR it touches, not by the lane. dQ's fragment pairs put four lanes in each
//    of eight sectors; one shuffle between rows g and g ^ 1 puts eight in each of four (13-21% of the kernel).
//  - the bias gradient's atomics from the registers, unconditionally in a full tile (a masked element's dS is exactly
//    0): a branch an element was a fifth of the issue.
//
// Semantics are flashBwdK's: s = (q . k) LOG2E + bias + 1e9 (mask_j - 1) LOG2E, P = exp2(s - lse) with the forward's
// log2-domain lse, dS = P (dP - D); dnbb [H][N][N] (or its int64 fixed point, nbbFixed's 2^40) accumulates dS over the
// batch; q carries the softmax scale, so dQ = dS K and dK = dS^T Q unscaled.
//
// MEASURED HERE AGAINST flashBwdK (lfbwdgate.cu, A100 40 GB; the A10 these kernels were tuned on is not measured):
//   D 32, N 256, 256 rows: flashBwdK + its dK/dV scatter 1.170 ms, this 0.506 (2.31x)
//   D 16, N 256, 256 rows:                                0.685 ms,      0.385 (1.78x)
//   D 32, N 522,  64 rows:                                1.281 ms,      0.625 (2.05x)
//   D 16, N 522,  64 rows:                                0.740 ms,      0.507 (1.46x)
// gradients against flashBwdK's: dQ, dK rel rms 1.1e-3 (dS rounded to f16 for the dQ/dK products, bf16 outputs),
// dV 4e-5, dnbb 2e-4.
//
// WIRING IT INTO triattn_body.cuh (not done: this box has none of the module oracles that gate it). In the row-chunk
// loop, in place of the dKf/dVf memsets, flashBwdK and scatterQKVK:
//   lfb::backward<D>(qkvgAt(cr0), 4 * C, biasH, davH, lseF + cr0 * H * N, lseD + cr0 * H * N, mask + cr0 * N,
//                    dcomb, TOT, dnbbF or nullptr, dnbbI or nullptr, dKf /* as dQ's f32 scratch */, cnt, crows, N, H, st);
// dQ, dK and dV are then all in dcomb's QKV columns, so after the loop only dnbb's columns are missing: the TA_SPLIT
// branch without its scatterQKVK (nbbScatterK, then the dx GEMM). scatterDxK, which fuses the scatter into the GEMM,
// reads dK/dV from f32 and does not apply. cnt is [crows][H] unsigned; dVf is no longer needed. msarow.cuh's call is the
// same kernel at D 32 if its mask and lse follow the same convention (not checked).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace lfb {

constexpr int BC = 64, QT = 64, THREADS = 128;
constexpr int LB = BC + 8;   // the bias tile's row stride (halves)
template <int D> __host__ __device__ constexpr int LK() { return D + 8; }
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void mma16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t h2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ uint32_t bf2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t word(const uint16_t* s, int i) { return *reinterpret_cast<const uint32_t*>(s + i); }
// the B fragment of rows k .. k + 15, columns n .. n + 7 of a row-major tile: ldmatrix.trans
__device__ __forceinline__ void ldT(uint32_t& b0, uint32_t& b1, const uint16_t* s, int k, int n, int ld) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(s + (k + (threadIdx.x & 15)) * ld + n);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(a));
}
// four 8 x 8 tiles transposed (lane l addresses row l % 8 of tile l / 8; tile j lands in r[j])
__device__ __forceinline__ void ldT4(uint32_t* r, const uint16_t* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void cp16(void* d, const void* s, bool on) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(d);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(s), "r"(on ? 16 : 0));
}
__device__ __forceinline__ void cp4(void* d, const void* s, bool on) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(d);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(a), "l"(s), "r"(on ? 4 : 0));
}
__device__ __forceinline__ void cpCommit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cpWait1() { asm volatile("cp.async.wait_group 1;"); }
__device__ __forceinline__ void cToA(uint32_t* f, const float* c0, const float* c1) {
  f[0] = h2(c0[0], c0[1]); f[1] = h2(c0[2], c0[3]); f[2] = h2(c1[0], c1[1]); f[3] = h2(c1[2], c1[3]);
}

struct Args {
  const half* qkvg; int projLD;          // [rows][N][projLD]: q at h D, k at W + h D, v at 2 W + h D
  const half* dO;                        // [rows][N][W]
  const half* bias; bool biasVec;        // [H][N][N], log2 units; biasVec: rows 16-byte aligned (N % 8 == 0)
  const float* mask;                     // [rows][N], 1 kept
  const float* lse; const float* Dv;     // [rows][H][N]: the forward's log2-domain lse, the row term
  float* dq;                             // [rows][N][W], zeroed: dQ's f32 accumulator
  __nv_bfloat16* dcomb; int tot;         // dQ, dK, dV land at columns h D, W + h D, 2 W + h D
  float* dnbb; long long* dnbbI;         // [H][N][N]: one of the two (dnbbI: int64 fixed point, 2^40)
  unsigned* cnt;                         // [rows][H], zeroed: key blocks finished (the last narrows dQ)
  int N, H;
};

template <int D>
__global__ void __launch_bounds__(THREADS, 3) lfFlashBwdK(Args a) {
  constexpr int L = LK<D>();
  extern __shared__ __align__(16) float sm[];
  uint16_t* Kb = reinterpret_cast<uint16_t*>(sm);                 // [BC][L]
  uint16_t* Qb = Kb + BC * L;                                     // [2][QT][L]
  uint16_t* Ob = Qb + 2 * QT * L;                                 // dO [2][QT][L]
  uint16_t* Tb = Ob + 2 * QT * L;                                 // [2][QT][LB]: the bias tile
  uint16_t* Sb = Tb + 2 * QT * LB;                                // dS^T [BC][QT], 16-byte chunks XOR-swizzled by row
  float* QV = reinterpret_cast<float*>(Sb + BC * QT);             // [2][QT][4]: lse, D
  const int N = a.N, H = a.H, W = H * D;
  const int nkb = (N + BC - 1) / BC;
  const int kb = blockIdx.x % nkb, h = (blockIdx.x / nkb) % H, b = blockIdx.x / (nkb * H);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int k0 = kb * BC, kl0 = warp * 16;
  const size_t bh = (size_t)b * H + h;
  const uint16_t* qkvg = reinterpret_cast<const uint16_t*>(a.qkvg) + (size_t)b * N * a.projLD;
  const uint16_t* dOb = reinterpret_cast<const uint16_t*>(a.dO) + (size_t)b * N * W;
  const uint16_t* biasH = reinterpret_cast<const uint16_t*>(a.bias) + (size_t)h * N * N + k0;
  const float* lseb = a.lse + bh * N;
  const float* Dvb = a.Dv + bh * N;
  // a [rows][D] tile of q (role 0) or dO into shared memory, zero past N
  auto loadRows = [&](uint16_t* dst, const uint16_t* src, int ld, int n0, int rows) {
    for (int i = threadIdx.x; i < rows * (D / 8); i += THREADS) {
      const int r = i / (D / 8), c = (i % (D / 8)) * 8;
      const bool on = n0 + r < N;
      cp16(dst + r * L + c, on ? src + (size_t)(n0 + r) * ld + c : src, on);
    }
  };
  loadRows(Kb, qkvg + W + h * D, a.projLD, k0, BC);
  auto issue = [&](int st, int q0) {
    loadRows(Qb + st * QT * L, qkvg + h * D, a.projLD, q0, QT);
    loadRows(Ob + st * QT * L, dOb + h * D, W, q0, QT);
    uint16_t* T = Tb + st * QT * LB;
    if (a.biasVec)
      for (int i = threadIdx.x; i < QT * BC / 8; i += THREADS) {
        const int r = i / (BC / 8), c = (i % (BC / 8)) * 8;
        const bool on = q0 + r < N && k0 + c < N;   // (N % 8 == 0: a chunk is all in or all out)
        cp16(T + r * LB + c, on ? biasH + (size_t)(q0 + r) * N + c : biasH, on);
      }
    else if (N % 2 == 0)   // (rows 4-byte aligned: cp.async a key pair at a time, still asynchronous - loaded
                           // synchronously the tile was most of the kernel at N 522)
      for (int i = threadIdx.x; i < QT * BC / 2; i += THREADS) {
        const int r = i / (BC / 2), c = (i % (BC / 2)) * 2;
        const bool on = q0 + r < N && k0 + c < N;
        cp4(T + r * LB + c, on ? biasH + (size_t)(q0 + r) * N + c : biasH, on);
      }
    else
      for (int i = threadIdx.x; i < QT * BC; i += THREADS) {
        const int r = i / BC, c = i % BC;
        T[r * LB + c] = q0 + r < N && k0 + c < N ? biasH[(size_t)(q0 + r) * N + c] : (uint16_t)0;
      }
    if (threadIdx.x < QT) {
      float* qv = QV + st * 4 * QT;
      const int j = threadIdx.x, q = q0 + j;
      const bool in = q < N;
      cp4(qv + 4 * j, in ? lseb + q : lseb, in);
      cp4(qv + 4 * j + 1, in ? Dvb + q : Dvb, in);
    }
  };
  issue(0, 0);
  cpCommit();
  // the thread's two keys: the mask term (log2 units), -inf past N
  float kadd[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int k = k0 + kl0 + g + r * 8;
    kadd[r] = k < N ? 1e9f * (a.mask[(size_t)b * N + k] - 1.f) * LOG2E : -INFINITY;
  }
  auto sOff = [](int k, int q) { return k * QT + (((q >> 3) ^ (k & (QT / 8 - 1))) << 3) + (q & 7); };
  // K and V as A fragments, rows k0 + kl0 + g (+ 8), columns 16 s + 2 t (+ 8)
  uint32_t ka[D / 16][4], va[D / 16][4];
#pragma unroll
  for (int s = 0; s < D / 16; ++s)
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int r = k0 + kl0 + g + (i & 1) * 8, c = s * 16 + 2 * t + (i >> 1) * 8;
      const uint16_t* row = qkvg + (size_t)r * a.projLD + h * D + c;
      ka[s][i] = r < N ? *reinterpret_cast<const uint32_t*>(row + W) : 0u;
      va[s][i] = r < N ? *reinterpret_cast<const uint32_t*>(row + 2 * W) : 0u;
    }
  float dka[D / 8][4], dva[D / 8][4];
#pragma unroll
  for (int n = 0; n < D / 8; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) dka[n][e] = dva[n][e] = 0.f;
  float* dqb = a.dq + (size_t)b * N * W + h * D;
  float* gF = a.dnbb ? a.dnbb + (size_t)h * N * N + k0 : nullptr;
  long long* gI = a.dnbbI ? a.dnbbI + (size_t)h * N * N + k0 : nullptr;
  const int nq = (N + QT - 1) / QT;
  for (int it = 0; it < nq; ++it) {
    const int st = it & 1, q0 = it * QT;
    if (it + 1 < nq) issue(st ^ 1, q0 + QT);
    cpCommit();
    cpWait1();
    __syncthreads();
    const uint16_t* Qs = Qb + st * QT * L;
    const uint16_t* Os = Ob + st * QT * L;
    const uint16_t* T = Tb + st * QT * LB;
    const float* qv = QV + st * 4 * QT;
#pragma unroll 1
    for (int hq = 0; hq < QT / 32; ++hq) {
      float sa[4][4], dpt[4][4];   // 16 keys x 32 queries
#pragma unroll
      for (int jj = 0; jj < 4; ++jj)
#pragma unroll
        for (int e = 0; e < 4; ++e) sa[jj][e] = dpt[jj][e] = 0.f;
#pragma unroll
      for (int s = 0; s < D / 16; ++s)
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
          const int i = ((hq * 4 + jj) * 8 + g) * L + s * 16 + 2 * t;
          mma16(sa[jj], ka[s], word(Qs, i), word(Qs, i + 8));
          mma16(dpt[jj], va[s], word(Os, i), word(Os, i + 8));
        }
      // the bias pairs (queries 2t, 2t + 1 at key g) of the warp's 16 keys: tile jj * 2 + hk, transposed
      uint32_t bw[8];
#pragma unroll
      for (int x = 0; x < 2; ++x) {
        const int m = x * 4 + (lane >> 3);
        ldT4(bw + 4 * x, T + ((hq * 4 + (m >> 1)) * 8 + (lane & 7)) * LB + kl0 + (m & 1) * 8);
      }
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        const int qe = (hq * 4 + jj) * 8 + 2 * t;   // the thread's two queries qe, qe + 1
        float Lq[2], Dq[2];
#pragma unroll
        for (int par = 0; par < 2; ++par) {
          const float2 v = *reinterpret_cast<const float2*>(qv + 4 * (qe + par));
          Lq[par] = q0 + qe + par < N ? v.x : INFINITY;   // (a padded query: every p exactly 0)
          Dq[par] = v.y;
        }
#pragma unroll
        for (int hk = 0; hk < 2; ++hk) {
          const int kl = kl0 + g + hk * 8;
          const __half2 bb = *reinterpret_cast<const __half2*>(&bw[jj * 2 + hk]);
          float dsp[2];
#pragma unroll
          for (int par = 0; par < 2; ++par) {
            const int e = hk * 2 + par;
            const float x = fmaf(sa[jj][e], LOG2E, kadd[hk] + __half2float(par ? __high2half(bb) : __low2half(bb)));
            const float p = ex2(x - Lq[par]);
            const float ds = p * (dpt[jj][e] - Dq[par]);
            sa[jj][e] = p;
            dpt[jj][e] = ds;
            dsp[par] = ds;
          }
          *reinterpret_cast<uint32_t*>(Sb + sOff(kl, qe)) = h2(dsp[0], dsp[1]);
        }
      }
      // dnbb from the registers, unconditionally in a full tile
      if (gF || gI) {
        const int qb = q0 + hq * 32 + 2 * t;
        const bool full = q0 + hq * 32 + 32 <= N && k0 + BC <= N;
#pragma unroll
        for (int jj = 0; jj < 4; ++jj)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int qo = jj * 8 + (e & 1), ko = (e >> 1) * 8;
            const int o = (qb + qo) * N + kl0 + g + ko;
            // (predicated, not redirected: pointing every out-of-range element at one address serialised a partial
            // key block's atomics on it - 7x the kernel at N 522)
            if (!full && !(qb + qo < N && k0 + kl0 + g + ko < N)) continue;
            if (gI) atomicAdd(reinterpret_cast<unsigned long long*>(gI + o),
                              (unsigned long long)(long long)((double)dpt[jj][e] * 1099511627776.0));
            else atomicAdd(gF + o, dpt[jj][e]);
          }
      }
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        uint32_t pa[4], da[4];
        cToA(pa, sa[2 * u], sa[2 * u + 1]);
        cToA(da, dpt[2 * u], dpt[2 * u + 1]);
#pragma unroll
        for (int n = 0; n < D / 8; ++n) {
          uint32_t b0, b1;
          ldT(b0, b1, Os, hq * 32 + u * 16, n * 8, L);
          mma16(dva[n], pa, b0, b1);
          ldT(b0, b1, Qs, hq * 32 + u * 16, n * 8, L);
          mma16(dka[n], da, b0, b1);
        }
      }
    }
    __syncthreads();
    // dQ for this warp's 16 queries over the block's 64 keys, from the accumulators: rows g and g ^ 1 trade a value
    // so each atomic covers four rows' eight columns, 32-byte sectors whole
    const int qw = warp * 16;
    float dqa[D / 8][4];
#pragma unroll
    for (int n = 0; n < D / 8; ++n) dqa[n][0] = dqa[n][1] = dqa[n][2] = dqa[n][3] = 0.f;
#pragma unroll
    for (int s = 0; s < BC / 16; ++s) {
      uint32_t A[4];
      {
        const int m = lane >> 3, k = s * 16 + (m >> 1) * 8 + (lane & 7);
        ldT4(A, Sb + sOff(k, qw + (m & 1) * 8));
      }
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        uint32_t b0, b1;
        ldT(b0, b1, Kb, s * 16, n * 8, L);
        mma16(dqa[n], A, b0, b1);
      }
    }
    const int odd = g & 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int qe = q0 + qw + (g & ~1) + r * 8;
#pragma unroll
      for (int n = 0; n < D / 8; ++n) {
        const float c0 = dqa[n][r * 2], c1 = dqa[n][r * 2 + 1];
        const float x = __shfl_xor_sync(~0u, odd ? c0 : c1, 4);
        float* p = dqb + (size_t)qe * W + n * 8 + 2 * t + odd;
        if (qe < N) atomicAdd(p, odd ? x : c0);
        if (qe + 1 < N) atomicAdd(p + W, odd ? c1 : x);
      }
    }
    __syncthreads();   // this stage is the one the next iteration's issue overwrites
  }
  // the last of a (b, h)'s key blocks narrows its dQ into dcomb
  {
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(a.cnt + bh, 1u) == (unsigned)(nkb - 1);
    __syncthreads();
    if (last) {
      __threadfence();
      for (int i = threadIdx.x; i < N * (D / 4); i += THREADS) {
        const int qq = i / (D / 4), c = (i % (D / 4)) * 4;
        const float4 v = __ldcg(reinterpret_cast<const float4*>(dqb + (size_t)qq * W + c));
        *reinterpret_cast<uint2*>(a.dcomb + ((size_t)b * N + qq) * a.tot + h * D + c) =
            make_uint2(bf2(v.x, v.y), bf2(v.z, v.w));
      }
    }
  }
  // dK and dV for this warp's 16 keys, once, as bf16 pairs
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int kk = k0 + kl0 + g + r * 8;
    if (kk >= N) continue;
    __nv_bfloat16* row = a.dcomb + ((size_t)b * N + kk) * a.tot + h * D;
#pragma unroll
    for (int n = 0; n < D / 8; ++n) {
      *reinterpret_cast<uint32_t*>(row + W + n * 8 + 2 * t) = bf2(dka[n][r * 2], dka[n][r * 2 + 1]);
      *reinterpret_cast<uint32_t*>(row + 2 * W + n * 8 + 2 * t) = bf2(dva[n][r * 2], dva[n][r * 2 + 1]);
    }
  }
}
template <int D>
inline size_t smem() { return 2 * (BC * LK<D>() + 4 * QT * LK<D>() + 2 * QT * LB + BC * QT) + 4 * (2 * 4 * QT); }

// rows batch rows of [N] positions. dq [rows][N][H D] and cnt [rows][H] scratch (zeroed here); dnbb or dnbbI as
// flashBwdK takes them (accumulated, not zeroed). Requires projLD, H D, tot multiples of 4 (8-byte rows), D 16 or 32.
template <int D>
inline void backward(const half* qkvg, int projLD, const half* bias, const half* dO, const float* lse,
                     const float* Dv, const float* mask, __nv_bfloat16* dcomb, int tot, float* dnbb,
                     long long* dnbbI, float* dq, unsigned* cnt, int rows, int N, int H, cudaStream_t st) {
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(lfFlashBwdK<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem<D>());
    cudaFuncSetAttribute(lfFlashBwdK<D>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    attr = true;
  }
  cudaMemsetAsync(dq, 0, (size_t)rows * N * H * D * 4, st);
  cudaMemsetAsync(cnt, 0, (size_t)rows * H * 4, st);
  Args a{qkvg, projLD, dO, bias, N % 8 == 0, mask, lse, Dv, dq, dcomb, tot, dnbb, dnbbI, cnt, N, H};
  const size_t grid = (size_t)((N + BC - 1) / BC) * H * rows;
  lfFlashBwdK<D><<<(unsigned)grid, THREADS, smem<D>(), st>>>(a);
}

}  // namespace lfb
