// WHICH MODULE LEAKS A MASKED ROW?
//
// Every gate in this directory runs with an ALL-ONES mask -- O.inputs() makes
// one -- so nothing here has ever tested masking.  It is not an academic gap:
// bindcraft pads its designed chain to a 32 bucket and masks the pad, and a
// campaign's target+binder length is 32-aligned only by luck, so the native
// path needs a masked pad to be usable at all.  Measured through the FFI, 64
// real residues embedded in a 96-wide masked input move out/pair by 2.0e-2
// and NaN the gradient, where stock JAX on the identical comparison moves by
// 0.000e+00.  So stock is mask-exact and we are not.
//
// This bisects it the way block.cu's prefix ladder does: run the same 64
// residues at N=64 and embedded in a masked N=96, with the first k pair steps
// and j msa steps enabled, and report the first step at which the 64x64
// corner diverges.
#include "block.cuh"
#include <cstdio>

__global__ void embedPairK(float* dst, const float* src, int n, int p, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)p * p * C) return;
  int c = (int)(i % C);
  size_t rc = i / C;
  int col = (int)(rc % p), row = (int)(rc / p);
  dst[i] = (row < n && col < n)
               ? src[((size_t)row * n + col) * C + c] : 0.f;
}
__global__ void embedMsaK(float* dst, const float* src, int S, int n, int p,
                          int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)S * p * C) return;
  int c = (int)(i % C);
  size_t rc = i / C;
  int col = (int)(rc % p), row = (int)(rc / p);
  dst[i] = (col < n) ? src[((size_t)row * n + col) * C + c] : 0.f;
}
__global__ void embedMask2K(float* dst, int n, int p) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)p * p) return;
  int col = (int)(i % p), row = (int)(i / p);
  dst[i] = (row < n && col < n) ? 1.f : 0.f;
}
__global__ void embedMask1K(float* dst, int S, int n, int p) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)S * p) return;
  dst[i] = ((int)(i % p) < n) ? 1.f : 0.f;
}
__global__ void cornerPairK(float* dst, const float* src, int n, int p, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)n * n * C) return;
  int c = (int)(i % C);
  size_t rc = i / C;
  int col = (int)(rc % n), row = (int)(rc / n);
  dst[((size_t)row * p + col) * C + c] = src[i];
}
__global__ void cornerMsaK(float* dst, const float* src, int S, int n, int p,
                           int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)S * n * C) return;
  int c = (int)(i % C);
  size_t rc = i / C;
  int col = (int)(rc % n), row = (int)(rc / n);
  dst[((size_t)row * p + col) * C + c] = src[i];
}
__global__ void fillK(float* d, size_t n, float v) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = v;
}
__global__ void randK(float* d, size_t n, uint32_t seed, float scale) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t h = dropHash(seed, (uint32_t)i, 0u, 0u);
  d[i] = (h * 2.3283064365386963e-10f - 0.5f) * 2.f * scale;
}

int main(int argc, char** argv) {
  Oracle o(argv[1]);
  const std::string P =
      o.has("param/evoformer_iteration/pair_transition/transition1/weights") ||
      o.has("param/evoformer_iteration/outer_product_mean/left_projection/weights")
          ? "param/evoformer_iteration/" : "param/";
  const int N = 64, Q = 96, S = 2;
  const int CZ_ = block::CZ, CM_ = block::CM;
  size_t npN = (size_t)N * N * CZ_, nmN = (size_t)S * N * CM_;
  size_t npQ = (size_t)Q * Q * CZ_, nmQ = (size_t)S * Q * CM_;

  float *pN, *mN, *pmN, *mmN, *pQ, *mQ, *pmQ, *mmQ;
  CU(cudaMalloc(&pN, npN * 4)); CU(cudaMalloc(&mN, nmN * 4));
  CU(cudaMalloc(&pmN, (size_t)N * N * 4)); CU(cudaMalloc(&mmN, (size_t)S * N * 4));
  CU(cudaMalloc(&pQ, npQ * 4)); CU(cudaMalloc(&mQ, nmQ * 4));
  CU(cudaMalloc(&pmQ, (size_t)Q * Q * 4)); CU(cudaMalloc(&mmQ, (size_t)S * Q * 4));
  auto G = [](size_t n) { return (int)((n + 255) / 256); };
  randK<<<G(npN), 256>>>(pN, npN, 1, 0.1f);
  randK<<<G(nmN), 256>>>(mN, nmN, 2, 0.1f);
  fillK<<<G((size_t)N * N), 256>>>(pmN, (size_t)N * N, 1.f);
  fillK<<<G((size_t)S * N), 256>>>(mmN, (size_t)S * N, 1.f);
  embedPairK<<<G(npQ), 256>>>(pQ, pN, N, Q, CZ_);
  embedMsaK<<<G(nmQ), 256>>>(mQ, mN, S, N, Q, CM_);
  embedMask2K<<<G((size_t)Q * Q), 256>>>(pmQ, N, Q);
  embedMask1K<<<G((size_t)S * Q), 256>>>(mmQ, S, N, Q);
  CU(cudaDeviceSynchronize());

  auto narrow = [&](const float* src, size_t n) {
    __nv_bfloat16* d; CU(cudaMalloc(&d, n * 2));
    toBf16K<<<G(n), 256>>>(d, src, n);
    return d;
  };
  __nv_bfloat16* pNb = narrow(pN, npN);
  __nv_bfloat16* pQb = narrow(pQ, npQ);

  // THE CONTROL.  N=64 against N=96 also changes tile counts, and on a bf16
  // chain a different reduction order alone moves ~1e-2 -- the rate-0 gates
  // read relL2 2.1e-02 against JAX for exactly that reason.  So a second
  // 96-wide input, identical except that the MASKED region holds large
  // values instead of zeros: same N, same tiles, same order, and the only
  // thing that differs is what the mask is supposed to be hiding.  A
  // mask-exact stack reads 0.000e+00 on this one whatever the first column
  // says.
  float* pQ2; CU(cudaMalloc(&pQ2, npQ * 4));
  float* mQ2; CU(cudaMalloc(&mQ2, nmQ * 4));
  randK<<<G(npQ), 256>>>(pQ2, npQ, 7, 100.f);
  randK<<<G(nmQ), 256>>>(mQ2, nmQ, 8, 100.f);
  cornerPairK<<<G(npN), 256>>>(pQ2, pN, N, Q, CZ_);   // the REAL corner only
  cornerMsaK<<<G(nmN), 256>>>(mQ2, mN, S, N, Q, CM_);
  CU(cudaDeviceSynchronize());
  __nv_bfloat16* pQ2b = narrow(pQ2, npQ);

  printf("  %-16s %12s %12s %12s\n", "steps", "out/pair", "out/msa",
         "pad-content");
  printf("  (64 real residues, alone and embedded in a masked 96; a "
         "mask-exact\n   stack reads 0.000e+00 on both, as stock JAX does)\n\n");

  for (int k = 0; k <= 5; ++k) {
    for (int j = (k == 0 ? 0 : 4); j <= 4; ++j) {
      block::Block bN, bQ;
      for (auto* b : {&bN, &bQ}) {
        b->do_msa = true; b->do_pair = true; b->n_pair = k; b->n_msa = j;
      }
      bN.alloc(N, S); bN.setWeights(o, P);
      bQ.alloc(Q, S); bQ.setWeights(o, P);
      bN.forward(pNb, mN, pmN, mmN);
      bQ.forward(pQb, mQ, pmQ, mmQ);
      CU(cudaDeviceSynchronize());
      static float* wC = nullptr;
      if (!wC) CU(cudaMalloc(&wC, npQ * 4));
      widenBf16K<<<G(npQ), 256>>>(wC, bQ.p6b, npQ);
      CU(cudaDeviceSynchronize());
      auto hC = fetch(wC, npQ);              // zero-filled pad
      bQ.forward(pQ2b, mQ2, pmQ, mmQ);       // large-filled pad, same N
      CU(cudaDeviceSynchronize());

      static float *wN = nullptr, *wQ = nullptr;
      if (!wN) { CU(cudaMalloc(&wN, npN * 4)); CU(cudaMalloc(&wQ, npQ * 4)); }
      widenBf16K<<<G(npN), 256>>>(wN, bN.p6b, npN);
      widenBf16K<<<G(npQ), 256>>>(wQ, bQ.p6b, npQ);
      CU(cudaDeviceSynchronize());
      auto hN = fetch(wN, npN), hQ = fetch(wQ, npQ);
      auto sN = fetch(bN.m3, nmN), sQ = fetch(bQ.m3, nmQ);

      static float* wC2 = nullptr;
      if (!wC2) CU(cudaMalloc(&wC2, npQ * 4));
      widenBf16K<<<G(npQ), 256>>>(wC2, bQ.p6b, npQ);
      CU(cudaDeviceSynchronize());
      auto hC2 = fetch(wC2, npQ);
      double dc = 0, mc2 = 0;
      for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
          for (int e = 0; e < CZ_; ++e) {
            size_t ix = ((size_t)r * Q + c) * CZ_ + e;
            dc = std::max(dc, std::abs((double)hC[ix] - (double)hC2[ix]));
            mc2 = std::max(mc2, std::abs((double)hC[ix]));
          }
      double dp = 0, mp = 0, dm = 0, mm2 = 0;
      for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
          for (int e = 0; e < CZ_; ++e) {
            double a = hN[((size_t)r * N + c) * CZ_ + e];
            double b = hQ[((size_t)r * Q + c) * CZ_ + e];
            dp = std::max(dp, std::abs(a - b)); mp = std::max(mp, std::abs(a));
          }
      for (int s = 0; s < S; ++s)
        for (int c = 0; c < N; ++c)
          for (int e = 0; e < CM_; ++e) {
            double a = sN[((size_t)s * N + c) * CM_ + e];
            double b = sQ[((size_t)s * Q + c) * CM_ + e];
            dm = std::max(dm, std::abs(a - b)); mm2 = std::max(mm2, std::abs(a));
          }
      if (envOn("BWD")) {
        static float* ctm = nullptr;
        static __nv_bfloat16* ctp = nullptr;
        if (!ctm) {
          CU(cudaMalloc(&ctm, nmQ * 4));
          CU(cudaMalloc(&ctp, npQ * 2));
          float* t; CU(cudaMalloc(&t, npQ * 4));
          randK<<<G(npQ), 256>>>(t, npQ, 11, 1.f);
          randK<<<G(nmQ), 256>>>(ctm, nmQ, 12, 1.f);
          toBf16K<<<G(npQ), 256>>>(ctp, t, npQ);
          CU(cudaDeviceSynchronize());
        }
        bQ.forward(pQb, mQ, pmQ, mmQ);
        bQ.backward(ctp, ctm, pmQ, mmQ);
        CU(cudaDeviceSynchronize());
      }
      char lbl[64];
      snprintf(lbl, sizeof lbl, "pair %d msa %d", k, j);
      printf("  %-16s %12.3e %12.3e %12.3e\n", lbl, dp / (mp > 0 ? mp : 1),
             dm / (mm2 > 0 ? mm2 : 1), dc / (mc2 > 0 ? mc2 : 1));
    }
  }
  return 0;
}
