// lf_flash_bwd.cuh against flashBwdK: the same inputs, the same contract, dQ/dK/dV/dnbb compared, both timed.
//   nvcc -O3 -std=c++17 -arch=sm_80 lfbwdgate.cu -o lfbwdgate -lcurand && ./lfbwdgate [N=256] [rows=N]
// D 32 (fb: triangle and MSA row attention) and D 16 (fb16: the template pair stack), H 4.
// (sm_86 on an A10.) flashBwdK's time includes the scatter it needs to land dK/dV in dcomb, which the port does not.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <curand.h>
#include "flash_bwd.cuh"
#include "lf_flash_bwd.cuh"

static __global__ void toHalfK(const float* x, half* y, size_t n, float s) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half(x[i] * s);
}
// the forward's log2-domain lse, naively: lse[b,h,i] = log2 sum_j exp2((q_i . k_j) LOG2E + bias + mask term)
static __global__ void lseK(const half* qkvg, int projLD, const half* bias, const float* mask, float* lse, int N, int H, int D,
                            size_t total) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int i = t % N, h = (t / N) % H; const size_t b = t / ((size_t)N * H);
  const int W = H * D;
  float m = -INFINITY, l = 0.f;
  for (int j = 0; j < N; ++j) {
    float s = 0.f;
    for (int d = 0; d < D; ++d)
      s += __half2float(qkvg[(b * N + i) * projLD + h * D + d]) * __half2float(qkvg[(b * N + j) * projLD + W + h * D + d]);
    s = s * 1.4426950408889634f + __half2float(bias[((size_t)h * N + i) * N + j]) + 1e9f * (mask[b * N + j] - 1.f) * 1.4426950408889634f;
    const float nm = fmaxf(m, s);
    l = l * exp2f(m - nm) + exp2f(s - nm); m = nm;
  }
  lse[t] = m + log2f(l);
}
static __global__ void scatterKV(const float* dK, const float* dV, __nv_bfloat16* dcomb, size_t M, int tot, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * (size_t)C) return;
  size_t r = i / C; int c = i % C;
  dcomb[r * tot + C + c] = __float2bfloat16(dK[i]);
  dcomb[r * tot + 2 * C + c] = __float2bfloat16(dV[i]);
}
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

template <int D>
static bool run(int N, int rows) {
  const int H = 4, W = H * D;
  const int projLD = 4 * W, TOT = 4 * W + H;
  const size_t M = (size_t)rows * N;
  curandGenerator_t gen; curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); curandSetPseudoRandomGeneratorSeed(gen, 7);
  auto rnd = [&](size_t n) { float* p; CK(cudaMalloc(&p, ((n + 1) & ~(size_t)1) * 4)); curandGenerateNormal(gen, p, (n + 1) & ~(size_t)1, 0.f, 1.f); return p; };
  half *qkvg, *dO, *bias;
  CK(cudaMalloc(&qkvg, M * projLD * 2)); CK(cudaMalloc(&dO, M * W * 2)); CK(cudaMalloc(&bias, (size_t)H * N * N * 2));
  { float* f = rnd(M * projLD); toHalfK<<<(M * projLD + 255) / 256, 256>>>(f, qkvg, M * projLD, 0.5f); cudaFree(f); }
  { float* f = rnd(M * W); toHalfK<<<(M * W + 255) / 256, 256>>>(f, dO, M * W, 0.1f); cudaFree(f); }
  { float* f = rnd((size_t)H * N * N); toHalfK<<<((size_t)H * N * N + 255) / 256, 256>>>(f, bias, (size_t)H * N * N, 1.4426950408889634f); cudaFree(f); }
  std::vector<float> mh(M, 1.f);
  for (size_t i = 0; i < M; ++i) if ((i * 2654435761u) % 13 == 0) mh[i] = 0.f;   // some masked keys
  float* mask; CK(cudaMalloc(&mask, M * 4)); CK(cudaMemcpy(mask, mh.data(), M * 4, cudaMemcpyHostToDevice));
  float* lse; CK(cudaMalloc(&lse, M * H * 4));
  lseK<<<(M * H + 127) / 128, 128>>>(qkvg, projLD, bias, mask, lse, N, H, D, M * H);
  float* Dv = rnd(M * H);   // (any row term: both kernels take it as given)
  { std::vector<float> d(M * H); CK(cudaMemcpy(d.data(), Dv, M * H * 4, cudaMemcpyDeviceToHost)); for (auto& x : d) x *= 0.01f;
    CK(cudaMemcpy(Dv, d.data(), M * H * 4, cudaMemcpyHostToDevice)); }
  __nv_bfloat16 *dc0, *dc1; CK(cudaMalloc(&dc0, M * TOT * 2)); CK(cudaMalloc(&dc1, M * TOT * 2));
  float *dKf, *dVf, *dQf, *nb0, *nb1, *dq; unsigned* cnt;
  CK(cudaMalloc(&dKf, M * W * 4)); CK(cudaMalloc(&dVf, M * W * 4)); CK(cudaMalloc(&dQf, M * W * 4));
  CK(cudaMalloc(&nb0, (size_t)H * N * N * 4)); CK(cudaMalloc(&nb1, (size_t)H * N * N * 4));
  CK(cudaMalloc(&dq, M * W * 4)); CK(cudaMalloc(&cnt, (size_t)rows * H * 4));
  // (the D 16 build is flashBwdK's fb16, the template pair stack's)
  auto fbK = D == 32 ? fb::flashBwdK<true, true, false, true> : fb16::flashBwdK<true, true, false, true>;
  const int fbBM = D == 32 ? fb::BM : fb16::BM, fbWARPS = D == 32 ? fb::WARPS : fb16::WARPS;
  const size_t fbSmem = D == 32 ? fb::bwdSmem() : fb16::bwdSmem();
  CK(cudaFuncSetAttribute((const void*)fbK, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)fbSmem));
  cudaStream_t st = 0;
  auto theirs = [&]() {
    cudaMemsetAsync(dKf, 0, M * W * 4, st); cudaMemsetAsync(dVf, 0, M * W * 4, st); cudaMemsetAsync(nb0, 0, (size_t)H * N * N * 4, st);
    dim3 g((N + fbBM - 1) / fbBM, H, rows);
    fbK<<<g, 32 * fbWARPS, fbSmem, st>>>(qkvg, projLD, bias, dO, lse, Dv, mask,
        nullptr, dQf, dKf, dVf, N, H, nb0, nullptr, nullptr, 0, dc0, TOT);
    scatterKV<<<(M * W + 255) / 256, 256, 0, st>>>(dKf, dVf, dc0, M, TOT, W);
  };
  auto ours = [&]() {
    cudaMemsetAsync(nb1, 0, (size_t)H * N * N * 4, st);
    lfb::backward<D>(qkvg, projLD, bias, dO, lse, Dv, mask, dc1, TOT, nb1, nullptr, dq, cnt, rows, N, H, st);
  };
  theirs(); ours(); CK(cudaDeviceSynchronize());
  // compare dQ, dK, dV (dcomb columns) and dnbb
  std::vector<__nv_bfloat16> a(M * TOT), b(M * TOT);
  CK(cudaMemcpy(a.data(), dc0, M * TOT * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dc1, M * TOT * 2, cudaMemcpyDeviceToHost));
  const char* nm[3] = {"dQ", "dK", "dV"};
  bool ok = true;
  for (int r = 0; r < 3; ++r) {
    double num = 0, den = 0;
    for (size_t i = 0; i < M; ++i)
      for (int c = 0; c < W; ++c) {
        const double x = __bfloat162float(a[i * TOT + r * W + c]), y = __bfloat162float(b[i * TOT + r * W + c]);
        num += (x - y) * (x - y); den += x * x;
      }
    const double rel = sqrt(num / (den + 1e-30));
    printf("  %s rel rms %.2e\n", nm[r], rel);
    ok &= rel < 2e-2;
  }
  {
    std::vector<float> x((size_t)H * N * N), y((size_t)H * N * N);
    CK(cudaMemcpy(x.data(), nb0, x.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(y.data(), nb1, y.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0; for (size_t i = 0; i < x.size(); ++i) { num += (x[i] - y[i]) * (x[i] - y[i]); den += x[i] * x[i]; }
    const double rel = sqrt(num / (den + 1e-30));
    printf("  dnbb rel rms %.2e\n", rel);
    ok &= rel < 2e-2;
  }
  // timing, interleaved
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  float t0 = 1e9, t1 = 1e9;
  for (int r = 0; r < 10; ++r) {
    float t;
    cudaEventRecord(e0, st); theirs(); cudaEventRecord(e1, st); cudaEventSynchronize(e1); cudaEventElapsedTime(&t, e0, e1); if (r) t0 = fminf(t0, t);
    cudaEventRecord(e0, st); ours(); cudaEventRecord(e1, st); cudaEventSynchronize(e1); cudaEventElapsedTime(&t, e0, e1); if (r) t1 = fminf(t1, t);
  }
  printf("D %d N %d rows %d: flashBwdK + scatter %.3f ms, lf %.3f ms (%.2fx)\n", D, N, rows, t0, t1, t0 / t1);
  CK(cudaGetLastError());
  cudaFree(qkvg); cudaFree(dO); cudaFree(bias); cudaFree(mask); cudaFree(lse); cudaFree(Dv); cudaFree(dc0); cudaFree(dc1);
  cudaFree(dKf); cudaFree(dVf); cudaFree(dQf); cudaFree(nb0); cudaFree(nb1); cudaFree(dq); cudaFree(cnt);
  return ok;
}
int main(int argc, char** argv) {
  const int N = argc > 1 ? atoi(argv[1]) : 256, rows = argc > 2 ? atoi(argv[2]) : N;
  const bool ok = run<32>(N, rows) & run<16>(N, rows);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
