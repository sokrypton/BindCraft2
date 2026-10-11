// Does the leading dimension explain why the attention GEMMs run at 8-9
// TFLOP/s when the triangle multiplication's reach 35-43?
//
// q, k and v are 32-wide slices of the one [M, 4C+H] projection, so a batched
// GEMM reading q[b,h] walks rows 516*2 = 1032 bytes apart to use 64 of them.
// This times the SAME arithmetic at three leading dimensions:
//
//   516 : what the fused projection gives (current)
//   128 : one array per projection, [b][n][h][d]
//    32 : fully packed per head, [b][h][n][d]
//
// If the 32 case is much faster, the fix is a layout change and does not need a
// flash kernel.  If it is not, the shape itself (a 32-deep reduction writing an
// N x N plane) is the limit and only fusion removes it.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 ldbench.cu -lcublas -o ldbench && ./ldbench 256
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CU(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("cuda %s\n",cudaGetErrorString(e));exit(1);} } while(0)
#define CUB(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cublas %d\n",(int)s);exit(1);} } while(0)

static const int H = 4, D = 32;

int main(int argc, char** argv) {
  int N = argc > 1 ? atoi(argv[1]) : 256;
  int M = N * N;
  cublasHandle_t h;
  CUB(cublasCreate(&h));

  __nv_bfloat16 *big, *logits;
  CU(cudaMalloc(&big, (size_t)M * 520 * 2));
  CU(cudaMalloc(&logits, (size_t)N * H * N * N * 2));
  CU(cudaMemset(big, 0, (size_t)M * 520 * 2));

  auto run = [&](int ld, const char* tag) {
    float alpha = 1.f, beta = 0.f;
    cudaEvent_t a, b;
    CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b));
    auto once = [&] {
      for (int hh = 0; hh < H; ++hh) {
        const __nv_bfloat16* q = big + hh * D;
        const __nv_bfloat16* k = big + (size_t)M * 0 + hh * D + ld / 4;  // any offset
        CUB(cublasGemmStridedBatchedEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, D, &alpha, k, CUDA_R_16BF, ld,
            (long long)N * ld, q, CUDA_R_16BF, ld, (long long)N * ld, &beta,
            logits + (size_t)hh * N * N, CUDA_R_16BF, N, (long long)H * N * N, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      }
    };
    for (int i = 0; i < 5; ++i) once();
    CU(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < 20; ++i) {
      CU(cudaEventRecord(a));
      once();
      CU(cudaEventRecord(b));
      CU(cudaEventSynchronize(b));
      float ms; CU(cudaEventElapsedTime(&ms, a, b));
      if (ms < best) best = ms;
    }
    double gf = 2.0 * (double)N * N * N * H * D / 1e9;
    printf("  ld=%-4d  %7.3f ms   %5.1f TFLOP/s   %s\n", ld, best, gf / best, tag);
  };

  printf("logits = q k^T, N=%d, H=%d, D=%d  (%.1f GFLOP)\n", N, H, D,
         2.0 * (double)N * N * N * H * D / 1e9);
  run(516, "the fused projection's stride (current)");
  run(128, "one array per projection");
  run(32, "packed per head, [b][h][n][d]");
  return 0;
}
