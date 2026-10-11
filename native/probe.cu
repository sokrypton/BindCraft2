// Which (dtype, epilogue, bias type) combinations does this cuBLASLt actually
// serve at the transition's shapes?  The heuristic returns NOT_SUPPORTED (7)
// without saying which attribute it objected to, so ask it one at a time.
//
//   nvcc -O3 -std=c++17 -arch=sm_86 probe.cu -lcublasLt -o probe && ./probe
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>

static const int M = 65536, C = 128, I = 512;

struct Case { const char* name; cublasLtEpilogue_t epi; bool aux, bias; };

static bool try_one(cudaDataType dt, cudaDataType bias_dt, bool transa,
                    const Case& c, int Mm, int Nn, int K, void* A, void* B,
                    void* D, void* bias, void* aux, const char** why) {
  cublasLtHandle_t h;
  cublasLtCreate(&h);
  cublasLtMatmulDesc_t op;
  if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) {
    *why = "desc"; return false;
  }
  cublasOperation_t ta = transa ? CUBLAS_OP_T : CUBLAS_OP_N, tn = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn));
  cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &c.epi,
                                 sizeof(c.epi));
  if (c.bias) {
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias,
                                   sizeof(bias));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                   &bias_dt, sizeof(bias_dt));
  }
  if (c.aux) {
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                   &aux, sizeof(aux));
    int64_t ld = Nn;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &ld,
                                   sizeof(ld));
  }
  cublasLtMatrixLayout_t la, lb, ldd;
  if (transa) cublasLtMatrixLayoutCreate(&la, dt, K, Nn, K);
  else cublasLtMatrixLayoutCreate(&la, dt, Nn, K, Nn);
  cublasLtMatrixLayoutCreate(&lb, dt, K, Mm, K);
  cublasLtMatrixLayoutCreate(&ldd, dt, Nn, Mm, Nn);

  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  size_t ws = 64u << 20;
  cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
  cublasLtMatmulHeuristicResult_t heur;
  int found = 0;
  cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, op, la, lb, ldd, ldd,
                                                     pref, 1, &heur, &found);
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(ldd);
  cublasLtMatmulDescDestroy(op);
  cublasLtDestroy(h);
  static char buf[64];
  snprintf(buf, sizeof(buf), "status %d found %d", (int)st, found);
  *why = buf;
  return st == CUBLAS_STATUS_SUCCESS && found > 0;
}

int main() {
  void *A, *B, *D, *bias, *aux;
  cudaMalloc(&A, (size_t)M * I * 4);
  cudaMalloc(&B, (size_t)M * I * 4);
  cudaMalloc(&D, (size_t)M * I * 4);
  cudaMalloc(&bias, I * 4);
  cudaMalloc(&aux, (size_t)M * I / 8 + 1024);

  Case cases[] = {
      {"DEFAULT", CUBLASLT_EPILOGUE_DEFAULT, false, false},
      {"BIAS", CUBLASLT_EPILOGUE_BIAS, false, true},
      {"RELU", CUBLASLT_EPILOGUE_RELU, false, false},
      {"RELU_BIAS", CUBLASLT_EPILOGUE_RELU_BIAS, false, true},
      {"RELU_AUX", CUBLASLT_EPILOGUE_RELU_AUX, true, false},
      {"RELU_AUX_BIAS", CUBLASLT_EPILOGUE_RELU_AUX_BIAS, true, true},
      {"DRELU", CUBLASLT_EPILOGUE_DRELU, true, false},
  };
  struct { const char* n; cudaDataType d; } dts[] = {
      {"bf16", CUDA_R_16BF}, {"f16", CUDA_R_16F}, {"f32", CUDA_R_32F}};
  struct { const char* n; cudaDataType d; } bts[] = {
      {"bias f32", CUDA_R_32F}, {"bias bf16", CUDA_R_16BF}};

  int ver = 0;
  cublasLtHandle_t h; cublasLtCreate(&h);
  printf("cublasLt version %zu\n", cublasLtGetVersion());
  (void)ver; cublasLtDestroy(h);

  printf("\nGEMM1 shape M=%d Nn=%d K=%d (the [M,C] @ [C,I])\n", M, I, C);
  for (auto& dt : dts)
    for (auto& bt : bts)
      for (int transa = 1; transa >= 0; --transa) {
        printf("  %-5s %-9s transa=%c :", dt.n, bt.n, transa ? 'T' : 'N');
        for (auto& c : cases) {
          const char* why;
          bool ok = try_one(dt.d, bt.d, transa, c, M, I, C, A, B, D, bias, aux, &why);
          printf("  %s=%s", c.name, ok ? "ok" : "NO");
        }
        printf("\n");
      }
  return 0;
}
