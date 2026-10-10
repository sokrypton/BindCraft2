// The few symbols localfold's flash.cuh takes from its own common.cuh.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// localfold runs everything on one stream; this prototype sets it to the stream
// the module was captured on so the kernels join its CUDA graph.
inline cudaStream_t LF_STREAM = nullptr;
#define STREAM LF_STREAM

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d cuda: %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + __expf(-x)); }

// The shared-memory opt-in, verbatim in behaviour from localfold's common.cuh:
// a fused kernel asks fitsSmem() before choosing its path, and smemAttr()
// refuses by name rather than letting the launch fail with "invalid argument".
#include <algorithm>
inline size_t SMEM_LIMIT = 0;
inline size_t smemLimit() {
  if (!SMEM_LIMIT) {
    int dev = 0, v = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    SMEM_LIMIT = (size_t)v;
    if (const char* e = getenv("LOCALFOLD_SMEM_LIMIT"))
      SMEM_LIMIT = std::min(SMEM_LIMIT, (size_t)atol(e));
  }
  return SMEM_LIMIT;
}
inline bool fitsSmem(size_t bytes) { return bytes <= smemLimit(); }
template <class K> void smemAttrNamed(K kernel, size_t bytes, const char* name) {
  if (!fitsSmem(bytes)) {
    fprintf(stderr, "%s wants %zu bytes of shared memory a block; this device "
                    "allows %zu\n", name, bytes, smemLimit());
    exit(1);
  }
  CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)bytes));
}
#define smemAttr(kernel, bytes) smemAttrNamed(kernel, (size_t)(bytes), #kernel)

// flashRegStaged() is defined inside flash.cuh itself.
