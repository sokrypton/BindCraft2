// What bandwidth can this card ACTUALLY reach?  Every "at the roof" claim in
// the README rests on ~450 GB/s taken from a copy kernel that was not written
// to be a bandwidth benchmark.  The A10's spec sheet says 600.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void copy4(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i];
}
__global__ void read4(const float4* __restrict__ a, float* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  float s = 0.f;
  if (i < n) { float4 v = a[i]; s = v.x + v.y + v.z + v.w; }
  if (s == 1e30f) out[0] = s;
}
// FIVE STREAMS, the shape the fused kernels actually have: dGateK reads
// three large arrays and writes two.  The 489 GB/s figure comes from ONE
// read and ONE write, and a DRAM scheduler does not treat those alike.
__global__ void five(const float4* __restrict__ a, const float4* __restrict__ b,
                     const float4* __restrict__ c, float4* __restrict__ d,
                     float4* __restrict__ e, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4 x = a[i], y = b[i], z = c[i];
  d[i] = make_float4(x.x + y.x, x.y + y.y, x.z + z.z, x.w + z.w);
  e[i] = make_float4(y.x + z.x, y.y + z.y, y.z + z.z, y.w + z.w);
}

__global__ void write4(float4* __restrict__ b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = make_float4(1.f, 2.f, 3.f, 4.f);
}

static cudaEvent_t ev0, ev1;
static float timeit(void (*launch)(const float4*, float4*, float*, size_t, int, int),
                    const float4* a, float4* b, float* o, size_t n, int gr, int tb) {
  for (int i = 0; i < 3; ++i) launch(a, b, o, n, gr, tb);
  cudaDeviceSynchronize();
  cudaEventRecord(ev0);
  for (int i = 0; i < 30; ++i) launch(a, b, o, n, gr, tb);
  cudaEventRecord(ev1);
  cudaDeviceSynchronize();
  float ms = 0; cudaEventElapsedTime(&ms, ev0, ev1);
  return ms / 30;
}
static void lcopy(const float4* a, float4* b, float*, size_t n, int gr, int tb) {
  copy4<<<gr, tb>>>(a, b, n); }
static void lread(const float4* a, float4*, float* o, size_t n, int gr, int tb) {
  read4<<<gr, tb>>>(a, o, n); }
static void lwrite(const float4*, float4* b, float*, size_t n, int gr, int tb) {
  write4<<<gr, tb>>>(b, n); }
static const float4 *A5, *B5, *C5; static float4 *D5, *E5;
static void lfive(const float4*, float4*, float*, size_t n, int gr, int tb) {
  five<<<gr, tb>>>(A5, B5, C5, D5, E5, n); }

int main() {
  size_t bytes = 1ull << 28;                 // 256 MB each way
  size_t n4 = bytes / sizeof(float4);
  float4 *a = nullptr, *b = nullptr; float* o = nullptr;
  if (cudaMalloc(&a, bytes) || cudaMalloc(&b, bytes) || cudaMalloc(&o, 4)) {
    printf("alloc failed\n"); return 1; }
  cudaMemset(a, 1, bytes);
  cudaEventCreate(&ev0); cudaEventCreate(&ev1);
  int tb = 256, gr = (int)((n4 + tb - 1) / tb);
  float c = timeit(lcopy, a, b, o, n4, gr, tb);
  float r = timeit(lread, a, b, o, n4, gr, tb);
  float w = timeit(lwrite, a, b, o, n4, gr, tb);
  printf("  copy (read+write)  %7.3f ms   %6.0f GB/s\n", c, 2.0 * bytes / 1e9 / (c * 1e-3));
  printf("  read only          %7.3f ms   %6.0f GB/s\n", r, 1.0 * bytes / 1e9 / (r * 1e-3));
  printf("  write only         %7.3f ms   %6.0f GB/s\n", w, 1.0 * bytes / 1e9 / (w * 1e-3));
  // five streams at a fifth the length each, so the TOTAL bytes match
  size_t n5 = n4 / 5;
  float4* big = nullptr;
  if (cudaMalloc(&big, 5 * n5 * sizeof(float4))) { printf("alloc5 failed\n"); return 1; }
  cudaMemset(big, 1, 5 * n5 * sizeof(float4));
  A5 = big; B5 = big + n5; C5 = big + 2 * n5; D5 = big + 3 * n5; E5 = big + 4 * n5;
  int gr5 = (int)((n5 + tb - 1) / tb);
  float f = timeit(lfive, a, b, o, n5, gr5, tb);
  printf("  five streams (3r+2w) %5.3f ms   %6.0f GB/s\n", f,
         5.0 * n5 * sizeof(float4) / 1e9 / (f * 1e-3));
  return 0;
}
