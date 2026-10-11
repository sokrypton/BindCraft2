// Standalone gate for the MSA column attention module (msacol.cuh).
//
// The module takes and returns bf16, which is what a block chains; this
// converts the oracle's float32 at the boundary, so the error reported here is
// the one a real stack would carry rather than a friendlier one.
#include "msacol.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: msacol <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& msa = o["in/msa"];
  int S = msa.shape[0], N = msa.shape[1];
  size_t M = (size_t)N * S, n = M * mcol::CM;
  printf("MSA column attention  S=%d N=%d  (attends over S)\n", S, N);

  msa.to_device();
  Tensor& mmask = o["in/msa_mask"]; mmask.to_device();
  Tensor& ct = o["ct/msa"]; ct.to_device();

  mcol::MsaCol m;
  m.alloc(N, S);
  m.setWeights(o, "param/msa_column_attention/");

  __nv_bfloat16* msaB;
  float* dmsa;
  CU(cudaMalloc(&msaB, n * 2)); CU(cudaMalloc(&dmsa, n * 4));
  mcol::toBf16K<<<(n + 255) / 256, 256>>>(msa.dev, msaB, n);
  const float* doutB = ct.dev;            // gradients cross the boundary in f32

  m.forward(msaB, mmask.dev);
  m.backward(doutB, msaB, mmask.dev, dmsa);
  CU(cudaDeviceSynchronize());
  {
    std::vector<__nv_bfloat16> h(n);
    CU(cudaMemcpy(h.data(), m.out, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> y(n);
    for (size_t i = 0; i < n; ++i) y[i] = msa.host[i] + __bfloat162float(h[i]);
    printf("forward\n");
    report("out/msa", compare(y, o["out/msa"].host));
    auto gh = fetch(dmsa, n);
    std::vector<float> g(n);
    for (size_t i = 0; i < n; ++i) g[i] = ct.host[i] + gh[i];
    printf("backward\n");
    report("grad/msa", compare(g, o["grad/msa"].host));
  }
  double tf = time_ms([&] { m.forward(msaB, mmask.dev); }, reps);
  double tb = time_ms([&] { m.backward(doutB, msaB, mmask.dev, dmsa); }, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb,
         time_ms([&] { m.forward(msaB, mmask.dev);
                       m.backward(doutB, msaB, mmask.dev, dmsa); }, reps));
  return 0;
}
