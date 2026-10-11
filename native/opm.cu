// Standalone gate for the outer product mean module (opm.cuh).
#include "opm.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: opm <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& msa = o["in/msa"];
  Tensor& pair = o["in/pair"];
  int S = msa.shape[0], N = msa.shape[1];
  size_t nm = (size_t)S * N * opm::CM, np = (size_t)N * N * opm::CZ;
  printf("outer product mean  S=%d N=%d  (reassociated)\n", S, N);

  msa.to_device(); pair.to_device();
  Tensor& mmask = o["in/msa_mask"]; mmask.to_device();
  Tensor& ctm = o["ct/msa"]; ctm.to_device();
  Tensor& ctp = o["ct/pair"]; ctp.to_device();

  opm::Opm m;
  m.alloc(N, S);
  m.setWeights(o, "param/outer_product_mean/");

  __nv_bfloat16* msaB;
  float* dmsa;
  CU(cudaMalloc(&msaB, nm * 2)); CU(cudaMalloc(&dmsa, nm * 4));
  opm::f32ToBf16K<<<(nm + 255) / 256, 256>>>(msa.dev, msaB, nm);

  m.forward(msaB, mmask.dev);
  m.backward(ctp.dev, msaB, mmask.dev, dmsa);
  CU(cudaDeviceSynchronize());
  {
    std::vector<__nv_bfloat16> h(np);
    CU(cudaMemcpy(h.data(), m.out, np * 2, cudaMemcpyDeviceToHost));
    std::vector<float> y(np);
    for (size_t i = 0; i < np; ++i) y[i] = pair.host[i] + __bfloat162float(h[i]);
    printf("forward\n");
    report("out/pair", compare(y, o["out/pair"].host));
    auto g = fetch(dmsa, nm);
    for (size_t i = 0; i < nm; ++i) g[i] += ctm.host[i];
    printf("backward\n");
    report("grad/msa", compare(g, o["grad/msa"].host));
  }
  double tf = time_ms([&] { m.forward(msaB, mmask.dev); }, reps);
  double tb = time_ms([&] { m.backward(ctp.dev, msaB, mmask.dev, dmsa); }, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb, tf + tb);
  return 0;
}
