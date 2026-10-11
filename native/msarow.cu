// Standalone gate for the MSA row attention module (msarow.cuh).
// Activations cross the boundary as bf16 and gradients as float32, so what this
// reports is the precision a chained block would carry.
#include "msarow.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: msarow <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& msa = o["in/msa"];
  Tensor& pair = o["in/pair"];
  int S = msa.shape[0], N = msa.shape[1];
  size_t Mm = (size_t)S * N, Mp = (size_t)N * N;
  size_t nm = Mm * mrow::CM, np = Mp * mrow::CZ;
  printf("MSA row attention with pair bias  S=%d N=%d\n", S, N);

  msa.to_device(); pair.to_device();
  Tensor& mmask = o["in/msa_mask"]; mmask.to_device();
  Tensor& ctm = o["ct/msa"]; ctm.to_device();
  Tensor& ctp = o["ct/pair"]; ctp.to_device();

  mrow::MsaRow m;
  m.alloc(N, S);
  m.setWeights(o, "param/msa_row_attention_with_pair_bias/");

  __nv_bfloat16 *msaB, *pairB;
  float *dmsa, *dpair;
  CU(cudaMalloc(&msaB, nm * 2)); CU(cudaMalloc(&pairB, np * 2));
  CU(cudaMalloc(&dmsa, nm * 4)); CU(cudaMalloc(&dpair, np * 4));
  mrow::f32ToBf16K<<<(nm + 255) / 256, 256>>>(msa.dev, msaB, nm);
  mrow::f32ToBf16K<<<(np + 255) / 256, 256>>>(pair.dev, pairB, np);

  m.forward(msaB, pairB, mmask.dev);
  m.backward(ctm.dev, msaB, pairB, mmask.dev, dmsa, dpair);
  CU(cudaDeviceSynchronize());
  {
    std::vector<__nv_bfloat16> h(nm);
    CU(cudaMemcpy(h.data(), m.out, nm * 2, cudaMemcpyDeviceToHost));
    std::vector<float> y(nm);
    for (size_t i = 0; i < nm; ++i) y[i] = msa.host[i] + __bfloat162float(h[i]);
    printf("forward\n");
    report("out/msa", compare(y, o["out/msa"].host));
    auto gm = fetch(dmsa, nm);
    for (size_t i = 0; i < nm; ++i) gm[i] += ctm.host[i];
    auto gp = fetch(dpair, np);
    for (size_t i = 0; i < np; ++i) gp[i] += ctp.host[i];
    printf("backward\n");
    report("grad/msa", compare(gm, o["grad/msa"].host));
    report("grad/pair", compare(gp, o["grad/pair"].host));
  }
  double tf = time_ms([&] { m.forward(msaB, pairB, mmask.dev); }, reps);
  double tb = time_ms([&] { m.backward(ctm.dev, msaB, pairB, mmask.dev, dmsa, dpair); }, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb, tf + tb);
  return 0;
}
