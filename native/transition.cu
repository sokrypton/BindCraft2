// Standalone gate for the transition module (transition.cuh).
//
// One binary covers both tracks: which transition is in the dump is read off
// the manifest, and the widths follow from the activation's own shape.
#include "transition.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: transition <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;

  const bool msa = o.has("param/msa_transition/transition1/weights") ||
                   o.has("param/msa_transition/~/transition1/weights");
  Tensor& x = o[msa ? "in/msa" : "in/pair"];
  Tensor& ct = o[msa ? "ct/msa" : "ct/pair"];
  int M = 1;
  for (size_t i = 0; i + 1 < x.shape.size(); ++i) M *= x.shape[i];
  int C = x.shape.back();
  size_t n = (size_t)M * C;
  // THE INTERMEDIATE FACTOR IS READ, NOT ASSUMED.  The Evoformer's transition
  // is 4x and the TEMPLATE pair stack's is 2x, and the weights say which:
  // transition1 is [C, I].
  const char* w1 = o.has("param/pair_transition/transition1/weights")
      ? "param/pair_transition/transition1/weights"
      : (o.has("param/msa_transition/transition1/weights")
         ? "param/msa_transition/transition1/weights" : nullptr);
  int I = w1 ? (int)o[w1].shape.back() : 4 * C;
  printf("%s transition  M=%d rows  C=%d  I=%d (%dx)\n",
         msa ? "MSA" : "pair", M, C, I, I / C);

  x.to_device(); ct.to_device();
  trans::Trans m;
  m.alloc(M, C, nullptr, I / C);
  m.setWeights(o, msa ? "param/msa_transition/" : "param/pair_transition/");

  __nv_bfloat16* xB;
  float* dx;
  CU(cudaMalloc(&xB, n * 2)); CU(cudaMalloc(&dx, n * 4));
  trans::f32ToBf16K<<<(n + 255) / 256, 256>>>(x.dev, xB, n);

  m.forward(xB);
  m.backward(ct.dev, xB, dx);
  CU(cudaDeviceSynchronize());
  {
    std::vector<__nv_bfloat16> h(n);
    CU(cudaMemcpy(h.data(), m.out, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> y(n);
    for (size_t i = 0; i < n; ++i) y[i] = x.host[i] + __bfloat162float(h[i]);
    printf("forward\n");
    report(msa ? "out/msa" : "out/pair", compare(y, o[msa ? "out/msa" : "out/pair"].host));
    auto g = fetch(dx, n);
    for (size_t i = 0; i < n; ++i) g[i] += ct.host[i];
    printf("backward\n");
    report(msa ? "grad/msa" : "grad/pair",
           compare(g, o[msa ? "grad/msa" : "grad/pair"].host));
  }
  double tf = time_ms([&] { m.forward(xB); }, reps);
  double tb = time_ms([&] { m.backward(ct.dev, xB, dx); }, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb,
         time_ms([&] { m.forward(xB); m.backward(ct.dev, xB, dx); }, reps));
  return 0;
}
