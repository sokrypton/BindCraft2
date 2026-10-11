// Standalone gate for the triangle multiplication module (trimul.cuh).
//
// Two widths, so two bodies: the Evoformer's tmul at C=128 and the template
// pair stack's tmul64 at C=64.  The dump says which -- the pair's last axis --
// and running the wrong one would be a silent 2x stride error, so the width
// is read rather than chosen by a flag.
#include "trimul.cuh"

// A namespace cannot be a template argument, so the body is a macro.  It is
// identical for both widths; only NS changes.
#define TRIMUL_GATE(NS)                                                        \
  do {                                                                         \
    size_t mc = (size_t)N * N * NS::C;                                         \
    NS::TriMul m;                                                              \
    m.alloc(N, outgoing);                                                      \
    m.setWeights(o, std::string("param/triangle_multiplication_") +            \
                        (outgoing ? "outgoing/" : "incoming/"));               \
                                                                               \
    __nv_bfloat16* pairB;                                                      \
    float* dpair;                                                              \
    CU(cudaMalloc(&pairB, mc * 2));                                            \
    CU(cudaMalloc(&dpair, mc * 4));                                            \
    NS::f32ToBf16K<<<(mc + 255) / 256, 256>>>(pair.dev, pairB, mc);            \
                                                                               \
    m.forward(pairB, pm.dev);                                                  \
    m.backward(ct.dev, pairB, pm.dev, dpair);                                  \
    CU(cudaDeviceSynchronize());                                               \
    {                                                                          \
      std::vector<__nv_bfloat16> h(mc);                                        \
      CU(cudaMemcpy(h.data(), m.out, mc * 2, cudaMemcpyDeviceToHost));         \
      std::vector<float> y(mc);                                                \
      for (size_t i = 0; i < mc; ++i)                                          \
        y[i] = pair.host[i] + __bfloat162float(h[i]);                          \
      printf("forward\n");                                                     \
      report("out/pair", compare(y, o["out/pair"].host));                      \
      auto g = fetch(dpair, mc);                                               \
      for (size_t i = 0; i < mc; ++i) g[i] += ct.host[i];                      \
      printf("backward\n");                                                    \
      report("grad/pair", compare(g, o["grad/pair"].host));                    \
    }                                                                          \
    double tf = time_ms([&] { m.forward(pairB, pm.dev); }, reps);              \
    double tb =                                                                \
        time_ms([&] { m.backward(ct.dev, pairB, pm.dev, dpair); }, reps);      \
    printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb,     \
           tf + tb);                                                           \
  } while (0)

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: trimul <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  bool outgoing = std::string(argv[1]).find("outgoing") != std::string::npos;
  Tensor& pair = o["in/pair"];
  int N = pair.shape[0];
  int C = pair.shape.back();
  printf("triangle multiplication (%s)  N=%d  C=%d\n",
         outgoing ? "outgoing" : "incoming", N, C);

  pair.to_device();
  Tensor& ct = o["ct/pair"]; ct.to_device();
  Tensor& pm = o["in/pair_mask"]; pm.to_device();

  if (C == tmul::C) {
    TRIMUL_GATE(tmul);
#if NE_TMUL64
  } else if (C == tmul64::C) {
    TRIMUL_GATE(tmul64);
#endif
  } else {
    fprintf(stderr, "no trimul build at C=%d\n", C);
    return 1;
  }
  return 0;
}
