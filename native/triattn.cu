// Standalone gate for the triangle attention module (triattn.cuh).
//
// Two widths, chosen by the dump rather than by a flag -- see trimul.cu for
// why.  A namespace is not a template argument, so the body is a macro.
#include "triattn.cuh"

#define TRIATTN_GATE(NS)                                                       \
  do {                                                                         \
    size_t mc = (size_t)N * N * NS::C;                                         \
    NS::TriAttn m;                                                             \
    m.alloc(N, ending);                                                        \
    m.allocOrientation();                                                      \
    m.setWeights(o, std::string("param/triangle_attention_") +                 \
                        (ending ? "ending_node/" : "starting_node/"));         \
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
      std::vector<float> y(mc), g(mc);                                         \
      auto gh = fetch(dpair, mc);                                              \
      for (int i = 0; i < N; ++i)                                              \
        for (int j = 0; j < N; ++j)                                            \
          for (int c = 0; c < NS::C; ++c) {                                    \
            size_t dst = ((size_t)i * N + j) * NS::C + c;                      \
            y[dst] = pair.host[dst] + __bfloat162float(h[dst]);                \
            g[dst] = gh[dst] + ct.host[dst];                                   \
          }                                                                    \
      printf("forward\n");                                                     \
      report("out/pair", compare(y, o["out/pair"].host));                      \
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
  if (argc < 2) { fprintf(stderr, "usage: triattn <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  bool ending = std::string(argv[1]).find("ending") != std::string::npos;
  Tensor& pair = o["in/pair"];
  int N = pair.shape[0];
  int C = pair.shape.back();
  printf("triangle attention (%s)  N=%d  C=%d  D=%d\n",
         ending ? "ending" : "starting", N, C, C / 4);

  pair.to_device();
  Tensor& ct = o["ct/pair"]; ct.to_device();
  Tensor& pm = o["in/pair_mask"]; pm.to_device();

  if (C == tattn::C) {
    TRIATTN_GATE(tattn);
#if NE_FD16
  } else if (C == tattn64::C) {
    TRIATTN_GATE(tattn64);
#endif
  } else {
    fprintf(stderr, "no triattn build at C=%d\n", C);
    return 1;
  }
  return 0;
}
