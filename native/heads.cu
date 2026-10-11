// Gate for the four design-relevant heads (heads.cuh).
#include "heads.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: heads <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& pair = o["in/pair"];
  int N = pair.shape[0];
  printf("heads  N=%d  distogram=%d  pae=%d  plddt=%d\n", N, heads::NDGRAM,
         heads::NPAE, heads::PL_BINS);
  for (const char* k : {"in/pair", "in/single", "in/single2", "ct/dgram",
                        "ct/pae", "ct/plddt", "ct/expres"})
    o[k].to_device();

  heads::Heads m;
  m.alloc(N);
  m.setWeights(o);
  float *dpair, *dsingle, *dsingle_evo;
  CU(cudaMalloc(&dpair, (size_t)N * N * heads::CZ * 4));
  CU(cudaMalloc(&dsingle, (size_t)N * heads::CS * 4));
  CU(cudaMalloc(&dsingle_evo, (size_t)N * heads::CS * 4));

  auto fwd = [&] {
    m.forward(pair.dev, o["in/single"].dev);
    m.expResFwd(o["in/single2"].dev);
  };
  auto bwd = [&] {
    m.backward(o["ct/dgram"].dev, o["ct/pae"].dev, o["ct/plddt"].dev,
               o["in/single"].dev, dpair, dsingle);
    // beta=0: nothing else writes dsingle_evo in this gate
    m.expResBwd(o["ct/expres"].dev, dsingle_evo, /*beta*/0.f);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/dgram", compare(fetch(m.dgram, (size_t)N * N * heads::NDGRAM),
                              o["out/dgram"].host));
  report("out/pae", compare(fetch(m.pae, (size_t)N * N * heads::NPAE),
                            o["out/pae"].host));
  report("out/plddt", compare(fetch(m.plddt, (size_t)N * heads::PL_BINS),
                              o["out/plddt"].host));
  report("out/expres", compare(fetch(m.expres, (size_t)N * heads::NER),
                               o["out/expres"].host));
  printf("backward\n");
  report("grad/pair", compare(fetch(dpair, (size_t)N * N * heads::CZ),
                              o["grad/pair"].host));
  report("grad/single", compare(fetch(dsingle, (size_t)N * heads::CS),
                                o["grad/single"].host));
  report("grad/single2", compare(fetch(dsingle_evo, (size_t)N * heads::CS),
                                 o["grad/single2"].host));
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms\n", tf, tb);
  return 0;
}
