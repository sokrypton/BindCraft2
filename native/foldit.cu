// Gate for one FoldIteration (foldit.cuh) against oracle_ipa.py --part fold.
#include "foldit.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: foldit <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  Tensor& single = o["in/single"];
  int N = single.shape[0];
  printf("fold iteration  N=%d  C=%d\n", N, foldit::C);

  for (const char* k : {"in/single", "in/pair", "in/pair_mask", "in/rot",
                        "in/trans", "ct/single", "ct/rot", "ct/trans"})
    o[k].to_device();

  foldit::FoldIteration m;
  m.alloc(N);
  // THE PREFIX IS THE DUMP'S, NOT A CONSTANT.  A --part ipa dump names its
  // modules at the top level; a --part fold dump scopes them under the haiku
  // module that owns them.  Reading which one is here beats hardcoding either.
  const std::string P = o.has("param/fold_iteration/transition/weights")
                            ? "param/fold_iteration/" : "param/";
  m.setWeights(o, P);
  float *rout, *tout, *dsingle, *dpair, *drot, *dtrans;
  CU(cudaMalloc(&rout, (size_t)N * 9 * 4));
  CU(cudaMalloc(&tout, (size_t)N * 3 * 4));
  CU(cudaMalloc(&dsingle, (size_t)N * foldit::C * 4));
  CU(cudaMalloc(&dpair, (size_t)N * N * ipa::CZ * 4));
  CU(cudaMalloc(&drot, (size_t)N * 9 * 4));
  CU(cudaMalloc(&dtrans, (size_t)N * 3 * 4));

  auto fwd = [&] {
    m.forward(single.dev, o["in/pair"].dev, o["in/pair_mask"].dev,
              o["in/rot"].dev, o["in/trans"].dev, rout, tout);
  };
  auto bwd = [&] {
    m.backward(o["ct/single"].dev, o["ct/rot"].dev, o["ct/trans"].dev,
               single.dev, o["in/pair"].dev, o["in/rot"].dev, o["in/trans"].dev,
               dsingle, dpair, drot, dtrans);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/single", compare(fetch(m.outAct(), (size_t)N * foldit::C),
                               o["out/single"].host));
  report("out/rot", compare(fetch(rout, (size_t)N * 9), o["out/rot"].host));
  report("out/trans", compare(fetch(tout, (size_t)N * 3), o["out/trans"].host));
  printf("backward\n");
  report("grad/single", compare(fetch(dsingle, (size_t)N * foldit::C),
                                o["grad/single"].host));
  report("grad/pair", compare(fetch(dpair, (size_t)N * N * ipa::CZ),
                              o["grad/pair"].host));
  report("grad/rot", compare(fetch(drot, (size_t)N * 9), o["grad/rot"].host));
  report("grad/trans", compare(fetch(dtrans, (size_t)N * 3),
                               o["grad/trans"].host));
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb, tf + tb);
  return 0;
}
