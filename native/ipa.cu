// Gate for invariant point attention (ipa.cuh) against oracle_ipa.py.
//
// FOUR GRADIENTS, and the two frame ones are the reason this gate exists: the
// rotation and the translation each enter the module twice, and a port that
// collects only one of the two routes still looks right on dsingle and dpair.
#include "ipa.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: ipa <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  Tensor& single = o["in/single"];
  int N = single.shape[0];
  printf("invariant point attention  N=%d  C=%d  H=%d\n", N, ipa::C, ipa::H);

  for (const char* k : {"in/single", "in/pair", "in/pair_mask", "in/rot",
                        "in/trans", "ct/single"})
    o[k].to_device();

  ipa::IPA m;
  m.alloc(N);
  m.setWeights(o);
  float *dsingle, *dpair, *drot, *dtrans;
  CU(cudaMalloc(&dsingle, (size_t)N * ipa::C * 4));
  CU(cudaMalloc(&dpair, (size_t)N * N * ipa::CZ * 4));
  CU(cudaMalloc(&drot, (size_t)N * 9 * 4));
  CU(cudaMalloc(&dtrans, (size_t)N * 3 * 4));

  auto fwd = [&] {
    m.forward(single.dev, o["in/pair"].dev, o["in/pair_mask"].dev,
              o["in/rot"].dev, o["in/trans"].dev);
  };
  auto bwd = [&] {
    m.backward(o["ct/single"].dev, single.dev, o["in/pair"].dev,
               o["in/rot"].dev, o["in/trans"].dev, dsingle, dpair, drot, dtrans);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/single", compare(fetch(m.out, (size_t)N * ipa::C),
                               o["out/single"].host));
  printf("backward\n");
  report("grad/single", compare(fetch(dsingle, (size_t)N * ipa::C),
                                o["grad/single"].host));
  report("grad/pair", compare(fetch(dpair, (size_t)N * N * ipa::CZ),
                              o["grad/pair"].host));
  report("grad/rot", compare(fetch(drot, (size_t)N * 9), o["grad/rot"].host));
  report("grad/trans", compare(fetch(dtrans, (size_t)N * 3),
                               o["grad/trans"].host));

  double tf = time_ms(fwd, reps);
  double tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb, tf + tb);
  return 0;
}
