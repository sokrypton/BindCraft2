// Gate for the 8-deep structure module (foldstack.cuh) against
// oracle_ipa.py --part fold --depth 8.
#include "foldstack.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: foldstack <oracle-dir> [depth] [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int depth = argc > 2 ? atoi(argv[2]) : 8;
  int reps = argc > 3 ? atoi(argv[3]) : 5;
  Tensor& single = o["in/single"];
  int N = single.shape[0];
  printf("structure module  N=%d  C=%d  depth=%d\n", N, foldstack::C, depth);

  for (const char* k : {"in/single", "in/pair", "in/pair_mask", "in/rot",
                        "in/trans", "ct/single", "ct/rot", "ct/trans"})
    o[k].to_device();

  foldstack::FoldStack s;
  s.alloc(N, depth);
  const std::string P = o.has("param/fold_iteration/transition/weights")
                            ? "param/fold_iteration/" : "param/";
  s.setWeights(o, P);

  float *dsingle, *dpair, *drot, *dtrans;
  CU(cudaMalloc(&dsingle, (size_t)N * foldstack::C * 4));
  CU(cudaMalloc(&dpair, (size_t)N * N * ipa::CZ * 4));
  CU(cudaMalloc(&drot, (size_t)N * 9 * 4));
  CU(cudaMalloc(&dtrans, (size_t)N * 3 * 4));

  auto fwd = [&] {
    s.forward(single.dev, o["in/pair"].dev, o["in/pair_mask"].dev,
              o["in/rot"].dev, o["in/trans"].dev);
  };
  auto bwd = [&] {
    s.backward(o["ct/single"].dev, o["ct/rot"].dev, o["ct/trans"].dev,
               single.dev, o["in/pair"].dev, dsingle, dpair, drot, dtrans);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/single", compare(fetch(s.outAct(), (size_t)N * foldstack::C),
                               o["out/single"].host));
  report("out/rot", compare(fetch(s.outRot(), (size_t)N * 9), o["out/rot"].host));
  report("out/trans", compare(fetch(s.outTrans(), (size_t)N * 3),
                              o["out/trans"].host));
  printf("backward\n");
  report("grad/single", compare(fetch(dsingle, (size_t)N * foldstack::C),
                                o["grad/single"].host));
  report("grad/pair", compare(fetch(dpair, (size_t)N * N * ipa::CZ),
                              o["grad/pair"].host));
  report("grad/rot", compare(fetch(drot, (size_t)N * 9), o["grad/rot"].host));
  report("grad/trans", compare(fetch(dtrans, (size_t)N * 3),
                               o["grad/trans"].host));
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  size_t fb, tb_;
  CU(cudaMemGetInfo(&fb, &tb_));
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms   VRAM %.2f GB\n",
         tf, tb, tf + tb, (tb_ - fb) / 1e9);
  return 0;
}
