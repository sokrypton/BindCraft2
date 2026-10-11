// Gate for the rigid update (quatrigid.cuh) against oracle_ipa.py --part rigid.
#include "quatrigid.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: quatrigid <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& single = o["in/single"];
  int N = single.shape[0];
  printf("quat rigid update  N=%d  C=%d\n", N, qrigid::C);

  for (const char* k : {"in/single", "in/rot", "in/trans", "ct/rot", "ct/trans"})
    o[k].to_device();

  qrigid::QuatRigid m;
  m.alloc(N);
  m.setWeights(o);
  float *rout, *tout, *dact, *drot, *dtrans;
  CU(cudaMalloc(&rout, (size_t)N * 9 * 4));
  CU(cudaMalloc(&tout, (size_t)N * 3 * 4));
  CU(cudaMalloc(&dact, (size_t)N * qrigid::C * 4));
  CU(cudaMalloc(&drot, (size_t)N * 9 * 4));
  CU(cudaMalloc(&dtrans, (size_t)N * 3 * 4));

  auto fwd = [&] {
    m.forward(single.dev, o["in/rot"].dev, o["in/trans"].dev, rout, tout);
  };
  auto bwd = [&] {
    m.backward(o["ct/rot"].dev, o["ct/trans"].dev, o["in/rot"].dev, dact, drot,
               dtrans);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/rot", compare(fetch(rout, (size_t)N * 9), o["out/rot"].host));
  report("out/trans", compare(fetch(tout, (size_t)N * 3), o["out/trans"].host));
  printf("backward\n");
  report("grad/single", compare(fetch(dact, (size_t)N * qrigid::C),
                                o["grad/single"].host));
  report("grad/rot", compare(fetch(drot, (size_t)N * 9), o["grad/rot"].host));
  report("grad/trans", compare(fetch(dtrans, (size_t)N * 3),
                               o["grad/trans"].host));
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms\n", tf, tb);
  return 0;
}
