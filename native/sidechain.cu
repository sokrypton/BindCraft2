// Gate for MultiRigidSidechain's MLP half (sidechain.cuh) against
// oracle_ipa.py --part sc.
//
// THE TWO INPUT REPRESENTATIONS ARE DIFFERENT TENSORS in the dump, on purpose:
// the real caller passes [act, initial_act] through two separate projections
// and sums them, so feeding the same tensor twice would make a swapped pair
// invisible.
#include "sidechain.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: sidechain <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& act = o["in/single"];
  int N = act.shape[0];
  printf("sidechain MLP  N=%d  Cin=%d  C=%d  torsions=%d\n", N, sidechain::CIN,
         sidechain::C, sidechain::NTOR);
  act.to_device();
  o["ct/single"].to_device();
  // the oracle's second representation is its own 384-wide activation, so
  // the two input projections are distinguishable
  o["in/single2"].to_device();
  float* initDev = o["in/single2"].dev;

  sidechain::Sidechain m;
  m.alloc(N);
  m.setWeights(o);
  float *dact, *dinit;
  CU(cudaMalloc(&dact, (size_t)N * sidechain::CIN * 4));
  CU(cudaMalloc(&dinit, (size_t)N * sidechain::CIN * 4));

  auto fwd = [&] { m.forward(act.dev, initDev); };
  auto bwd = [&] {
    m.backward(o["ct/single"].dev, act.dev, initDev, dact, dinit, false);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/angles", compare(fetch(m.ang, (size_t)N * sidechain::NANG),
                               o["out/single"].host));
  printf("backward\n");
  report("grad/single", compare(fetch(dact, (size_t)N * sidechain::CIN),
                                o["grad/single"].host));
  report("grad/single2", compare(fetch(dinit, (size_t)N * sidechain::CIN),
                                 o["grad/single2"].host));
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms\n", tf, tb);
  return 0;
}
