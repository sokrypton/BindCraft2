// Gate for the atom builder (atom14.cuh) against oracle_ipa.py --part atom.
//
// The gradient here goes to the ANGLES and the BACKBONE FRAME only: aatype is
// an index and every table is a constant.
#include "atom14.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: atom14 <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  Tensor& ang = o["in/angles"];
  int N = ang.shape[0];
  printf("atom builder  N=%d  groups=%d  atoms=%d\n", N, atom14::NG, atom14::NA);
  for (const char* k : {"in/angles", "in/rot", "in/trans", "in/aatype",
                        "ct/atom"})
    o[k].to_device();

  atom14::Atom14 m;
  m.alloc(N);
  m.setTables(o);
  float *dang, *drot, *dtrans;
  CU(cudaMalloc(&dang, (size_t)N * atom14::NTOR * 2 * 4));
  CU(cudaMalloc(&drot, (size_t)N * 9 * 4));
  CU(cudaMalloc(&dtrans, (size_t)N * 3 * 4));

  auto fwd = [&] {
    m.forward(ang.dev, o["in/rot"].dev, o["in/trans"].dev, o["in/aatype"].dev);
  };
  auto bwd = [&] {
    m.backward(o["ct/atom"].dev, ang.dev, o["in/rot"].dev, o["in/aatype"].dev,
               dang, drot, dtrans);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/atom", compare(fetch(m.out, (size_t)N * atom14::NA * 3),
                             o["out/atom"].host));
  printf("backward\n");
  report("grad/angles", compare(fetch(dang, (size_t)N * atom14::NTOR * 2),
                                o["grad/angles"].host));
  report("grad/rot", compare(fetch(drot, (size_t)N * 9), o["grad/rot"].host));
  report("grad/trans", compare(fetch(dtrans, (size_t)N * 3),
                               o["grad/trans"].host));
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms\n", tf, tb);
  return 0;
}
