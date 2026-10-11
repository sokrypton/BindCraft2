// Gate for the whole StructureModule (structmod.cuh) against
// oracle_ipa.py --part sm --depth 8.
#include "structmod.cuh"
#if NE_CUPTI
#include "cupti_prof.cuh"
#endif

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: structmod <oracle-dir> [depth] [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int depth = argc > 2 ? atoi(argv[2]) : 8;
  int reps = argc > 3 ? atoi(argv[3]) : 3;
  Tensor& single = o["in/single"];
  int N = single.shape[0];
  printf("structure module (whole)  N=%d  C=%d  CZ=%d  depth=%d\n", N,
         structmod::C, structmod::CZ, depth);
  for (const char* k : {"in/single", "in/pair", "in/pair_mask"})
    o[k].to_device();
  // A --part smatom dump drives the backward from the ATOMS ALONE -- no
  // cotangent on the act or the frame.  That is deliberate: it isolates the
  // sidechain/atom path, which the act and frame gates have already covered,
  // so a bug there cannot hide behind them.  The missing cotangents are
  // zeros, supplied here rather than padded into the dump.
  auto ctOrZero = [&](const char* k, size_t n) -> const float* {
    if (o.has(k)) { o[k].to_device(); return o[k].dev; }
    float* z;
    CU(cudaMalloc(&z, n * 4));
    CU(cudaMemset(z, 0, n * 4));
    return z;
  };
  const float* ctS = ctOrZero("ct/single", (size_t)N * structmod::C);
  const float* ctR = ctOrZero("ct/rot", (size_t)N * 9);
  const float* ctT = ctOrZero("ct/trans", (size_t)N * 3);

  // ATOMS ARE ON WHEN THE DUMP CARRIES THEM.  A --part sm dump stops at the
  // act and the frame; --part smatom adds the last iteration's 14 atom
  // positions and the residue-constant tables they need.
  const bool atoms = o.has("ct/atom");
  // A STREAM OF ITS OWN, because a CUDA graph cannot be captured on the
  // legacy default stream -- and the launch gap is what this module is
  // actually paying.  The profile reads 6.5 ms of kernel time over a 10.5 ms
  // span: ~600 launches at a few microseconds each.
  cudaStream_t st;
  CU(cudaStreamCreate(&st));
  structmod::StructureModule m;
  m.do_atoms = atoms;
  m.alloc(N, depth, st);
  m.setWeights(o);
  float *dsingle, *dpair;
  CU(cudaMalloc(&dsingle, (size_t)N * structmod::C * 4));
  CU(cudaMalloc(&dpair, (size_t)N * N * structmod::CZ * 4));

  const float* aat = nullptr;
  const float* dat = nullptr;
  if (atoms) {
    o["in/aatype"].to_device();
    o["ct/atom"].to_device();
    aat = o["in/aatype"].dev;
    dat = o["ct/atom"].dev;
  }
  auto fwd = [&] {
    m.forward(single.dev, o["in/pair"].dev, o["in/pair_mask"].dev, aat);
  };
  auto sync = [&] { CU(cudaStreamSynchronize(st)); };
  auto bwd = [&] {
    m.backward(ctS, ctR, ctT, single.dev, dsingle, dpair, dat, aat);
  };
  fwd(); bwd();
  CU(cudaDeviceSynchronize());
  printf("forward\n");
  report("out/single", compare(fetch(m.outAct(), (size_t)N * structmod::C),
                               o["out/single"].host));
  report("out/rot", compare(fetch(m.outRot(), (size_t)N * 9), o["out/rot"].host));
  report("out/trans", compare(fetch(m.outTrans(), (size_t)N * 3),
                              o["out/trans"].host));
  if (atoms)
    report("out/atom", compare(fetch(m.outAtom(), (size_t)N * 14 * 3),
                               o["out/atom"].host));
  printf("backward\n");
  report("grad/single", compare(fetch(dsingle, (size_t)N * structmod::C),
                                o["grad/single"].host));
  report("grad/pair", compare(fetch(dpair, (size_t)N * N * structmod::CZ),
                              o["grad/pair"].host));
  // NE_CUPTI gives the per-kernel table.  The structure module is 1.6x SLOWER
  // than XLA on the same shape, and that is a claim about WHICH kernels, so
  // it gets measured rather than guessed at.
#if NE_CUPTI
  if (envOn("NE_CUPTI")) {
    fwd(); bwd(); CU(cudaDeviceSynchronize());     // warm
    cuprof::init(); cuprof::start();
    fwd(); bwd();
    CU(cudaDeviceSynchronize());
    cuprof::stop(28);
  }
#endif
  // NE_GRAPH replays the whole forward+backward as one graph launch.  The
  // shapes are static and there is no data-dependent control flow, so the
  // capture is the obvious shape for this module; the point of measuring it
  // separately is that the gap, not the arithmetic, is the thing to beat.
  double tg = 0;
  if (envOn("NE_GRAPH")) {
    fwd(); bwd();
    CU(cudaStreamSynchronize(st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CU(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    fwd(); bwd();
    CU(cudaStreamEndCapture(st, &graph));
    CU(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CU(cudaGraphLaunch(exec, st));
    CU(cudaStreamSynchronize(st));
    int gr = 20;
    cudaEvent_t e0, e1;
    CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
    CU(cudaEventRecord(e0, st));
    for (int i = 0; i < gr; ++i) CU(cudaGraphLaunch(exec, st));
    CU(cudaEventRecord(e1, st));
    CU(cudaStreamSynchronize(st));
    float ms; CU(cudaEventElapsedTime(&ms, e0, e1));
    tg = ms / gr;
  }
  double tf = time_ms(fwd, reps), tb = time_ms(bwd, reps);
  if (tg > 0)
    printf("\n  GRAPH fwd+bwd %7.3f ms  (against %7.3f ms launched one by "
           "one)\n", tg, tf + tb);
  size_t fb, tot;
  CU(cudaMemGetInfo(&fb, &tot));
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms   VRAM %.2f GB\n",
         tf, tb, tf + tb, (tot - fb) / 1e9);
  return 0;
}
