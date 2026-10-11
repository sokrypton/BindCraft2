// Gate for the Evoformer stack (stack.cuh).
//
// Two oracle directories, because the weights and the activations come from
// different places: the activations from oracle.py --blocks K --no-params, the
// 48 blocks of weights from weights.py straight out of the checkpoint.
#include "stack.cuh"
#if NE_CUPTI
#include "cupti_prof.cuh"
#endif

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: stack <acts-dir> <weights-dir> [reps]\n");
    return 1;
  }
  Oracle o(argv[1]);
  Oracle w(argv[2]);
  int reps = argc > 3 ? atoi(argv[3]) : 3;

  Tensor& msa = o["in/msa"];
  Tensor& pair = o["in/pair"];
  int S = msa.shape[0], N = msa.shape[1];
  size_t nm = (size_t)S * N * block::CM, np = (size_t)N * N * block::CZ;

  // the depth is in the dump's own name: "stack<K>-n<N>"
  std::string dir(argv[1]);
  size_t a = dir.find("stack"), z = dir.find("-n", a);
  int depth = (a == std::string::npos) ? 1 : atoi(dir.substr(a + 5, z - a - 5).c_str());
  printf("Evoformer stack  depth=%d  S=%d N=%d\n", depth, S, N);

  msa.to_device(); pair.to_device();
  Tensor& mmask = o["in/msa_mask"]; mmask.to_device();
  Tensor& pmask = o["in/pair_mask"]; pmask.to_device();
  Tensor& ctm = o["ct/msa"]; ctm.to_device();
  Tensor& ctp = o["ct/pair"]; ctp.to_device();

  stack::Stack s;
  // PAIR_ONLY is the extra-MSA stack: four blocks whose MSA side is dead and
  // whose outer product mean is the constant `output_b / eps`. The pair track
  // is dimensionally identical to the main stack's, which is why the same
  // Block serves both.
  if (envOn("PAIR_ONLY")) { s.b.pair_only = true; s.b.do_msa = false; }
  // OPM_LAST is the MONOMER (alphafold2_ptm) block order: the outer product
  // mean at the END, reading the msa after its transition, with the row
  // attention reading the block's own input pair.  Must be set before
  // alloc(), which is where the two extra bf16 views are allocated.
  if (envOn("OPM_LAST")) s.b.opm_first = false;
  s.alloc(N, S, depth);
  s.setWeights(w);
  printf("  %zu weight slots x %d depths\n", s.slots.size(), depth);

  // The stack's own input IS bf16 -- it is the one place AF2 rounds -- but the
  // chain after it is f32, so round once and hand the f32 value on.  Rounding
  // to bf16 and back is what the model does to the activations entering its
  // first module; what it does NOT do is round again at each of the 432 steps
  // that follow.
  __nv_bfloat16 *msaH, *pairH;
  float *msaB, *pairB;
  CU(cudaMalloc(&msaH, nm * 2)); CU(cudaMalloc(&pairH, np * 2));
  CU(cudaMalloc(&msaB, nm * 4)); CU(cudaMalloc(&pairB, np * 4));
  trans::f32ToBf16K<<<(nm + 255) / 256, 256>>>(msa.dev, msaH, nm);
  trans::f32ToBf16K<<<(np + 255) / 256, 256>>>(pair.dev, pairH, np);
  opm::bf16ToF32K<<<(nm + 255) / 256, 256>>>(msaH, msaB, nm);
  opm::bf16ToF32K<<<(np + 255) / 256, 256>>>(pairH, pairB, np);

  // CT_SCALE scales the cotangent.  The backward is LINEAR in it, so cos is
  // invariant and only the dynamic range moves -- which is how to tell a wrong
  // gradient from one that overflowed.  The attention path is f16 (max 65504)
  // and the monomer network amplifies the msa gradient 7.5x more per block
  // than the multimer one does.
  if (const char* cs = getenv("CT_SCALE")) {
    float f = atof(cs);
    scaleK<<<(np + 255) / 256, 256>>>(ctp.dev, f, np);
    scaleK<<<(nm + 255) / 256, 256>>>(ctm.dev, f, nm);
    CU(cudaDeviceSynchronize());
    printf("  cotangent scaled by %g\n", f);
  }
  // the keep re-plan, which in the FFI runs at the top of a save forward.
  // Off unless REGRANT_AFTER is set -- see regrant() in stack.cuh.
  s.regrant();
#if NE_CUPTI
  // a warm pass first: the profile should see the steady state, not the
  // cuBLASLt tuning and the first-touch page faults
  s.forward(pairB, msaB, pmask.dev, mmask.dev);
  s.backward(ctp.dev, ctm.dev, pmask.dev, mmask.dev);
  CU(cudaDeviceSynchronize());
  if (envOn("NE_CUPTI")) { cuprof::init(); cuprof::start(); }
#endif
  s.forward(pairB, msaB, pmask.dev, mmask.dev);
  // SHARE_PROBE: THE HAZARD THE SHARED BLOCK SCRATCH WOULD HAVE, MADE REAL.
  // In a design step two stacks adopt one set of pair buffers, and the risky
  // moment is between a stack's forward and its backward -- the OTHER stack
  // runs in between and writes all over them.  The claim is that nothing
  // there has to survive, because the replay recomputes it.  So: build a
  // second stack that adopts the same scratch, run its forward right here,
  // and see whether the gradient below still lands in band.
  if (envOn("SHARE_PROBE")) {
    static stack::Stack other;
    other.b.opm_first = s.b.opm_first;
    // ON THE SAME STREAM, which is what the FFI does: both stacks are handed
    // XLA's stream and setStream() points the SHARED cuBLAS handles at it on
    // every call.  Letting this one create its own stream instead leaves
    // those handles aimed somewhere else and the probe measures a race that
    // nothing ships -- the third time private streams have cost a day here.
    other.alloc(N, S, 1, s.b.st);
    other.setWeights(w);
    other.setStream(s.b.st);
    other.forward(pairB, msaB, pmask.dev, mmask.dev);
    CU(cudaStreamSynchronize(s.b.st));
    s.setStream(s.b.st);          // as the FFI does at the top of every call
    fprintf(stderr, "SHARE_PROBE: a second stack ran over the shared "
            "scratch between the forward and the backward\n");
  }
  s.backward(ctp.dev, ctm.dev, pmask.dev, mmask.dev);
  CU(cudaDeviceSynchronize());
#if NE_CUPTI
  if (envOn("NE_CUPTI")) cuprof::stop(30);
#endif
  {
    auto grab = [&](const float* d, size_t n) { return fetch(d, n); };
    printf("forward\n");
    report("out/pair", compare(grab(s.pout, np), o["out/pair"].host));
    report("out/msa", compare(grab(s.mout, nm), o["out/msa"].host));
    printf("backward\n");
    report("grad/pair", compare(fetchAny(s.b.dp, np), o["grad/pair"].host));
    report("grad/msa", compare(fetch(s.b.dm, nm), o["grad/msa"].host));

    // A MASKED DUMP NEEDS A SECOND SCORE.  The padded region's activations are
    // enormous (rms 3797 against an unmasked run's 70.8 -- a masked row never
    // gets renormalised), so they dominate the norm and the whole-tensor score
    // above would read green with the real residues arbitrarily wrong.  These
    // rows restrict to the residues the mask keeps, and they are the ones that
    // say whether masking works.
    size_t kept = 0;
    for (int i = 0; i < N; ++i) kept += pmask.host[(size_t)i * N + i] > 0.5f;
    if (kept != (size_t)N) {
      printf("masked: %zu of %d residues real\n", kept, N);
      std::vector<char> kp(np), km(nm);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
          char v = pmask.host[(size_t)i * N + j] > 0.5f;
          for (int c = 0; c < block::CZ; ++c) kp[((size_t)i * N + j) * block::CZ + c] = v;
        }
      for (int t = 0; t < S; ++t)
        for (int i = 0; i < N; ++i) {
          char v = mmask.host[(size_t)t * N + i] > 0.5f;
          for (int c = 0; c < block::CM; ++c) km[((size_t)t * N + i) * block::CM + c] = v;
        }
      report("  real out/pair", compare_where(grab(s.pout, np), o["out/pair"].host, kp));
      report("  real out/msa", compare_where(grab(s.mout, nm), o["out/msa"].host, km));
      report("  real grad/pair", compare_where(fetchAny(s.b.dp, np), o["grad/pair"].host, kp));
      report("  real grad/msa", compare_where(fetch(s.b.dm, nm), o["grad/msa"].host, km));
    }
  }
  // STACK_OUT writes the stack's own output, so it can be fed to an oracle as
  // an input.  That is how "does the next block amplify OUR error, or is our
  // composition wrong" gets asked: run the oracle's block i+1 on our block i's
  // output and compare with the oracle's own depth i+1.
  if (const char* od = getenv("STACK_OUT")) {
    auto wr = [&](const char* name, const float* d, size_t n) {
      std::vector<float> h = fetch(d, n);
      std::string fn = std::string(od) + "/" + name;
      FILE* fp = fopen(fn.c_str(), "wb");
      if (!fp) { fprintf(stderr, "cannot write %s\n", fn.c_str()); return; }
      fwrite(h.data(), 4, n, fp); fclose(fp);
      printf("  wrote %s (%zu floats)\n", fn.c_str(), n);
    };
    wr("out_pair.f32", s.pout, np);
    wr("out_msa.f32", s.mout, nm);
    // every block's INPUT too: pin[i] is what block i was handed, so pin[1]
    // from a depth-2 run must equal pout from a depth-1 run, exactly
    for (int i = 0; i < depth; ++i) {
      char nm1[64]; snprintf(nm1, sizeof nm1, "in_pair_%d.f32", i);
      // pin is bf16 now; widen for the dump, which the cross-check reads as f32
      static float* wide = nullptr;
      if (!wide) CU(cudaMalloc(&wide, np * 4));
      widenBf16K<<<(np + 255) / 256, 256>>>(wide, s.pin[i], np);
      CU(cudaDeviceSynchronize());
      wr(nm1, wide, np);
    }
  }
  size_t free_b, total_b;
  CU(cudaMemGetInfo(&free_b, &total_b));
  printf("\n  VRAM in use %.2f GB of %.2f\n", (total_b - free_b) / 1e9,
         total_b / 1e9);
  double tf = time_ms([&] { s.forward(pairB, msaB, pmask.dev, mmask.dev); }, reps);
  prof().reset();
  s.forward(pairB, msaB, pmask.dev, mmask.dev);
  prof().report("stack forward");
  double tb = time_ms([&] { s.backward(ctp.dev, ctm.dev, pmask.dev, mmask.dev); }, reps);
  prof().reset();
  s.backward(ctp.dev, ctm.dev, pmask.dev, mmask.dev);
  prof().report("stack backward");
  printf("  fwd %8.2f ms   bwd %8.2f ms   fwd+bwd %8.2f ms   (%.2f ms a block)\n",
         tf, tb, tf + tb, (tf + tb) / depth);
  if (envOn("ISSUE_TIME")) {
    double qf = issue_ms([&] { s.forward(pairB, msaB, pmask.dev, mmask.dev); }, reps);
    double qb = issue_ms([&] { s.backward(ctp.dev, ctm.dev, pmask.dev, mmask.dev); }, reps);
    printf("  host issue: fwd %7.2f ms (%4.1f%% of gpu)   bwd %7.2f ms (%4.1f%%)\n",
           qf, 100 * qf / tf, qb, 100 * qb / tb);
  }
  return 0;
}
