// Gate for the whole Evoformer block (block.cuh) against the JAX oracle's
// `--part full` dump: nine residual steps forward, nine in reverse.
#include "block.cuh"

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: block <oracle-dir> [reps]\n"); return 1; }
  Oracle o(argv[1]);
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  Tensor& msa = o["in/msa"];
  Tensor& pair = o["in/pair"];
  int S = msa.shape[0], N = msa.shape[1];
  size_t nm = (size_t)S * N * block::CM, np = (size_t)N * N * block::CZ;
  printf("Evoformer block  S=%d N=%d  (9 residual steps)\n", S, N);

  msa.to_device(); pair.to_device();
  Tensor& mmask = o["in/msa_mask"]; mmask.to_device();
  Tensor& pmask = o["in/pair_mask"]; pmask.to_device();
  Tensor& ctm = o["ct/msa"]; ctm.to_device();
  Tensor& ctp = o["ct/pair"]; ctp.to_device();

  // which track: read off the dump, so one binary gates full / msa / pair
  const bool has_opm = o.has("param/evoformer_iteration/outer_product_mean/left_projection/weights") ||
                       o.has("param/outer_product_mean/left_projection/weights");
  const bool has_tri = o.has("param/evoformer_iteration/triangle_multiplication_outgoing/projection/weights") ||
                       o.has("param/triangle_multiplication_outgoing/projection/weights");
  const std::string P = o.has("param/evoformer_iteration/pair_transition/transition1/weights") ||
                        o.has("param/evoformer_iteration/outer_product_mean/left_projection/weights")
                        ? "param/evoformer_iteration/" : "param/";
  printf("  track: %s%s   prefix %s\n", has_opm ? "msa " : "", has_tri ? "pair" : "", P.c_str());

  block::Block b;
  b.do_msa = has_opm; b.do_pair = has_tri;
  // a prefix ladder: "block <dir> <reps> <n_msa> <n_pair>"
  if (argc > 3) b.n_msa = atoi(argv[3]);
  if (argc > 4) b.n_pair = atoi(argv[4]);
  // DROPOUT=1 turns the masks on for a by-hand check; the gates run at rate 0
  // so they stay comparable to a JAX oracle, whose threefry masks we cannot
  // reproduce (see common.cuh).
  b.use_dropout = envOn("DROPOUT");
  b.alloc(N, S);
  b.setWeights(o, P);  // reads only the modules the prefix runs
  b.setSeed(b.use_dropout ? (uint32_t)(getenv("DROP_SEED") ? atoi(getenv("DROP_SEED")) : 0) : 0u);

  // round ONCE to bf16, as AF2 does to the activations entering its first
  // module, then carry the f32 value -- the chain is f32 (see common.cuh)
  __nv_bfloat16 *msaH, *pairH;
  float *msaB, *pairB;
  CU(cudaMalloc(&msaH, nm * 2)); CU(cudaMalloc(&pairH, np * 2));
  CU(cudaMalloc(&msaB, nm * 4)); CU(cudaMalloc(&pairB, np * 4));
  trans::f32ToBf16K<<<(nm + 255) / 256, 256>>>(msa.dev, msaH, nm);
  trans::f32ToBf16K<<<(np + 255) / 256, 256>>>(pair.dev, pairH, np);
  opm::bf16ToF32K<<<(nm + 255) / 256, 256>>>(msaH, msaB, nm);
  opm::bf16ToF32K<<<(np + 255) / 256, 256>>>(pairH, pairB, np);

  // the block's pair cotangent is bf16 now; the gate's comes from the oracle
  // as f32, so narrow it once here the way the stack does
  __nv_bfloat16* ctpB;
  CU(cudaMalloc(&ctpB, np * 2));
  toBf16K<<<(np + 255) / 256, 256>>>(ctpB, ctp.dev, np);

  // the block's INPUT pair is bf16 now too -- it is a save-set slice, and the
  // save-set is stored narrow.  Narrow it once here the way the stack does.
  __nv_bfloat16* pairIn;
  CU(cudaMalloc(&pairIn, np * 2));
  toBf16K<<<(np + 255) / 256, 256>>>(pairIn, pairB, np);

  b.forward(pairIn, msaB, pmask.dev, mmask.dev);
  b.backward(ctpB, ctm.dev, pmask.dev, mmask.dev);
  CU(cudaDeviceSynchronize());
  {
    auto grab = [&](const float* d, size_t n) { return fetch(d, n); };
    printf("forward\n");
    // THE PAIR OUTPUT LIVES WHEREVER THE CHAIN ENDS, and the bf16 residual
    // chain moved it.  With BLK_BF16CHAIN -- the default -- the nine steps
    // hand each other bf16 views and `p6` is never written, so reading it
    // reported the block's INPUT as its output (cos 0.05) while the backward,
    // which reads the bf16 views, stayed green.  A gate pointed at a buffer
    // an optimisation retired fails loudly here, but it would have gone
    // unnoticed longer still if the stale value had been plausible.
    static float* p6w = nullptr;
    const float* pout = b.p6;
    if (BLK_BF16CHAIN) {
      if (!p6w) CU(cudaMalloc(&p6w, np * 4));
      widenBf16K<<<(np + 255) / 256, 256>>>(p6w, b.p6b, np);
      CU(cudaDeviceSynchronize());
      pout = p6w;
    }
    report("out/pair", compare(grab(pout, np), o["out/pair"].host));
    report("out/msa", compare(grab(b.m3, nm), o["out/msa"].host));
    printf("backward\n");
    report("grad/pair", compare(fetchAny(b.dp, np), o["grad/pair"].host));
    report("grad/msa", compare(fetch(b.dm, nm), o["grad/msa"].host));
  }
  double tf = time_ms([&] { b.forward(pairIn, msaB, pmask.dev, mmask.dev); }, reps);
  double tb = time_ms([&] { b.backward(ctpB, ctm.dev, pmask.dev, mmask.dev); }, reps);
  printf("\n  fwd %7.3f ms   bwd %7.3f ms   fwd+bwd %7.3f ms\n", tf, tb,
         time_ms([&] { b.forward(pairIn, msaB, pmask.dev, mmask.dev);
                       b.backward(ctpB, ctm.dev, pmask.dev, mmask.dev); }, reps));
  return 0;
}
