// A design step with NO JAX: sequence -> loss -> sequence, entirely in CUDA.
//
//   soft_seq(params)        -> pseudo                     seqxform.cuh
//   scatter                 -> target_feat, msa_feat
//   embed                   -> msa, pair                  embed.cuh
//   extra-MSA stack         -> pair                       stack.cuh (pair_only)
//   Evoformer, 48 blocks    -> msa, pair                  stack.cuh
//   single_activations      -> single                     embed.cuh (post-trunk)
//   structure module        -> act, frame, atoms          structmod.cuh
//   heads                   -> dgram, pae, plddt logits   heads.cuh
//   losses                  -> i_con, plddt, pae          losses.cuh
//   ... and the whole thing in reverse, then norm_seq_grad + Adam.
//
// WHAT THIS IS FOR: the JIT compile of a design step, which is the thing
// being driven to zero.  Every piece here was gated on its own against the
// real model; this is the chain, and the chain is its own claim -- a
// component that is exact in isolation can still be wired to the wrong
// neighbour, which is how single_activations ended up on the embedder's msa
// instead of the trunk's.
//
// NOT HERE YET, and both are real gaps rather than simplifications:
//   - RECYCLING.  Design runs at zero recycles, so the prev_* terms are
//     absent rather than zeroed; a run that wants recycling needs them.
//   - TEMPLATES.  The template pair stack is ported and gated, but the
//     embedder's template path (template_single_embedding, the dgram, the
//     concat onto the msa rows) is not.
#include "embed.cuh"
#include "heads.cuh"
#include "losses.cuh"
#include "seqxform.cuh"
#include "stack.cuh"
#include "structmod.cuh"

// the 49-wide msa_feat layout: 23 one-hot, has_deletion, deletion_value,
// 23 profile, deletion_mean.  A designed sequence goes into the one-hot slots
// AND the profile slots, which is what v1's _update_seq does.
__global__ void scatterSeqK(const float* __restrict__ pseudo,
                            float* __restrict__ target_feat,
                            float* __restrict__ msa_feat, int L, int S) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L * 20) return;
  int a = i % 20, n = i / 20;
  float v = pseudo[i];
  target_feat[(size_t)n * embed::NAA + a] = v;     // 21-wide, pad stays 0
  for (int s = 0; s < S; ++s) {
    size_t row = ((size_t)s * L + n) * embed::NMSA_F;
    msa_feat[row + a] = v;          // the one-hot slots
    msa_feat[row + 25 + a] = v;     // the profile slots
  }
}

__global__ void gatherSeqGradK(const float* __restrict__ dtf,
                               const float* __restrict__ dmf,
                               float* __restrict__ dpseudo, int L, int S) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L * 20) return;
  int a = i % 20, n = i / 20;
  float g = dtf[(size_t)n * embed::NAA + a];
  for (int s = 0; s < S; ++s) {
    size_t row = ((size_t)s * L + n) * embed::NMSA_F;
    g += dmf[row + a] + dmf[row + 25 + a];
  }
  dpseudo[i] = g;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr,
            "usage: design <embed-dump> <stack-weights> <structmod-dump> "
            "[heads-dump] [steps]\n");
    return 1;
  }
  Oracle emb_o(argv[1]);
  Oracle stk_w(argv[2]);
  Oracle sm_o(argv[3]);
  Oracle hd_o(argc > 4 ? argv[4] : argv[3]);
  int steps = argc > 5 ? atoi(argv[5]) : 5;

  const int L = emb_o["in/target_feat"].shape[0];
  const int S = 2;                     // the design msa: query + one row
  printf("design step, no JAX   L=%d  S=%d  steps=%d\n", L, S, steps);

  cudaStream_t st;
  CU(cudaStreamCreate(&st));

  // ---- the pieces
  embed::Embed em;
  em.alloc(L, S, st);
  em.setWeights(emb_o);

  stack::Stack trunk;
  trunk.alloc(L, S, 48, st);
  trunk.setWeights(stk_w);

  structmod::StructureModule sm;
  sm.do_atoms = false;                 // the losses here read logits, not atoms
  sm.alloc(L, 8, st);
  sm.setWeights(sm_o);

  heads::Heads hd;
  hd.alloc(L, st);
  hd.setWeights(hd_o);

  // ---- the design variables
  float *params, *bias, *pseudo, *soft, *dpseudo, *adam_m, *adam_v;
  const size_t nseq = (size_t)L * 20;
  CU(cudaMalloc(&params, nseq * 4));
  CU(cudaMalloc(&bias, nseq * 4));
  CU(cudaMalloc(&pseudo, nseq * 4));
  CU(cudaMalloc(&soft, nseq * 4));
  CU(cudaMalloc(&dpseudo, nseq * 4));
  CU(cudaMalloc(&adam_m, nseq * 4));
  CU(cudaMalloc(&adam_v, nseq * 4));
  CU(cudaMemset(bias, 0, nseq * 4));
  CU(cudaMemset(adam_m, 0, nseq * 4));
  CU(cudaMemset(adam_v, 0, nseq * 4));
  {
    std::vector<float> init(nseq);
    srand(0);
    for (size_t i = 0; i < nseq; ++i) init[i] = (rand() / (float)RAND_MAX - .5f) * 0.1f;
    CU(cudaMemcpy(params, init.data(), nseq * 4, cudaMemcpyHostToDevice));
  }

  float *tf, *mf, *dtf, *dmf;
  CU(cudaMalloc(&tf, (size_t)L * embed::NAA * 4));
  CU(cudaMalloc(&mf, (size_t)S * L * embed::NMSA_F * 4));
  CU(cudaMalloc(&dtf, (size_t)L * embed::NAA * 4));
  CU(cudaMalloc(&dmf, (size_t)S * L * embed::NMSA_F * 4));
  CU(cudaMemset(tf, 0, (size_t)L * embed::NAA * 4));
  CU(cudaMemset(mf, 0, (size_t)S * L * embed::NMSA_F * 4));

  // the masks: one chain is the binder, the rest the target
  const int binder = L / 2;
  std::vector<float> hrows(L), hcols(L), hmpair((size_t)L * L), hres(L, 1.f),
      hidx(L), hasym(L);
  for (int i = 0; i < L; ++i) {
    hrows[i] = i < binder;
    hcols[i] = i >= binder;
    hidx[i] = (float)i;
    hasym[i] = i < binder ? 0.f : 1.f;
  }
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) hmpair[(size_t)i * L + j] = hrows[i] * hcols[j];
  auto up = [&](const std::vector<float>& h) {
    float* d;
    CU(cudaMalloc(&d, h.size() * 4));
    CU(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
  };
  float *rows = up(hrows), *mpair = up(hmpair), *ridx = up(hidx),
        *asym = up(hasym), *seqmask = up(hres);
  std::vector<float> hpm((size_t)L * L, 1.f), hmm((size_t)S * L, 1.f);
  float *pmask = up(hpm), *mmask = up(hmm);

  // loss scratch
  float *p, *pxs, *pxms, *pk, *icon, *dp, *dpk, *ddg;
  int *pick1, *cnt1, *pick2, *cnt2;
  const size_t LL = (size_t)L * L;
  CU(cudaMalloc(&p, LL * 4)); CU(cudaMalloc(&pxs, LL * 64 * 4));
  CU(cudaMalloc(&pxms, LL * 64 * 4)); CU(cudaMalloc(&pk, L * 4));
  CU(cudaMalloc(&icon, 4)); CU(cudaMalloc(&dp, LL * 4));
  CU(cudaMalloc(&dpk, L * 4)); CU(cudaMalloc(&ddg, LL * 64 * 4));
  CU(cudaMalloc(&pick1, (size_t)L * losses::MAXK * 4));
  CU(cudaMalloc(&cnt1, L * 4));
  CU(cudaMalloc(&pick2, (size_t)losses::MAXK * 4));
  CU(cudaMalloc(&cnt2, 4));
  float *dpair_h, *dsingle_h, *dpairAll, *dsingleSM, *dmsa0, *dmsaTrunk;
  CU(cudaMalloc(&dpair_h, LL * heads::CZ * 4));
  CU(cudaMalloc(&dsingle_h, (size_t)L * heads::CS * 4));
  CU(cudaMalloc(&dpairAll, LL * heads::CZ * 4));
  CU(cudaMalloc(&dsingleSM, (size_t)L * heads::CS * 4));
  CU(cudaMalloc(&dmsa0, (size_t)L * embed::CM * 4));
  CU(cudaMalloc(&dmsaTrunk, (size_t)S * L * embed::CM * 4));
  float *single, *dact_zero, *drot_zero, *dtrans_zero;
  CU(cudaMalloc(&single, (size_t)L * heads::CS * 4));
  CU(cudaMalloc(&dact_zero, (size_t)L * heads::CS * 4));
  CU(cudaMalloc(&drot_zero, (size_t)L * 9 * 4));
  CU(cudaMalloc(&dtrans_zero, (size_t)L * 3 * 4));
  CU(cudaMemset(drot_zero, 0, (size_t)L * 9 * 4));
  CU(cudaMemset(dtrans_zero, 0, (size_t)L * 3 * 4));

  const int gseq = (int)((nseq + 255) / 256);
  const int gll = (int)((LL + 255) / 256);

  // per-step wall time, which is the number that compares against a JAX
  // design step.  The first step is excluded: it pays cuBLAS autotuning and
  // first-touch page faults that a 300-step run pays once.
  double t_total = 0;
  cudaEvent_t ev0, ev1;
  CU(cudaEventCreate(&ev0));
  CU(cudaEventCreate(&ev1));
  for (int step = 1; step <= steps; ++step) {
    if (step == 2) { CU(cudaStreamSynchronize(st)); CU(cudaEventRecord(ev0, st)); }
    const float alpha = 2.f, temp = 1.f, sw = 1.f, hw = 0.f;
    // ---------------------------------------------------------- forward
    seqx::softSeqK<<<(L + 255) / 256, 256, 0, st>>>(params, bias, alpha, temp,
                                                    sw, hw, soft, pseudo, L);
    scatterSeqK<<<gseq, 256, 0, st>>>(pseudo, tf, mf, L, S);
    em.forward(tf, mf, ridx, asym, asym, asym);
    trunk.forward(em.pair, em.msa, pmask, mmask);
    em.singleFromTrunk(trunk.mout, single);
    sm.forward(single, trunk.pout, pmask);
    hd.forward(trunk.pout, sm.outAct());

    losses::conPerPairK<<<gll, 256, 0, st>>>(hd.dgram, 14.f, 0, p, pxs, pxms, L);
    losses::minKK<true><<<L, 32, L, st>>>(p, mpair, 2, pk, pick1, cnt1, L, L);
    losses::minKK<false><<<1, 32, L, st>>>(pk, rows, 9, icon, pick2, cnt2, L, 0);
    CU(cudaStreamSynchronize(st));
    printf("  step %2d  i_con %8.4f", step, fetch(icon, 1)[0]);

    // --------------------------------------------------------- backward
    {
      std::vector<float> one(1, 1.f);
      float* d1 = up(one);
      CU(cudaMemsetAsync(dpk, 0, L * 4, st));
      losses::minKBwdK<<<1, 32, 0, st>>>(pick2, cnt2, d1, dpk, 1, L, 0);
      CU(cudaMemsetAsync(dp, 0, LL * 4, st));
      losses::minKBwdK<<<L, 32, 0, st>>>(pick1, cnt1, dpk, dp, L, L, 1);
      losses::conPerPairBwdK<<<gll, 256, 0, st>>>(dp, pxs, pxms, p, 14.f, 0,
                                                  ddg, L);
      CU(cudaFree(d1));
    }
    // the heads: only the distogram carries a cotangent in this objective
    CU(cudaMemsetAsync(dsingle_h, 0, (size_t)L * heads::CS * 4, st));
    {
      static float* zpae = nullptr;
      static float* zpl = nullptr;
      if (!zpae) {
        CU(cudaMalloc(&zpae, LL * heads::NPAE * 4));
        CU(cudaMemset(zpae, 0, LL * heads::NPAE * 4));
        CU(cudaMalloc(&zpl, (size_t)L * heads::PL_BINS * 4));
        CU(cudaMemset(zpl, 0, (size_t)L * heads::PL_BINS * 4));
      }
      hd.backward(ddg, zpae, zpl, sm.outAct(), dpair_h, dsingle_h);
    }
    // the structure module, driven by the pLDDT head's gradient on its act
    CU(cudaMemsetAsync(dact_zero, 0, (size_t)L * heads::CS * 4, st));
    sm.backward(dsingle_h, drot_zero, dtrans_zero, single, dsingleSM, dpairAll);
    // the trunk's pair is read by BOTH the heads and the structure module
    foldit::accK<<<(int)((LL * heads::CZ + 255) / 256), 256, 0, st>>>(
        dpairAll, dpair_h, LL * heads::CZ);
    // single_activations sits between the trunk's msa and the structure module
    em.singleFromTrunkBwd(dsingleSM, dmsa0);
    CU(cudaMemsetAsync(dmsaTrunk, 0, (size_t)S * L * embed::CM * 4, st));
    CU(cudaMemcpyAsync(dmsaTrunk, dmsa0, (size_t)L * embed::CM * 4,
                       cudaMemcpyDeviceToDevice, st));
    trunk.backward(dpairAll, dmsaTrunk, pmask, mmask);
    em.backward(trunk.b.dm, (const float*)trunk.b.dp, nullptr, tf, dtf, dmf);
    gatherSeqGradK<<<gseq, 256, 0, st>>>(dtf, dmf, dpseudo, L, S);
    seqx::softSeqBwdK<<<(L + 255) / 256, 256, 0, st>>>(dpseudo, soft, alpha,
                                                       temp, sw, hw, dpseudo, L);
    seqx::normSeqGradK<<<1, 32, 0, st>>>(dpseudo, L);
    seqx::adamK<<<gseq, 256, 0, st>>>(params, dpseudo, adam_m, adam_v, 0.1f,
                                      0.9f, 0.999f, 1e-8f, step, nseq);
    CU(cudaStreamSynchronize(st));
    printf("   (updated)\n");
  }
  if (steps >= 2) {
    CU(cudaEventRecord(ev1, st));
    CU(cudaEventSynchronize(ev1));
    float ms = 0;
    CU(cudaEventElapsedTime(&ms, ev0, ev1));
    t_total = ms / (steps - 1);
    printf("\n  %.1f ms a design step (steps 2..%d), JIT compiles: 0\n",
           t_total, steps);
  }
  size_t fb, tot;
  CU(cudaMemGetInfo(&fb, &tot));
  printf("\n  VRAM %.2f GB    JIT compiles: 0\n", (tot - fb) / 1e9);
  return 0;
}
