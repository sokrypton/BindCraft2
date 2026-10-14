// One iteration of AF2's TEMPLATE PAIR STACK in CUDA, forward and backward.
//
// THE ORDER is modules.TemplateEmbeddingIteration, which is exactly the
// Evoformer block's PAIR track with no MSA side at all:
//
//   p2 = p1 + TriMulOutgoing(p1)      <- p1 is the block's input
//   p3 = p2 + TriMulIncoming(p2)
//   p4 = p3 + TriAttnStarting(p3)
//   p5 = p4 + TriAttnEnding(p4)
//   p6 = p5 + PairTransition(p5)      <- pair out
//
// WHY THIS IS NOT block.cuh WITH pair_only AND A WIDTH.  It could be: the
// modules are the same five, the residual discipline is the same, and Block
// already has a pair-only mode for the extra-MSA stack.  What Block also has
// is the keep-tier machinery -- keepSlots, regrant, donated scratch, the
// save-set -- and all of that exists because the 48-block stack cannot afford
// to store 48 blocks of a 128-channel pair.  This stack is TWO blocks of a
// 64-channel one: 4 MB a pair at N=128 against the Evoformer's 100 MB at
// N=256.  There is nothing to remat and nothing to tier, so chaining the five
// modules directly is both smaller and honest about which problem it solves.
//
// THE SAVE-SET IS THEREFORE TOTAL: every one of the five inputs is kept, which
// is what lets backward() run with no replay.
#pragma once
#include <vector>
#include "trimul.cuh"
#include "triattn.cuh"
#include "transition.cuh"

namespace tplblock {

static const int CZ = 64;   // pair_channel, model.embeddings_and_evoformer.template
// num_intermediate_factor on the template stack's pair_transition.  The
// Evoformer's is 4; this is the one dimension that is not just "half".
static const int TRANS_FACTOR = 2;

struct TplBlock {
  int N = 0;
  cudaStream_t st = nullptr;

  tmul64::TriMul tmo, tmi;
  tattn64::TriAttn tas, tae;
  trans::Trans ptr;

  // the chain links: p2b..p6b are the four residual outputs plus the block's.
  // p1b is NOT here -- it is the caller's input, and forward() keeps the
  // pointer so backward() can hand it back to the first module.
  __nv_bfloat16 *p2b = nullptr, *p3b = nullptr, *p4b = nullptr,
                *p5b = nullptr, *p6b = nullptr;
  __nv_bfloat16* dp = nullptr;            // the cotangent, in place down the chain
  const __nv_bfloat16* p1b = nullptr;     // borrowed, see above

  // DROPOUT, because TemplateEmbeddingIteration wraps all five of these in
  // dropout_wrapper exactly as the Evoformer's pair track does.  Running
  // without it would not be a numerical difference, it would be a different
  // model whenever a design stage has dropout on -- which bindcraft's early
  // stages do.  Same masks as block.cuh: 0.25, shared per row for the two
  // triangle multiplications and the starting-node attention, per column for
  // the ending-node one, none on the transition.  The module ids are their
  // own range so the template stack does not draw the Evoformer's masks.
  bool use_dropout = false;
  const uint32_t* drop_seed = nullptr;    // [seed, enabled], device
  uint32_t drop_depth = 0;
  __nv_bfloat16* ddrop = nullptr;         // the masked cotangent
  static constexpr float DROP = 0.25f;
  enum { DM_TMO = 11, DM_TMI = 12, DM_TAS = 13, DM_TAE = 14 };

  void alloc(int N_, cudaStream_t s = nullptr) {
    N = N_; st = s;
    size_t np = (size_t)N * N * CZ;
    for (auto q : {&p2b, &p3b, &p4b, &p5b, &p6b, &dp})
      CU(cudaMalloc(q, np * 2));
    CU(cudaMalloc(&ddrop, np * 2));
    tmo.alloc(N, true);
    tmi.alloc(N, false);
    tas.alloc(N, false); tas.allocOrientation();
    tae.alloc(N, true);  tae.allocOrientation();
    ptr.is_pair = true;
    ptr.alloc((size_t)N * N, CZ, st, TRANS_FACTOR);
    // ONE STREAM, for the reason block.cuh gives at length: trimul::alloc and
    // triattn::alloc each make a stream of their own for their standalone
    // gates, and a block that never called setStream runs its modules
    // unordered against its own residual adds.
    setStream(st);
  }

  // RUN OUT OF THE EVOFORMER'S BYTES.
  //
  // The template stack is first in the forward and last in the backward and
  // never runs beside the Evoformer; four conditions were measured before
  // this was written (NB_TPL_SCRIBBLE / NB_EVO_SCRIBBLE / NB_TPL_DIRTY /
  // NB_EVO_DIRTY), covering both "the contents need not survive" and "the
  // scratch is written before it is read".
  //
  // Templated on the donor so this header does not have to include
  // block.cuh, which includes this one's neighbours.  The donor's tri-muls
  // are the WIDE instantiation of the same source file, so the member lists
  // correspond element by element and every one of its buffers is at least
  // as large -- adoptScratchFrom checks that and aborts if it ever stops
  // being true.
  // THE BLOCK'S OWN CHAIN BUFFERS, as well as its modules'.
  //
  // p2b..p6b, dp and ddrop are this block's residual chain at CZ = 64; the
  // Evoformer block holds the same chain at 128, so each of its is exactly
  // twice the size.  `ddrop` is the one asymmetry: this block allocates it
  // unconditionally and block.cuh only under use_dropout, so when the donor
  // has none we keep ours -- the same rule trimul's `out` needed.
  template <class EvoBlock>
  void adoptChainFrom(EvoBlock& b) {
    const size_t np = (size_t)N * N * CZ * 2;
    if (b.chainBytes() < np) {
      fprintf(stderr, "tplblock: donor chain link is %zu bytes, needs %zu\n",
              b.chainBytes(), np);
      exit(1);
    }
    struct Pair { __nv_bfloat16** mine; __nv_bfloat16** theirs; };
    const Pair ps[] = {{&p2b, &b.p2b}, {&p3b, &b.p3b}, {&p4b, &b.p4b},
                       {&p5b, &b.p5b}, {&p6b, &b.p6b}, {&dp, &b.dp},
                       {&ddrop, &b.ddrop}};
    for (const Pair& q : ps) {
      if (!*q.theirs) continue;          // the donor does not have this one
      if (*q.mine) CU(cudaFree(*q.mine));
      *q.mine = *q.theirs;
    }
  }

  template <class EvoBlock>
  void adoptScratch(EvoBlock& b) {
    adoptChainFrom(b);
    tmo.adoptScratchFrom(b.tmo, "tplblock.tmo");
    tmi.adoptScratchFrom(b.tmi, "tplblock.tmi");
    tas.adoptScratchFrom(b.tas, "tplblock.tas");
    tae.adoptScratchFrom(b.tae, "tplblock.tae");
    ptr.adoptScratchFrom(b.ptr, "tplblock.ptr");
  }

  void setStream(cudaStream_t s) {
    st = s;
    ptr.st = s; tmo.st = s; tmi.st = s; tas.st = s; tae.st = s;
    CUB(cublasSetStream(tmo.cub, s));
    CUB(cublasSetStream(tmi.cub, s));
    CUB(cublasSetStream(tas.cub, s));
    CUB(cublasSetStream(tae.cub, s));
  }

  // Every module's weight pointers, in a fixed order -- block.cuh's
  // weightSlots(), minus the MSA modules a template block does not have.
  // TplStack uses it to run ONE block at every depth, swapping the pointers
  // instead of holding a block (and a module scratch set) per depth.
  size_t chainBytes() const { return (size_t)N * N * CZ * 2; }

  std::vector<void**> weightSlots() {
    std::vector<void**> v;
    auto app = [&](std::vector<void**> w) { v.insert(v.end(), w.begin(), w.end()); };
    app(tmo.weightSlots()); app(tmi.weightSlots());
    app(tas.weightSlots()); app(tae.weightSlots()); app(ptr.weightSlots());
    return v;
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    tmo.setWeights(o, P + "triangle_multiplication_outgoing/");
    tmi.setWeights(o, P + "triangle_multiplication_incoming/");
    tas.setWeights(o, P + "triangle_attention_starting_node/");
    tae.setWeights(o, P + "triangle_attention_ending_node/");
    ptr.setWeights(o, P + "pair_transition/");
  }

  void forward(const __nv_bfloat16* pair, const float* pair_mask) {
    size_t np = (size_t)N * N * CZ;
    int gp = (int)((np + 255) / 256);
    p1b = pair;
    tmo.forward(p1b, pair_mask);
    if (use_dropout)
      addDropNarrowBK<true><<<gp, 256, 0, st>>>(p2b, p1b, tmo.out, N, CZ,
          drop_seed, drop_depth, DM_TMO, DROP, np);
    else
      addNarrowBK<<<gp, 256, 0, st>>>(p2b, p1b, tmo.out, np);
    tmi.forward(p2b, pair_mask);
    if (use_dropout)
      addDropNarrowBK<true><<<gp, 256, 0, st>>>(p3b, p2b, tmi.out, N, CZ,
          drop_seed, drop_depth, DM_TMI, DROP, np);
    else
      addNarrowBK<<<gp, 256, 0, st>>>(p3b, p2b, tmi.out, np);
    tas.forward(p3b, pair_mask);
    if (use_dropout)
      addDropNarrowBK<true><<<gp, 256, 0, st>>>(p4b, p3b, tas.out, N, CZ,
          drop_seed, drop_depth, DM_TAS, DROP, np);
    else
      addNarrowBK<<<gp, 256, 0, st>>>(p4b, p3b, tas.out, np);
    tae.forward(p4b, pair_mask);
    if (use_dropout)
      addDropNarrowBK<false><<<gp, 256, 0, st>>>(p5b, p4b, tae.out, N, CZ,
          drop_seed, drop_depth, DM_TAE, DROP, np);
    else
      addNarrowBK<<<gp, 256, 0, st>>>(p5b, p4b, tae.out, np);
    ptr.forward(p5b);
    addNarrowBK<<<gp, 256, 0, st>>>(p6b, p5b, ptr.out, np);
  }

  // `ct` is the cotangent on p6; `dp` ends up holding the one on p1.  Must
  // follow a forward() with the same input.
  //
  // Each module takes dp as BOTH its incoming cotangent and its accumulator:
  // the residual's input gradient is the delta's gradient plus what passed by,
  // and these backwards each read their cotangent once, at the top, into a
  // bf16 copy -- which is what makes the aliasing safe.  See block.cuh.
  void backward(const __nv_bfloat16* ct, const float* pair_mask) {
    size_t np = (size_t)N * N * CZ;
    CU(cudaMemcpyAsync(dp, ct, np * 2, cudaMemcpyDeviceToDevice, st));
    ptr.backward(dp, p5b, dp, /*acc*/true);
    if (use_dropout) {
      dropScaleBK<false><<<(np + 255) / 256, 256, 0, st>>>(ddrop, dp, N, CZ,
          drop_seed, drop_depth, DM_TAE, DROP, np);
      tae.backward(ddrop, p4b, pair_mask, dp, /*acc*/true);
    } else tae.backward(dp, p4b, pair_mask, dp, /*acc*/true);
    if (use_dropout) {
      dropScaleBK<true><<<(np + 255) / 256, 256, 0, st>>>(ddrop, dp, N, CZ,
          drop_seed, drop_depth, DM_TAS, DROP, np);
      tas.backward(ddrop, p3b, pair_mask, dp, /*acc*/true);
    } else tas.backward(dp, p3b, pair_mask, dp, /*acc*/true);
    if (use_dropout) {
      dropScaleBK<true><<<(np + 255) / 256, 256, 0, st>>>(ddrop, dp, N, CZ,
          drop_seed, drop_depth, DM_TMI, DROP, np);
      tmi.backward(ddrop, p2b, pair_mask, dp, /*acc*/true);
    } else tmi.backward(dp, p2b, pair_mask, dp, /*acc*/true);
    if (use_dropout) {
      dropScaleBK<true><<<(np + 255) / 256, 256, 0, st>>>(ddrop, dp, N, CZ,
          drop_seed, drop_depth, DM_TMO, DROP, np);
      tmo.backward(ddrop, p1b, pair_mask, dp, /*acc*/true);
    } else tmo.backward(dp, p1b, pair_mask, dp, /*acc*/true);
  }
};

}  // namespace tplblock
