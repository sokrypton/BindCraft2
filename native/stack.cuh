// The 48-deep Evoformer stack, forward and backward, on one block's scratch.
//
// This is where the design step's time actually is: ablating the stack depth
// measured 6.21 ms a block plus 38.9 ms of everything else, so 48 blocks are
// 88% of a 338 ms step.
//
// TWO THINGS MAKE IT FIT.
//
// WEIGHTS are per depth, scratch is per step.  One Block holds the scratch and
// the save-set; the 48 depths are 48 sets of weight POINTERS, snapshotted out
// of setWeights() once each and swapped in by use().  48 Blocks would be 48
// copies of several hundred MB of scratch; this is 48 x ~3.5 MB of bf16
// weights.  It is the only reason each module keeps its weight pointers in a
// group separate from its activations.
//
// ACTIVATIONS are rematerialised.  Each block's backward needs the nine
// intermediates its forward produced, which is ~100 MB -- times 48 is 4.8 GB,
// before anything else on the card.  So the forward keeps only each block's
// INPUT (pair + msa, ~17 MB a depth, 830 MB in all) and the backward replays
// one block's forward immediately before running its backward.  That is
// exactly what AF2's own use_remat does, and it costs one extra forward: the
// stack's backward is fwd+bwd per block, not bwd.
#pragma once
#include "block.cuh"

namespace stack {

// ONE BLOCK'S SCRATCH FOR BOTH STACKS.  A design step builds TWO of these:
// the 4-block extra-MSA stack and the 48-block main one.  Each allocated its
// own block scratch, and at N=384 that is 3.16 + 3.32 = 6.48 GB -- for
// buffers that are, by definition, scratch: everything that has to survive a
// call is in the save-set or a keep tier, and the replay RECOMPUTES whatever
// a tier did not get.  The two stacks never run at the same time (extra
// forward, main forward, main backward, extra backward are four separate
// custom calls), so one set of buffers serves both.
//
// WHAT IS SHARED IS THE DEVICE MEMORY, NOT THE OBJECT, AND ONLY THE PART
// THAT DOES NOT DEPEND ON MSA DEPTH.  Each Stack keeps its
// own `Block` -- its own flags, its own weight slots, its own recorded
// `wsets`, its own MSA modules -- and adopts only the five PAIR modules and
// the pair-shaped buffers, which depend on N alone.  Those are 95% of the
// bytes.  A first version shared everything by inflating S and copying the
// whole struct, and the gate read grad/pair 0.12: a module captures S when
// it allocates, so a block told S=8 cannot then be run at S=1.
static block::Block* g_donor = nullptr;
static int g_donor_N = 0;
static bool g_donor_opm = false;

struct Stack {
  block::Block b;
  bool shared_scratch = false;
  int N = 0, S = 0, depth = 0;
  // the saved INPUTS, one pair and one msa a depth, plus the stack's output
  // FLOAT32 throughout, because AF2's carry between blocks is float32: only
  // the activations entering block 0 are bf16 (see common.cuh:addF32K).  The
  // save-set doubles to 1.6 GB at N=256 depth 48, which is the price of it.
  // THE PAIR SAVE-SET IS bf16.  It is the biggest allocation here -- np*4*
  // depth was 3.62 GB at N=384 -- and the block already narrows its input
  // before any module sees it, so the only thing storing f32 bought was
  // the base of the f32 residual chain.  Measured against a DESIGN step
  // (not against the f32 oracle, which is what rejected this before): the
  // median step-1 metric moves 8.1e-04, 27x less than the native stack
  // already moves them against XLA.  The msa side stays f32: it is
  // S*N*Cm, 0.02 GB, and not worth a second code path.
  std::vector<__nv_bfloat16*> pin;
  std::vector<float*> min_;
  __nv_bfloat16* psave = nullptr;
  float* msave = nullptr;                            // the default storage
  float *pout = nullptr, *mout = nullptr;      // where forward() writes
  float *pown = nullptr, *mown = nullptr;      // ours, only if nobody lends one

  // Point the outputs at buffers we own.  The standalone gates need this;
  // the FFI sets pout/mout to XLA's own and never calls it.
  void useOwnOut() {
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    if (!pown) CU(cudaMalloc(&pown, np * 4));
    if (!mown) CU(cudaMalloc(&mown, nm * 4));
    pout = pown; mout = mown;
  }
  __nv_bfloat16* dpc = nullptr;
  float* dmc = nullptr;   // the cotangent carried between depths
  // the 48 weight sets, as the values the block's slots should hold
  std::vector<void**> slots;
  // KEEP INTERMEDIATES PER DEPTH instead of recomputing them in the replay,
  // GREEDILY AND BY TIER.  See KeepTier in common.cuh for the measured
  // ms-per-GB of each; the short version is that the three tiers differ by 6x
  // in value per byte, and this used to be ONE all-or-nothing allocation, so a
  // length that could not afford the expensive tier lost the cheap ones too.
  // Now each tier is granted for as many DEPTHS as the remaining budget
  // allows, cheapest per byte first, so the keep degrades instead of
  // vanishing.  The pointers are swapped per depth in use(), the same
  // mechanism as the weights.
  std::vector<void**> fslots;
  std::vector<size_t> fsize;
  std::vector<int> ftier;
  std::vector<void*> forig;          // what alloc() gave the module, to restore
  char* tbase[KEEP_NTIER] = {};
  size_t tstride[KEEP_NTIER] = {};   // bytes a depth, for one tier
  // BYTES ACTUALLY HELD in each tier's arena.  The stride changes when the
  // step is re-derived at another length; the arena does not, and knowing
  // its size is what lets the new stride be fitted into it instead of
  // allocating a second one.
  size_t tbytes[KEEP_NTIER] = {};
  int tdepth[KEEP_NTIER] = {};       // depths actually granted
  bool have(int tier, int i) const { return i < tdepth[tier]; }
  bool full(int tier) const { return tdepth[tier] >= depth; }
  std::vector<std::vector<void*>> wsets;

  void alloc(int N_, int S_, int depth_, cudaStream_t st = nullptr) {
    N = N_; S = S_; depth = depth_;
    // A REAL STREAM, ALWAYS.  The legacy default stream cannot be captured
    // into a graph, so a standalone caller that passes nothing would be
    // unable to use STACK_GRAPH at all. cudaStreamCreate (not the
    // non-blocking flavour) still synchronises with the legacy default
    // stream, so the gate's own host-side fetches keep working -- the
    // non-blocking flavour would silently break them, which is the stream
    // trap this port has already paid for twice.
    if (!st) { CU(cudaStreamCreate(&owned)); st = owned; }
    // WHOSE MEMORY IS IT.  BLK_MEM reads the card before the stack has
    // allocated anything, which is the only way to separate what the model
    // holds from what this does -- the tier report below is read AFTER the
    // module scratch, so it has never been able to answer that.
    size_t fb0 = 0, tb0 = 0;
    if (getenv("BLK_MEM")) {
      CU(cudaMemGetInfo(&fb0, &tb0));
      fprintf(stderr, "stack: before any stack allocation, %.2f GB of %.2f "
              "is already in use\n", (tb0 - fb0) / 1e9, tb0 / 1e9);
    }
    // THE DONOR, OR A FRESH SET.  NE_SHARE_BLOCK=0 opts out.
    const bool share = envOnDef("NE_SHARE_BLOCK", true);
    const bool take = share && g_donor && g_donor_N == N &&
                      g_donor_opm == b.opm_first;
    b.alloc(N, S, st, take ? g_donor : nullptr);
    shared_scratch = take;
    if (share && !take) {
      g_donor = new block::Block(b);
      g_donor_N = N; g_donor_opm = b.opm_first;
      // the DONOR's own forig buffers must outlive it too
      shared_scratch = true;
    }
    if (getenv("BLK_MEM") && take)
      fprintf(stderr, "stack: adopted the other stack's pair scratch\n");
    if (getenv("BLK_MEM")) {
      size_t fb1 = 0, tb1 = 0;
      CU(cudaMemGetInfo(&fb1, &tb1));
      fprintf(stderr, "stack: the block's scratch took %.2f GB\n",
              (fb0 - fb1) / 1e9);
    }
    slots = b.weightSlots();
    for (auto& k : b.keepSlots()) {
      fslots.push_back(k.slot); fsize.push_back(k.bytes); ftier.push_back(k.tier);
      forig.push_back(*k.slot);          // the module's own buffer, to restore
    }
    for (size_t k = 0; k < fslots.size(); ++k)
      tstride[ftier[k]] += (fsize[k] + 255) / 256 * 256;   // keep each aligned

    // THE MANDATORY BUFFERS GO FIRST.  The remat save-set is not optional and
    // it is the big one -- np * 4 * depth is 3.6 GB at N=384 -- so taking the
    // optional keep before it meant the keep succeeded and then the save-set
    // died, which is the one outcome the check below was written to prevent.
    // L=384 OOMed at `common.cuh:49 cuda: out of memory` while the same run
    // with NO_FLASH_KEEP=1 fitted in 18.57 GB and beat XLA by 1.36x.
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    pin.resize(depth); min_.resize(depth);
    // NOT ALLOCATED HERE.  np*4*depth is 1.23 GB at L=224 and 3.62 at N=384,
    // and in the in-model path it is never read: own_save follows
    // STACK_GRAPH_UNSAFE, which is off, so the FFI hands us XLA's save-set
    // instead.  Allocating it anyway was what made L=375 + the extra-MSA
    // stack die, AT THIS LINE, with 3.62 GB it would have thrown away.
    // ensureSaveSet() makes it on demand for the standalone path, which has
    // nobody to hand it one.
    // `pout`/`mout` ARE NOT ALLOCATED HERE EITHER, for the same reason the
    // save set is not: in the in-model path the FFI has an output buffer
    // already and we can write it directly.  pout is np*4 -- 139 MB at N=522
    // -- and ffi.cu used to copy this buffer straight into it, so it was 139
    // MB of memory and a 139 MB device-to-device copy for nothing.
    //
    // WRITING THE CALLER'S BUFFER IS SAFE HERE, and that is worth spelling
    // out because the obvious objection is real elsewhere.  The widen at the
    // end of forward_ is `widenBf16K(pout, pb, np)`, and if its SOURCE could
    // alias its destination that would corrupt: thread i reads 2 bytes at 2i
    // and writes 4 bytes at 4i, landing on the elements threads 2i and 2i+1
    // have not read yet.  It cannot alias.  `pb` there is `b.p6b`, a
    // block-internal buffer, not the stack's `pair` argument -- the loop
    // reassigns it after the first block.  The caller's `pair` is read only
    // by block 0's toBf16K, which is long finished.
    //
    // useOwnOut() covers the standalone path, which has nobody to hand it a
    // buffer.

    // ONLY WHAT COMFORTABLY FITS, TIER BY TIER.  Every buffer here grows as
    // N^2, so at N=384 the flash tier alone is 3.9 GB on top of a scratch that
    // has itself tripled, and a design that would have run would instead die
    // at startup.  A speedup that turns a working length into an
    // out-of-memory is not a speedup.
    //
    // WHAT CHANGED: this was one all-or-nothing allocation of all three tiers.
    // The flash tier is 97% of the bytes and the WORST value per byte of the
    // three (22 ms/GB against 136 and 104), so at N=384 the card refused 1.8 GB
    // and the stack lost the 0.1 GB that was six times better value with it.
    // Now the tiers are granted cheapest-first, and a tier that does not fit
    // whole is granted for as many DEPTHS as do fit -- half a keep is half a
    // saving, where before it was none.
    //
    // TWO GUARDS, because one is not enough.  `cudaMemGetInfo` is read before
    // the weights are converted and before JAX grows into the backward, so a
    // margin on top of what is free today is the only way to leave room for
    // what is allocated tomorrow -- and even then the estimate can be wrong,
    // so each allocation is attempted with a plain cudaMalloc and a failure
    // falls back to recomputing instead of aborting.
    size_t freeb = 0, totalb = 0;
    CU(cudaMemGetInfo(&freeb, &totalb));
    size_t margin = np * 4 * depth / 2 + totalb / 8;
    // A SHALLOW STACK RUNS FIRST AND MUST LEAVE ROOM FOR THE REST OF THE STEP.
    // The 4-block extra-MSA stack allocates before the 48-deep trunk, the
    // structure module and the heads, and none of those are in `freeb` yet --
    // so its own margin (np*4*4/2, 303 MB at N=544) let it grant itself a
    // depth-set the rest of the step then could not have.  Measured at 537
    // residues: the shallow stack took 3.66 GB of tiers and the structure
    // module died in foldit/ipa with 1.7 GB free; with the grant suppressed
    // the same length RUNS, at 2036 ms against the 5919 the JAX fallback
    // costs.  Reserve the trunk's save-set, which is what follows it.
    //
    // The REGRANT is why this is not simply giving the memory up: it runs on
    // the first save forward, once everything else has allocated and XLA has
    // stopped growing, and hands the tiers back what is genuinely spare.
    // AND IT STAYS np*2*48 UNDER STRIDE REMAT, which is deliberate and was
    // measured.  The trunk's save-set is only saveSlots(48) slots now -- 15
    // at NB_STRIDE=4 -- so this looks like 2.8 GB of over-reserve at N=576,
    // and shrinking it to the real figure is a REGRESSION: the shallow
    // stack spends the difference on its own keep tiers and the structure
    // module dies instead.  Measured, NB_STRIDE=4, three lengths that ran
    // before the change: 552 foldit cublas:3, 567 ipa OOM, 587 foldit OOM,
    // all reproducible twice.  The number is a proxy for "the rest of the
    // step", not for the save-set it is written in terms of.
    //
    // AND IT IS DEAD FOR A PREALLOCATED STACK.  `margin` only ever reaches
    // `budget`, and `budget` only feeds the keep-tier loop below, which
    // `defer_keep` skips outright -- so a stack built by
    // native_stack_prealloc never reads this number.  It still governs the
    // stacks nobody preallocated (NB_NO_PREALLOC, or a shape the trace did
    // not announce), which is why the measurement above still stands and
    // why this is not the place to go looking for memory.
    if (depth <= 8) margin += np * 2 * 48;
    size_t budget = freeb > margin ? freeb - margin : 0;
    // KEEP_BUDGET_MB FORCES the budget, so the partial-grant path can be
    // GATED rather than hoped for.  Everything below only fires under memory
    // pressure, which on this card means N=384 -- an allocator branch that is
    // exercised exactly once, in the hardest configuration, is how a silent
    // wrong gradient gets shipped.  With it, `KEEP_BUDGET_MB=120 ./stack
    // out/stack48-n256 stack_weights` grants stats and bias whole and the
    // flash tier a fraction of its depths, on a card with room to spare.
    // KEEP_ACTS IS OFF BY DEFAULT, and that is a memory decision, not a
    // measurement one.  It works and it is the largest single win left:
    // standalone at N=256 the backward goes 717.1 -> 667.0 ms (-7.0%), and
    // in-model on a PD1 binder 963.5 -> 921.5 (-4.4%), gradient unchanged.
    // It costs 5 x N^2 x I x 2 bytes a depth -- 8.1 GB at N=256 standalone,
    // 5.3 GB in-model at N=288, which took that step to 19.3 GB of 23.68.
    // 82% of the card for 4.4% is the wrong default for a tool that runs
    // multi-hour campaigns and whose weak side is already memory; it is the
    // right option for anyone with room.  NE_KEEP_ACTS=1.
    if (const char* kb = getenv("KEEP_BUDGET_MB"))
      budget = (size_t)atoll(kb) * 1000000;
    // DEFER THE KEEP, which is what a trace-time prealloc needs.  alloc()
    // sizes the tiers from cudaMemGetInfo, and at trace time the card is
    // EMPTY -- so a stack built then grants itself everything and the step
    // that follows has nothing.  Measured: the N=192 depth-48 stack took
    // 6.10 GB preallocated against 0.84 lazily, and the N=576 pair-only
    // 7.48 against 4.30.  With this set the prealloc takes only what the
    // block cannot run without, and regrant() -- which exists for exactly
    // this, and runs on the first save forward once XLA has stopped growing
    // -- hands out the tiers from what is genuinely spare.
    // A RE-DERIVE AT A NEW LENGTH INVALIDATES THE GRANT.
    //
    // `tstride[t]` is recomputed from the new length by the code above;
    // `tdepth[t]` is the number of depths the PREVIOUS length's budget paid
    // for, and `tbase[t]` is an arena carved at the previous length's stride.
    // Keeping the second two while replacing the first is the one
    // combination that is not merely slow: use() hands every module slot an
    // address computed from the new stride inside an arena laid out for the
    // old one, and the depth count is the old one's too.  It reads as an
    // illegal access two kernels later.
    //
    // So the tiers are dropped here and regrant() hands them out again for
    // the length the stack now runs at.  The arena is not freed -- the pool
    // gives the same bytes back when the grant asks, because a shorter
    // length's stride is smaller.  The cost is that the short state runs
    // untiered until the next save forward completes a grant.
    if (nbrealloc::active()) {
      // RE-FIT THE ARENA, do not throw the grant away.
      //
      // The first version of this set tdepth to 0, regranted to false and
      // save_fwds to 0, so that regrant() would hand the tiers out again at
      // the new length.  Both of those were wrong, and the first one
      // silently:
      //
      //   * `save_fwds = 0` on every re-derive, and with ONE object serving
      //     two states there is a re-derive on every call, so the counter
      //     never reached regrant_after and regrant() returned early for the
      //     life of the process.  Zero keep tiers, permanently -- this
      //     morning's "a deferred grant that nobody completes" bug, put back
      //     by the one-object work and worth the 15% measured at complex 215.
      //     A flat per-call curve is what it looks like from outside:
      //     19378 / 14250 / 14292 ... with no warm-up step at all.
      //
      //   * dropping tdepth without freeing tbase would have made the NEXT
      //     grant cudaMalloc a second arena and leak the first, once a call
      //     ever got that far.
      //
      // The arena is already the right bytes; only the stride moved.  So
      // keep it and recompute how many depths now fit.
      for (int t = 0; t < KEEP_NTIER; ++t)
        tdepth[t] = (tbase[t] && tstride[t])
                      ? (int)std::min((size_t)depth, tbytes[t] / tstride[t]) : 0;
    }
    const bool no_keep = envOn("NO_FLASH_KEEP") || defer_keep;
    // NO_FLASH_KEEP keeps its old meaning -- drop EVERYTHING -- because it is
    // what the scaling sweep and the L=384 fallback measurement used, and a
    // flag that silently started meaning "drop one tier" would make those
    // numbers incomparable.  NO_KEEP_TIER=<n> drops one.
    const char* drop = getenv("NO_KEEP_TIER");
    // ON BY DEFAULT since 2026-10-08.  It is -8.3% of the backward (703.3 ->
    // 644.9 ms at N=256 depth 48, gradient equivalent) and the greedy budget
    // already degrades it safely: a card that cannot afford the 9.71 GB
    // depth-set gets a PARTIAL grant, whose value is linear in the depths
    // granted (48 -62.2 ms, 25 -30.7, 10 -8.9, 3 -1.6), and one that can
    // afford nothing gets zero and replays as before.  NE_KEEP_ACTS=0 opts
    // out -- which now means what it says; see envOn().
    const bool want_acts = envOnDef("NE_KEEP_ACTS", true);
    for (int t = 0; t < KEEP_NTIER && !no_keep; ++t) {
      if (drop && atoi(drop) == t) continue;
      if ((t == KEEP_ACTS || t == KEEP_TMUL_T) && !want_acts) continue;
      // KEEP_RELU IS OFF BY DEFAULT AND THE REASON IS A MEASUREMENT, not
      // memory: it is a wash.  See the relu-mask section in the README.
      if (t == KEEP_RELU && !envOn("NE_KEEP_RELU")) continue;
      if (!tstride[t]) continue;
      int n = (int)std::min((size_t)depth, budget / tstride[t]);
      if (keepWholeOnly(t) && n < depth) n = 0;
      while (n > 0 && cudaMalloc(&tbase[t], (size_t)n * tstride[t]) != cudaSuccess) {
        cudaGetLastError();                   // a failed cudaMalloc, cleared
        tbase[t] = nullptr; n /= 2;           // the estimate was wrong; halve
      }
      tdepth[t] = tbase[t] ? n : 0;
      tbytes[t] = (size_t)tdepth[t] * tstride[t];
      budget -= (size_t)tdepth[t] * tstride[t];
    }
    // A TIER GRANTED IN FULL MAKES THE MODULE'S OWN BUFFER DEAD.  use() points
    // each slot at the arena for every depth it covers and only falls back to
    // forig[k] for the depths it does not, so when tdepth == depth that
    // fallback is unreachable and the buffer alloc() made is never read again.
    // It was still held: 188 MB at N=256 depth 48, and it grows as N^2 like
    // everything else here, so it is 420 MB at N=384 -- out of exactly the
    // headroom the margin exists to protect.  forig is nulled rather than left
    // dangling, so a mistake here is a null dereference and not silent
    // corruption.
    size_t freed = 0;
    for (size_t k = 0; k < fslots.size(); ++k)
      if (full(ftier[k]) && forig[k] && !shared_scratch) {
        CU(cudaFree(forig[k]));
        freed += fsize[k];
        forig[k] = nullptr;
      }
    if (freed && getenv("BLK_MEM"))
      fprintf(stderr, "stack: freed %.1f MB of module buffers made dead by "
              "fully-granted tiers\n", freed / 1e6);
    // A DEFERRED GRANT THAT NOBODY COMPLETES IS A STACK WITH NO TIERS, and
    // for one release that is exactly what the fused design step was: it
    // owns its stacks, so it never reached the handler where regrant() is
    // called, and `no_keep` here meant the first grant said nothing either.
    // Zero tiers, in silence, costing 181 ms a step at complex 215.  Say it
    // instead, and name the thing that has to happen next.
    if (defer_keep)
      fprintf(stderr, "stack: first grant took NO keep tiers (defer_keep) -- "
              "regrant() must be called on a save forward or this stack "
              "replays every block for the life of the process\n");
    if (!no_keep)
      for (int t = 0; t < KEEP_NTIER; ++t)
        if (tdepth[t] < depth && !(t == KEEP_RELU && !envOn("NE_KEEP_RELU")))
          // A TIER THAT IS SWITCHED OFF IS NOT A TIER THAT DID NOT FIT.
          // This line used to say "the replay recomputes the rest" either
          // way, which reads like memory pressure and is how a reader
          // concludes the card is full when nothing was ever asked for.
          fprintf(stderr, "stack: keep tier '%s' granted %d of %d depths "
                  "(%.2f GB a depth-set, %.2f GB free, %.2f GB held back) -- "
                  "%s\n", keepName(t), tdepth[t], depth,
                  (double)tstride[t] * depth / 1e9, freeb / 1e9, margin / 1e9,
                  ((t == KEEP_ACTS || t == KEEP_TMUL_T) && !want_acts)
                      ? "switched OFF by NE_KEEP_ACTS=0"
                      : "the replay recomputes the rest");
  }

  // THE SECOND GRANT: RE-PLAN THE KEEP ONCE THE PROCESS IS WARM.
  //
  // alloc() runs inside the FIRST forward, when `cudaMemGetInfo` has not yet
  // seen JAX grow into the backward -- so it holds back
  // `np*4*depth/2 + totalb/8`, which is 4.77 GB at N=384, against growth it
  // cannot measure.  By the third step there is nothing left to guess: XLA's
  // BFC allocator has reached its high-water mark and never returns a byte,
  // so free memory IS what is spare.
  //
  // Called from the FFI at the top of a SAVE forward, which is the only safe
  // point: the kept buffers are written by the forward and read by the
  // backward's replay, so moving a depth between "kept" and "recomputed"
  // in between would have the replay read a slot the forward never wrote.
  //
  // It is a no-op on a card that already granted everything, which is every
  // length up to ~L=300 on a 24 GB A10 -- this exists for the top of the
  // range, where the blind margin costs the acts tier 42 of its 48 depths.
  int regrant_after = 2;      // steps to let XLA's arena settle
  int save_fwds = 0;
  bool regranted = false;
  void regrant() {
    // A CAPTURED GRAPH BAKES THESE POINTERS IN.  Moving a tier's storage
    // under a graph that already holds the old addresses is how the graph
    // path's existing divergence would get a sibling, so the regrant simply
    // does not run there.
    if (regranted || graph_on || envOn("NO_REGRANT")) return;
    // REGRANT_AFTER=0 makes it fire on the first call, which is how the
    // standalone gate reaches this path at all: pair it with KEEP_BUDGET_MB
    // to starve the blind grant and the re-plan is exercised on a card with
    // room to spare, rather than only at N=384 where it is hardest to debug.
    if (const char* a = getenv("REGRANT_AFTER")) regrant_after = atoi(a);
    if (++save_fwds <= regrant_after) return;
    regranted = true;
    bool anything = false;
    for (int t = 0; t < KEEP_NTIER; ++t)
      if (tstride[t] && !full(t)) anything = true;
    if (!anything) return;
    // THE SAVE-SET MUST EXIST BEFORE THE READING, or the margin below is
    // short by exactly it.  In the FFI it always does (XLA's buffer, or
    // useOwnSaveSet's); the standalone harness allocates it lazily inside
    // forward(), which is AFTER this.
    ensureSaveSet();
    const bool no_keep = envOn("NO_FLASH_KEEP");
    const bool want_acts = envOnDef("NE_KEEP_ACTS", true);
    const char* drop = getenv("NO_KEEP_TIER");
    if (no_keep) return;
    size_t freeb = 0, totalb = 0;
    CU(cudaMemGetInfo(&freeb, &totalb));
    // A SMALLER MARGIN, AND THE REASON IT IS DEFENSIBLE.  alloc()'s margin
    // covers (a) a save-set it has not allocated yet and (b) JAX growth it
    // cannot see.  Neither applies here: the save-set exists by now (it is
    // XLA's output buffer, or ours), and two complete steps have run at the
    // shapes every later step will use.  What is left is the ordinary
    // fragmentation reserve.
    size_t margin = totalb / 16;
    if (const char* kb = getenv("REGRANT_BUDGET_MB"))
      margin = freeb > (size_t)atoll(kb) * 1000000
                   ? freeb - (size_t)atoll(kb) * 1000000 : 0;
    // A RE-PLAN, NOT AN EXTENSION, and that is the whole value of it.  The
    // blind pass spends its budget cheapest-tier-first, so at N=384 it gave
    // 1.75 GB to `acts` and `trimul t` (9.3 and 8.5 ms/GB) and then refused
    // the flash tier, which is 22 ms/GB but may only be granted WHOLE.
    // Simply adding to what it chose cannot undo that.  Handing back the
    // partial grants first can: 1.75 GB returned plus 2.9 GB that was never
    // spoken for is 4.6, and the flash tier is 3.85.
    //
    // WHAT MAY NOT BE HANDED BACK: a tier that alloc() granted in FULL had
    // the module's own buffer freed under it (see above), so `use()` has no
    // fallback for it any more.  Those tiers are LOCKED at what they have.
    // In practice they are `stats` and `bias`, 0.05 GB between them.
    bool locked[KEEP_NTIER] = {};
    for (size_t k = 0; k < fslots.size(); ++k)
      if (!forig[k]) locked[ftier[k]] = true;
    size_t given_back = 0;
    for (int t = 0; t < KEEP_NTIER; ++t)
      if (!locked[t] && tbase[t]) {
        CU(cudaFree(tbase[t]));
        given_back += (size_t)tdepth[t] * tstride[t];
        tbase[t] = nullptr; tdepth[t] = 0;
      }
    if (given_back) CU(cudaMemGetInfo(&freeb, &totalb));
    size_t budget = freeb > margin ? freeb - margin : 0;
    size_t took = 0;
    for (int t = 0; t < KEEP_NTIER; ++t) {
      if (drop && atoi(drop) == t) continue;
      if ((t == KEEP_ACTS || t == KEEP_TMUL_T) && !want_acts) continue;
      if (t == KEEP_RELU && !envOn("NE_KEEP_RELU")) continue;
      if (!tstride[t] || locked[t] || full(t)) continue;
      int n = (int)std::min((size_t)depth, budget / tstride[t]);
      if (keepWholeOnly(t) && n < depth) n = 0;
      while (n > 0 && cudaMalloc(&tbase[t], (size_t)n * tstride[t]) != cudaSuccess) {
        cudaGetLastError();
        tbase[t] = nullptr; n /= 2;
        if (keepWholeOnly(t)) { n = 0; break; }
      }
      tdepth[t] = tbase[t] ? n : 0;
      tbytes[t] = (size_t)tdepth[t] * tstride[t];
      budget -= (size_t)tdepth[t] * tstride[t];
      took += (size_t)tdepth[t] * tstride[t];
    }
    for (int t = 0; t < KEEP_NTIER; ++t)
      if (tstride[t] && !locked[t])
        fprintf(stderr, "stack: regrant tier '%s' now %d of %d depths "
                "(%.2f GB free after handing %.2f GB back, %.2f GB held "
                "back)\n", keepName(t), tdepth[t], depth, freeb / 1e9,
                given_back / 1e9, margin / 1e9);
    // a tier the regrant completed makes its module buffer dead, exactly as a
    // first-pass full grant does
    for (size_t k = 0; k < fslots.size(); ++k)
      if (full(ftier[k]) && forig[k] && !shared_scratch) {
        CU(cudaFree(forig[k]));
        forig[k] = nullptr;
      }
    if (!took)
      fprintf(stderr, "stack: regrant found nothing it could take "
              "(%.2f GB free, %.2f GB held back)\n", freeb / 1e9, margin / 1e9);
  }

  // setWeights() allocates and converts a fresh set of device buffers each
  // call, so calling it once a depth and recording where it put them gives 48
  // independent sets that share one block's scratch.
  void setWeights(Oracle& o, const std::string& prefix = "param/block") {
    wsets.assign(depth, {});
    for (int i = 0; i < depth; ++i) {
      b.setWeights(o, prefix + std::to_string(i) + "/");
      wsets[i].resize(slots.size());
      for (size_t k = 0; k < slots.size(); ++k) wsets[i][k] = *slots[k];
    }
  }

  // SHARED SCRATCH MEANS SHARED cuBLAS HANDLES, and that is the one way the
  // sharing can bite.  The SHARE_PROBE gate read grad/pair 0.911 against a
  // control of 0.978 until the second stack was put on the same stream: its
  // setStream had pointed the handles both stacks now own at a stream the
  // first stack was not running on.  In the FFI every call gets XLA's stream
  // and every call calls this, so in practice there is one stream -- but
  // "in practice" is not a guarantee, so if the stream ever CHANGES under a
  // shared block, the previous one is drained first.  It costs one
  // synchronise on a transition that normally never happens.
  void setStream(cudaStream_t s) {
    static cudaStream_t last = nullptr;
    if (shared_scratch && last && last != s) CU(cudaStreamSynchronize(last));
    if (shared_scratch) last = s;
    b.setStream(s);
  }

  // Point the per-depth save-set at storage the CALLER owns, so a custom call
  // can hand back the remat inputs as a real output and take them back as a
  // real input.  That makes the forward and the backward independent calls --
  // no shared state, re-entrant under JAX's async dispatch -- and costs no
  // extra forward, which carrying only the inputs across would.
  // `single` points EVERY depth at the same one block's worth, for a forward
  // whose gradient will never be taken.  A design step runs two forwards and
  // one backward -- the first forward is AF2's detached recycle pass, so its
  // save-set is written and thrown away: 811 MB at N=256 depth 48, allocated
  // by XLA for a dead output.  A stack in this mode cannot be backward()ed,
  // which is the caller's business to know.
  // THE SAVE SET MUST BE OURS WHEN A GRAPH IS CAPTURED.  `pin[i]` and
  // `min_[i]` are slices of XLA's psave/msave, and those pointers MOVE: a
  // probe over twelve design iterations (NE_PTRS in ffi.cu) shows `msave`
  // changing on every single call and `pair` on every call but one.  A
  // captured graph bakes the address it saw, so from the third call the
  // forward writes its remat save-set somewhere XLA has since reused, and
  // the backward's replay recomputes from whatever is there now.  That is
  // the open divergence: finite, repeatable, and invisible to a single-shot
  // gate, exactly as observed (i_con 4.4977 against 2.3456 over 40
  // iterations).  The inputs were already staged; this is the one buffer set
  // that was not.
  //
  // Only allocated when the graph is on, because it is depth * N^2 * CZ * 4
  // -- 1.6 GB at N=256, 100 MB at N=64.  That is the right shape: the graph
  // is worth 10% at L=64 and 3% at L=256, so it costs least where it pays
  // most.
  __nv_bfloat16* osp = nullptr;
  float* osm = nullptr;
  void useOwnSaveSet() {
    // REUSE alloc()'s psave/msave rather than allocate a SECOND set.  osp/osm
    // used to be their own cudaMalloc, so a graph-captured run held the
    // save-set twice -- np*4*depth is 1.61 GB at N=256 and 3.62 at N=384.
    // alloc() already made exactly this buffer and, in this mode, nothing
    // else uses it.
    if (psave) { setSaveBuffers(psave, msave); return; }
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    if (!osp) {
      CU(cudaMalloc(&osp, (size_t)saveSlots() * np * 2));
      CU(cudaMalloc(&osm, (size_t)saveSlots() * nm * 4));
    }
    setSaveBuffers(osp, osm);
  }

  // the standalone harness has no one to hand it a save-set, so it makes its
  // own on first use.  The FFI never reaches this: it calls setSaveBuffers
  // (XLA's) or useOwnSaveSet() before the first forward.
  void ensureSaveSet() {
    if (pin[0] || psave) return;
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    CU(cudaMalloc(&psave, np * 2 * saveSlots()));
    CU(cudaMalloc(&msave, nm * 4 * saveSlots()));
    setSaveBuffers(psave, msave);
  }

  // STRIDE REMAT.  `NB_STRIDE=k` keeps only every k-th block's input and
  // rebuilds the rest of a segment, into a scratch of k-1 slots, just
  // before the backward walks that segment.  k = 1 is the historical
  // behaviour -- one slot per depth -- and is the default.
  //
  // The save-set is the biggest single allocation in the step (np*2*depth
  // is 3.35 GB at N=512, most of XLA's 4.69 GB largest_alloc_size), and it
  // is XLA that allocates it, so `saveSlots` is DUPLICATED on the Python
  // side in bindcraft/af/native.py.  The two must agree: a mismatch is not
  // a compile error, it is a buffer that is too small and a backward that
  // reads whatever is past the end of it.
  //
  // AND IT TURNS ITSELF ON, for the same reason the chunk policy does: a
  // knob nobody sets is a knob that does not run.  The stride costs 21% of
  // a step at 215 and buys nothing below the length where memory is the
  // binding constraint, so it is off (k = 1, bit-for-bit the old path)
  // up to `NB_STRIDE_N` and k = 4 above it.  `NB_STRIDE` forces a value
  // either way, which is how the gates run the segment logic at 215.
  //
  // THE POLICY READS N, AND N IS THE SAME NUMBER ON BOTH SIDES: the FFI
  // takes it from `pair.dimensions()[0]` and Python from `pair.shape[0]`.
  // Anything else here -- a length the caller passes, a padded figure one
  // side computes -- would be a save-set the two disagree about.
  // HOW MANY BLOCKS THE SAVE SET SKIPS, chosen from the CARD'S BUDGET rather
  // than from a length threshold tuned on one card.
  //
  // The old rule was `k = 4 above N=512`, with the 512 scaled per device by
  // native.py's _scaled().  That moves WHEN the stride turns on but not HOW
  // FAR it goes, so a bigger card running a proportionally longer sequence
  // still got k = 4 -- and complex 1087 on a 40 GB A100 needed k = 6, which
  // had to be set by hand.
  //
  // The quantity that actually matters is the save set AS A FRACTION OF THE
  // BUDGET: slots(k) * N^2 * CZ * 2, where slots(k) = ceil(depth/k) + k - 1.
  // Picking the smallest k that fits a fraction of the budget is both
  // per-card and per-length, which is what the length threshold was a proxy
  // for.
  //
  // THE BUDGET IS PASSED IN, NOT READ HERE, and that is load-bearing.
  // native.py's _save_slots DUPLICATES this function -- XLA allocates the
  // buffer from the shape Python declares and this code slices it -- and the
  // two must agree EXACTLY: if Python says fewer slots than this slices, the
  // backward reads past the end of somebody else's buffer, which is not a
  // compile error and not a crash, just a wrong gradient.  Both sides
  // reading cudaMemGetInfo would observe different free memory at different
  // moments and could disagree.  Python reads the device budget once and
  // exports it as NB_BUDGET_BYTES, so both compute from the same number.
  // MIRRORS native.py's _stride_k EXACTLY -- see the note there, and the
  // one at the export of NB_BUDGET_BYTES.  Only a SATURATED base escalates:
  // below the length threshold nothing changes, and at k = 4 the stride
  // tightens only if the save-set at k = 4 does not fit a ninth of the
  // budget.  k > 4 costs about 4% a step, so it is not paid where k = 4 fits.
  static int slotsAt(int depth_, int k) {
    return k <= 1 ? depth_ : (depth_ + k - 1) / k + k - 1;
  }
  static int strideForBudget(int depth_, int base, size_t slot, size_t budget) {
    if (base < 4) return base;
    if (!budget || !slot) return base;
    const size_t cap = budget / 9;
    if ((size_t)slotsAt(depth_, 4) * slot <= cap) return std::min(4, depth_);
    for (int k = 5; k <= 8 && k <= depth_; ++k)
      if ((size_t)slotsAt(depth_, k) * slot <= cap) return k;
    int best = 1;
    for (int k = 2; k <= 8 && k <= depth_; ++k)
      if (slotsAt(depth_, k) < slotsAt(depth_, best)) best = k;
    return best;
  }

  int strideK() const {
    static const char* e = getenv("NB_STRIDE");
    static const char* t = getenv("NB_STRIDE_N");
    static const char* b = getenv("NB_BUDGET_BYTES");
    int k;
    if (e) {
      k = atoi(e);                          // an explicit override still wins
    } else if (b) {
      const int base = N > (t ? atoi(t) : 512) ? 4 : 1;
      const size_t slot = (size_t)N * N * block::CZ * 2;
      k = strideForBudget(depth, base, slot, (size_t)atoll(b));
    } else {
      k = N > (t ? atoi(t) : 512) ? 4 : 1;  // no budget exported: as before
    }
    if (k < 1) k = 1;
    if (k > depth) k = depth;
    return k;
  }
  int saveSlots() const {
    const int k = strideK();
    return k <= 1 ? depth : (depth + k - 1) / k + k - 1;
  }
  // depth i's input lives in slot i/k when i is a checkpoint, and in the
  // shared scratch slot otherwise.  The forward writes every slot (the
  // scratch ones get overwritten segment after segment and nobody reads
  // them until the rebuild puts them back); only the checkpoints survive.
  int slotOf(int i) const {
    const int k = strideK();
    if (k <= 1) return i;
    const int r = i % k;
    return r == 0 ? i / k : (depth + k - 1) / k + r - 1;
  }

  void setSaveBuffers(__nv_bfloat16* p, float* m, bool single = false) {
    // HANDED SOMEONE ELSE'S SAVE-SET, OURS IS DEAD.  own_save follows
    // STACK_GRAPH_UNSAFE, which is off by default, so the in-model path runs
    // on XLA's buffers -- while alloc() had already allocated ours and
    // nothing ever reads it.  That is np*4*depth: 1.23 GB at N=224, 1.61 at
    // N=256, 3.62 at N=384, allocated and untouched for the whole run.
    if (!single && psave && p != psave) {
      CU(cudaFree(psave)); CU(cudaFree(msave));
      psave = nullptr; msave = nullptr;
    }
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    for (int i = 0; i < depth; ++i) {
      pin[i] = single ? p : p + (size_t)slotOf(i) * np;
      min_[i] = single ? m : m + (size_t)slotOf(i) * nm;
    }
  }

  // one block's worth, for the `single` mode above
  //
  // SIZED BY THE ALLOCATION'S LENGTH, not by the current one.  `N` is what
  // setN last re-derived to, and a design that predicts a 224-residue
  // binder_alone before a 608-residue complex would otherwise size this
  // slot at 224 and then let the complex's blocks write 95 MB into a 12 MB
  // buffer -- no error, somebody else's memory.  `nalloc` defaults to N for
  // the staged path, which has no setN and is always at its own length.
  __nv_bfloat16* p1save = nullptr;
  float* m1save = nullptr;
  void allocOneSave(int nalloc = 0) {
    if (p1save) return;
    const int NA = nalloc > N ? nalloc : N;
    size_t np = (size_t)NA * NA * block::CZ, nm = (size_t)S * NA * block::CM;
    CU(cudaMalloc(&p1save, np * 2));
    CU(cudaMalloc(&m1save, nm * 4));
  }

  // ...and the same for the outputs, for the same reason: in the fused path
  // pout/mout are XLA'S buffers, handed over per call by setTrunkCarry, and
  // a predict call that does not hand over new ones would write through the
  // pointers the last DESIGN call left.
  // ...and back again.  `single` mode leaves every depth pointing at one
  // buffer; a BACKWARD run against that walks a trail of one block.
  void restoreSaveSet() {
    if (!psave) return;                 // never had our own: nothing to undo
    setSaveBuffers(psave, msave);
    save_full = true;
  }

  void useOwnOut(int nalloc) {
    const int NA = nalloc > N ? nalloc : N;
    size_t np = (size_t)NA * NA * block::CZ, nm = (size_t)S * NA * block::CM;
    if (!pown) CU(cudaMalloc(&pown, np * 4));
    if (!mown) CU(cudaMalloc(&mown, nm * 4));
    pout = pown; mout = mown;
  }


  // Point the block at depth i's weights, and at whichever kept buffers this
  // depth was granted.  A slot whose tier did not reach depth i is RESTORED to
  // the buffer alloc() gave the module -- per-depth grants mean a pointer can
  // be a keep at depth 3 and scratch at depth 40, so leaving a stale slab
  // pointer in place would have the forward write depth 40's values into depth
  // 3's keep and the replay read them back.
  void use(int i) {
    b.drop_depth = (uint32_t)i;   // each depth gets its own masks
    for (size_t k = 0; k < slots.size(); ++k) *slots[k] = wsets[i][k];
    size_t off[KEEP_NTIER] = {};
    for (size_t k = 0; k < fslots.size(); ++k) {
      int t = ftier[k];
      if (have(t, i)) {
        *fslots[k] = tbase[t] + (size_t)i * tstride[t] + off[t];
      } else {
        *fslots[k] = forig[k];   // nulled above iff the tier covers every depth
      }
      off[t] += (fsize[k] + 255) / 256 * 256;
    }
  }

  // ---------------------------------------------------------------- graphs
  // PRICING A CUDA GRAPH.  The stack issues on the order of 36k kernels a
  // step and the host needs ~1.9 us to launch each, so roughly 68 ms of CPU
  // work -- 8% of the stack at N=224 and most of it at N=64.  That cost is
  // meant to overlap with execution, and whether any of it is actually on the
  // critical path cannot be read off this box (no nsys, no CUPTI).  So
  // capture the whole 48-deep stack once and replay it, and measure.
  //
  // The whole stack is capturable because every pointer is already fixed:
  // `use(i)` swaps weights HOST-side, so a capture bakes the right set per
  // depth, and the save buffers are preallocated.  Only the caller's input and
  // cotangent vary, and those are staged.
  cudaStream_t owned = nullptr;
  // THREE GRAPHS, NOT TWO: the forward runs in two shapes.  A detached
  // recycle pass (save=0) writes its remat set into one block's scratch; the
  // differentiated pass (save=1) writes 48 slices of the save set.  Same
  // kernels, different destination pointers, so they cannot share a captured
  // graph.  They did, and it was correct only by the accident of which call
  // captured first -- the save=1 pass happened to be the second forward, so
  // the save=0 pass replayed ITS graph and merely wrote 1.6 GB it did not
  // need.  Reverse the order and the save set would never be written at all.
  cudaGraphExec_t gexec[3] = {nullptr, nullptr, nullptr};
  // DO NOT TURN THIS ON FOR DESIGN.  It is correct in the standalone harness
  // and for a single forward+backward through the FFI, and it is WRONG over a
  // repeated design loop -- see the block comment on replay() for the two
  // failures and what is still unexplained. Kept because the standalone win
  // is real and measured, and because the next attempt should start from this
  // evidence rather than rediscover it.
  bool graph_on = envOn("STACK_GRAPH_UNSAFE");
  bool defer_keep = false;   // set before alloc(); see the first grant
  bool save_probe = envOn("SAVE_BF16");  // missed by the envOn sweep

  // CAPTURE ON THE SECOND CALL, NOT THE FIRST.  The obvious shape -- run it
  // once to warm the plans, then capture a second run -- executes the body
  // TWICE back to back, and the backward cannot survive that: its remat
  // replay is defined only against the forward that preceded it, so a second
  // backward with no forward between reads buffers the first one consumed.
  // The standalone gate missed this because it times forward and backward in
  // separate loops, so a stale pairing still produced finite numbers; through
  // the FFI it came out as a NaN gradient with a perfect forward.
  //
  // Instead the first call just runs, warming everything, and the capture
  // happens on the second -- by which time the caller's own alternation has
  // put a matching forward in front of it.
  int seen[3] = {0, 0, 0};
  bool save_full = true;   // the FFI sets this before each forward
  cudaEvent_t ein = nullptr, eout = nullptr;

  template <class F>
  void replay(int which, F&& issue) {
    if (!graph_on || seen[which]++ == 0) { issue(); return; }
    // TWO FAILURES, ONE FIXED, ONE NOT.
    //
    // (1) Capturing the CALLER's stream. JAX's stream is not ours to put into
    // capture mode: anything XLA touches on it inside the window is swept
    // into the graph and replayed every step. That surfaced as a NaN
    // gradient under a perfect forward. Fixed below by capturing on `owned`.
    //
    // (2) STILL OPEN. Even on a private stream, a 40-iteration design
    // diverges: i_con 4.4977 against the stream path's 2.3456, loss 7.2404
    // against 4.2824 -- finite, repeatable, and far outside the +-0.4 that
    // n=1 noise allows. A single forward+backward through the FFI
    // (test_pad.py) agrees to 1.697e-01 vs 1.690e-01, so whatever is wrong
    // needs the repeated call pattern to show itself. The obvious suspect -- a
    // module enqueueing onto a stream that is not `st`, which capture would
    // silently leave out of the graph -- was CHECKED AND CLEARED: every
    // launch without a stream argument is in setWeights, none in forward or
    // backward. What the design loop does that the single-shot gate does not
    // is two forwards a step (num_recycle=1) and a few hundred alternations.
    // Start there.
    //
    // CAPTURE ON A PRIVATE STREAM, NEVER THE CALLER'S.  The first version
    // captured JAX's stream directly, and that is not ours to put into
    // capture mode: anything XLA touches on it inside the window is swept
    // into the graph and then replayed every step forever. It passed the
    // single-shot FFI gate and produced NaN loss, pae and pLDDT a few
    // hundred iterations into a real design.
    //
    // So the graph lives on `owned` and is stitched to the caller's stream
    // with events: wait for what the caller has queued, run, and let the
    // caller wait for us. The capture then sees only our own work.
    if (!ein) { CU(cudaEventCreateWithFlags(&ein, cudaEventDisableTiming));
                CU(cudaEventCreateWithFlags(&eout, cudaEventDisableTiming)); }
    cudaStream_t caller = b.st;
    CU(cudaEventRecord(ein, caller));
    CU(cudaStreamWaitEvent(owned, ein, 0));
    if (!gexec[which]) {
      b.setStream(owned);                  // the modules' cublas handles too
      cudaGraph_t g;
      CU(cudaStreamBeginCapture(owned, cudaStreamCaptureModeThreadLocal));
      issue();
      CU(cudaStreamEndCapture(owned, &g));
      CU(cudaGraphInstantiate(&gexec[which], g, nullptr, nullptr, 0));
      CU(cudaGraphDestroy(g));
      b.setStream(caller);
    }
    CU(cudaGraphLaunch(gexec[which], owned));
    CU(cudaEventRecord(eout, owned));
    CU(cudaStreamWaitEvent(caller, eout, 0));
  }

  // A GRAPH BAKES ITS POINTERS, AND JAX'S BUFFERS MOVE.  The standalone gate
  // hands the same addresses every call, but the FFI does not, so everything
  // the caller supplies is copied into a staging buffer the capture owns. The
  // copies are 4 arrays a call against the ~36k launches they replace.
  cudaStream_t ostream = nullptr;     // PROBE_OVERLAP only
  cudaStream_t b_st_main = nullptr;
  float *sp = nullptr, *sm = nullptr, *spm = nullptr, *smm = nullptr;
  float *sctp = nullptr, *sctm = nullptr;

  void stage(const float* pair, const float* msa, const float* pair_mask,
             const float* msa_mask) {
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    if (!sp) {
      CU(cudaMalloc(&sp, np * 4));   CU(cudaMalloc(&sm, nm * 4));
      CU(cudaMalloc(&spm, (size_t)N * N * 4));
      CU(cudaMalloc(&smm, (size_t)S * N * 4));
    }
    CU(cudaMemcpyAsync(sp, pair, np * 4, cudaMemcpyDeviceToDevice, b.st));
    CU(cudaMemcpyAsync(sm, msa, nm * 4, cudaMemcpyDeviceToDevice, b.st));
    CU(cudaMemcpyAsync(spm, pair_mask, (size_t)N * N * 4,
                       cudaMemcpyDeviceToDevice, b.st));
    CU(cudaMemcpyAsync(smm, msa_mask, (size_t)S * N * 4,
                       cudaMemcpyDeviceToDevice, b.st));
  }

  void forward(const float* pair, const float* msa,
               const float* pair_mask, const float* msa_mask) {
    if (!graph_on) { forward_(pair, msa, pair_mask, msa_mask); return; }
    stage(pair, msa, pair_mask, msa_mask);
    replay(save_full ? 0 : 2, [&] { forward_(sp, sm, spm, smm); });
  }

  void backward(const float* ct_pair, const float* ct_msa,
                const float* pair_mask, const float* msa_mask) {
    if (!graph_on) { backward_(ct_pair, ct_msa, pair_mask, msa_mask); return; }
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    if (!sctp) { CU(cudaMalloc(&sctp, np * 4)); CU(cudaMalloc(&sctm, nm * 4)); }
    CU(cudaMemcpyAsync(sctp, ct_pair, np * 4, cudaMemcpyDeviceToDevice, b.st));
    CU(cudaMemcpyAsync(sctm, ct_msa, nm * 4, cudaMemcpyDeviceToDevice, b.st));
    replay(1, [&] { backward_(sctp, sctm, spm, smm); });
  }

  void forward_(const float* pair, const float* msa,
               const float* pair_mask, const float* msa_mask) {
    ensureSaveSet();
    if (!pout || !mout) useOwnOut();
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    const float *p = pair, *m = msa;
    // BLK_BF16CHAIN: from block 1 on, the previous block's output is ALREADY
    // bf16, so the narrowing below is a copy.  Only the stack's own input is
    // f32.
    const __nv_bfloat16* pb = nullptr;
    for (int i = 0; i < depth; ++i) {
      const __nv_bfloat16* p_in = nullptr;
      const float* m_in = nullptr;
      if (save_full || !BLK_BF16CHAIN || save_probe) {
        // narrow on save: this IS the approximation, and it is also what
        // makes the forward and the replay see the same values
        if (pb) CU(cudaMemcpyAsync(pin[i], pb, np * 2, cudaMemcpyDeviceToDevice, b.st));
        else toBf16K<<<(np + 255) / 256, 256, 0, b.st>>>(pin[i], p, np);
        CU(cudaMemcpyAsync(min_[i], m, nm * 4, cudaMemcpyDeviceToDevice, b.st));
        if (save_probe)
          roundInPlaceK<<<(nm + 255) / 256, 256, 0, b.st>>>(min_[i], nm);
        p_in = pin[i];
        m_in = min_[i];
      } else {
        if (!pb) {
          toBf16K<<<(np + 255) / 256, 256, 0, b.st>>>(b.p6b, p, np);
          pb = b.p6b;
        }
        p_in = pb;
        m_in = m;
      }
      use(i);
      // THE COMMENT HERE SAID keepGates=false AND THE ARGUMENT WAS haveActs.
      // Nine positional arguments reach `haveActs`, not `keepGates`, so the
      // tri-muls have been writing their gate sigmoids in this pass all
      // along.  That is not a bug in effect -- it is REQUIRED now, because
      // gl/gr are KEEP_ACTS and this forward is what fills the tier -- but
      // the saving the comment claims (0.482 -> 0.423 ms a tri-mul call) is
      // only available on depths the tier does NOT cover, so it is a
      // function of i and not a constant.
      //
      // `wantMask` is KEEP_RELU's pack, and it is gated twice: on the depth
      // being covered, and on this being the differentiated forward.
      b.forward(p_in, m_in, pair_mask, msa_mask, /*tail*/true,
                /*haveFlash*/false, /*haveStats*/false, /*haveBias*/false,
                /*haveActs*/false, /*haveT*/false, /*haveMsa*/false,
                /*haveRelu*/false,
                /*wantMask*/have(KEEP_RELU, i) && save_full,
                /*keepGates*/have(KEEP_ACTS, i) && save_full,
                /*keepFlash*/have(KEEP_FLASH, i) && save_full,
                /*haveOpmNz*/(i > 0), /*haveMaskT*/(i > 0));
      if (BLK_BF16CHAIN) pb = b.p6b; else p = b.p6;
      m = b.m3;
    }
    if (BLK_BF16CHAIN) widenBf16K<<<(np + 255) / 256, 256, 0, b.st>>>(pout, pb, np);
    else CU(cudaMemcpyAsync(pout, p, np * 4, cudaMemcpyDeviceToDevice, b.st));
    CU(cudaMemcpyAsync(mout, m, nm * 4, cudaMemcpyDeviceToDevice, b.st));
  }

  // STRIDE REMAT's second forward.  The backward is walking down and has
  // just reached the TOP of segment [s, i]; only `pin[s]` survived the
  // first forward, so run blocks s .. i-1 again and park each one's output
  // in the next block's scratch slot.  After this the segment looks exactly
  // as it did under k = 1 and the loop below needs no other change.
  //
  // This is a THIRD pass over those blocks -- forward, rebuild, replay --
  // so it is paid in time to be saved in memory.  It is bit-identical to
  // the first forward by construction: same weights, same input bytes
  // (pin[s] is the narrowed bf16 the first forward wrote), same
  // deterministic kernels.  `wantMask` is false because the KEEP_RELU pack
  // is still there from the differentiated forward, and nothing here
  // overwrites it.
  void rebuildSegment(int i, const float* pair_mask, const float* msa_mask) {
    const int k = strideK();
    if (k <= 1) return;
    const int s = (i / k) * k;
    if (i == s) return;                 // a checkpoint: its input was kept
    // only on the way IN to a segment, which is its highest index
    if (i != depth - 1 && (i % k) != k - 1) return;
    const size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    for (int j = s; j < i; ++j) {
      use(j);
      b.forward(pin[j], min_[j], pair_mask, msa_mask, /*tail*/true,
                /*haveFlash*/false, /*haveStats*/false, /*haveBias*/false,
                /*haveActs*/false, /*haveT*/false, /*haveMsa*/false,
                /*haveRelu*/false, /*wantMask*/false,
                /*keepGates*/false, /*keepFlash*/false,
                /*haveOpmNz*/(j > s), /*haveMaskT*/(j > s));
      if (BLK_BF16CHAIN)
        CU(cudaMemcpyAsync(pin[j + 1], b.p6b, np * 2,
                           cudaMemcpyDeviceToDevice, b.st));
      else
        toBf16K<<<(np + 255) / 256, 256, 0, b.st>>>(pin[j + 1], b.p6, np);
      CU(cudaMemcpyAsync(min_[j + 1], b.m3, nm * 4,
                         cudaMemcpyDeviceToDevice, b.st));
      if (save_probe)
        roundInPlaceK<<<(nm + 255) / 256, 256, 0, b.st>>>(min_[j + 1], nm);
    }
  }

  // Must follow forward() with the same inputs.  b.dp / b.dm end up holding the
  // cotangents on the stack's own pair and msa inputs.
  void backward_(const float* ct_pair, const float* ct_msa,
                const float* pair_mask, const float* msa_mask) {
    ensureSaveSet();
    b_st_main = b.st;
    size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
    // the caller's cotangent is f32 (it crosses the FFI that way); narrow it
    // once here, because the chain inside is bf16
    __nv_bfloat16* dp = b.dp;
    float* dm = b.dm;
    toBf16K<<<(np + 255) / 256, 256, 0, b.st>>>(dp, ct_pair, np);
    CU(cudaMemcpyAsync(dm, ct_msa, nm * 4, cudaMemcpyDeviceToDevice, b.st));
    // GRAD_TRACE prints ||dp|| and ||dm|| as the backward walks from block 47
    // back to block 0.  If the gradient decays, backprop through only the last
    // K blocks is nearly free accuracy-wise and cuts the backward AND its
    // replay by (depth-K)/depth -- the same bet v1 made on recycles, where
    // detaching all but the last pass was strictly better and 1.6x faster.
    const bool trace = envOn("GRAD_TRACE");
    auto nrm = [&](auto* d, size_t n) {
      std::vector<float> h = fetchAny(d, n);
      double a = 0; for (size_t j = 0; j < n; ++j) a += (double)h[j] * h[j];
      return sqrt(a / n);
    };
    if (trace) printf("  %5s %14s %14s\n", "block", "rms(dp)", "rms(dm)");
    for (int i = depth - 1; i >= 0; --i) {
      if (trace) {
        CU(cudaStreamSynchronize(b.st));
        printf("  %5d %14.6e %14.6e\n", i, nrm(dp, np), nrm(dm, nm));
      }
      rebuildSegment(i, pair_mask, msa_mask);
      use(i);
      // Rematerialise: this block's save-set is gone, only its input was kept.
      // `tail` false because the replay exists to rebuild what backward() reads
      // and the block's own outputs are not that -- the cotangents on them are
      // already in hand.  It skips a whole GEMM in each transition, the outer
      // product mean's output bmm, and both gate-applies.
      // NO_FLASH_KEEP (in alloc) makes the replay recompute the flash, which
      // is the A/B this was decided on (tuner pinned off so it is the only
      // variable): recomputed 1002.2 ms and grad/pair 1.045e-01, kept 968.7 ms
      // and 1.099e-01.  The forward is bit-identical either way; keeping costs
      // 5% of the gradient and 1.31 GB for 3.3% of the stack.
      // PER DEPTH, PER TIER: a tier granted only 20 of 48 depths lets the
      // first 20 replays skip its work and makes the other 28 recompute, so
      // the flags are a function of i and not of the allocation as a whole.
      // PROBE_OVERLAP: THE REPLAY DOES NOT DEPEND ON THE BACKWARD.  Depth
      // i-1's replay reads the save-set and the weights, neither of which
      // the backward touches, so in principle it can run concurrently with
      // depth i's backward -- and flashBwd, 33% of the backward, holds only
      // 3 blocks an SM, so there are idle SMs to run it on.  What stops it
      // is that both want the same block scratch.
      //
      // This issues the replay on a SECOND STREAM with no ordering against
      // the backward: the math is wrong (they race over one set of buffers)
      // and the instruction count and traffic are identical, so the delta is
      // the overlap headroom -- i.e. what a double-buffered block would be
      // worth before paying 1.5 GB at N=256 for it.
      //
      // **IT IS WORTH 3.1%: 604.42 -> 585.52 ms.**  And this is the UPPER
      // bound, with no events and no ordering at all; a correct version pays
      // for both.  So the GPU is already saturated through the backward --
      // flashBwd's 3 blocks an SM is a SHARED MEMORY limit, not idle SMs --
      // which is the same thing the CUDA-graph path says from the other
      // side (0.6%).  Not built: 1.5 GB at N=256 and 3.4 at N=384, and at
      // the lengths where that memory exists the keep tiers want it more.
      static const bool ovl = envOn("PROBE_OVERLAP");   // once, not per depth
      if (ovl) { if (!ostream) CU(cudaStreamCreate(&ostream)); b.setStream(ostream); }
      b.forward(pin[i], min_[i], pair_mask, msa_mask, /*tail*/false,
                /*haveFlash*/have(KEEP_FLASH, i),
                /*haveStats*/have(KEEP_STATS, i),
                /*haveBias*/have(KEEP_BIAS, i),
                /*haveActs*/have(KEEP_ACTS, i),
                /*haveT*/have(KEEP_TMUL_T, i),
                /*haveMsa*/have(KEEP_MSA, i),
                /*haveRelu*/have(KEEP_RELU, i), /*wantMask*/false,
                /*keepGates*/true, /*keepFlash*/true,
                /*haveOpmNz*/(i < depth - 1), /*haveMaskT*/(i < depth - 1));
      if (ovl) b.setStream(b_st_main);
      b.backward(dp, dm, pair_mask, msa_mask, /*haveReluB*/have(KEEP_RELU, i));
    }
  }
};

}  // namespace stack
