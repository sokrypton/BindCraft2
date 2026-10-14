// One AF2 Evoformer block in CUDA: the seven module types chained into the nine
// residual steps EvoformerIteration runs, forward and backward.
//
// THE ORDER is modules.EvoformerIteration with outer_product_mean.first=True,
// which is the multimer setting and so the binder-design setting:
//
//   p1 = p0 + OPM(m0)            <- the pair the MSA row attention then reads
//   m1 = m0 + MSARow(m0, p1)
//   m2 = m1 + MSACol(m1)
//   m3 = m2 + MSATransition(m2)  <- msa out
//   p2 = p1 + TriMulOutgoing(p1)
//   p3 = p2 + TriMulIncoming(p2)
//   p4 = p3 + TriAttnStarting(p3)
//   p5 = p4 + TriAttnEnding(p4)
//   p6 = p5 + PairTransition(p5) <- pair out
//
// WHAT THE BLOCK OWNS that a module does not: the residual, in both directions.
// A module writes its DELTA and reads the delta's cotangent; the block adds the
// input to the one and accumulates into the other.  That is the whole reason
// the interface is shaped that way -- it is what lets nine steps chain without
// any module knowing where it sits.
//
// THE SAVE-SET is the nine intermediate activations, because every module's
// backward recomputes its LayerNorm from its own input.  At N=256 that is
// ~100 MB of bf16 pair and a rounding error of msa, for one block.  A 48-block
// stack keeps one block's worth and recomputes the rest -- that is remat, and
// it is the stack's business, not this file's.
#pragma once
#include "opm.cuh"
#include "msarow.cuh"
#include "msacol.cuh"
#include "trimul.cuh"
#include "triattn.cuh"
#include "transition.cuh"

namespace block {

static const int CM = 256;  // msa channels
static const int CZ = 128;  // pair channels

#ifndef BLK_ONEPF
#define BLK_ONEPF 1
#endif
#ifndef BLK_SHARE_DSCRATCH
#define BLK_SHARE_DSCRATCH 1
#endif

struct Block {
  // Which half of the block to run.  The two tracks only meet at the outer
  // product mean and the MSA row attention's pair bias, so each can be gated
  // on its own against the oracle's --part msa / --part pair dumps -- which is
  // what localises an error to a track instead of to "the block".
  bool do_msa = true, do_pair = true;
  // AF2 EVOFORMER DROPOUT.  Off unless a caller sets it, because every gate
  // in this directory compares against a rate-0 oracle and a silently
  // stochastic stack would make all of them flap.  `drop_depth` is the
  // block's index in the stack: the mask must differ per depth, and a Block
  // is reused across depths by the stack's use().
  bool use_dropout = false;
  __nv_bfloat16* ddrop = nullptr;   // the masked cotangent, see alloc()
  float* mdrop = nullptr;           // the msa half of the same thing (f32: dm is)
  uint32_t* drop_seed = nullptr;    // ONE DEVICE WORD -- see common.cuh
  uint32_t drop_depth = 0;
  // the host side of it: a design loop calls this once a step, which is a
  // 4-byte copy rather than the recompile a changed FFI attribute would cost
  void setSeed(uint32_t v, bool enabled = true) {
    uint32_t w[2] = {v, enabled ? 1u : 0u};
    if (drop_seed)
      CU(cudaMemcpyAsync(drop_seed, w, 8, cudaMemcpyHostToDevice, st));
  }
  // the five rates AF2 actually uses; everything else is 0
  static constexpr float DROP_MSA_ROW = 0.15f;
  static constexpr float DROP_PAIR = 0.25f;
  // module ids, so two modules at the same depth get different masks
  enum { DM_MSA_ROW = 1, DM_TMO = 2, DM_TMI = 3, DM_TAS = 4, DM_TAE = 5 };
  // PAIR-ONLY, FOR THE EXTRA-MSA STACK. In design that stack's MSA side
  // cannot reach its pair -- its own msa output is discarded by the caller and
  // the only msa->pair path is the outer product mean, whose projections are
  // `extra_msa_mask * Linear(...)` and `extra_msa_mask` is all zeros. So the
  // four MSA modules are skipped and the OPM collapses to the per-channel
  // CONSTANT `bias / eps` (see addConstNarrowK). This also means the extra
  // stack needs no CM = 64 variant of opm/msarow/msacol and no kernel for
  // msa_column_GLOBAL_attention, which is a module this directory does not
  // have -- the pair track is dimensionally identical to the main stack's.
  //
  // It is a SPECIALISATION, not a general port: with a real extra alignment
  // the mask is not zero and this is wrong. install_hook gates it.
  bool pair_only = false;
  // false is the MONOMER (alphafold2_ptm) order: the outer product mean runs
  // at the END of the block.  backend.enable reads it off the model's config.
  bool opm_first = true;
  float* opm_k = nullptr;        // [CZ], per depth, swapped like a weight
  // How many steps of each track to run, as a PREFIX of the block's order.
  // Every module gates green on its own, so an error that only appears in a
  // chain belongs to a junction between two of them -- and the only way to say
  // which junction is to lengthen the chain one module at a time.
  int n_msa = 4, n_pair = 5;
  int N = 0, S = 0;
  cudaStream_t st = nullptr;
  opm::Opm opm_;
  mrow::MsaRow row;
  mcol::MsaCol col;
  trans::Trans mtr, ptr;
  tmul::TriMul tmo, tmi;
  tattn::TriAttn tas, tae;

  // The save-set, FLOAT32 because AF2's residual chain is float32 (see
  // common.cuh:addF32K).  `pb`/`mb` are the bf16 view a module is handed: its
  // LayerNorm sees the same 8-bit mantissa it always did, but the value its
  // delta is added back to is exact, which is where the port's error lived.
  // p0 IS bf16 NOW: it is the stack's save-set slice, and the save-set is
  // stored narrow.  Everything downstream of the first add stays f32.
  const __nv_bfloat16* p0 = nullptr;
  float *p1, *p2, *p3, *p4, *p5, *p6;
  float *m0 = nullptr, *m1, *m2, *m3;
  // the bf16 view of each intermediate, written by the add that produced it
  // (see common.cuh:addNarrowK).  p6/m3 need none: nothing reads the outputs.
  __nv_bfloat16 *p1b, *p2b, *p3b, *p4b, *p5b, *m0b, *m1b, *m2b;
  // ptm ONLY: the monomer config puts the outer product mean at the END of
  // the block, so the MSA row attention reads the block's INPUT pair and the
  // OPM reads the UPDATED msa.  Neither view exists in the multimer order.
  __nv_bfloat16 *p0b = nullptr, *m3b = nullptr;
  // BLK_BF16CHAIN: the block's pair OUTPUT, in the chain's own width.  The
  // stack's carry is already bf16 (the save-set is), so this retires the
  // narrowing the stack did on the way out as well.
  __nv_bfloat16* p6b = nullptr;
  float* opmNz = nullptr;
  // gradients: one accumulator a track, plus a scratch for a module's write
  // `dp` IS bf16 (see common.cuh:toF).  A probe said it costs nothing --
  // grad/pair 1.111/1.043/1.067e-01 against f32's 1.112/1.121/1.105 -- and it
  // halves the traffic at seven touch points a block AND retires the separate
  // narrowed copy the modules used to be handed, because the accumulator is
  // now already at the width they read.  `dm` stays f32: 0.46 MB, nothing to
  // win.  `sp` is gone -- every pair module accumulates in place.
  __nv_bfloat16* dp = nullptr;
  float *dm = nullptr, *sm = nullptr;

  // `donor` is another Block, already allocated at the same N, whose
  // S-INDEPENDENT scratch this one adopts instead of allocating its own: the
  // five pair modules and the pair-shaped buffers, which are 95% of the
  // bytes and depend only on N.  See the note on g_donor in stack.cuh for
  // why two stacks may share them.  Everything MSA-shaped is still this
  // block's own, because a module captures S at alloc time -- the first
  // attempt inflated S instead and the gate read grad/pair 0.12.
  void alloc(int N_, int S_, cudaStream_t s = nullptr,
             const Block* donor = nullptr) {
    N = N_; S = S_; st = s;
    size_t fb0 = 0, tb0 = 0;
    const bool mem = envOn("BLK_MEM");
    if (mem) { CU(cudaMemGetInfo(&fb0, &tb0)); fprintf(stderr, "BLK_MEM N=%d S=%d\n", N, S); }
    size_t np = (size_t)N * N * CZ, nm = (size_t)S * N * CM;
    auto f32 = [&](float** q, size_t n) { CU(cudaMalloc(q, n * 4)); };
    // ONE f32 PAIR BUFFER, NOT SIX.  p1..p6 are the forward's residual chain
    // and nothing else reads them: every backward takes the bf16 NARROW
    // (p1b..p5b), which is kept per step as before.  Each link is
    // `p_{i+1} = p_i + out`, elementwise, so aliasing them all onto one buffer
    // makes each add in-place and is safe -- a thread reads its own index
    // before writing it.  p0 is the block INPUT and is NOT part of this: it
    // belongs to the stack's save-set, so p1 = p0 + opm() still writes
    // somewhere else, which is the one aliasing constraint there is.
    // 5 x N^2 x CZ x 4 bytes = 168 MB at N=256, and it is N^2 like the rest.
    if (donor) {
      p1 = donor->p1; p2 = donor->p2; p3 = donor->p3;
      p4 = donor->p4; p5 = donor->p5; p6 = donor->p6;
      p1b = donor->p1b; p2b = donor->p2b; p3b = donor->p3b;
      p4b = donor->p4b; p5b = donor->p5b;
      dp = donor->dp;
    } else if (BLK_ONEPF) {
      f32(&p1, np); p2 = p3 = p4 = p5 = p6 = p1;
    } else {
      for (auto q : {&p1, &p2, &p3, &p4, &p5, &p6}) f32(q, np);
    }
    for (auto q : {&m1, &m2, &m3}) f32(q, nm);
    if (!donor)
      for (auto q : {&p1b, &p2b, &p3b, &p4b, &p5b}) CU(cudaMalloc(q, np * 2));
    for (auto q : {&m0b, &m1b, &m2b}) CU(cudaMalloc(q, nm * 2));
    if (!opm_first) {
      // the extra-MSA stack's constant-OPM specialisation lives in the
      // multimer branch only, and the monomer network has no extra stack to
      // run anyway -- so refuse the combination rather than silently skip it
      if (pair_only) {
        fprintf(stderr, "block: pair_only needs the multimer order\n");
        abort();
      }
      if (donor) p0b = donor->p0b; else CU(cudaMalloc(&p0b, np * 2));
      CU(cudaMalloc(&m3b, nm * 2));
    }
    if (!donor) CU(cudaMalloc(&p6b, np * 2)); else p6b = donor->p6b;
    // the OPM's per-(r,c) normaliser, built once a call instead of once an
    // element -- see opmNzK in common.cuh
    if (!donor) CU(cudaMalloc(&opmNz, (size_t)N * N * 4)); else opmNz = donor->opmNz;
    if (!donor) CU(cudaMalloc(&dp, np * 2));
    CU(cudaMalloc(&dm, nm * 4)); CU(cudaMalloc(&sm, nm * 4));
    // THE MASKED COTANGENT NEEDS ITS OWN BUFFER.  Forward is
    // out = in + mask * delta, so on the way back the DELTA's cotangent is
    // mask * dp while the residual's passthrough is dp unscaled -- the two
    // differ, and the aliased dp-as-both form cannot express that.  The
    // module still ACCUMULATES into dp, so the 57.6 ms the scratch-then-accK
    // form used to cost is not reintroduced; only the scaling is new.
    if (use_dropout) CU(cudaMalloc(&ddrop, np * 2));   // bf16, as dp is
    if (use_dropout) CU(cudaMalloc(&mdrop, nm * 4));
    if (use_dropout) { CU(cudaMalloc(&drop_seed, 8)); CU(cudaMemset(drop_seed, 0, 8)); }

    // BLK_MEM reports what each module's scratch costs.  The stack holds ONE
    // block's worth and remats the rest, so this table is the whole stack's
    // scratch, and it is what caps the length a design can run at: every
    // pair-shaped buffer grows as N^2.
    auto took = [&](const char* who) {
      if (!mem) return;
      size_t f, t; CU(cudaMemGetInfo(&f, &t));
      // signed -- "shared d-scratch (negative = freed)" means it, and
      // size_t subtraction turned a 757 MB saving into 18446744072952.5 MB
      fprintf(stderr, "  %-14s %7.1f MB\n", who,
              ((double)f - (double)fb0) / -1e6);
      fb0 = f;
    };
    took("block");
    // A PAIR-ONLY BLOCK DOES NOT ALLOCATE THE MSA MODULES. The outer product
    // mean's scratch alone is N^2-shaped, and with the extra-MSA stack there
    // are now TWO blocks live on the card -- the main 48-deep one and this
    // 4-deep one. Allocating four modules it will never call cost L=224 an
    // out-of-memory at stack.cuh:88 the first time both were installed.
    if (!pair_only) {
      opm_.alloc(N, S, st);                                 took("opm");
      row.alloc(N, S, st);                                  took("msa_row");
      col.alloc(N, S, st);                                  took("msa_col");
      mtr.alloc(S * N, CM, st);                             took("msa_trans");
    }
    if (donor) {
      // the five pair modules are N-only, so a memberwise copy IS "use the
      // same buffers"; their weight pointers are overwritten per depth by
      // the stack's own use(), and keepSlots()/weightSlots() hand out
      // addresses of THIS copy's fields.
      tmo = donor->tmo; tmi = donor->tmi;
      tas = donor->tas; tae = donor->tae; ptr = donor->ptr;
      took("adopted pair modules");
    } else {
    // An Evoformer block applies the output gate inside its own residual add
    // (`fuseGate` below), so trimul's gated `out` is dead weight here -- 105 MB
    // at N=640 for the pair.  NO_FUSE_OUTGATE turns that off, and then it is
    // not; the decision has to be made HERE because the buffer is allocated at
    // alloc() time now, not on first use.  See trimul_body.cuh.
    tmo.need_out = tmi.need_out = envOn("NO_FUSE_OUTGATE");
    tmo.alloc(N, true);                                     took("trimul_out");
    tmi.alloc(N, false);                                    took("trimul_in");
    tas.alloc(N, false); tas.allocOrientation();            took("triattn_start");
    tae.alloc(N, true);  tae.allocOrientation();            took("triattn_end");
    ptr.is_pair = true;                       // KEEP_RELU, see transition.cuh
    ptr.alloc(N * N, CZ, st);                               took("pair_trans");
    }
    // THE PAIRS SHARE THEIR BACKWARD SCRATCH.  backward() runs
    // d:pair_trans, d:triattn_end, d:triattn_start, d:trimul_in, d:trimul_out
    // in that order on one stream, so the second instance of each pair is
    // finished with its d* buffers before the first instance starts.  Only
    // the BACKWARD ones: the forward intermediates these modules keep are
    // written for all nine before any backward reads one, and the keep tiers
    // are built from keepSlots() below, which this does not touch.
    // ...and a donated block has already had this done to it, so doing it
    // again would free the donor's buffers a second time.
    if (BLK_SHARE_DSCRATCH && !donor) {
      tmi.adoptBackwardScratch(tmo);
      tae.adoptBackwardScratch(tas);
      took("shared d-scratch (negative = freed)");
    }
    // PUT EVERY MODULE ON ONE STREAM, here rather than only in the FFI path.
    // trimul::alloc and triattn::alloc each create a stream of their own (their
    // standalone gates need one), so a Block that never called setStream ran
    // the four pair-attention modules unordered against its own residual adds
    // and against each other.  It gave right answers only because CUDA's
    // legacy default stream implicitly synchronises with blocking streams --
    // accidental, and it hid a real ordering hazard: fusing the gradient
    // accumulation into the modules read 8.8e-01 in the gate and 2.6e-01 under
    // one stream, correct code failing a harness that tested a configuration
    // nothing ships.  This is the second time private streams have cost a day;
    // the first was the cuBLAS handles, which setStream also fixes.
    setStream(st);
  }

  // Everything a stack can KEEP per depth so the replay need not recompute it,
  // each slot with its own size.  Ordered, and the order is the only contract:
  // the stack lays them out back to back a depth and swaps the pointers in
  // use().
  std::vector<KeepSlot> keepSlots() {
    std::vector<KeepSlot> v;
    for (auto w : {tas.keepSlots(), tae.keepSlots(),
                   tmo.keepSlots(), tmi.keepSlots(), ptr.keepSlots()})
      v.insert(v.end(), w.begin(), w.end());
    // KEEP_MSA, and only where there IS an msa track: a pair_only block never
    // allocated these modules, so asking them for slots would hand the stack
    // the addresses of buffers that do not exist.
    if (!pair_only) {
      size_t nm = (size_t)S * N * CM;
      for (auto w : {row.keepSlots(), col.keepSlots(), mtr.keepSlots()})
        v.insert(v.end(), w.begin(), w.end());
      // the two chain LINKS the skipped adds would have written.  m0b is not
      // here: it is narrowed from the block's own msa input, which the replay
      // is handed.  m3b is the monomer order's and the opm reads m0b in the
      // multimer one.
      v.push_back({(void**)&m1b, nm * 2, KEEP_MSA});
      v.push_back({(void**)&m2b, nm * 2, KEEP_MSA});
    }
    return v;
  }

  // Every module's weight pointers, in a fixed order.
  // One link of the residual chain, in bytes -- what a narrower block has to
  // fit inside to borrow it.  See tplblock::adoptChainFrom.
  size_t chainBytes() const { return (size_t)N * N * CZ * 2; }

  std::vector<void**> weightSlots() {
    std::vector<void**> v;
    if (pair_only) v.push_back((void**)&opm_k);
    auto app = [&](std::vector<void**> w) { v.insert(v.end(), w.begin(), w.end()); };
    app(opm_.weightSlots()); app(row.weightSlots()); app(col.weightSlots());
    app(mtr.weightSlots()); app(tmo.weightSlots()); app(tmi.weightSlots());
    app(tas.weightSlots()); app(tae.weightSlots()); app(ptr.weightSlots());
    return v;
  }

  // Every module captured a stream in alloc(), so changing the block's own is
  // not enough -- its kernels would keep launching on the one they were built
  // with.  An XLA custom call is handed a different stream than the default,
  // and a block half on each is a race that reads as garbage output.
  void setStream(cudaStream_t s) {
    st = s;
    ptr.st = s; tmo.st = s; tmi.st = s; tas.st = s; tae.st = s;
    // the MSA modules exist only when alloc() built them; a pair-only block
    // has a NULL cublas handle there, and cublasSetStream on it is an error
    if (!pair_only) { opm_.st = s; row.st = s; col.st = s; mtr.st = s; }
    // AND the cuBLAS handles.  A handle carries its own stream, so the modules
    // that use cublas<...>StridedBatchedEx rather than cuBLASLt (which takes a
    // stream per call) would keep issuing those GEMMs on the DEFAULT stream
    // while every kernel around them ran on XLA's.  That is a race, and it
    // read as a garbage pair output with a correct msa output -- which the
    // standalone gates can never catch, because they run everything on the
    // default stream and so are accidentally consistent.
    if (!pair_only) CUB(cublasSetStream(opm_.cub, s));
    CUB(cublasSetStream(tmo.cub, s));
    CUB(cublasSetStream(tmi.cub, s));
    CUB(cublasSetStream(tas.cub, s));
    CUB(cublasSetStream(tae.cub, s));
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    // only the modules the prefix actually runs: a ladder dump holds no
    // weights for the ones it stops short of
    int j = do_msa ? n_msa : 0, k = do_pair ? n_pair : 0;
    if (pair_only) {
      // a FRESH allocation a call, like every module's setWeights, because the
      // stack snapshots the pointer once a depth and swaps it back in use()
      auto& ob = o[P + "outer_product_mean/output_b"];
      ob.to_device();
      CU(cudaMalloc(&opm_k, CZ * 4));
      opmConstK<<<1, CZ, 0, st>>>(ob.dev, opm_k, opm::OPM_EPS, CZ);
    }
    if (j > 0) opm_.setWeights(o, P + "outer_product_mean/");
    if (j > 1) row.setWeights(o, P + "msa_row_attention_with_pair_bias/");
    if (j > 2) col.setWeights(o, P + "msa_column_attention/");
    if (j > 3) mtr.setWeights(o, P + "msa_transition/");
    if (k > 0) tmo.setWeights(o, P + "triangle_multiplication_outgoing/");
    if (k > 1) tmi.setWeights(o, P + "triangle_multiplication_incoming/");
    if (k > 2) tas.setWeights(o, P + "triangle_attention_starting_node/");
    if (k > 3) tae.setWeights(o, P + "triangle_attention_ending_node/");
    if (k > 4) ptr.setWeights(o, P + "pair_transition/");
  }

  // `pair` and `msa` are the block's inputs; the outputs are p6 and m3.
  // `tail` false is the REPLAY form: p6 and m3 -- the block's two outputs --
  // are not written, and the two transitions stop before the GEMM that would
  // produce them.  The stack's backward replays each block to rebuild the state
  // the module backwards read, and it already holds the cotangents on the
  // outputs, so those are the one thing the replay can skip.
  //
  // IT IS ONLY THE OUTPUTS.  Every other module's `out` is added to the running
  // activation to make the next intermediate -- p1..p5, m1, m2 -- and those are
  // precisely what the module backwards read.  Skipping them leaves the
  // backward reading stale buffers; the stack gate reports grad/pair rms 0.34
  // against the oracle's 10.8, with the forward still perfect.
  void forward(const __nv_bfloat16* pair, const float* msa,
               const float* pair_mask, const float* msa_mask, bool tail = true,
               bool haveFlash = false, bool haveStats = false,
               bool haveBias = false, bool haveActs = false, bool haveT = false,
               // KEEP_MSA: the three msa modules' backwards have everything
               // they read, and m1b/m2b are held, so the replay can skip the
               // whole msa track.  The OPM is NOT skipped -- it feeds p1,
               // which is f32 and aliased (see KEEP_MSA in common.cuh).
               bool haveMsa = false,
               // KEEP_RELU: the pair transition's relu sign is held, so the
               // replay can skip it outright.  `wantMask` is the SAVE pass --
               // the only forward whose mask anyone will read.
               bool haveRelu = false, bool wantMask = false,
               // the tri-muls write their two gate sigmoids only when a
               // backward will read them -- the stack's own forward does not
               // need them and passes false (0.423 against 0.482 ms a call)
               bool keepGates = true) {
    p0 = pair;
    m0 = const_cast<float*>(msa);
    // nz is a function of the mask alone, so one pass over N^2 serves every
    // depth and every channel.  65536 threads against the 8.4M that used to
    // each redo it.
    if (do_msa && n_msa > 0 && !pair_only)
      opmNzK<<<((size_t)N * N + 255) / 256, 256, 0, st>>>(msa_mask, opmNz, N, S);
    size_t np = (size_t)N * N * CZ, nm = (size_t)S * N * CM;
    int gp = (np + 255) / 256, gm = (nm + 255) / 256;

    auto copy = [&](float* d, const float* s, size_t n) {
      CU(cudaMemcpyAsync(d, s, n * 4, cudaMemcpyDeviceToDevice, st));
    };
    // m0 is the block's INPUT, so nothing produced its view -- it is the one
    // conversion left, and the msa is small.
    toBf16K<<<gm, 256, 0, st>>>(m0b, m0, nm);
    const bool fuseGate = !envOn("NO_FUSE_OUTGATE");
#define STEP(NAME, ...) do { if (!prof().at(1)) { __VA_ARGS__; } else {        \
                               int _s = prof().slot(NAME); prof().begin(_s, st); \
                               __VA_ARGS__; prof().end(_s, st); } } while (0)

// NAN_TRACE names the first backward step whose output is not finite.  A
// diagnostic: it syncs the stream and counts, so it is never on a measured
// path.  `which` is just the step name.
#define NAN_CHECK(NAME)                                                        \
  do { if (nanTrace()) {                                                       \
         static unsigned* cnt = nullptr;                                       \
         if (!cnt) CU(cudaMalloc(&cnt, 8));                                    \
         unsigned h[2];                                                        \
         CU(cudaMemsetAsync(cnt, 0, 8, st));                                   \
         countBadBfK<<<(np + 255) / 256, 256, 0, st>>>(dp, cnt, np);           \
         countBadK<<<(nm + 255) / 256, 256, 0, st>>>(dm, cnt + 1, nm);         \
         CU(cudaStreamSynchronize(st));                                        \
         CU(cudaMemcpy(h, cnt, 8, cudaMemcpyDeviceToHost));                    \
         if (h[0] || h[1])                                                     \
           fprintf(stderr, "  NAN_TRACE after %-18s dp %u/%zu  dm %u/%zu\n",   \
                   NAME, h[0], np, h[1], nm);                                  \
       } } while (0)
    // THE MSA TRACK AND THE OUTER PRODUCT MEAN SWAP PLACES BETWEEN THE TWO
    // CONFIGS, and it is not just the OPM's position --
    // the dependencies differ too.
    //
    //   multimer (opm_first):  p1 = p0 + opm(m0);  m1 = m0 + row(m0, p1)
    //   ptm      (!opm_first): m1 = m0 + row(m0, p0); ...; p1 = p0 + opm(m3)
    //
    // so under ptm the row attention reads the block's INPUT pair and the OPM
    // reads the msa AFTER its transition.  `embeddings_and_evoformer.evoformer
    // .outer_product_mean.first` is the only field that differs between the
    // monomer and multimer presets -- checked, all nine others match.
    if (!opm_first) {
      p0b = const_cast<__nv_bfloat16*>(p0);   // already narrow; no pass needed
      if (do_msa && n_msa > 1) {
        STEP("msa_row", row.forward(m0b, p0b, msa_mask);
                        if (use_dropout)
                          addDropNarrowK<true><<<gm, 256, 0, st>>>(
                              m1, m1b, m0, row.out, N, CM, drop_seed,
                              drop_depth, DM_MSA_ROW, DROP_MSA_ROW, nm);
                        else
                          addNarrowK<<<gm, 256, 0, st>>>(m1, m1b, m0, row.out, nm));
      } else { copyNarrowK<<<gm, 256, 0, st>>>(m1, m1b, m0, nm); }
      if (do_msa && n_msa > 2) {
        STEP("msa_col", col.forward(m1b, msa_mask);
                        addNarrowK<<<gm, 256, 0, st>>>(m2, m2b, m1, col.out, nm));
      } else { copyNarrowK<<<gm, 256, 0, st>>>(m2, m2b, m1, nm); }
      // ALWAYS produce m3 AND its bf16 view: the OPM below reads it, so the
      // `tail` shortcut the multimer order can take is not available here.
      if (do_msa && n_msa > 3) {
        STEP("msa_trans", mtr.forward(m2b, /*tail*/true);
                          addNarrowK<<<gm, 256, 0, st>>>(m3, m3b, m2, mtr.out, nm));
      } else { copyNarrowK<<<gm, 256, 0, st>>>(m3, m3b, m2, nm); }
      if (do_msa && n_msa > 0) {
        STEP("opm", opm_.forward(m3b, msa_mask, /*fusedOut*/true);
                    addOpmNarrowK<__nv_bfloat16, CZ><<<gp, 256, 0, st>>>(p1, p1b, p0,
                                                      opm_.out, opm_.ob, opmNz,
                                                      opm::OPM_EPS, np, BLK_BF16CHAIN));
      } else { copyNarrowK<<<gp, 256, 0, st>>>(p1, p1b, p0, np); }
    } else {
    if (do_msa && n_msa > 0) {
      STEP("opm", opm_.forward(m0b, msa_mask, /*fusedOut*/true);
                  addOpmNarrowK<__nv_bfloat16, CZ><<<gp, 256, 0, st>>>(p1, p1b, p0,
                                                    opm_.out, opm_.ob, opmNz,
                                                    opm::OPM_EPS, np, BLK_BF16CHAIN));
    } else if (pair_only) {
      // the OPM's whole contribution, as a constant. Dropping it instead
      // would silently change the model by rms 5.04 a block, four times.
      STEP("opm_const",
        addConstNarrowK<<<gp, 256, 0, st>>>(p1, p1b, p0, opm_k, CZ, np));
    } else {
      copyNarrowK<<<gp, 256, 0, st>>>(p1, p1b, p0, np);
    }
    if (do_msa && n_msa > 1 && !haveMsa) {
      STEP("msa_row", row.forward(m0b, p1b, msa_mask);
                      if (use_dropout)
                        addDropNarrowK<true><<<gm, 256, 0, st>>>(
                            m1, m1b, m0, row.out, N, CM, drop_seed, drop_depth,
                            DM_MSA_ROW, DROP_MSA_ROW, nm);
                      else
                        addNarrowK<<<gm, 256, 0, st>>>(m1, m1b, m0, row.out, nm));
    } else if (!haveMsa) { copyNarrowK<<<gm, 256, 0, st>>>(m1, m1b, m0, nm); }
    if (do_msa && n_msa > 2 && !haveMsa) {
      STEP("msa_col", col.forward(m1b, msa_mask);
                      addNarrowK<<<gm, 256, 0, st>>>(m2, m2b, m1, col.out, nm));
    } else if (!haveMsa) { copyNarrowK<<<gm, 256, 0, st>>>(m2, m2b, m1, nm); }
    if (do_msa && n_msa > 3 && !haveMsa) {
      STEP("msa_trans", mtr.forward(m2b, tail);
                        if (tail) addF32K<<<gm, 256, 0, st>>>(m3, m2, mtr.out, nm));
    } else if (!haveMsa) { copy(m3, m2, nm); }
    }
    int k = do_pair ? n_pair : 0;
    if (k > 0) { STEP("trimul_out", tmo.forward(p1b, pair_mask, haveStats, fuseGate, keepGates, haveActs, haveT);
                               if (fuseGate && BLK_BF16CHAIN) PROF_STEP("blk residual", addGateNarrowBK<<<gp, 256, 0, st>>>(p2b, p1b, tmo.o, tmo.t2, np));
                               else if (fuseGate) PROF_STEP("blk residual", addGateNarrowK<<<gp, 256, 0, st>>>(p2, p2b, p1, tmo.o, tmo.t2, np));
                               else if (BLK_BF16CHAIN) { if (use_dropout) addDropNarrowBK<true><<<gp, 256, 0, st>>>(p2b, p1b, tmo.out, N, CZ, drop_seed, drop_depth, DM_TMO, DROP_PAIR, np); else addNarrowBK<<<gp, 256, 0, st>>>(p2b, p1b, tmo.out, np); }
                               else addNarrowK<<<gp, 256, 0, st>>>(p2, p2b, p1, tmo.out, np)); }
    else copyNarrowK<<<gp, 256, 0, st>>>(p2, p2b, p1, np);
    if (k > 1) { STEP("trimul_in", tmi.forward(p2b, pair_mask, haveStats, fuseGate, keepGates, haveActs, haveT);
                              if (fuseGate && BLK_BF16CHAIN) PROF_STEP("blk residual", addGateNarrowBK<<<gp, 256, 0, st>>>(p3b, p2b, tmi.o, tmi.t2, np));
                              else if (fuseGate) PROF_STEP("blk residual", addGateNarrowK<<<gp, 256, 0, st>>>(p3, p3b, p2, tmi.o, tmi.t2, np));
                              else if (BLK_BF16CHAIN) { if (use_dropout) addDropNarrowBK<true><<<gp, 256, 0, st>>>(p3b, p2b, tmi.out, N, CZ, drop_seed, drop_depth, DM_TMI, DROP_PAIR, np); else addNarrowBK<<<gp, 256, 0, st>>>(p3b, p2b, tmi.out, np); }
                              else addNarrowK<<<gp, 256, 0, st>>>(p3, p3b, p2, tmi.out, np)); }
    else copyNarrowK<<<gp, 256, 0, st>>>(p3, p3b, p2, np);
    if (k > 2) { STEP("triattn_start", tas.forward(p3b, pair_mask, haveFlash, haveBias); PROF_STEP("blk residual", if (BLK_BF16CHAIN) { if (use_dropout) addDropNarrowBK<true><<<gp, 256, 0, st>>>(p4b, p3b, tas.out, N, CZ, drop_seed, drop_depth, DM_TAS, DROP_PAIR, np); else addNarrowBK<<<gp, 256, 0, st>>>(p4b, p3b, tas.out, np); } else addNarrowK<<<gp, 256, 0, st>>>(p4, p4b, p3, tas.out, np))); }
    else copyNarrowK<<<gp, 256, 0, st>>>(p4, p4b, p3, np);
    if (k > 3) { STEP("triattn_end", tae.forward(p4b, pair_mask, haveFlash, haveBias); PROF_STEP("blk residual", if (BLK_BF16CHAIN) { if (use_dropout) addDropNarrowBK<false><<<gp, 256, 0, st>>>(p5b, p4b, tae.out, N, CZ, drop_seed, drop_depth, DM_TAE, DROP_PAIR, np); else addNarrowBK<<<gp, 256, 0, st>>>(p5b, p4b, tae.out, np); } else addNarrowK<<<gp, 256, 0, st>>>(p5, p5b, p4, tae.out, np))); }
    else copyNarrowK<<<gp, 256, 0, st>>>(p5, p5b, p4, np);
    if (k > 4 && !haveRelu) {
      STEP("pair_trans", ptr.forward(p5b, tail, wantMask);
                         if (tail && BLK_BF16CHAIN) addNarrowBK<<<gp, 256, 0, st>>>(p6b, p5b, ptr.out, np);
                         else if (tail) addF32K<<<gp, 256, 0, st>>>(p6, p5, ptr.out, np));
    }
    else if (BLK_BF16CHAIN) CU(cudaMemcpyAsync(p6b, p5b, np * 2, cudaMemcpyDeviceToDevice, st));
    else copy(p6, p5, np);
  }

  // `ct_pair`/`ct_msa` are the cotangents on p6/m3; `dp`/`dm` end up holding
  // the cotangents on p0/m0.  Must follow a forward() with the same inputs.
  // `haveReluB` says the relu sign came from KEEP_RELU rather than from a
  // replay that rebuilt `h`.
  void backward(const __nv_bfloat16* ct_pair, const float* ct_msa,
                const float* pair_mask, const float* msa_mask,
                bool haveReluB = false) {
    // NO_ACC is a PROBE: it drops every residual accumulation, so the gradient
    // is wrong.  It prices accK, which is 288 calls of 100 MB a stack.
    const bool noacc = envOn("NO_ACC");
    size_t np = (size_t)N * N * CZ, nm = (size_t)S * N * CM;
    int gp = (np + 255) / 256, gm = (nm + 255) / 256;
    CU(cudaMemcpyAsync(dp, ct_pair, np * 2, cudaMemcpyDeviceToDevice, st));
    CU(cudaMemcpyAsync(dm, ct_msa, nm * 4, cudaMemcpyDeviceToDevice, st));
    // NO CONVERSIONS HERE.  The replay that must precede this backward wrote
    // every bf16 view as a side effect of the add that produced it, so each
    // module is handed exactly the input its forward saw.

    // --- the pair track, in reverse.  dp arrives as the gradient on the step's
    // OUTPUT, which is also the gradient on its delta; the module's own write
    // is the gradient on its input, which the residual adds to what passed by.
    //
    // THE FIVE PAIR STEPS ADD THAT IN THEMSELVES, with dp passed as both the
    // incoming cotangent and the accumulator.  The scratch-then-accK form cost
    // an extra 33 MB write and 33 MB read a call -- 57.6 ms over the stack's
    // 288 calls, priced with NO_ACC -- and accK is already near bandwidth, so
    // the only way to get it back is not to do it.  Aliasing is safe because
    // each of these backwards reads its cotangent once, at the top, into a
    // bf16 copy; it is NOT safe for a module that reads dout again later.
    int k = do_pair ? n_pair : 0, j = do_msa ? n_msa : 0;
    if (k > 4) { NAN_CHECK("before d:pair_trans"); STEP("d:pair_trans", ptr.backward(dp, p5b, dp, /*acc*/!noacc, /*use_bits*/haveReluB)); }
    if (k > 3) { NAN_CHECK("before d:triattn_end"); STEP("d:triattn_end", (use_dropout ? (dropScaleBK<false><<<gp, 256, 0, st>>>(ddrop, dp, N, CZ, drop_seed, drop_depth, DM_TAE, DROP_PAIR, np), tae.backward(ddrop, p4b, pair_mask, dp, /*acc*/!noacc)) : tae.backward(dp, p4b, pair_mask, dp, /*acc*/!noacc))); }
    if (k > 2) { NAN_CHECK("before d:triattn_start"); STEP("d:triattn_start", (use_dropout ? (dropScaleBK<true><<<gp, 256, 0, st>>>(ddrop, dp, N, CZ, drop_seed, drop_depth, DM_TAS, DROP_PAIR, np), tas.backward(ddrop, p3b, pair_mask, dp, /*acc*/!noacc)) : tas.backward(dp, p3b, pair_mask, dp, /*acc*/!noacc))); }
    if (k > 1) { NAN_CHECK("before d:trimul_in"); STEP("d:trimul_in", (use_dropout ? (dropScaleBK<true><<<gp, 256, 0, st>>>(ddrop, dp, N, CZ, drop_seed, drop_depth, DM_TMI, DROP_PAIR, np), tmi.backward(ddrop, p2b, pair_mask, dp, /*acc*/!noacc)) : tmi.backward(dp, p2b, pair_mask, dp, /*acc*/!noacc))); }
    if (k > 0) { NAN_CHECK("before d:trimul_out"); STEP("d:trimul_out", (use_dropout ? (dropScaleBK<true><<<gp, 256, 0, st>>>(ddrop, dp, N, CZ, drop_seed, drop_depth, DM_TMO, DROP_PAIR, np), tmo.backward(ddrop, p1b, pair_mask, dp, /*acc*/!noacc)) : tmo.backward(dp, p1b, pair_mask, dp, /*acc*/!noacc))); }
    // dp is now the gradient on p1 -- which the MSA row attention also read, so
    // its pair gradient lands on top of this one rather than replacing it.

    // --- the MSA track, in reverse
    // ptm: the OPM produced p1, so its backward comes FIRST -- dp is the
    // gradient on p1 here and becomes the gradient on p0 straight after,
    // which is what the row attention's pair gradient then lands on.
    if (!opm_first && j > 0) {
      NAN_CHECK("before d:opm"); STEP("d:opm", opm_.backward(dp, m3b, msa_mask, sm);
                    if (!noacc) accK<<<gm, 256, 0, st>>>(dm, sm, nm));
    }
    if (j > 3) { NAN_CHECK("before d:msa_trans"); STEP("d:msa_trans", mtr.backward(dm, m2b, sm); if (!noacc) accK<<<gm, 256, 0, st>>>(dm, sm, nm)); }
    if (j > 2) { NAN_CHECK("before d:msa_col"); STEP("d:msa_col", col.backward(dm, m1b, msa_mask, sm); if (!noacc) accK<<<gm, 256, 0, st>>>(dm, sm, nm)); }
    if (j > 1) {
      // the PAIR half accumulates in place like the five pair steps do; the
      // msa half still goes through `sm`, because `dm` is also the input this
      // module reads and it is 500x smaller, so the round trip costs nothing
      NAN_CHECK("before d:msa_row"); STEP("d:msa_row",
           if (use_dropout)
             dropScaleK<true><<<gm, 256, 0, st>>>(mdrop, dm, N, CM, drop_seed,
                                                  drop_depth, DM_MSA_ROW,
                                                  DROP_MSA_ROW, nm);
           row.backward(use_dropout ? mdrop : dm, m0b,
                        opm_first ? p1b : p0b, msa_mask, sm, dp,
                        /*accPair*/!noacc);
           if (!noacc) accK<<<gm, 256, 0, st>>>(dm, sm, nm));
    }
    // --- and the outer product mean, which closes the loop back onto the msa
    if (opm_first && j > 0) {
      NAN_CHECK("before d:opm"); STEP("d:opm", opm_.backward(dp, m0b, msa_mask, sm);
                    if (!noacc) accK<<<gm, 256, 0, st>>>(dm, sm, nm));
    }
#undef STEP
    // dp is now the gradient on p0: the OPM's delta carried it no further.
  }
};

}  // namespace block
