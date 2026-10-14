// THE WHOLE DESIGN STEP AS ONE CALL.
//
// A sequence in, the nine arrays a loss reads out.  Everything between is
// native: the sequence features, the input features, the embedder, both
// stacks, the template embedding, the trunk, the structure module, the
// heads, the confidence arithmetic and the Kabsch alignment.  The loss
// stays in JAX, which is the point -- a user writes one in Python and
// never touches CUDA.  The contract is bindcraft/af/boundary.py.
//
// WHY ONE CALL AND NOT TEN.  An XLA custom call takes STATIC shapes.  Ten
// calls, each padded to 32 in JAX, leave the graph specialised to the
// padded length: the compile still recurs per bucket and the glue between
// them is still XLA's.  One call whose buffers are sized NMAX and whose
// LENGTH IS A RUNTIME INPUT collapses every bucket into one graph, which is
// the only part of this worth the work -- the compile saving alone is about
// 2.4 s, paid a handful of times per campaign (native/GLUE_PORT.md).
//
// So `n` below is read from device memory, never from a template parameter
// or an attribute.  A kernel that specialises on it gives the saving back.
#pragma once
#include <vector>
#include "common.cuh"
#include "seqfeat.cuh"
#include "inputfeat.cuh"
#include "confidence.cuh"
#include "kabsch.cuh"
#include "embed.cuh"
#include "stack.cuh"
#include "structmod.cuh"
#include "heads.cuh"
#include "tplembed.cuh"
#include "tpl1d.cuh"
#include "recycle.cuh"

namespace design_step {

constexpr int A = 20;        // amino acids
constexpr int ATOM = 37;     // atom37
constexpr int DBINS = 64;    // distogram bins

// Everything the step reads.  Buffers are [NMAX, ...]; only the first `n`
// rows are meaningful and the rest must be masked, not merely ignored --
// a padded row that reaches a reduction changes the answer.
struct Inputs {
  const float*   sequence;        // [NMAX, A]      designed logits
  const float*   atoms;           // [NMAX, 37, 3]  template coordinates
  const float*   atom_mask;       // [NMAX, 37]
  const uint8_t* flags;           // [NMAX]         DESIGN/TEMPLATE/...
  const int32_t* residue_index;   // [NMAX]
  const int32_t* asym_id;         // [NMAX]         chain
  const int32_t* entity_id;       // [NMAX]
  const int32_t* interface_asym;  // [NMAX]         for ipTM
  const float*   seq_mask;        // [NMAX]
  const float*   act_const;       // [NMAX, NMAX, 64] the hoisted template 8
  const int32_t* n;               // [] RUNTIME length, read on device
  const uint32_t* seed;           // [2] {seed, dropout_enabled}
  // THE FOUR DESIGN SCALARS, AS A DEVICE BUFFER [4].  They are arguments
  // to bc2's jitted step and a trajectory ramps them, so an attribute
  // would recompile the graph on every ramp step -- the same mistake as
  // passing the length as a shape.  See the note in seqfeat.cuh.
  const float* scalars;
};

// The nine, in the order bindcraft/af/boundary.py lists them.
struct Outputs {
  float* atoms;                   // [NMAX, 37, 3]
  bool*  atom_mask;               // [NMAX, 37]
  float* plddt;                   // [NMAX]
  float* pae;                     // [NMAX, NMAX]
  float* ptm;                     // []
  float* iptm;                    // []
  float* iptm_per_residue;        // [NMAX]
  float* expres_ca;               // [NMAX]
  float* distogram;               // [NMAX, NMAX, DBINS]
};

// Cotangents come back in the same shape; the step returns d/d sequence.
// act_const takes none -- it is constant by construction, which is why it
// is hoisted.  THE BACKWARD MUST REPLAY, not read what the forward left in
// this object: a design step runs forwards for every state and every
// recycle before any backward, and a backward that reads member state
// gates green and silently ruins a campaign (see tplembed.cuh).
struct Cotangents {
  const float* atoms;
  const float* plddt;
  const float* pae;
  const float* ptm;
  const float* iptm;
  const float* iptm_per_residue;
  const float* expres_ca;
  const float* distogram;
};

// ---- the orchestration -------------------------------------------------
// Every component already has the same shape -- alloc / setWeights /
// setStream / forward / backward -- so composing them is mechanical.  The
// order is the one the JAX takes:
//
//   seqfeat   logits -> sequence features and profile
//   inputfeat -> aatype, the gathers, the template arrays, msa_feat, offsets
//   embed     -> msa [S, N, 256], pair [N, N, 128]
//   tplembed  pair + act_const -> a pair contribution
//   stack     extra-MSA (pair only), then the 48-block trunk -> single, pair
//   structmod + heads -> atoms, distogram, pae, plddt, experimentally resolved
//   confidence -> plddt, pae, ptm, iptm, iptm_per_residue, expres_ca
//   kabsch    -> the atoms, aligned to the target template
//
// RECYCLING IS FORWARD-ONLY.  design_recycles defaults to 1 -- NOT 0, which
// is what a harness that sets num_recycle=0 will mislead you into assuming
// -- and recycled_alphafold_outputs wraps every pass but the last in
// stop_gradient.  So the recycle passes need no saved activations and no
// backward: only the FINAL pass does.  That is a large saving and it is
// easy to miss, because a loop that saved every pass would be correct and
// cost several times the memory.
//
// The backward runs the final pass's components in reverse, and each of
// them REPLAYS rather than reading what its forward left behind -- see the
// note on Cotangents above, and tplembed.cuh for what happens otherwise.
struct Plan {
  int nmax;            // buffer size; one graph serves every length <= this
  int recycles;        // forward-only passes before the differentiated one
  int trunk_depth;     // 48
  int extra_depth;     // 4
  int tpl_depth;       // 2
  int sm_depth;        // 8
  int msa_rows;        // S
  // DEFER THE KEEP TIERS, for a trace-time build.  Stack::alloc sizes them
  // from cudaMemGetInfo, and on an empty card they take everything -- the
  // staged prealloc hit exactly this and 4.51 GB went to a NMAX=128 step
  // that needs about 1.2.  regrant() hands them out later, once XLA has
  // stopped growing.
  bool defer_keep = false;
  // DOES THIS STEP EVER DROP?  Not "is dropout on now" -- that rides in
  // seed[1] and changes per call, because bc2 flips it between the stages
  // of one trajectory and a static flag would build a SECOND DesignStep
  // beside this one (two graphs is tolerable; two sets of N^2 buffers is
  // not).  This decides only whether the two Evoformer stacks RESERVE the
  // two mask scratches, which has to happen before alloc and therefore has
  // to be in the cache key.  The template and structure-module blocks
  // allocate theirs unconditionally, so they need nothing here.
  bool dropout = false;
};

// ffi.cu keeps its own bf16ToF32OutK next to the custom calls; a header
// cannot reach into it, so this carries its own -- the same arrangement
// tplembed.cuh already uses for the narrowing direction.
__global__ void widenK(float* __restrict__ dst,
                       const __nv_bfloat16* __restrict__ src, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

// THE EMBEDDER TAKES ITS INDEX FEATURES AS FLOATS.  The step carries them as
// int32 (that is what the gathers want), and pairInitK dereferences them as
// `const float*` -- so handing the int32 pointers straight over compiles,
// reads the same bytes as floats, and produces garbage relative positions
// with no error anywhere.  The three arrays are tiny; widen them.
__global__ void idxToFloatK(const int32_t* __restrict__ src,
                            float* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (float)src[i];
}

// TARGET_FEAT IS 21 WIDE, THE SEQUENCE FEATURES ARE 20.  modules_multimer
// pads with a zero column before the three Linears
// (`jnp.pad(batch["target_feat"], [[0,0],[0,1]])`), and the native embedder
// asserts NAA == 21.  Handing it the [N, 20] buffer compiles -- it is a
// `const float*` either way -- and reads every row one column off, which
// garbles the embedding without ever touching unmapped memory.
__global__ void reColsK(const float* __restrict__ src, int sc,
                        float* __restrict__ dst, int dc, int N) {
  const size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (size_t)N * dc) return;
  const int i = (int)(q / dc), c = (int)(q % dc);
  dst[q] = c < sc ? src[(size_t)i * sc + c] : 0.f;
}

// THE STRUCTURE MODULE EMITS ATOM14, THE BOUNDARY IS ATOM37.  The per-piece
// post-trunk call hands atom14 back and lets JAX scatter it; a single call
// has no JAX to scatter in.  AF2's all_atom.atom14_to_atom37 is a gather by
// residx_atom37_to_atom14 followed by a multiply by atom37_atom_exists --
// NOT a scatter by residx_atom14_to_atom37, which lands the same atoms in
// the same places only for the residues where both tables agree.
__global__ void atom14To37K(const float* __restrict__ a14,   // [NMAX,14,3]
                            const int32_t* __restrict__ a37_14,  // [NMAX,37]
                            const float* __restrict__ exists37, // [NMAX,37]
                            const int32_t* __restrict__ np,
                            float* __restrict__ out, int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * ATOM) return;
  const int i = (int)(p / ATOM);
  const float keep = (i < *np) ? exists37[p] : 0.f;
  const int k = a37_14[p];
  for (int c = 0; c < 3; ++c)
    out[p * 3 + c] = keep * a14[((size_t)i * 14 + k) * 3 + c];
}

// the gather's transpose: every atom37 slot sends its cotangent back to the
// atom14 slot it read, and several slots can read the same one
__global__ void atom37To14BwdK(const float* __restrict__ d37,  // [NMAX,37,3]
                               const int32_t* __restrict__ a37_14,
                               const float* __restrict__ exists37,
                               const int32_t* __restrict__ np,
                               float* __restrict__ d14, int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * ATOM) return;
  const int i = (int)(p / ATOM);
  if (i >= *np || exists37[p] == 0.f) return;
  const int k = a37_14[p];
  for (int c = 0; c < 3; ++c)
    atomicAdd(&d14[((size_t)i * 14 + k) * 3 + c], d37[p * 3 + c]);
}

// ---- DS_DUMP: the intermediates, for bisecting a composition error ----
// Every component here is gated against JAX in isolation and the chain
// still came out 5% wrong on the distogram.  That difference can only be
// localised by looking at what each stage HANDS THE NEXT, and the JAX side
// already exposes those three points through its own native hooks
// (NATIVE_EMBEDDER, NATIVE_EXTRA_MSA_STACK, NATIVE_EVOFORMER).
//
// Off unless DS_DUMP names a directory.  Synchronous and slow, deliberately.
// A CALL COUNTER, because the interesting question is whether two calls in
// ONE PROCESS agree: without it the second call overwrites the first and
// the comparison has nothing to compare.
inline int& dsDumpCall() { static int n = 0; return n; }

inline void dsDump(const char* name, const void* dev, size_t bytes) {
  const char* dir = getenv("DS_DUMP");
  if (!dir || !dev) return;
  std::vector<char> host(bytes);
  CU(cudaDeviceSynchronize());
  CU(cudaMemcpy(host.data(), dev, bytes, cudaMemcpyDeviceToHost));
  char path[1024];
  snprintf(path, sizeof(path), "%s/c%d_%s.f32", dir, dsDumpCall(), name);
  FILE* f = fopen(path, "wb");
  if (!f) { fprintf(stderr, "DS_DUMP: cannot write %s\n", path); return; }
  fwrite(host.data(), 1, bytes, f);
  fclose(f);
}

// seq_mask[i] * seq_mask[j], masked past the runtime length
__global__ void pairMaskK(const float* __restrict__ smask,
                          const int32_t* __restrict__ np,
                          float* __restrict__ out, int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * NMAX) return;
  const int i = (int)(p / NMAX), j = (int)(p % NMAX);
  const int n = *np;
  out[p] = (i < n && j < n) ? smask[i] * smask[j] : 0.f;
}

// ---- the step ----------------------------------------------------------
// DECLARATION ONLY.  The members and the shape of the call are settled;
// runOnce and backward have NO BODIES yet and alloc does not yet allocate
// the glue buffers.  This compiles, which proves nothing except that the
// components fit together as declared -- do not mistake it for a working
// step.  What is missing:
//
//   * alloc: the glue buffers listed below, sized from nmax
//   * setWeights: five components, each with its own directory layout
//   * runOnce: the chain in the order given above, threading `n` through
//   * backward: that chain reversed, each component REPLAYING
//
// The pieces it calls are all verified (glue_gate.py, 13/13); what is
// untested is them working together, and that can only be gated
// end-to-end against a real design step.
struct DesignStep {
  Plan plan{};
  cudaStream_t st = nullptr;

  embed::Embed emb;
  tplembed::TplEmbed tpl;
  stack::Stack extra;          // the 4-block extra-MSA stack, pair only
  stack::Stack trunk;          // the 48-block Evoformer
  structmod::StructureModule sm;
  heads::Heads hd;
  tpl1d::Tpl1D t1d;            // the template's MSA row
  recycle::Recycle rec;        // the three prev_* terms
  tpl1d::Tables t1d_tab{};

  // the glue's own buffers; all sized NMAX, all masked past `n`
  float *feat = nullptr, *prof = nullptr, *msa_feat = nullptr;
  float *a14e = nullptr, *a37e = nullptr, *pbeta = nullptr, *pbeta_m = nullptr;
  float *t_pos = nullptr, *t_mask = nullptr, *chain_len = nullptr;
  float *pae_err = nullptr, *tm_term = nullptr, *per_align = nullptr;
  float *d_plddt_logits = nullptr, *d_pae_logits = nullptr;
  float *d_expres_logits = nullptr, *d_pair = nullptr, *d_single = nullptr;
  float *d_sm_act = nullptr, *d_msa = nullptr, *d_msa_feat = nullptr;
  float *d_feat = nullptr, *d_seqfeat = nullptr, *d_proffeat = nullptr;
  float *d_tpl_query = nullptr, *pae_centres = nullptr, *tm_bins = nullptr;
  float *out_iptm_per_residue = nullptr;
  float *pair_mask = nullptr;  // [NMAX, NMAX]: the components want 2-D
  float *pair_all = nullptr;   // the pair WITH the template contribution
  // THE BACKWARD NEEDS THE FORWARD'S STATE, so a backward-only entry point
  // has to run the forward itself -- and the forward writes eight outputs
  // nobody downstream wants.  These are where they go.
  Outputs scratch{};
  float *tpl_out = nullptr;    // the template embedding's contribution
  float *kab = nullptr;        // [24] the Kabsch stats and rotation
  int32_t *aatype = nullptr, *t_aatype = nullptr, *offset = nullptr;
  // the index features widened for the embedder; sym_id is asym_id, which is
  // what bc2's alphafold_input_features sets it to unconditionally
  float *ridx_f = nullptr, *asym_f = nullptr, *ent_f = nullptr;
  // AND SO IS AATYPE.  atom14.cuh does `int a = (int)aatype[n]` on a
  // `const float*`: handing it the int32 buffer reinterprets the bits and
  // indexes the residue-constant tables out of bounds.
  float *aatype_f = nullptr;
  float *tf21 = nullptr, *d_tf21 = nullptr;   // target_feat, 20 -> 21 -> 20
  float *atom37 = nullptr;                   // the prediction, atom37
  // THE TRUNK'S MSA IS NOT THE EMBEDDER'S.  modules_multimer concatenates
  // the template row onto both the msa and its mask before the Evoformer,
  // so the trunk runs at msa_rows + 1 and these two buffers are what it
  // actually reads.  Running it on the embedder's msa alone is a different
  // model, and no per-component gate can see it: they all take an MSA that
  // JAX already assembled.
  float *trunk_msa = nullptr, *trunk_msa_mask = nullptr;
  float *d_atom37 = nullptr, *d_atom14 = nullptr;
  float *d_pair_sm = nullptr;   // the structure module's own pair cotangent
  // THE FRAME COTANGENTS ARE ZERO, BUT THEY ARE NOT NULL.  bc2 reads
  // positions, not frames, so nothing downstream differentiates the output
  // rotation or translation -- and foldstack.cuh:113 memcpys them
  // unconditionally, so passing nullptr is `cuda: invalid argument` rather
  // than the no-op it looks like.  ffi.cu's SmBwdImpl allocates and memsets
  // them for exactly this reason.
  float *d_rot = nullptr, *d_trans = nullptr;
  // THE TEMPLATE STACK'S REPLAY NEEDS SOMEWHERE TO PUT ITS INPUT.
  // tplembed::backward re-runs its own forward into `save` and then
  // tplstack::backward memcpys FROM it -- so nullptr, which the forward
  // accepts as "do not save", is `cuda: invalid argument` on the way back.
  __nv_bfloat16 *tpl_save = nullptr;
  int32_t *a14_37 = nullptr, *a37_14 = nullptr;
  inputfeat::Tables tables{};

  // RUN THIS OBJECT AT A SHORTER LENGTH.
  //
  // A step runs several STATES and they are not the same length -- a binder
  // alone is 220 where its complex is 607.  Giving each its own DesignStep
  // costs a second allocation (4.34 GB at N=256 beside 15.67 at N=640, which
  // is exactly what puts complex 617 per-state out of memory); padding them
  // all up to the biggest and sharing one costs the short state the long
  // state's runtime, 25.8 s against 14.2.
  //
  // Neither is needed.  alloc() already derives every N-dependent stride,
  // extent, chunk policy and carved offset in all 23 modules -- that is its
  // whole job -- and at a smaller n every buffer it would ask for is smaller
  // than the one it has.  So run it again, with nbrealloc handing each
  // pointer member back the buffer it already owns.  The object comes out
  // CONSISTENTLY at n: pair_all, the save sets and the stacks all stride by
  // n together, which is what keeps the step-level code (already written
  // against in.n) and the stacks (which run at nmax and mask) in agreement.
  //
  // Not free, and not for free: see the keep tiers below.
  void setN(int n) {
    if (n == plan.nmax) return;
    if (n > nmax_alloc) {
      fprintf(stderr, "design_step: setN(%d) past the allocation's %d\n",
              n, nmax_alloc);
      exit(1);
    }
    Plan p = plan;
    p.nmax = n;
    // The chunk policies follow the ALLOCATION's length from here on, so a
    // shorter step cannot ask for a bigger buffer than it has.  Set once and
    // left set: it is what every later re-derive must agree with.
    nbrealloc::allocN() = nmax_alloc;
    nbrealloc::Scope reuse;
    alloc(p, st);
  }

  // The length alloc() was FIRST called with -- the size of the buffers,
  // which setN never grows.
  int nmax_alloc = 0;
  size_t tpl_ledger_first = 0, tpl_ledger_last = 0;
  size_t evo_ledger_first = 0, evo_ledger_last = 0;

  void alloc(const Plan& p, cudaStream_t s) {
    if (!nmax_alloc) nmax_alloc = p.nmax;
    // DS_MEM itemises the STEP's allocation the way BLK_MEM itemises a
    // block's.  BLK_MEM covers the Evoformer modules and nothing else, which
    // at N=736 is 8.9 GB of a 20.6 GB step -- the other 11.7 was being
    // guessed at.  Same shape as block.cuh's took().
    size_t ds_fb = 0, ds_tb = 0;
    const bool ds_mem = getenv("DS_MEM") && !nbrealloc::active();
    if (ds_mem) CU(cudaMemGetInfo(&ds_fb, &ds_tb));
    auto dstook = [&](const char* who) {
      if (!ds_mem) return;
      size_t f, t; CU(cudaMemGetInfo(&f, &t));
      fprintf(stderr, "  ds %-16s %8.1f MB\n", who, (ds_fb - f) / 1e6);
      ds_fb = f;
    };
    plan = p; st = s;
    const int N = p.nmax;
    emb.alloc(N, p.msa_rows, s);  dstook("embed");
    // The template stack's buffers, by ledger position, so NB_TPL_SCRIBBLE
    // can prove its scratch is throwaway without enumerating 60-odd members
    // in a list that would drift.  See nbrealloc::scribble.
    tpl_ledger_first = nbrealloc::ledger().size();
    tpl.alloc(N, p.tpl_depth, s);  dstook("tplembed");
    tpl_ledger_last = nbrealloc::ledger().size();
    // THE EXTRA-MSA STACK IS NOT A SMALL TRUNK.  It uses
    // msa_column_GLOBAL_attention, and its outer-product-mean comes first.
    // Allocating it like the trunk aborts in the Oracle looking for
    // msa_column_attention/attention/query_w, which is the first thing the
    // end-to-end gate found.
    extra.b.pair_only = true;
    extra.b.opm_first = true;
    // AND do_msa FALSE.  pair_only alone decides which modules are
    // ALLOCATED; it does not stop setWeights looking the msa modules up,
    // and the Oracle aborts on msa_column_attention -- which the extra
    // stack does not have, because it uses msa_column_GLOBAL_attention.
    // ffi.cu's `get()` has carried this third line all along; the end-to-end
    // gate found it missing here twice, because the first fix copied only
    // the two lines the comment mentioned.
    extra.b.do_msa = false;
    extra.b.use_dropout = p.dropout;
    trunk.b.use_dropout = p.dropout;
    extra.defer_keep = p.defer_keep;
    trunk.defer_keep = p.defer_keep;
    evo_ledger_first = nbrealloc::ledger().size();
    extra.alloc(N, 1, p.extra_depth, s);  dstook("extra stack");     // pair-only: one msa row
    // THE SAME ORDERING FAILURE THE TRIMUL `out` BUFFER HAD, one stack up.
    //
    // Stack::alloc deliberately leaves the save set and pout/mout unallocated,
    // because in the STAGED path XLA owns both and allocating them would be
    // 3.62 GB thrown away (see stack.cuh).  The fused step hands the TRUNK
    // XLA's buffers through setTrunkCarry -- but nothing ever hands them to
    // `extra`, so its first forward fell into the lazy ensureSaveSet() /
    // useOwnOut() inside forward_().  By then prealloc has given the rest of
    // the card to XLA (17.89 GB taken, 4.44 left, XLA takes it) and a ~0.5 GB
    // request mid-step has nowhere to go: `stack.cuh:540 cuda: out of memory`
    // at complex 687.  Same bytes, asked for where there is room.
    extra.ensureSaveSet();  dstook("extra saveset");
    extra.useOwnOut();  dstook("extra out");
    trunk.alloc(N, p.msa_rows + 1, p.trunk_depth, s);   // + the template row
    dstook("trunk stack");
    evo_ledger_last = nbrealloc::ledger().size();

    // THE TEMPLATE PAIR STACK RUNS OUT OF THE EVOFORMER'S SCRATCH.
    //
    // 6.2 GB at N=736 of pair-module scratch that was live for the whole
    // step and used in a fraction of it.  The donation is declared buffer by
    // buffer in trimul_body.cuh's scratchBuffers(), not decided by an
    // allocator: the two instantiations of that file have the same member
    // list and the wide one's entries are never smaller, and adoptScratchFrom
    // checks each pair and aborts rather than truncating.
    //
    // `extra.b` is the donor because the trunk already aliases it -- one set
    // of pair scratch serves all three stacks.
    // ON BY DEFAULT since 2026-10-14.  |grad| at complex 487, 547, 587 and
    // 607 is unchanged bit for bit, and the ceiling goes 687 -> 747.
    // NB_TPL_SHARE=0 gives the template stack its own scratch back.
    if (envOnDef("NB_TPL_SHARE", true)) {
      tpl.adoptScratch(extra.b);
      dstook("tplembed share");
    }
    t1d.alloc(N, s);  dstook("tpl1d");
    // Zero recycles is what design runs, and then the two normed buffers
    // hold zeros forever -- 277 MB of them at N=736.  See recycle.cuh.
    rec.prev_zero = (p.recycles == 0);
    rec.alloc(N, s);  dstook("recycle");
    // BEFORE alloc().  do_atoms decides whether the last fold iteration
    // ALLOCATES the atom builder's output at all -- ffi.cu's sm_get sets it
    // before alloc for exactly this reason.  Setting it in setWeights
    // instead leaves sm.outAtom() null, and the first thing that reads it
    // is an illegal access with no symbol anywhere near the cause.
    sm.do_atoms = true;              // bc2 reads final_atom_positions
    sm.alloc(N, p.sm_depth, s);  dstook("structmod");
    hd.alloc(N, s);  dstook("heads");
    // Every buffer is sized from NMAX.  None of them is sized from the
    // runtime length: that is what lets one allocation serve every length.
    const size_t n1 = (size_t)N, nn = (size_t)N * N;
    CU(cudaMalloc(&feat, n1 * inputfeat::A * 4));
    CU(cudaMalloc(&prof, n1 * inputfeat::A * 4));
    CU(cudaMalloc(&msa_feat, n1 * inputfeat::MSA * 4));
    CU(cudaMalloc(&a14e, n1 * inputfeat::ATOM14 * 4));
    CU(cudaMalloc(&a37e, n1 * inputfeat::ATOM * 4));
    CU(cudaMalloc(&pbeta, n1 * 3 * 4));
    CU(cudaMalloc(&pbeta_m, n1 * 4));
    CU(cudaMalloc(&t_pos, n1 * inputfeat::ATOM * 3 * 4));
    CU(cudaMalloc(&t_mask, n1 * inputfeat::ATOM * 4));
    CU(cudaMalloc(&chain_len, n1 * 4));
    CU(cudaMalloc(&pae_err, nn * 4));
    CU(cudaMalloc(&tm_term, nn * 4));
    CU(cudaMalloc(&per_align, n1 * 4));
    CU(cudaMalloc(&kab, 24 * 4));
    CU(cudaMalloc(&pair_mask, nn * 4));
    CU(cudaMalloc(&tpl_out, nn * (size_t)embed::CZ * 4));
    CU(cudaMalloc(&pair_all, nn * (size_t)embed::CZ * 4));
    CU(cudaMalloc(&aatype, n1 * 4));
    CU(cudaMalloc(&t_aatype, n1 * 4));
    CU(cudaMalloc(&offset, nn * 4));
    CU(cudaMalloc(&a14_37, n1 * inputfeat::ATOM14 * 4));
    CU(cudaMalloc(&a37_14, n1 * inputfeat::ATOM * 4));
    CU(cudaMalloc(&ridx_f, n1 * 4));
    CU(cudaMalloc(&asym_f, n1 * 4));
    CU(cudaMalloc(&ent_f, n1 * 4));
    CU(cudaMalloc(&aatype_f, n1 * 4));
    CU(cudaMalloc(&tf21, n1 * embed::NAA * 4));
    CU(cudaMalloc(&d_tf21, n1 * embed::NAA * 4));
    CU(cudaMalloc(&atom37, n1 * ATOM * 3 * 4));
    CU(cudaMalloc(&trunk_msa, (size_t)(p.msa_rows + 1) * n1 * embed::CM * 4));
    CU(cudaMalloc(&trunk_msa_mask, (size_t)(p.msa_rows + 1) * n1 * 4));
    CU(cudaMalloc(&d_atom37, n1 * ATOM * 3 * 4));
    CU(cudaMalloc(&d_atom14, n1 * 14 * 3 * 4));
    // THE BACKWARD'S BUFFERS.  They were declared and never allocated, which
    // a forward-only gate cannot see: every one of these is a null pointer a
    // kernel writes through the first time a gradient is asked for.
    CU(cudaMalloc(&d_plddt_logits, n1 * heads::PL_BINS * 4));
    CU(cudaMalloc(&d_pae_logits, nn * heads::NPAE * 4));
    CU(cudaMalloc(&d_expres_logits, n1 * heads::NER * 4));
    CU(cudaMalloc(&d_pair, nn * (size_t)embed::CZ * 4));
    CU(cudaMalloc(&d_pair_sm, nn * (size_t)embed::CZ * 4));
    // AND `d_tpl_query` IS THAT SAME BUFFER.  Both are "a module's pair
    // cotangent, written to its own buffer and then added into d_pair"
    // -- the structure module's at the top of the backward, the template
    // embedder's at the bottom -- and the two are never live together:
    // d_pair_sm is consumed into d_pair some forty lines before
    // tpl.backward writes d_tpl_query, and neither is read again after its
    // accumulate.  N*N*CZ*4 is 277 MB at N=608 and 443 at 768.
    d_tpl_query = d_pair_sm;
    CU(cudaMalloc(&tpl_save, nn * (size_t)tplembed::CT * 2));
    CU(cudaMalloc(&d_rot, n1 * 9 * 4));
    CU(cudaMalloc(&d_trans, n1 * 3 * 4));
    CU(cudaMalloc(&d_single, n1 * heads::CS * 4));
    CU(cudaMalloc(&d_sm_act, n1 * structmod::C * 4));
    CU(cudaMalloc(&d_msa, (size_t)(p.msa_rows + 1) * n1 * embed::CM * 4));
    CU(cudaMalloc(&d_msa_feat, (size_t)p.msa_rows * n1 * inputfeat::MSA * 4));
    CU(cudaMalloc(&d_feat, n1 * inputfeat::A * 4));
    CU(cudaMalloc(&d_seqfeat, n1 * inputfeat::A * 4));
    CU(cudaMalloc(&d_proffeat, n1 * inputfeat::A * 4));
    CU(cudaMalloc(&out_iptm_per_residue, n1 * 4));
    CU(cudaMalloc(&scratch.atoms, n1 * ATOM * 3 * 4));
    CU(cudaMalloc(&scratch.plddt, n1 * 4));
    dstook("step buffers");
    CU(cudaMalloc(&scratch.pae, nn * 4));
    CU(cudaMalloc(&scratch.ptm, 4));
    CU(cudaMalloc(&scratch.iptm, 4));
    CU(cudaMalloc(&scratch.iptm_per_residue, n1 * 4));
    CU(cudaMalloc(&scratch.expres_ca, n1 * 4));
    CU(cudaMalloc(&scratch.distogram, nn * DBINS * 4));
    CU(cudaMalloc(&pae_centres, confidence::PBINS * 4));
    CU(cudaMalloc(&tm_bins, confidence::PBINS * 4));
  }

  // THE RECYCLE PASSES ARE FORWARD ONLY.  Every pass but the last is
  // stop_gradient'd upstream, so they need no save set -- which is where
  // most of the memory would otherwise go.
  void forward(const Inputs& in, const Outputs& out, bool want_save) {
    // the recycled state starts at ZERO, and zero is not neutral -- see the
    // note at the top of recycle.cuh
    rec.zeroPrev();
    for (int r = 0; r < plan.recycles; ++r) runOnce(in, out, /*save=*/false);
    runOnce(in, out, want_save);
  }

  // Each component keeps its own directory, exactly as the per-piece FFI
  // factories do -- one Oracle per directory, shared across components that
  // read the same one.
  // REBIND EVERY MEMBER'S STREAM.  This is what lets the whole step be
  // BUILT AT TRACE TIME, before XLA's stream exists -- without it `DsKey`
  // has to carry the stream, and then a trace-time build is a second copy
  // the real call never finds.  Nothing here is stream-ordered: every
  // buffer comes from plain cudaMalloc and the stream is used only for
  // launches and cublasSetStream.  See the prealloc note in ffi.cu.
  // CARRY THE TRUNK ACROSS THE fwd -> bwd BOUNDARY.
  //
  // The backward handler replays the WHOLE forward before doing anything,
  // because a design step runs every state's forward before any backward
  // and C++ member state is stale by then -- the bug that cost i_pTM
  // 0.79 -> 0.23.  Measured, that replay is the entire speed gap against
  // the staged path: complex 487 goes 8766.8 -> 7385.5 ms with it skipped
  // (the staged path is 7308.2), and 537 goes 12150.5 -> 10427.2 (10350.7).
  //
  // The fix is not to delete it but to make the expensive part
  // unnecessary.  The trunk is 48 of the 54 blocks, and everything after
  // `trunk.forward` needs only its pout/mout -- so when the save set and
  // those two outputs live in XLA BUFFERS rather than in members, they
  // survive the boundary as data, interleaving and all, and the replay can
  // skip the trunk while emb/tpl/extra/sm/hd (cheap) still run honestly.
  // That is the same arrangement the staged path has always had: its
  // `psave` is an ffi::Result the caller owns.
  bool skip_trunk = false;

  void setTrunkCarry(__nv_bfloat16* psave, float* msave, float* pout,
                     float* mout) {
    trunk.setSaveBuffers(psave, msave);
    trunk.pout = pout;
    trunk.mout = mout;
  }

  // THE SECOND KEEP GRANT, FOR THE FUSED STEP.
  //
  // `defer_keep` tells both Evoformer stacks to take NO keep tiers in their
  // first grant -- alloc() runs at trace time, before XLA has grown into the
  // backward, and a tier taken then is a tier taken from the thing that has
  // to fit.  regrant() is what hands them out once the arena has settled.
  //
  // It was never called.  regrant() lived at exactly one call site, the top
  // of the STACK handler's save forward, and the fused step does not go
  // through that handler -- it owns its stacks.  So the fused step ran with
  // zero kept activations for its whole life, and its backward recomputed
  // per block what the staged path rebuilds from avH: flash attention in
  // full instead of gateFromAvK, and a second projGateK for the trimul
  // gates.  Measured at complex 215 that is +104 flashGridHalf launches and
  // +208 projGateK<false>, which is most of the 181 ms a step the fused path
  // was behind the staged one -- the gap that had been blamed first on the
  // replay and then on the structure module, and was neither.
  //
  // Same safe point as the staged path, for the same reason: a save forward
  // is the one moment when no kept buffer is live, because the backward that
  // reads them has not run yet.
  void regrant() {
    extra.regrant();
    trunk.regrant();
  }

  void setStream(cudaStream_t s) {
    st = s;
    emb.setStream(s); tpl.setStream(s);
    extra.setStream(s); trunk.setStream(s);
    sm.setStream(s); hd.setStream(s);
    t1d.setStream(s); rec.setStream(s);
  }

  void setWeights(Oracle& trunk_w, Oracle& extra_w, Oracle& tpl_w,
                  Oracle& post_w, Oracle& emb_w, inputfeat::Tables t,
                  tpl1d::Tables t1) {
    emb.setWeights(emb_w);
    // the two template_*_1d Linears live under the embedder's scope
    t1d.setWeights(emb_w);
    rec.setWeights(emb_w);     // prev_* also live under the embedder's scope
    t1d_tab = t1;
    tpl.setWeights(tpl_w);
    extra.setWeights(extra_w);
    trunk.setWeights(trunk_w);
    sm.setWeights(post_w);
    hd.setWeights(post_w);
    tables = t;
  }

  void runOnce(const Inputs& in, const Outputs& out, bool save);
  void backward(const Inputs& in, const Cotangents& ct, float* d_sequence);
};

// ---- one pass, sequence to the nine ------------------------------------
inline void DesignStep::runOnce(const Inputs& in, const Outputs& out,
                                bool save) {
  const int N = plan.nmax;
  const int g1 = (N + 127) / 128;
  const int gn = (int)(((size_t)N * N + 127) / 128);
  const size_t nn = (size_t)N * N;
  const size_t nn_cz = nn * embed::CZ;

  // 0. DROPOUT, pointed at the caller's seed buffer.  The two Evoformer
  //    stacks take the device pointer directly (b.drop_seed, the same
  //    thing ffi.cu's stack handler does); the template embedding and the
  //    structure module take it through their own setDropout.  Whether a
  //    mask is actually drawn is seed[1] at RUNTIME -- dropScale returns 1
  //    when it is zero -- so a trajectory that turns dropout off between
  //    stages costs no rebuild and no second object.
  //
  //    EVERY PASS OF A RECYCLE DRAWS THE SAME MASKS, because the hash is
  //    (seed, depth, module, index) and nothing here varies per pass.  AF2
  //    splits a fresh key per recycle, so this is a real difference -- and
  //    an unexercised one: every gate here runs at recycles=0.  The
  //    backward's replay DEPENDS on the masks being reproducible, so
  //    whatever varies per pass must stay a function of the pass index and
  //    never of a counter.
  if (plan.dropout) {
    trunk.b.drop_seed = const_cast<uint32_t*>(in.seed);
    extra.b.drop_seed = const_cast<uint32_t*>(in.seed);
    tpl.setDropout(true, in.seed);
    sm.setDropout(true, in.seed);
  }

  // 1. the designed sequence -> features AlphaFold reads, and the profile
  seqfeat::Scalars sc{};
  seqfeat::seqFeatK<<<g1, 128, 0, st>>>(
      in.sequence, in.flags, /*bias*/nullptr, sc, seqfeat::OMITTED_HALF,
      feat, prof, N, in.scalars);

  // 2. aatype, the residue_constants gathers, the template arrays
  inputfeat::perResidueK<<<g1, 128, 0, st>>>(
      feat, in.atoms, in.atom_mask, in.flags, in.seq_mask, tables, in.n,
      aatype, a14e, a37e, a14_37, a37_14, t_aatype, t_pos, t_mask,
      pbeta, pbeta_m, N);
  inputfeat::chainLenK<<<N, 32, 0, st>>>(
      in.asym_id, in.seq_mask, in.n, chain_len, N);
  inputfeat::offsetK<<<gn, 128, 0, st>>>(
      in.residue_index, in.asym_id, in.flags, in.seq_mask, chain_len, in.n,
      offset, N);
  inputfeat::msaFeatK<<<(int)(((size_t)N * inputfeat::MSA + 127) / 128),
                        128, 0, st>>>(feat, prof, in.n, msa_feat, N);

  // 3. the embedder: target_feat and msa_feat -> msa and pair.
  //    The relative encoding reads the OFFSET MATRIX computed above, not
  //    residue_index -- see the note on pairInitK in embed.cuh.
  reColsK<<<(int)(((size_t)N * embed::NAA + 127) / 128), 128, 0, st>>>(
      feat, inputfeat::A, tf21, embed::NAA, N);
  //    AF2 casts BOTH of the embedder's inputs to bf16 before the Linears
  //    (`batch['msa_feat'].astype(dtype)`, and target_feat inside the pad)
  {
    const size_t n21 = (size_t)N * embed::NAA;
    const size_t nmf = (size_t)plan.msa_rows * N * inputfeat::MSA;
    roundBf16K<<<(int)((n21 + 255) / 256), 256, 0, st>>>(tf21, n21);
    roundBf16K<<<(int)((nmf + 255) / 256), 256, 0, st>>>(msa_feat, nmf);
  }
  idxToFloatK<<<g1, 128, 0, st>>>(in.residue_index, ridx_f, N);
  idxToFloatK<<<g1, 128, 0, st>>>(in.asym_id, asym_f, N);
  idxToFloatK<<<g1, 128, 0, st>>>(in.entity_id, ent_f, N);
  emb.forward(tf21, msa_feat, ridx_f, asym_f, ent_f, /*sym*/asym_f, offset);
  //    AND ROUND.  bc2's own embedder hook does this
  //    (`msa_activations.astype(dtype)`) because the block it replaces runs
  //    in bf16; keeping f32 is more accurate and not the same model.
  roundBf16K<<<(int)((nn_cz + 255) / 256), 256, 0, st>>>(emb.pair, nn_cz);
  {
    const size_t nm = (size_t)plan.msa_rows * N * embed::CM;
    roundBf16K<<<(int)((nm + 255) / 256), 256, 0, st>>>(emb.msa, nm);
  }

  // 3a. the three recycle terms, added in place.  On the first pass they
  //     are the LayerNorms' offsets and the zero-distance distogram's
  //     column -- NOT zero.
  rec.forward(aatype, in.n, emb.pair, emb.msa);

  // 3b. THE COMPONENTS WANT A 2-D PAIR MASK, not seq_mask.  Passing the
  //     1-D one compiles wherever the type matches and silently masks the
  //     wrong thing.
  pairMaskK<<<gn, 128, 0, st>>>(in.seq_mask, in.n, pair_mask, N);

  // 4. the template embedding's contribution is ADDED to the pair track,
  //    as AF2 does: pair += template_embedding(pair, ...)
  //    THE SUM GOES IN ITS OWN BUFFER.  Adding it into emb.pair in place
  //    would be the same forward and a wrong backward: tpl.backward
  //    REPLAYS from its query, and its query was the pair BEFORE its own
  //    contribution was added.  An in-place += leaves emb.pair holding
  //    query + output, and the replay then runs the template embedding on
  //    an input it never saw -- green forward, quietly wrong gradient,
  //    which is the shape of the bug that broke a campaign once already.
  // NB_TPL_DIRTY: the test the two scribbles above do NOT make.  They
  // destroy a stack's scratch AFTER it has run, which shows the contents
  // need not survive.  Sharing imposes the opposite condition: the template
  // would START on whatever bytes the Evoformer left behind, so every
  // module must WRITE its scratch before it reads it.  A module that leans
  // on a zeroed buffer, or on its own leftovers from the previous call,
  // passes both scribbles and breaks the moment the bytes are shared.
  if (envOn("NB_TPL_DIRTY"))
    nbrealloc::scribble(tpl_ledger_first, tpl_ledger_last, st);
  tpl.forward(emb.pair, in.act_const, pair_mask, tpl_out, /*save*/nullptr);
  // NB_TPL_SCRIBBLE: destroy the template stack's scratch the instant its
  // forward returns.  tpl.backward REPLAYS from its query (see the note
  // above), so nothing downstream should be able to tell -- and if the
  // gradient is bit-identical with these bytes filled with 0x5A, they are
  // free to be shared with the Evoformer's, which is 6.2 GB at N=736.
  if (envOn("NB_TPL_SCRIBBLE"))
    nbrealloc::scribble(tpl_ledger_first, tpl_ledger_last, st);
  const int gacc = (int)((nn_cz + 255) / 256);
  roundBf16K<<<gacc, 256, 0, st>>>(tpl_out, nn_cz);
  CU(cudaMemcpyAsync(pair_all, emb.pair, nn_cz * 4,
                     cudaMemcpyDeviceToDevice, st));
  foldit::accK<<<gacc, 256, 0, st>>>(pair_all, tpl_out, nn_cz);
  roundBf16K<<<gacc, 256, 0, st>>>(pair_all, nn_cz);

  dsDump("tf21", tf21, (size_t)N * embed::NAA * 4);
  dsDump("msa_feat", msa_feat, (size_t)N * inputfeat::MSA * 4);
  dsDump("emb_pair", emb.pair, nn_cz * 4);
  dsDump("emb_msa", emb.msa, (size_t)plan.msa_rows * N * embed::CM * 4);
  dsDump("tpl_out", tpl_out, nn_cz * 4);
  dsDump("pair_all", pair_all, nn_cz * 4);

  // 5. the extra-MSA stack (pair only), then the trunk.
  //    The extra stack takes the embedder's msa and ignores it -- it is
  //    pair_only, so its outer product mean is the constant the weights
  //    carry -- but the TRUNK takes the msa with the template row on the
  //    end, and the mask with the template's row mask on the end.
  const size_t rowc = (size_t)N * embed::CM;
  // NB_EVO_DIRTY: the fourth condition, symmetric to NB_TPL_DIRTY.  Shared
  // bytes arrive at the Evoformer holding whatever the template stack left.
  if (envOn("NB_EVO_DIRTY"))
    nbrealloc::scribble(evo_ledger_first, evo_ledger_last, st);
  extra.forward(pair_all, emb.msa, pair_mask, in.seq_mask);
  for (int r = 0; r < plan.msa_rows; ++r) {
    CU(cudaMemcpyAsync(trunk_msa + r * rowc, emb.msa + r * rowc, rowc * 4,
                       cudaMemcpyDeviceToDevice, st));
    CU(cudaMemcpyAsync(trunk_msa_mask + (size_t)r * N, in.seq_mask,
                       (size_t)N * 4, cudaMemcpyDeviceToDevice, st));
  }
  t1d.forward(t_aatype, t_pos, t_mask, in.n, t1d_tab,
              trunk_msa + (size_t)plan.msa_rows * rowc,
              trunk_msa_mask + (size_t)plan.msa_rows * N);
  // SKIPPED when the caller carried the trunk across the boundary: its
  // save set and pout/mout are XLA buffers holding the forward's results
  // already, so recomputing 48 blocks would only overwrite them with the
  // same numbers.
  if (!skip_trunk)
    trunk.forward(extra.pout, trunk_msa, pair_mask, trunk_msa_mask);

  dsDump("extra_pout", extra.pout, nn_cz * 4);
  dsDump("trunk_msa", trunk_msa,
         (size_t)(plan.msa_rows + 1) * N * embed::CM * 4);
  dsDump("trunk_pout", trunk.pout, nn_cz * 4);
  dsDump("trunk_mout", trunk.mout,
         (size_t)(plan.msa_rows + 1) * N * embed::CM * 4);

  // 6. THE SINGLE COMES FROM THE TRUNK, not the embedder.  embed.cuh holds
  //    the single_activations weights but singleFromTrunk is a separate
  //    call taking the TRUNK's msa row 0 -- wiring emb.single here would
  //    hand the structure module the pre-trunk representation and quietly
  //    produce a worse structure with no error anywhere.
  emb.singleFromTrunk(trunk.mout, emb.single);
  idxToFloatK<<<g1, 128, 0, st>>>(aatype, aatype_f, N);
  sm.forward(emb.single, trunk.pout, pair_mask, aatype_f);
  hd.forward(trunk.pout, sm.outAct());
  //    and the experimentally-resolved head is a SEPARATE call reading the
  //    EVOFORMER single, not the structure module's act -- see heads.cuh,
  //    which says so because AlphaFoldIteration does.
  hd.expResFwd(emb.single);

  dsDump("single", emb.single, (size_t)N * embed::CS * 4);
  dsDump("sm_atom14", sm.outAtom(), (size_t)N * 14 * 3 * 4);
  dsDump("hd_plddt", hd.plddt, (size_t)N * heads::PL_BINS * 4);
  dsDump("hd_pae", hd.pae, nn * (size_t)heads::NPAE * 4);
  dsDump("hd_dgram", hd.dgram, nn * (size_t)heads::NDGRAM * 4);
  dsDump("hd_expres", hd.expres, (size_t)N * heads::NER * 4);

  // 7. confidence, off the head logits.  The bin tables are built here and
  //    not passed in: tm_bin's d0 depends on sum(seq_mask), so it is a
  //    function of the RUNTIME length and cannot be baked.
  confidence::binTablesK<<<1, 64, 0, st>>>(
      in.seq_mask, in.n, pae_centres, tm_bins, N, DBINS);
  confidence::plddtK<<<g1, 128, 0, st>>>(hd.plddt, in.n, out.plddt, N);
  confidence::paeTermK<<<gn, 128, 0, st>>>(
      hd.pae, pae_centres, tm_bins, in.n, pae_err, tm_term, N, DBINS);
  confidence::symmetriseK<<<gn, 128, 0, st>>>(pae_err, in.n, out.pae, N);
  confidence::perAlignmentK<<<N, 32, 0, st>>>(
      tm_term, in.seq_mask, in.interface_asym, 0, in.n, per_align, N);
  confidence::weightedMaxK<<<1, 32, 0, st>>>(
      per_align, in.seq_mask, in.n, out.ptm, N);
  // into the MEMBER, not straight into the output: the backward needs this
  // vector and never sees the Outputs struct
  confidence::perAlignmentK<<<N, 32, 0, st>>>(
      tm_term, in.seq_mask, in.interface_asym, 1, in.n,
      out_iptm_per_residue, N);
  confidence::weightedMaxK<<<1, 32, 0, st>>>(
      out_iptm_per_residue, in.seq_mask, in.n, out.iptm, N);
  CU(cudaMemcpyAsync(out.iptm_per_residue, out_iptm_per_residue, (size_t)N * 4,
                     cudaMemcpyDeviceToDevice, st));
  confidence::expresCaK<<<g1, 128, 0, st>>>(
      hd.expres, in.n, out.expres_ca, N);

  // 8. and the atoms, superposed onto the target template
  const int ga = (int)(((size_t)N * ATOM + 127) / 128);
  atom14To37K<<<ga, 128, 0, st>>>(sm.outAtom(), a37_14, a37e, in.n, atom37, N);
  kabsch::covK<<<1, 32, 0, st>>>(atom37, t_pos, t_mask, in.flags, in.n,
                                 kab, N);
  kabsch::rotationK<<<1, 32, 0, st>>>(kab, kab + 15);
  kabsch::applyK<<<ga, 128, 0, st>>>(
      atom37, a37e, kab, kab + 15, in.n, out.atoms, N);
  // the distogram is the head's logits unchanged
  CU(cudaMemcpyAsync(out.distogram, hd.dgram,
                     (size_t)N * N * DBINS * 4,
                     cudaMemcpyDeviceToDevice, st));
  // and what the next recycle pass reads
  rec.save(atom37, trunk.pout, trunk.mout);
  (void)save;
}

// ---- the backward ------------------------------------------------------
// The chain reversed.  Three things this has to get right, each of which
// fails quietly rather than loudly:
//
//  * CALL ORDER IN THE CONFIDENCE BLOCK.  paeBwdK WRITES d_pae_logits and
//    tmBwdK ACCUMULATES into it, so pae goes first.  The other order
//    overwrites the atomicAdds and leaves ipTM training on nothing.
//  * THE TRUNK SINGLE RECEIVES FROM TWO PLACES -- the structure module and
//    the experimentally-resolved head -- so the second accumulates
//    (expResBwd's beta = 1) rather than overwrites.
//  * EACH COMPONENT REPLAYS.  None of them may read what its own forward
//    left behind, because the recycle passes ran after it.  That is the
//    bug that gated green and broke a campaign; see tplembed.cuh.
//
// Kabsch contributes nothing: its rotation is under stop_gradient.
inline void DesignStep::backward(const Inputs& in, const Cotangents& ct,
                                 float* d_sequence) {
  const int N = plan.nmax;
  const int g1 = (N + 127) / 128;
  const int gn = (int)(((size_t)N * N + 127) / 128);
  const size_t nn = (size_t)N * N;
  const size_t nn_cz = nn * embed::CZ;

  // 1. confidence -> the head logits' cotangents.  The bin tables are
  //    rebuilt here rather than read from the forward: they are a function
  //    of the inputs alone, so rebuilding is cheap and cannot go stale.
  confidence::binTablesK<<<1, 64, 0, st>>>(
      in.seq_mask, in.n, pae_centres, tm_bins, N, DBINS);
  confidence::plddtBwdK<<<g1, 128, 0, st>>>(
      hd.plddt, ct.plddt, in.n, d_plddt_logits, N);
  confidence::paeBwdK<<<gn, 128, 0, st>>>(           // WRITES
      hd.pae, pae_centres, ct.pae, in.n, d_pae_logits, N, DBINS);
  confidence::expresCaBwdK<<<(int)(((size_t)N * 37 + 127) / 128), 128, 0, st>>>(
      hd.expres, ct.expres_ca, in.n, d_expres_logits, N);
  // ptm, iptm and iptm_per_residue ACCUMULATE, so they must follow pae.
  // The two scalars' cotangents are DEVICE buffers -- this passed a
  // literal 0.f, which is a loss on pTM or ipTM training on nothing.
  confidence::tmBwdK<<<g1, 128, 0, st>>>(
      hd.pae, tm_bins, per_align, in.seq_mask, in.interface_asym, 0,
      ct.ptm, in.n, d_pae_logits, N, DBINS);
  confidence::tmBwdK<<<g1, 128, 0, st>>>(
      hd.pae, tm_bins, out_iptm_per_residue, in.seq_mask, in.interface_asym,
      1, ct.iptm, in.n, d_pae_logits, N, DBINS);
  // and the per-residue vector is its own output, not just the thing the
  // max is taken over
  confidence::perAlignBwdK<<<N, 128, 0, st>>>(
      hd.pae, tm_bins, ct.iptm_per_residue, in.seq_mask, in.interface_asym,
      1, in.n, d_pae_logits, N, DBINS);

  // 2. the heads -> the trunk pair and the structure module's act.
  //    THE pLDDT HEAD'S LAYERNORM READ sm.outAct(), so that is what its
  //    backward has to be handed; emb.single is the structure module's
  //    INPUT and the two are the same shape, so the wrong one compiles.
  hd.backward(ct.distogram, d_pae_logits, d_plddt_logits, sm.outAct(),
              d_pair, d_sm_act);

  // 2b. the atoms: back through the superposition, then back through the
  //     atom37 gather.  `kab + 15` is the rotation the forward left; it is
  //     a function of stop_gradient'd CAs, so reusing it is not the
  //     read-stale-state trap -- it is a constant of this input.
  const int ga = (int)(((size_t)N * ATOM + 127) / 128);
  kabsch::applyBwdK<<<ga, 128, 0, st>>>(ct.atoms, a37e, kab + 15, in.n,
                                        d_atom37, N);
  CU(cudaMemsetAsync(d_atom14, 0, (size_t)N * 14 * 3 * 4, st));
  atom37To14BwdK<<<ga, 128, 0, st>>>(d_atom37, a37_14, a37e, in.n,
                                     d_atom14, N);

  dsDump("b_pae_logits", d_pae_logits, nn * (size_t)heads::NPAE * 4);
  dsDump("b_after_heads_dpair", d_pair, nn_cz * 4);
  dsDump("b_sm_act", d_sm_act, (size_t)N * structmod::C * 4);

  // 3. the structure module -> the single and the pair.
  //    ITS dpair IS WRITTEN, NOT ACCUMULATED (a LayerNorm backward ends in
  //    `DX[c] = ...`), and the two pair heads have already put theirs in
  //    d_pair -- so it goes to its own buffer and is added on.
  CU(cudaMemsetAsync(d_rot, 0, (size_t)N * 9 * 4, st));
  CU(cudaMemsetAsync(d_trans, 0, (size_t)N * 3 * 4, st));
  sm.backward(d_sm_act, d_rot, d_trans, emb.single,
              d_single, d_pair_sm, d_atom14, aatype_f);
  foldit::accK<<<(int)((nn_cz + 255) / 256), 256, 0, st>>>(
      d_pair, d_pair_sm, nn_cz);
  // and NOW the experimentally-resolved head, which reads the EVOFORMER
  // single: its beta = 1 accumulates onto what the structure module just
  // wrote, so it has to come after.  Before it, the accumulation lands in
  // an uninitialised buffer and is then overwritten.
  hd.expResBwd(d_expres_logits, d_single, /*beta*/1.f);

  // 4. the single came from the trunk's msa row 0.  Every other row --
  //    the rest of the MSA and the template row -- takes no cotangent from
  //    anywhere, so they have to be ZEROED, not left as the last step's.
  CU(cudaMemsetAsync(d_msa, 0,
                     (size_t)(plan.msa_rows + 1) * N * embed::CM * 4, st));
  emb.singleFromTrunkBwd(d_single, d_msa, /*beta*/0.f);

  dsDump("b_after_sm_dpair", d_pair, nn_cz * 4);
  dsDump("b_dsingle", d_single, (size_t)N * heads::CS * 4);
  dsDump("b_dmsa", d_msa,
         (size_t)(plan.msa_rows + 1) * N * embed::CM * 4);

  // 5. the trunk, then the extra-MSA stack.
  //    THE STACK'S PAIR COTANGENT COMES OUT bf16 (b.dp); the msa one is
  //    f32 (b.dm).  Everything either side of it is f32, so it is widened
  //    at each hand-off rather than silently reinterpreted.
  trunk.backward(d_pair, d_msa, pair_mask, trunk_msa_mask);
  widenK<<<(int)((nn_cz + 255) / 256), 256, 0, st>>>(
      d_pair, trunk.b.dp, nn_cz);
  dsDump("b_after_trunk_dpair", d_pair, nn_cz * 4);
  dsDump("b_trunk_dm", trunk.b.dm,
         (size_t)(plan.msa_rows + 1) * N * embed::CM * 4);
  extra.backward(d_pair, trunk.b.dm, pair_mask, in.seq_mask);
  widenK<<<(int)((nn_cz + 255) / 256), 256, 0, st>>>(
      d_pair, extra.b.dp, nn_cz);

  // 6. the template embedding, whose contribution was ADDED to the pair, so
  //    the same cotangent reaches both it and the embedder
  // NB_EVO_SCRIBBLE: the other half of the lifetime proof.  If the template
  // stack is to run out of the Evoformer's bytes, then tpl.backward -- the
  // LAST thing in the step that touches them -- destroys the extra and
  // trunk stacks' scratch.  Nothing after this point may depend on it.
  if (envOn("NB_EVO_SCRIBBLE"))
    nbrealloc::scribble(evo_ledger_first, evo_ledger_last, st);
  // and the same for the template's BACKWARD, which replays its forward and
  // so must also tolerate dirty scratch on entry
  if (envOn("NB_TPL_DIRTY"))
    nbrealloc::scribble(tpl_ledger_first, tpl_ledger_last, st);
  tpl.backward(d_pair, emb.pair, in.act_const, pair_mask,
               tpl_save, d_tpl_query);
  foldit::accK<<<(int)((nn_cz + 255) / 256), 256, 0, st>>>(
      d_pair, d_tpl_query, nn_cz);

  dsDump("b_after_tpl_dpair", d_pair, nn_cz * 4);

  // 7. the embedder -> target_feat and msa_feat.
  //    THE MSA COTANGENT COMES FROM THE TRUNK, not the extra stack: the
  //    extra stack is pair_only and never touched the embedder's msa, so
  //    its `dm` is whatever its allocation happened to hold.  Only the
  //    embedder's own rows -- the first plan.msa_rows of the trunk's --
  //    belong to it; the template row's cotangent stops here.
  emb.backward(trunk.b.dm, d_pair, d_single, tf21, d_tf21, d_msa_feat);
  // and back down to the 20 the sequence features have
  reColsK<<<(int)(((size_t)N * inputfeat::A + 127) / 128), 128, 0, st>>>(
      d_tf21, embed::NAA, d_feat, inputfeat::A, N);

  // 8. msa_feat and target_feat -> the sequence features and the profile
  inputfeat::msaFeatBwdK<<<(int)(((size_t)N * inputfeat::A + 127) / 128),
                           128, 0, st>>>(
      d_msa_feat, d_feat, in.n, d_seqfeat, d_proffeat, N);

  // 9. and the sequence features -> the designed logits
  seqfeat::Scalars sc{};
  seqfeat::seqFeatBwdK<<<g1, 128, 0, st>>>(
      in.sequence, in.flags, /*bias*/nullptr, sc, seqfeat::OMITTED_HALF,
      d_seqfeat, d_proffeat, d_sequence, N, in.scalars);
  dsDump("b_dseqfeat", d_seqfeat, (size_t)N * inputfeat::A * 4);
  dsDump("b_dsequence", d_sequence, (size_t)N * A * 4);
  ++dsDumpCall();
}

}  // namespace design_step
