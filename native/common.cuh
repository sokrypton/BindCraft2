// Shared scaffolding for the native-block prototype: the oracle loader, the
// error metrics the gate reads, and a timer.
//
// The oracle is written by dev/native_block/oracle.py --dump: a manifest naming
// every array and a flat float32 .f32 file per array.  Nothing here knows about
// any particular module.
#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

// A BOOLEAN ENV FLAG THAT HONOURS ITS VALUE.  Every flag in this module was
// `getenv(X) != nullptr`, so `X=0` turned the flag ON.  That is not a
// theoretical trap: an A/B of the activation-keep tier run as
// `NE_KEEP_ACTS=0` against `NE_KEEP_ACTS=1` had the tier enabled in BOTH
// arms, measured 640.6 vs 643.8 ms, and was about to be written up as "the
// documented 7% win no longer delivers".  Run properly it is 700.2 -> 642.7,
// -8.2%.  An unset flag is off; a set one is on unless its value says
// otherwise.
inline bool envOnDef(const char* name, bool dflt) {
  const char* v = getenv(name);
  if (!v) return dflt;
  return !(v[0] == '\0' || !strcmp(v, "0") || !strcmp(v, "false") ||
           !strcmp(v, "no") || !strcmp(v, "off"));
}
inline bool envOn(const char* name) {
  return envOnDef(name, false);
}

// ------------------------------------------------------------ auto-chunking
// ONE DECISION, FROM N, BEFORE ANYTHING ALLOCATES.
//
// Triangle attention, both tri-muls and the pair transition can each run a
// chunk of their N^2 rows at a time instead of materialising every
// intermediate whole.  Each is exact -- no replay, no cross-chunk accumulator
// -- and each costs a few percent.  This says WHEN to do it.
//
// THE DECISION CANNOT BE PER-MODULE, and that is the whole reason this lives
// here rather than in three `alloc()`s.  The tempting version is for each
// module to call cudaMemGetInfo and chunk if memory looks tight.  That is the
// greedy first grant wearing a new hat: it is what the shallow extra-MSA
// stack did when it sized its keep-tier margin from its own depth, granted
// itself 3.66 GB and starved the trunk, the structure module and the heads.
// Whichever module allocates FIRST always sees plenty free, so it would not
// chunk and the last one would still run out -- and the chunk size would then
// depend on allocation ORDER, so a run would not reproduce.  One threshold,
// read before the first cudaMalloc, has neither problem.
//
// THE NUMBER IS A PADDED N, AND GETTING THAT WRONG COSTS A WHOLE BUCKET.
// Every caller pads to a multiple of 32 (`pad = (-n) % 32`, four places in
// bindcraft/af/native.py), so these modules are never handed the complex's
// real length: a 457-residue complex arrives here as 480.  The threshold was
// first set to 457, the longest complex measured to fit unchunked -- and that
// complex then chunked anyway, because 480 > 457.  It cost nothing but 3% and
// a |grad| that moved in the fourth digit (7.24374 -> 7.24925, the keep tiers
// granting differently once the chunking freed room), which is exactly why it
// would have shipped unnoticed.  480 is the same measurement expressed in the
// units this function is actually called with.
//
// MEASURED on a 23.7 GB A10 with a real target and the interface loss:
// complex 457 (padded 480) is the longest that fits unchunked, and complex
// 512 (padded 512) the longest that fits chunked.  Below the threshold
// nothing changes at all -- chunking costs 0.7-8.5% and buys nothing when the
// step already fits.
//
// NB_CHUNK_N moves the threshold (0 disables chunking entirely); each
// module's own knob -- TM_ROWS, TA_ROWS, TR_ROWS -- still overrides the chunk
// SIZE outright, at any length.
// The length the buffers were SIZED for.  Zero until a step is re-derived at
// a shorter one; see nbrealloc below, which is declared after this only
// because nbChunkOn is.
namespace nbrealloc { inline int& allocN() { static int n = 0; return n; } }

// NO BACKWARD BEHIND THIS ONE.
//
// A predict-only object never runs a backward, so every buffer that exists
// only to hold a cotangent is allocated and never read.  At complex 607 the
// step is 11.13 GB where the JAX predict it replaces runs in about 4, and
// the difference is almost all of this.
//
// It is a GLOBAL rather than a parameter because the alternative is a `lean`
// argument through twenty-odd alloc() signatures, most of which forward it
// untouched.  It is set around DesignStep::alloc and read only there --
// nothing consults it at run time, so there is no state to get out of step
// with a call.
//
// THE DANGER IS NOT THE MEMORY, IT IS THE SHARING.  This code lends buffers
// between phases on purpose (block.cuh's "shared d-scratch", tplblock's
// adopted scratch), so a d-named buffer is not automatically backward-only
// and nulling one a forward quietly uses is a segfault at best and a wrong
// number at worst.  Every buffer dropped under this flag has to be checked
// against the forward, and ds_predict_gate.py -- which compares this arm's
// eight outputs with the design forward's at max|d| 0 -- is what says so.
//
// A lean object must never serve a design, so `lean` is part of the DsKey.
namespace nblean {
inline bool& building() { static bool v = false; return v; }
// `true` when this allocation will never back-propagate.
inline bool on() { return building(); }
struct Scope {
  bool was;
  explicit Scope(bool v) : was(building()) { building() = v; }
  ~Scope() { building() = was; }
};
}  // namespace nblean

inline bool nbChunkOnAt(size_t n);   // defined just below nbChunkOn

inline bool nbChunkOn(size_t n) {
  // CHUNK AS THE ALLOCATION DID, not as this length would on its own.
  //
  // Every chunk policy here turns itself OFF below a threshold, so a SHORTER
  // length asks for MORE scratch: unchunked at n=224 trimul wants all 50176
  // rows where chunked at N=608 it wanted 22528.  Re-deriving a step at a
  // shorter length then has to allocate a bigger buffer -- which spends the
  // memory the whole exercise was meant to save, and MOVES buffers that
  // another module aliased in the first pass (adoptBackwardScratch), which
  // is how this surfaced: cuBLASLt failing with execution and internal
  // errors on operands that were all non-null.
  //
  // Chunking is a loop, not a correctness condition.  Running the short
  // length in the long length's chunks is the same arithmetic in more
  // iterations, and keeps every buffer inside what it already has.
  if (nbrealloc::allocN()) n = (size_t)nbrealloc::allocN();
  return nbChunkOnAt(n);
}

// The policy for a length, with no pin applied.  The pair TRANSITION asks
// this instead, because chunking is not neutral there the way it is for the
// tri-muls and the triangle attentions: it flips `h_chunked`, and with it
// whether the 512-wide hidden activation is materialised whole or
// RECOMPUTED per chunk in the backward.  Pinning it made a short state take
// the long state's backward path and the gradient moved 0.4% at complex
// 487 -- reproducible in both arms, so a real divergence and not the
// bf16 run-to-run noise.  The buffer it sizes is 51 MB against 47 at the
// lengths where that costs anything, so leaving it unpinned costs one small
// fresh allocation and buys a gradient identical to the per-state path.
inline bool nbChunkOnAt(size_t n) {
  static const char* e = getenv("NB_CHUNK_N");
  static const size_t thresh = e ? (size_t)atoll(e) : 480;
  return thresh > 0 && n > thresh;
}


#define CU(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d cuda: %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUB(x)                                                                 \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "%s:%d cublas: %d\n", __FILE__, __LINE__, (int)s_);      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// =====================================================================
// RUNNING A SMALLER N ON A BIGGER ALLOCATION.
//
// A design step runs several STATES -- for a binder, the complex (say 607)
// and the binder alone (220).  Each state's length gets its own DesignStep,
// and the second one is not free: 4.34 GB at N=256 beside 15.67 at N=640,
// which is exactly what puts complex 617 per-state out of memory (XLA then
// asks for 2.15 GB with 2.33 free and misses).  The alternative on offer was
// to pad every state up to the biggest and share one object -- which fits,
// and makes the 220-residue state execute as a 608-residue one: 25.8 s
// against 14.2.
//
// Neither is necessary.  An allocation sized for N=640 is big enough to run
// a step at n=256; nothing about the memory requires the padding.  What
// stands in the way is only that N is baked in at alloc() time, in 23
// modules, as strides, grid extents, chunk policies and carved offsets.
//
// So: call alloc() AGAIN with the smaller n.  It already derives every one
// of those correctly -- that is its job -- and the buffers it would ask for
// are smaller than the ones it has.  `active()` makes the second pass hand
// each pointer member back the buffer it already owns instead of allocating.
//
// KEYED ON THE ADDRESS OF THE POINTER MEMBER, not on call order.  `&x->psave`
// is stable for the life of the module instance, so the mapping is
// self-describing: a branch that allocated last time and not this time
// simply is not asked, and one asked for the first time allocates honestly.
// Call-order keying would have silently crossed two members the first time
// an `if (use_dropout)` differed between passes.
//
// A SMALLER REQUEST IS SERVED, A BIGGER ONE IS NOT.  n <= the recorded size
// reuses; anything larger allocates fresh and re-records, so this cannot
// hand back a buffer that is too small -- the one failure that would be
// silent rather than loud.
namespace nbrealloc {

inline bool& active() { static bool a = false; return a; }

struct Slot { void* p; size_t n; };

// Every buffer this process has handed out, in allocation order.  It exists
// for the reuse keying above, and it doubles as an INVENTORY: a caller can
// snapshot it around one module's alloc() and learn exactly which buffers
// are that module's, without enumerating them by hand in a header that will
// drift.  tplembed uses that to prove its scratch is throwaway.
inline std::vector<Slot>& ledger() { static std::vector<Slot> v; return v; }

// DONATION.  The template stack holds a second complete set of pair scratch
// -- 6.2 GB at N=736 -- and never runs beside the Evoformer: it is first in
// the forward and last in the backward.  Four conditions were measured
// before this existed (NB_TPL_SCRIBBLE / NB_EVO_SCRIBBLE / NB_TPL_DIRTY /
// NB_EVO_DIRTY in design_step.cuh): neither stack's scratch has to survive
// the other's use of the bytes, and neither reads its scratch before
// writing it.  So the template can run out of the Evoformer's allocations.
//
// A bump pointer per donor buffer, 256-byte aligned as cudaMalloc is, so
// the ranges handed out are disjoint and every one is as aligned as the
// buffer it came from.  Anything that does not fit falls through to a real
// cudaMalloc: the worst case is exactly the behaviour before this.
struct Donor { char* base; size_t size, used; };
inline std::vector<Donor>& donors() { static std::vector<Donor> d; return d; }
inline bool& donating() { static bool b = false; return b; }
// Donated pointers are INTERIOR pointers into somebody else's allocation,
// and cudaFree cannot take one.  TriMul::adoptBackwardScratch frees its own
// scratch before aliasing the other instance's, so the template stack hits
// this the moment it is donated to -- it surfaced as cuBLAS NOT_SUPPORTED
// two calls later rather than as a failed free.
inline std::set<void*>& donated() { static std::set<void*> s; return s; }

// NOT EVERY BUFFER IN THE DONOR'S RANGE MAY BE LENT.  The scribble proofs
// show that no stack's scratch has to SURVIVE the other's use of the bytes,
// which is the right question for an activation and the wrong one for a
// WORKSPACE: cuBLASLt's is written and read inside a single GEMM, so an
// operand placed on top of it is corrupted within the call, and scribbling
// between calls never notices.  lt.cuh marks it, and anything else that is
// live for the duration of a kernel rather than between kernels belongs
// here too.
inline std::set<void*>& neverLend() { static std::set<void*> s; return s; }
inline void neverDonate(void* p) { neverLend().insert(p); }

inline void donorOpen(size_t first, size_t last) {
  donors().clear();
  // ONLY THE BIG N^2 SCRATCH IS LENT.  The small buffers in a stack's range
  // are the ones most likely to be written ONCE at alloc and relied on
  // afterwards -- seeds, masks, scalars, the memsets scattered through
  // block.cuh -- and lending one of those puts the template's garbage
  // somewhere nothing ever rewrites.  That is invisible to the scribble
  // proofs, which only ever ran between kernels.  The 6 GB this exists for
  // is all in the N^2 buffers, so the threshold costs nothing.
  const size_t floor_bytes =
      getenv("NB_DONOR_MIN") ? (size_t)atoll(getenv("NB_DONOR_MIN")) : (8u << 20);
  for (size_t i = first; i < last && i < ledger().size(); ++i) {
    if (neverLend().count(ledger()[i].p)) continue;
    if (ledger()[i].n < floor_bytes) continue;
    donors().push_back(Donor{(char*)ledger()[i].p, ledger()[i].n, 0});
  }
  // NB_DONOR_LIMIT=k lends only the first k, which bisects a corruption to
  // one donor buffer instead of reasoning about which of thirty it is.
  if (const char* lim = getenv("NB_DONOR_LIMIT")) {
    const size_t k = (size_t)atoll(lim);
    if (donors().size() > k) donors().resize(k);
  }
  if (getenv("NB_DONOR_LIST"))
    for (size_t i = 0; i < donors().size(); ++i)
      fprintf(stderr, "  donor[%2zu] %p %7.1f MB\n", i,
              (void*)donors()[i].base, donors()[i].size / 1e6);
  donating() = true;
}
inline void donorClose(size_t* served, size_t* bytes) {
  donating() = false;
  if (!served) return;
  for (const Donor& d : donors()) *bytes += d.used;
}

// Best fit, so a small request does not eat the one buffer a big one needs.
inline void* donate(size_t n) {
  const size_t ALIGN = 256;
  Donor* best = nullptr;
  for (Donor& d : donors()) {
    const size_t off = (d.used + ALIGN - 1) & ~(ALIGN - 1);
    if (off + n > d.size) continue;
    if (!best || (d.size - off) < (best->size - ((best->used + ALIGN - 1) & ~(ALIGN - 1))))
      best = &d;
  }
  if (!best) return nullptr;
  const size_t off = (best->used + ALIGN - 1) & ~(ALIGN - 1);
  best->used = off + n;
  return best->base + off;
}

inline cudaError_t mallocReusing(void** p, size_t n) {
  static std::map<void**, Slot> pool;
  auto it = pool.find(p);
  if (active() && it != pool.end() && n <= it->second.n) {
    *p = it->second.p;
    return cudaSuccess;
  }
  // GREW.  A shorter length does NOT imply smaller buffers: the chunk
  // policies turn themselves OFF below their threshold, so an unchunked
  // n=224 asks for more rows of scratch than a chunked N=608.  Allocating
  // fresh is correct, but it MOVES the buffer, and anything that aliased the
  // old address in the first pass (TriMul::adoptBackwardScratch and its
  // triattn twin hand one instance's scratch to another) is now pointing at
  // the smaller one.  NB_REALLOC_LOUD names them.
  if (active() && it != pool.end() && getenv("NB_REALLOC_LOUD"))
    fprintf(stderr, "nbrealloc: GREW %p: %zu -> %zu bytes\n",
            (void*)p, it->second.n, n);
  if (donating()) {
    if (void* d = donate(n)) {
      *p = d;
      donated().insert(d);
      pool[p] = Slot{d, n};
      ledger().push_back(Slot{d, n});
      return cudaSuccess;
    }
  }
  cudaError_t e = ::cudaMalloc(p, n);
  if (e == cudaSuccess) { pool[p] = Slot{*p, n}; ledger().push_back(Slot{*p, n}); }
  return e;
}

// SCRIBBLE, for proving a lifetime rather than arguing one.  Fill a range of
// the ledger with a pattern that is not a plausible activation: if a later
// stage's output is bit-identical with these buffers scribbled over, their
// contents provably did not need to survive, and the bytes can be shared
// with something else.  If it is not identical, the sharing is unsound and
// one cheap run said so.
inline void scribble(size_t first, size_t last, cudaStream_t st) {
  for (size_t i = first; i < last && i < ledger().size(); ++i) {
    if (!ledger()[i].p || !ledger()[i].n) continue;
    if (neverLend().count(ledger()[i].p)) continue;
    CU(cudaMemsetAsync(ledger()[i].p, 0x5A, ledger()[i].n, st));
  }
}

// Streams and cuBLAS handles are NOT sized by N, so a second alloc() must
// reuse them rather than make another -- a module that creates a stream per
// pass would run its kernels unordered against residual adds that are still
// on the first one.  Keyed the same way, and on the member's ADDRESS rather
// than its value, because not every module initialises these to null and
// reading an uninitialised handle to decide is how that becomes a crash in
// cuBLAS instead of a bug here.
inline cudaError_t streamCreateOnce(cudaStream_t* s) {
  static std::map<void*, cudaStream_t> made;
  auto it = made.find((void*)s);
  if (active() && it != made.end()) { *s = it->second; return cudaSuccess; }
  cudaError_t e = ::cudaStreamCreate(s);
  if (e == cudaSuccess) made[(void*)s] = *s;
  return e;
}

inline cublasStatus_t cublasCreateOnce(cublasHandle_t* h) {
  static std::map<void*, cublasHandle_t> made;
  auto it = made.find((void*)h);
  if (active() && it != made.end()) { *h = it->second; return CUBLAS_STATUS_SUCCESS; }
  cublasStatus_t e = ::cublasCreate(h);
  if (e == CUBLAS_STATUS_SUCCESS) made[(void*)h] = *h;
  return e;
}

// AND NOTHING IS FREED WHILE RE-DERIVING.  alloc() is not purely allocating:
// TriMul::adoptBackwardScratch and its triattn twin hand their own backward
// scratch back and alias the other instance's, and the keep-tier grant frees
// a tier before resizing it.  On a second pass every one of those pointers is
// either already freed or belongs to somebody else -- freeing it is a double
// free that surfaces as `cuda: invalid argument` three lines later.  Nothing
// NEW is created on a reuse pass, so nothing on it needs releasing.
inline cudaError_t freeUnlessReusing(void* p) {
  if (active()) return cudaSuccess;
  if (donated().count(p)) return cudaSuccess;   // not ours to free
  neverLend().erase(p);
  for (Slot& s : ledger())
    if (s.p == p) { s.p = nullptr; s.n = 0; }
  return ::cudaFree(p);
}

// RAII so an abort inside alloc() cannot leave the whole process reusing.
struct Scope {
  bool prev;
  Scope() : prev(active()) { active() = true; }
  ~Scope() { active() = prev; }
};

}  // namespace nbrealloc

#define cudaMalloc(p, n) nbrealloc::mallocReusing((void**)(p), (n))
#define cudaStreamCreate(s) nbrealloc::streamCreateOnce(s)
#define cublasCreate(h) nbrealloc::cublasCreateOnce(h)
#define cudaFree(p) nbrealloc::freeUnlessReusing(p)

struct Tensor {
  std::vector<int> shape;
  std::vector<float> host;
  float* dev = nullptr;

  size_t n() const {
    size_t t = 1;
    for (int d : shape) t *= (size_t)d;
    return t;
  }
  void to_device() {
    // ONCE.  This had no guard, so every setWeights that read a tensor
    // allocated ANOTHER float32 copy of it on the device and dropped the
    // last -- about 400 MB a model swap, on top of the packs.  The design
    // pool's note that "each Oracle keeps its own device copy" is what
    // this is for: one copy per tensor, for the life of the Oracle.
    if (dev) return;
    CU(cudaMalloc(&dev, n() * sizeof(float)));
    CU(cudaMemcpy(dev, host.data(), n() * sizeof(float), cudaMemcpyHostToDevice));
  }
};

// Every array the oracle dumped, by its dotted name ("param/pair_transition/
// transition1/weights", "in/pair", "grad/pair", ...).
// ---- WHO OWNS A MODEL'S PACKED WEIGHTS ---------------------------------
//
// Nearly every module here PACKS its weights on setWeights: it allocates
// device buffers and converts the oracle's float32 into the bf16/f16/
// transposed/fragment-permuted forms its kernels read.  Those allocations
// were never freed, so every model swap leaked the last model's pack --
// measured at 1.29 GB per swap across a whole step, which took three
// designs with two held-out models each from 10.5 GB to 20.9 GB and would
// not have fit a fourth.  The design pool's own comment, "setWeights only
// rebinds pointers, so a swap between steps costs nothing", was false.
//
// The owner cannot be the module: `EvoStack::setWeights` calls one block's
// setWeights ONCE PER DEPTH and records where each call put its buffers,
// so 48 live packs come out of one instance and a module that freed its
// own previous pack would delete depth b-1 while packing depth b.  (That
// is exactly what the first version of this fix did, and every loss went
// NaN on the first step.)  The owner is the thing whose weights are being
// REPLACED -- the DesignStep -- so it publishes a collector for the
// duration of its setWeights and frees what the last one collected.
//
// A module that takes its pointers from another instance (shareWeights,
// which setWeights calls immediately after) allocates nothing and records
// nothing, so nothing is freed twice.
inline std::vector<void*>*& weightPackCollector() {
  static std::vector<void*>* p = nullptr;
  return p;
}
inline void recordWeightPack(void* p) {
  if (weightPackCollector() && p) weightPackCollector()->push_back(p);
}

struct Oracle {
  std::string dir;
  std::map<std::string, Tensor> t;

  explicit Oracle(const std::string& d) : dir(d) {
    std::string mf = dir + "/manifest.txt";
    FILE* f = fopen(mf.c_str(), "r");
    if (!f) {
      fprintf(stderr, "no manifest at %s -- run oracle.py --dump first\n",
              mf.c_str());
      exit(1);
    }
    char name[512], file[512];
    int ndim;
    while (fscanf(f, "%511s %511s %d", name, file, &ndim) == 3) {
      Tensor x;
      for (int i = 0; i < ndim; ++i) {
        int d;
        if (fscanf(f, "%d", &d) != 1) { fprintf(stderr, "bad manifest\n"); exit(1); }
        x.shape.push_back(d);
      }
      std::string path = dir + "/" + file;
      FILE* b = fopen(path.c_str(), "rb");
      if (!b) { fprintf(stderr, "missing %s\n", path.c_str()); exit(1); }
      x.host.resize(x.n());
      if (fread(x.host.data(), sizeof(float), x.n(), b) != x.n()) {
        fprintf(stderr, "short read on %s\n", path.c_str());
        exit(1);
      }
      fclose(b);
      t[name] = std::move(x);
    }
    fclose(f);
  }

  Tensor& operator[](const std::string& k) {
    auto it = t.find(k);
    if (it == t.end()) {
      fprintf(stderr, "oracle has no '%s'; it has:\n", k.c_str());
      for (auto& kv : t) fprintf(stderr, "  %s\n", kv.first.c_str());
      exit(1);
    }
    return it->second;
  }
  bool has(const std::string& k) const { return t.count(k) != 0; }
};

// What the gate reads.  max|d| alone is the wrong measure for a GRADIENT: a
// gradient that points the right way but is scaled, or that is right everywhere
// except one huge element, are different failures.  Cosine is the one that
// says "this would steer the optimiser the same way", so it leads.
struct Err {
  double cos, rel_l2, max_abs, max_rel, rms_a, rms_b;
};

inline Err compare(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0, dn = 0, maxa = 0, maxr = 0;
  size_t n = a.size();
  for (size_t i = 0; i < n; ++i) {
    double x = a[i], y = b[i], d = x - y;
    dot += x * y; na += x * x; nb += y * y; dn += d * d;
    double ad = fabs(d);
    if (ad > maxa) maxa = ad;
    double den = fabs(y) > 1e-6 ? fabs(y) : 1e-6;
    if (ad / den > maxr && fabs(y) > 1e-4) maxr = ad / den;
  }
  Err e;
  e.cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
  e.rel_l2 = sqrt(dn) / (sqrt(nb) + 1e-30);
  e.max_abs = maxa;
  e.max_rel = maxr;
  e.rms_a = sqrt(na / n);
  e.rms_b = sqrt(nb / n);
  return e;
}

// The same comparison over a SUBSET of the elements.  A padded run needs it:
// the masked region's values are huge (a 48-block stack at pad 96 reports rms
// 3797 against the unmasked run's 70.8, because a masked row is never
// renormalised), so they dominate the norm and a perfect score there would
// hide any error in the residues that matter.  `keep[i]` selects.
inline Err compare_where(const std::vector<float>& a, const std::vector<float>& b,
                         const std::vector<char>& keep) {
  std::vector<float> xa, xb;
  xa.reserve(a.size()); xb.reserve(b.size());
  for (size_t i = 0; i < a.size(); ++i)
    if (keep[i]) { xa.push_back(a[i]); xb.push_back(b[i]); }
  if (xa.empty()) return Err{};
  return compare(xa, xb);
}

inline void report(const char* what, const Err& e) {
  printf("  %-22s cos %.9f  relL2 %.3e  max|d| %.3e  maxrel %.3e  "
         "(rms %.4f vs %.4f)\n",
         what, e.cos, e.rel_l2, e.max_abs, e.max_rel, e.rms_a, e.rms_b);
}

// HOW LONG DOES THE HOST SPEND ISSUING, against how long the GPU spends
// running?  This is the probe that prices a CUDA graph: the stack launches
// thousands of kernels a step, and if the CPU can issue them faster than the
// GPU drains them then there is no launch overhead to recover and a graph
// buys nothing.  Returns the host-side seconds to issue one call, having
// synchronised first so the queue starts empty.
template <class F>
double issue_ms(F&& f, int reps, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) f();
  CU(cudaDeviceSynchronize());
  double best = 1e30;
  for (int i = 0; i < reps; ++i) {
    CU(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    f();                                  // issue only -- no sync
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    CU(cudaDeviceSynchronize());          // drain before the next reading
    if (ms < best) best = ms;
  }
  return best;
}

// min-of-reps wall time in ms for a device-side callable
template <class F>
double time_ms(F&& f, int reps, int warmup = 3) {
  for (int i = 0; i < warmup; ++i) f();
  CU(cudaDeviceSynchronize());
  double best = 1e30;
  cudaEvent_t a, b;
  CU(cudaEventCreate(&a));
  CU(cudaEventCreate(&b));
  for (int i = 0; i < reps; ++i) {
    CU(cudaEventRecord(a));
    f();
    CU(cudaEventRecord(b));
    CU(cudaEventSynchronize(b));
    float ms;
    CU(cudaEventElapsedTime(&ms, a, b));
    if (ms < best) best = ms;
  }
  CU(cudaEventDestroy(a));
  CU(cudaEventDestroy(b));
  return best;
}

// read a device buffer as float, whatever width it is stored at
// d += a, f32.  foldit.cuh has its own accK over a bf16 chain; this is the
// plain f32 one the structure module's residuals want.
__global__ inline void accSumK(float* __restrict__ d, const float* __restrict__ a,
                               size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] += a[i];
}

// A BIAS BROADCAST OVER ROWS, f32: D[r, c] += b[c].  The trunk's modules have
// no equivalent because every one of them fuses its bias into a cuBLASLt
// epilogue; the STRUCTURE MODULE's matmuls are plain f32 Sgemms, which have
// no epilogue to fuse into.
__global__ inline void addRowBiasK(float* __restrict__ d,
                                   const float* __restrict__ b, int width,
                                   size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] += b[i % width];
}

inline std::vector<float> fetchAny(const float* dev, size_t n) {
  std::vector<float> h(n);
  CU(cudaMemcpy(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

inline std::vector<float> fetchAny(const __nv_bfloat16* dev, size_t n) {
  std::vector<__nv_bfloat16> b(n);
  CU(cudaMemcpy(b.data(), dev, n * 2, cudaMemcpyDeviceToHost));
  std::vector<float> h(n);
  for (size_t i = 0; i < n; ++i) h[i] = __bfloat162float(b[i]);
  return h;
}

inline std::vector<float> fetch(const float* dev, size_t n) {
  std::vector<float> h(n);
  CU(cudaMemcpy(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

// BLK_PROF attributes the stack's time to the nine steps.  The stack measures
// 398 ms forward and 1108 ms backward at N=256 depth 48, and those two numbers
// cannot say which module to work on next.  Each step is bracketed on the
// block's own stream and summed over however many times it runs, so a 48-block
// stack reports the whole stack's per-module cost in one go.
//
// IT SYNCHRONISES at every step, to read an event pair before reusing it.  So
// the TOTAL under BLK_PROF is larger than the real one (the steps no longer
// overlap at all), and the right way to read the table is as a SHARE.  Keeping
// 2 x 9 x depth event pairs alive instead would avoid that, and would also
// stop measuring what it set out to measure: whether these kernels overlap is
// a question for the profile, not a thing to design the profiler around.
// KEEP TIERS.  A module offers the stack buffers it could hold per depth so
// the backward's replay need not recompute them, and they are NOT equally
// worth holding.  Measured at N=256 depth 48 by running the stack with and
// without each (the ms is what the replay stops doing, the GB is what it
// costs over all 48 depths):
//
//   KEEP_STATS  trimul cmu/crstd   6.8 ms   0.05 GB   136 ms/GB
//   KEEP_BIAS   triattn biasH      5.2      0.05      104
//   KEEP_FLASH  triattn avH/lseF  37.8      1.71       22
//
// and, for scale, every candidate that is NOT offered -- trimul's pre, its
// a/bb/t2, its t, its o -- prices at 5-7 ms/GB, so the list below is the whole
// of what is worth holding, not a subset of it.
//
// The order IS the priority: the stack grants tiers greedily, cheapest per
// byte first, and within a tier grants as many DEPTHS as the budget allows.
// Before this the keep was one all-or-nothing allocation, so a length that
// could not afford the 1.71 GB tier also lost the 0.05 GB ones that are six
// times better value -- at N=384 on a 24 GB card, which is exactly where the
// replay costs most.
// KEEP_ACTS is tri-mul's a/bb/gl/gr/t2 plus its input LayerNorm statistics:
// everything the replay would otherwise rebuild with lnFwd + projGate.  It is
// the WORST value per byte of the four and is ranked last, which is the whole
// point -- the greedy budget grants it only out of memory the better tiers
// and the model did not want.  45.9 ms for 5 x N^2 x I x 2 bytes a depth
// (5.1 GB at N=288, 8.1 at N=256... which is 6.7 ms/GB, the number the keep
// enumeration rejected these at when the question was "should this be kept
// always".  It is not always; it is when there is room.)
// KEEP_TMUL_T IS ITS OWN TIER, AND THE REASON IS THE GREEDY'S UNIT.  Measured
// at N=256 depth 48 by dropping each half (NO_KEEP_TIER=4, NE_KEEP_ACTS=0):
//
//   the five projection tensors   5.69 GB   52.7 ms   9.3 ms/GB
//   `t`, the contraction's output 1.61      13.7      8.5
//
// Inside ONE tier the budget can only trade whole DEPTHS, so a card with
// 5.69 GB spare had to take all six tensors for 78% of the depths (51.8 ms)
// when it could take the better five for all of them (52.7).  Ranked after
// KEEP_ACTS, `t` is simply dropped first.
//
// **That is worth about 2%, not the 8% a first estimate claimed.**  The
// estimate priced `t` at the 10.6 ms its bmm costs STANDALONE and got
// 6.6 ms/GB; in the stack it is 13.7 ms, so the two halves are much closer in
// value than they looked and the split mostly just makes the ranking correct
// and automatic rather than buying a lot.  It is kept for that: the decision
// is now made in the right unit, and the gap widens with N.
// KEEP_MSA IS THE TIER THE ENUMERATION MISSED, and the reason it missed it is
// worth keeping: every candidate it priced was PAIR-shaped, where everything
// costs N^2.  The three MSA modules are S*N-shaped and this model runs S=2.
// `BLK_PROF` on the stack's backward says they are replayed in full --
// msa_row 8.8 ms of a 8.9 ms forward, msa_col 2.4 of 2.5, msa_trans 0.8 of
// 1.7 -- because nothing of theirs was ever offered to the budget.  What
// their backwards actually read (grepped, not assumed: not `x`, not `pn`,
// not `nbb`, not `ag`, not `out`) is 6.3 MB a depth at N=256, so:
//
//   KEEP_MSA   the three msa modules   12.0 ms   0.35 GB   34 ms/GB
//
// which ranks third, behind stats and bias and ahead of flash.
//
// WHAT IS NOT HERE, AND WHY.  The outer product mean is replayed in full too
// (14.2 ms of 14.7) and skipping it needs `p1` -- in f32, because the pair
// chain's next link is an f32 add, and BLK_ONEPF aliases p1..p6 onto one
// buffer so the value is gone by the time the backward wants it.  That is
// 33.6 + 16.8 MB a depth for 14.2 ms, ~6 ms/GB, below everything on the
// board.  The pair transition is worse: its backward reads `h`, which is
// M*4C, 67 MB a depth, for 12.8 ms -- 3 ms/GB.
// KEEP_RELU is the PAIR transition's relu sign, one bit an element.  Its
// backward reads `h` for exactly one thing -- `dr *= (h > 0)` -- and `h` is
// M*4C, 67 MB a depth at N=256.  As bits it is 8.4, which buys two things at
// once: the replay skips `gemm1+relu` (mu/rstd are the only other thing the
// backward needs), and maskDx stops reading 44% of its traffic for one bit an
// element.  Priced end to end at -9.5 - 6 + 7.4 = -8 ms, the +7.4 being the
// pack, which only the SAVE forward pays.
//
//   KEEP_RELU   the pair transition   8.0 ms   0.23 GB   35 ms/GB
enum KeepTier { KEEP_STATS = 0, KEEP_BIAS = 1, KEEP_MSA = 2, KEEP_RELU = 3,
                KEEP_FLASH = 4, KEEP_ACTS = 5, KEEP_TMUL_T = 6, KEEP_NTIER = 7 };
// A TIER THAT MAY ONLY BE GRANTED WHOLE.  Partial grants are fine value per
// byte -- the flash tier at 11 of 48 depths measured 27 ms/GB, the same rate
// as a whole grant -- and that is the wrong unit at the top of the length
// range.  Measured at L=384 on a 24 GB card: letting the flash tier take
// whatever the budget left bought 16 ms of 3898 (0.4%) and cost 1.0 GB of the
// headroom that is the only thing standing between that length and the
// cudaMalloc failure this guard exists for, on an estimate read before JAX
// has grown into the backward.  The two cheap tiers are ~1% of the bytes and
// keep their partial grants.
inline bool keepWholeOnly(int t) { return t == KEEP_FLASH; }
inline const char* keepName(int t) {
  return t == KEEP_STATS ? "stats" : t == KEEP_BIAS ? "bias"
       : t == KEEP_MSA ? "msa" : t == KEEP_RELU ? "relu"
       : t == KEEP_FLASH ? "flash"
       : t == KEEP_ACTS ? "acts" : "trimul t";
}
// What a module offers: where the pointer lives, how big it is, which tier.
struct KeepSlot { void** slot; size_t bytes; int tier; };

struct Prof {
  // BLK_PROF=1 times the block's nine steps, =2 times the kernels INSIDE the
  // modules.  Not both at once: the step brackets enclose the kernel brackets,
  // so a combined report would count every kernel twice and give percentages
  // that sum to 200.
  int lvl = 0;
  const char* names[64];
  double acc[64] = {0};
  long cnt[64] = {0};
  cudaEvent_t a[64], b[64];
  int n = 0;

  void init() {
    const char* e = getenv("BLK_PROF");
    lvl = e ? atoi(e) : 0;
  }
  bool at(int l) const { return lvl == l; }
  int slot(const char* name) {
    for (int i = 0; i < n; ++i) if (names[i] == name) return i;
    if (n >= 64) { fprintf(stderr, "Prof: out of slots for %s\n", name); abort(); }
    names[n] = name;
    CU(cudaEventCreate(&a[n]));
    CU(cudaEventCreate(&b[n]));
    return n++;
  }
  void begin(int i, cudaStream_t st) { CU(cudaEventRecord(a[i], st)); }
  void end(int i, cudaStream_t st) {
    CU(cudaEventRecord(b[i], st));
    CU(cudaEventSynchronize(b[i]));
    float ms = 0; CU(cudaEventElapsedTime(&ms, a[i], b[i]));
    acc[i] += ms; cnt[i] += 1;
  }
  void report(const char* tag) {
    if (!lvl) return;
    double tot = 0;
    for (int i = 0; i < n; ++i) tot += acc[i];
    fprintf(stderr, "\nBLK_PROF %s   total %8.1f ms\n", tag, tot);
    for (int i = 0; i < n; ++i)
      fprintf(stderr, "  %-22s %8.1f ms  %5.1f%%  n=%ld\n", names[i], acc[i],
              100.0 * acc[i] / tot, cnt[i]);
  }
  void reset() { for (int i = 0; i < n; ++i) { acc[i] = 0; cnt[i] = 0; } }
  // Every caller that reports also resets, so anything still accumulated at
  // exit was never reported -- which is the case for all six MODULE drivers.
  // They have had PROF_STEP brackets all along and silently dropped them.
  ~Prof() {
    double t = 0; for (int i = 0; i < n; ++i) t += acc[i];
    if (lvl && t > 0) report("at exit");
  }
};

// A WMMA FRAGMENT SHOULD BE CONTIGUOUS, and it costs nothing to make it so.
// `tmul::projGateK` loads its weight fragment straight out of the [C][PRE] matrix
// with ldm = PRE, which is 16 rows of 32 bytes 1280 bytes apart -- 16 sectors
// a fragment, 40 fragments a warp.  Replacing the stride with a contiguous
// read (a deliberately wrong-math control) measured 0.473 -> 0.418 ms, 12%,
// so the layout is worth having for real.  This builds it once a depth:
//
//     wcatG[((g * BLKS + w) * K + k) * 16 + c] = wcat[k * PRE + g * I + w * 16 + c]
//
// which is a permutation -- same bytes, same count -- and lets the kernel load
// with ldm = 16.
// TM_FRAGPERM=0 restores the strided fragment load everywhere, for the A/B.
#ifndef TM_FRAGPERM
#define TM_FRAGPERM 1
#endif
__global__ void permuteFragK(const __nv_bfloat16* __restrict__ src,
                             __nv_bfloat16* __restrict__ dst, int K, int W,
                             int off, int span) {
  // src is [K][W] row-major; dst takes the `span` columns from `off` and lays
  // them out as [span/16 blocks][K][16]
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)K * span) return;
  int k = (int)(i / span), j = (int)(i % span);
  int blk = j / 16, c = j % 16;
  dst[((size_t)blk * K + k) * 16 + c] = src[(size_t)k * W + off + j];
}

// ---------------------------------------------------------------- dropout
//
// AF2's Evoformer dropout, which the port assumed away until now (the comment
// below used to say "with rate 0").  Five modules carry it and every one uses
// a SHARED mask -- apply_dropout zeroes a broadcast dimension, so the mask is
// not per element:
//
//   msa_row_attention_with_pair_bias  0.15  per_row     mask [1, N, C]
//   triangle_attention_starting_node  0.25  per_row     mask [1, N, C]
//   triangle_attention_ending_node    0.25  per_column  mask [N, 1, C]
//   triangle_multiplication_out / in  0.25  per_row     mask [1, N, C]
//
// It scales the module's DELTA before the residual add, which is exactly
// where these kernels already have a seam.
//
// THE MASK IS REGENERATED, NOT STORED.  A counter-based hash of
// (seed, depth, module, index) gives the same mask in the forward and the
// backward for free, where storing it would cost an N*C buffer per module per
// depth -- and the backward needs precisely the same mask, since dropout's
// VJP is the same scaling.
//
// IT DOES NOT REPRODUCE JAX'S MASKS.  jax.random.bernoulli is threefry over a
// split key chain; matching it bit for bit would mean reimplementing that
// chain exactly.  So a dropout run cannot be compared element-wise against
// stock -- only in distribution, and a design run is stochastic anyway.
__device__ __forceinline__ uint32_t dropHash(uint32_t a, uint32_t b,
                                             uint32_t c, uint32_t d) {
  uint32_t h = a * 0x9e3779b9u;
  h ^= b + 0x85ebca6bu + (h << 6) + (h >> 2);
  h ^= c + 0xc2b2ae35u + (h << 6) + (h >> 2);
  h ^= d + 0x27d4eb2fu + (h << 6) + (h >> 2);
  h ^= h >> 15; h *= 0x2c1b3c6du;
  h ^= h >> 12; h *= 0x297a2d39u;
  h ^= h >> 15;
  return h;
}

// THE SEED IS A DEVICE VALUE, not a kernel argument.  A design loop reseeds
// every step; as a launch argument that is free, but the FFI boundary above
// would have had to carry it as a static attribute, and a new attribute value
// is a new XLA computation -- a recompile of the whole stack, per step.  As a
// one-word buffer it is a memcpy instead.
// keep ? 1/keep_rate : 0 -- the scale apply_dropout applies
// THE WORD AFTER THE SEED IS AN ENABLE FLAG, and it has to be here rather
// than in the `dropout` attribute next to it.  BindCraft2 flips dropout
// BETWEEN STAGES of one trajectory -- hardening and mutation run without it,
// the earlier stages with -- and it does so by tracing `use_dropout` into the
// batch, so stock pays nothing for the flip.  As an attribute it would be a
// different XLA computation each way, i.e. a full recompile of the 48-block
// stack at every stage boundary.  The attribute now only decides whether the
// scratch is ALLOCATED; whether the masks apply is this word.
__device__ __forceinline__ float dropScale(const uint32_t* seedp,
                                           uint32_t depth, uint32_t mod,
                                           uint32_t idx, float rate) {
  uint32_t seed = seedp[0];
  if (rate <= 0.f || seedp[1] == 0u) return 1.f;
  float keep_rate = 1.f - rate;
  float u = dropHash(seed, depth, mod, idx) * 2.3283064365386963e-10f;  // /2^32
  return u < keep_rate ? 1.f / keep_rate : 0.f;
}

// dst = src + drop(delta).  `cols` is the mask's fastest axis (C), and `unit`
// selects which index the mask varies over: the delta's row for per_column,
// its column for per_row.  PER_ROW here means the mask is shared ACROSS rows,
// which is the opposite of what the name first suggests -- apply_dropout sets
// shape[broadcast_dim] = 1, so per_row (broadcast_dim 0) shares over dim 0.
template <bool PER_ROW, bool TRANSPOSE_DELTA = false>
__global__ void addDropNarrowBK(__nv_bfloat16* __restrict__ dst,
                                const __nv_bfloat16* __restrict__ src,
                                const __nv_bfloat16* __restrict__ delta,
                                int N, int C, const uint32_t* __restrict__ seedp, uint32_t depth,
                                uint32_t mod, float rate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t c, rc;
  if ((C & (C - 1)) == 0) {
    c = (uint32_t)i & (uint32_t)(C - 1);
    rc = (uint32_t)(i >> (__ffs(C) - 1));
  } else {
    c = (uint32_t)(i % (size_t)C);
    rc = (uint32_t)(i / (size_t)C);
  }
  uint32_t col = rc % (uint32_t)N;
  uint32_t row = (PER_ROW && !TRANSPOSE_DELTA) ? 0u : (rc / (uint32_t)N);
  uint32_t idx = (PER_ROW ? col : row) * (uint32_t)C + c;
  float s = dropScale(seedp, depth, mod, idx, rate);
  size_t di = TRANSPOSE_DELTA ? ((size_t)col * N + row) * C + c : i;
  dst[i] = __float2bfloat16(__bfloat162float(src[i]) +
                            s * __bfloat162float(delta[di]));
}

// the bf16 form, which is what the block's chain actually carries: a module's
// backward templates its cotangent and its accumulator on the SAME type, so a
// float scratch cannot be paired with a bf16 accumulator.
// THE STRUCTURE MODULE'S DROPOUT IS NOT THE EVOFORMER'S.  AF2 applies it to
// the WHOLE act after the residual add (folding_multimer.FoldIteration, after
// the IPA residual and after the transition residual), with broadcast_dim
// None -- so the mask is elementwise over [N, C] and shares nothing, where
// every Evoformer site scales a module DELTA before the residual and shares
// the mask across rows or columns.  Same hash, different shape of use, so it
// gets its own kernel rather than a fourth template parameter on dropScaleK.
__global__ void dropFlatK(float* __restrict__ dst, const float* __restrict__ src,
                          const uint32_t* __restrict__ seedp, uint32_t depth,
                          uint32_t mod, float rate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = src[i] * dropScale(seedp, depth, mod, (uint32_t)i, rate);
}

template <bool PER_ROW, bool TRANSPOSE_OUT = false>
__global__ void dropScaleBK(__nv_bfloat16* __restrict__ dst,
                            const __nv_bfloat16* __restrict__ src, int N, int C,
                            const uint32_t* __restrict__ seedp, uint32_t depth, uint32_t mod,
                            float rate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t c, rc;
  if ((C & (C - 1)) == 0) {
    c = (uint32_t)i & (uint32_t)(C - 1);
    rc = (uint32_t)(i >> (__ffs(C) - 1));
  } else {
    c = (uint32_t)(i % (size_t)C);
    rc = (uint32_t)(i / (size_t)C);
  }
  uint32_t col = rc % (uint32_t)N;
  uint32_t row = (PER_ROW && !TRANSPOSE_OUT) ? 0u : (rc / (uint32_t)N);
  uint32_t idx = (PER_ROW ? col : row) * (uint32_t)C + c;
  size_t di = TRANSPOSE_OUT ? ((size_t)col * N + row) * C + c : i;
  dst[di] = __float2bfloat16(__bfloat162float(src[i]) *
                             dropScale(seedp, depth, mod, idx, rate));
}

// the same mask on the way back: the cotangent on the DELTA is mask * dout,
// while the residual's passthrough is unscaled -- so this writes a scaled COPY
// and the module's backward accumulates from it into the unscaled dp.
template <bool PER_ROW>
__global__ void dropScaleK(float* __restrict__ dst, const float* __restrict__ src,
                           int N, int C, const uint32_t* __restrict__ seedp, uint32_t depth,
                           uint32_t mod, float rate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t c, rc;
  if ((C & (C - 1)) == 0) {
    c = (uint32_t)i & (uint32_t)(C - 1);
    rc = (uint32_t)(i >> (__ffs(C) - 1));
  } else {
    c = (uint32_t)(i % (size_t)C);
    rc = (uint32_t)(i / (size_t)C);
  }
  uint32_t col = rc % (uint32_t)N;
  uint32_t row = PER_ROW ? 0u : (rc / (uint32_t)N);
  uint32_t idx = (PER_ROW ? col : row) * (uint32_t)C + c;
  dst[i] = src[i] * dropScale(seedp, depth, mod, idx, rate);
}

// dst = src + delta, the residual add when the rate IS zero (see
// addDropNarrowBK above for the rest).

// The msa residual's dropped twin.  The msa is [S, N, C] and AF2 asks for
// `per_row` here, i.e. shared_dropout along axis 0 -- one mask over (residue,
// channel) reused for every sequence in the stack, which is exactly the
// PER_ROW index addDropNarrowBK uses for the pair.
template <bool PER_ROW>
__global__ void addDropNarrowK(float* __restrict__ dst,
                               __nv_bfloat16* __restrict__ dstb,
                               const float* __restrict__ src,
                               const __nv_bfloat16* __restrict__ delta,
                               int N, int C, const uint32_t* __restrict__ seedp, uint32_t depth,
                               uint32_t mod, float rate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t c, rc;
  if ((C & (C - 1)) == 0) {
    c = (uint32_t)i & (uint32_t)(C - 1);
    rc = (uint32_t)(i >> (__ffs(C) - 1));
  } else {
    c = (uint32_t)(i % (size_t)C);
    rc = (uint32_t)(i / (size_t)C);
  }
  uint32_t col = rc % (uint32_t)N;
  uint32_t row = PER_ROW ? 0u : (rc / (uint32_t)N);
  uint32_t idx = (PER_ROW ? col : row) * (uint32_t)C + c;
  float s = dropScale(seedp, depth, mod, idx, rate);
  float v = src[i] + s * __bfloat162float(delta[i]);
  dst[i] = v;
  dstb[i] = __float2bfloat16(v);
}

__global__ void addK(__nv_bfloat16* __restrict__ dst,
                     const __nv_bfloat16* __restrict__ src,
                     const __nv_bfloat16* __restrict__ delta, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(__bfloat162float(src[i]) +
                                       __bfloat162float(delta[i]));
}

// THE RESIDUAL CHAIN IS FLOAT32, because AF2's is.  Under
// global_config.bfloat16 only the activations entering the first module are
// bf16: the weights stay float32, so every einsum promotes and a module
// RETURNS float32 (see the README).  Keeping the chain in bf16 re-rounded the
// whole activation at each of the nine steps and at each of the 48 blocks, and
// that single term dominated the port's error -- one bf16 rounding is 1.95e-03
// relative, and one module measured 1.53e-03 against an oracle's 2.06e-04.
// The module's DELTA stays bf16: it is small beside the activation it is added
// to, so its own rounding is diluted, and keeping it bf16 leaves every module's
// internals (and its GEMMs) untouched.
__global__ void addF32K(float* __restrict__ dst, const float* __restrict__ src,
                        const __nv_bfloat16* __restrict__ delta, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i] + __bfloat162float(delta[i]);
}

// the bf16 view a module reads.  Its LayerNorm sees the same 8-bit mantissa it
// always did; what changed is that the value it is added back to is exact.
// THE BLOCK INPUT ARRIVES IN bf16 NOW, so the three kernels that read it are
// templated on the source type.  `src` is p0 -- the stack's save-set slice --
// and everything else they touch stays f32.  widen() is the only difference
// between the two instantiations.
__device__ __forceinline__ float widen(float v) { return v; }
__device__ __forceinline__ float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

// cudaMemsetAsync IS NOT FAST HERE.  The triangle attention's f32 dK/dV
// accumulators are zeroed at the top of every backward -- 6.54 GB over a
// fwd+bwd at N=256 -- and PROBE_NOZERO prices that at 26.3 ms, which is
// 249 GB/s.  For a WRITE-ONLY stream on a card whose read+write roof is 489
// that is about half what it should be.  A vectorised store does the same
// job: 16 bytes a thread, nothing else.
__global__ void zero16K(float4* __restrict__ p, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) p[i] = make_float4(0.f, 0.f, 0.f, 0.f);
}
inline void zeroFast(void* p, size_t bytes, cudaStream_t st) {
  if (bytes % 16) { CU(cudaMemsetAsync(p, 0, bytes, st)); return; }
  size_t n4 = bytes / 16;
  zero16K<<<(unsigned)((n4 + 255) / 256), 256, 0, st>>>((float4*)p, n4);
}
__global__ void widenBf16K(float* __restrict__ dst,
                           const __nv_bfloat16* __restrict__ src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void toBf16K(__nv_bfloat16* __restrict__ dst,
                        const float* __restrict__ src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// ROUND IN PLACE TO bf16 PRECISION, staying f32.
//
// AF2 multimer runs its whole embedder region under
// `utils.bfloat16_context()` -- global_config.bfloat16 is True -- so every
// Linear, every LayerNorm and every `+=` between the input features and the
// extra-MSA stack lands on a bf16 grid.  A port that keeps f32 there is
// MORE accurate and still WRONG as a drop-in: the trunk amplifies the
// difference, and bc2's own native embedder hook rounds its f32 result back
// (`msa_activations.astype(dtype)`) for exactly this reason.
__global__ void roundBf16K(float* __restrict__ x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __bfloat162float(__float2bfloat16(x[i]));
}

// THE ADD AND THE NARROWING ARE ONE KERNEL.  Every module is handed a bf16
// view of the f32 activation, and doing that in its own pass costs a 33.6 MB
// read plus a 16.8 MB write every time -- in the forward, again in the
// backward's replay, and again in the backward itself.  Writing the view here,
// out of a value already in registers, costs one extra 16.8 MB write and
// retires the read entirely; and because the replay writes the views too, the
// backward needs no conversion at all.  33.3 GB of traffic a stack becomes
// 7.4 GB.
// MEASURED AND REJECTED: four elements a thread.  These three kernels are the
// residual chain and they are ~48 ms of a 286 ms forward, one element a
// thread -- four memory instructions for twelve bytes.  A float4 + two bf16x2
// form, which quarters the instruction count, measured 1004.0 against the
// scalar path's 1006.3 ms on the whole stack: nothing.  Unlike the fused
// kernels above, this chain really is at the bandwidth roof (100.8 MB a call
// at ~450 GB/s), so there was no issue rate to recover.  Worth knowing before
// anyone vectorises it again.
// NAN_TRACE's counter: how many non-finite entries a buffer holds.  A
// diagnostic, so it syncs -- never on a measured path.
inline bool nanTrace() {
  static const bool on = envOn("NAN_TRACE");
  return on;
}

__global__ void countBadK(const float* a, unsigned* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !isfinite(a[i])) atomicAdd(out, 1u);
}
__global__ void countBadBfK(const __nv_bfloat16* a, unsigned* out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && !isfinite(__bfloat162float(a[i]))) atomicAdd(out, 1u);
}

// ------------------------------------------------- dynamic loss scaling
// The MSA attention's backward runs in f16, whose max is 65504, and it is
// LINEAR in its incoming cotangent -- so scaling that cotangent by a power of
// two and unscaling the gradients afterwards is exact except for range.  A
// power of two touches only the exponent, which is why this is a range fix and
// not a numerics change.
//
// WHY IT IS NEEDED: the MONOMER network amplifies the msa gradient far harder
// than the multimer one.  Walking the 48 blocks backward at N=256 S=1, rms(dm)
// reaches 18.7 by block 2 on multimer weights and 139 on ptm weights -- 7.5x --
// and the f16 attention backward overflows to NaN two blocks later.
//
// Done on the device end to end: reading the max on the host would mean a
// stream sync per block per backward, which is what this port spends its
// effort avoiding.
__global__ void maxAbsHalfK(const half* __restrict__ a, unsigned* out, size_t n) {
  __shared__ unsigned sm;
  if (threadIdx.x == 0) sm = 0u;
  __syncthreads();
  unsigned local = 0u;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    // for a NON-NEGATIVE float the IEEE bit pattern is monotonic, so an
    // integer max over |x| is a float max over |x|
    float v = fabsf(__half2float(a[i]));
    if (!isfinite(v)) v = 0.f;        // an inf already here is not ours to fix
    local = max(local, __float_as_uint(v));
  }
  atomicMax(&sm, local);
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(out, sm);
}

// s[0] = 2^k brings `mx` to about `target`; s[1] = 2^-k undoes it
__global__ void pickScaleK(const unsigned* mx, float* s, float target) {
  float m = __uint_as_float(*mx);
  if (!(m > 0.f) || !isfinite(m)) { s[0] = 1.f; s[1] = 1.f; return; }
  int e, want;
  frexpf(m, &e);
  frexpf(target, &want);
  s[0] = ldexpf(1.0f, want - e);
  s[1] = ldexpf(1.0f, e - want);
}

__global__ void scaleHalfByK(half* a, const float* s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __float2half(__half2float(a[i]) * s[0]);
}

__global__ void scaleFloatByK(float* a, const float* s, int which, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= s[which];
}

// CT_SCALE's helper in stack.cu: the backward is linear in the cotangent, so
// cos is invariant under this and only the dynamic range moves -- which is how
// to tell a wrong gradient from one that overflowed.
__global__ void scaleK(float* a, float f, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] *= f;
}

// PROBE_BF16CHAIN: THE PRICE OF THE f32 RESIDUAL CHAIN, measured rather than
// assumed.  `blk residual` is 42.1 ms in the forward AND 42.1 in the
// backward's replay -- 9.5% of a fwd+bwd -- and it sits at 94% of this card's
// bandwidth roof, so the only lever is BYTES.  Each link reads `src` in f32
// (33.6 MB at N=256) and writes `dst` in f32 (33.6) plus the bf16 narrow
// (16.8), against a `delta` of 16.8: 100.8 MB, two thirds of it the f32
// chain.  A bf16-only chain would move 50.4.
//
// This probe reads the narrow in place of the f32 source and skips the f32
// store, so the traffic is a bf16 chain's and the math is WRONG for anything
// downstream that still reads f32 (the block's output copy, the OPM's add).
// The delta is therefore the upper bound on what dropping the f32 chain is
// worth, before asking what it costs the gradient.
#ifndef PROBE_BF16CHAIN
#define PROBE_BF16CHAIN 0
#endif
// BLK_BF16CHAIN is the REAL version of that probe: the within-block pair
// residual chain carries bf16 instead of f32.  The chain BETWEEN blocks is
// already bf16 -- the save-set is -- so this is 5 roundings a block where
// there was 1, not a new kind of error.
// ON BY DEFAULT, and decided against a DESIGN STEP rather than the oracle --
// the same question the save-set's bf16 was re-asked with, for the same
// reason: a stack that is 0.978 against its own oracle has no exact
// trajectory to deviate from, so the oracle number is not the one that
// matters.  8 seeds, step-1 metrics, L=224 PD1 binder:
//
//                            median      p90
//   native vs XLA           1.87e-02   8.36e-02     <- what the port already moves
//   bf16 chain vs native    2.25e-03   1.74e-02     <- what this adds
//
// 8.3x smaller than the deviation already there, and ~10x smaller on every
// individual metric including rg (3.91e-02 against 3.08e-01).  It costs the
// ORACLE plenty -- out/pair relL2 1.669e-02 -> 3.357e-02, grad/pair 0.9782 ->
// 0.9755 -- which is exactly the reading that would have rejected it.
//
// NE_BF16CHAIN=0 switches it off.
static const bool BLK_BF16CHAIN = envOnDef("NE_BF16CHAIN", true);
// `srcb` is `src`'s bf16 view, and under BLK_BF16CHAIN it IS the chain: the
// f32 store goes away and the read halves.  Callers that have no narrow of
// their source (the msa chain, which is 0.5 MB and not worth changing) pass
// nullptr and keep the f32 path whatever the flag says.
template <bool TRANSPOSE_DELTA = false>
__global__ void addNarrowBK(__nv_bfloat16* __restrict__ dstb,
                            const __nv_bfloat16* __restrict__ srcb,
                            const __nv_bfloat16* __restrict__ delta, size_t n,
                            int N = 0, int C = 0) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  size_t di = i;
  if (TRANSPOSE_DELTA) {
    int c = (int)(i % C);
    size_t rc = i / C;
    int col = (int)(rc % N), row = (int)(rc / N);
    di = ((size_t)col * N + row) * C + c;
  }
  dstb[i] = __float2bfloat16(__bfloat162float(srcb[i]) +
                             __bfloat162float(delta[di]));
}
__global__ void addNarrowK(float* __restrict__ dst, __nv_bfloat16* __restrict__ dstb,
                           const float* __restrict__ src,
                           const __nv_bfloat16* __restrict__ delta, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (PROBE_BF16CHAIN) {
    // same instruction count, a bf16 chain's bytes
    float v = __bfloat162float(dstb[i]) + __bfloat162float(delta[i]);
    dstb[i] = __float2bfloat16(v);
    return;
  }
  float v = src[i] + __bfloat162float(delta[i]);
  dst[i] = v;
  dstb[i] = __float2bfloat16(v);
}

// THE RESIDUAL ADD ABSORBS TRIANGLE MULTIPLICATION'S OUTPUT GATE.  That module
// ended with out = o * t2 (t2 already holds the sigmoid) and the block then
// added `out` to the running activation -- so `out` was written once and read
// once, 12.8 MB each way, and never looked at again.  Reading o and t2 here
// instead retires it: 25.6 MB a call, and the call happens 96 times in a
// forward and 96 more in the backward's replay.
// THE TWO GATES ARE SIGMOIDS, SO THEY DO NOT NEED 16 BITS.  gl and gr are
// written by projGateK purely for the backward -- the forward folds them into
// `a`/`bb` and never reads them again -- and they live in [0, 1], where a
// uint8 is uniform to 1/255 while bf16's 8-bit mantissa is 1/256 RELATIVE,
// i.e. coarser than the uint8 above 0.5 and finer below it.  They are two of
// the six tensors in the KEEP_ACTS tier, so this is 1.61 GB of 9.72 at N=256
// depth 48, at no cost in time: the loads stay 4-byte (four gates a uchar4
// where there were two in a bf16x2).  TM_GATE8=0 restores bf16.
#ifndef TM_GATE8
#define TM_GATE8 1
#endif
// t2 IS A SIGMOID TOO -- it is the OUTPUT gate, written by the same projection
// and read by outGateK as `out = o * t2`.  Same argument as gl/gr, with one
// difference that matters: gl/gr are read only by the backward, so narrowing
// them cannot touch the forward, while t2 multiplies the module's output.
// Its own flag, so the forward cost is measurable on its own.
#ifndef TM_GATE8T
#define TM_GATE8T 1
#endif
#if TM_GATE8T
typedef unsigned char gt2_t;
#else
typedef __nv_bfloat16 gt2_t;
#endif
__device__ __forceinline__ float gt2Load(gt2_t v) {
#if TM_GATE8T
  return (float)v * (1.f / 255.f);
#else
  return __bfloat162float(v);
#endif
}
#if TM_GATE8
typedef unsigned char gate_t;
__device__ __forceinline__ float gateLoad(gate_t v) { return (float)v * (1.f / 255.f); }
__device__ __forceinline__ gate_t gateStore(float g) {
  return (gate_t)__float2uint_rn(__saturatef(g) * 255.f);
}
#else
typedef __nv_bfloat16 gate_t;
__device__ __forceinline__ float gateLoad(gate_t v) { return __bfloat162float(v); }
__device__ __forceinline__ gate_t gateStore(float g) { return __float2bfloat16(g); }
#endif

__global__ void addGateNarrowK(float* __restrict__ dst,
                               __nv_bfloat16* __restrict__ dstb,
                               const float* __restrict__ src,
                               const __nv_bfloat16* __restrict__ o,
                               const gt2_t* __restrict__ t2, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // THE PRODUCT IS ROUNDED TO bf16 BEFORE THE ADD, and that is not an
  // oversight to clean up later -- it is what AF2 does.  Leaving it in f32 is
  // strictly more accurate arithmetic and measures 58% WORSE against the
  // oracle (stack forward 1.096e-02 against 6.942e-03), while rounding
  // reproduces the unfused path to the last digit.  So triangle
  // multiplication's delta really is bf16 in the model, whatever
  // dropout_wrapper's `output_act + residual` looks like in float32.
  float prod = __bfloat162float(o[i]) * gt2Load(t2[i]);
  float v = src[i] + __bfloat162float(__float2bfloat16(prod));
  dst[i] = v;
  dstb[i] = __float2bfloat16(v);
}

// THE RESIDUAL ADD ABSORBS THE OUTER PRODUCT MEAN'S BIAS AND NORMALISE.  That
// module ended with two full passes over its output -- out += bias, then
// out /= the per-(i,j) count of sequences present in both rows -- and the
// block then added `out` to the running activation.  Three reads and three
// writes of 12.8 MB where one read suffices.
//
// BOTH bf16 ROUNDINGS ARE KEPT, deliberately.  Computing bias-then-divide in
// one f32 expression is better arithmetic and would NOT match: the same
// mistake, made and measured on triangle multiplication's output gate, cost
// 58% on the stack's forward.  Matching the model beats being accurate.
__global__ void addGateNarrowBK(__nv_bfloat16* __restrict__ dstb,
                                const __nv_bfloat16* __restrict__ srcb,
                                const __nv_bfloat16* __restrict__ o,
                                const gt2_t* __restrict__ t2, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // the product is rounded to bf16 before the add, exactly as the f32 path
  // does -- see addGateNarrowK for why that is AF2's arithmetic and not ours
  float prod = __bfloat162float(o[i]) * gt2Load(t2[i]);
  prod = __bfloat162float(__float2bfloat16(prod));
  dstb[i] = __float2bfloat16(__bfloat162float(srcb[i]) + prod);
}

template <bool PER_ROW>
__global__ void addDropGateNarrowBK(__nv_bfloat16* __restrict__ dstb,
                                    const __nv_bfloat16* __restrict__ srcb,
                                    const __nv_bfloat16* __restrict__ o,
                                    const gt2_t* __restrict__ t2,
                                    int N, int C, const uint32_t* __restrict__ seedp,
                                    uint32_t depth, uint32_t mod, float rate, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t c, rc;
  if ((C & (C - 1)) == 0) {
    c = (uint32_t)i & (uint32_t)(C - 1);
    rc = (uint32_t)(i >> (__ffs(C) - 1));
  } else {
    c = (uint32_t)(i % (size_t)C);
    rc = (uint32_t)(i / (size_t)C);
  }
  uint32_t col = rc % (uint32_t)N;
  uint32_t row = PER_ROW ? 0u : (rc / (uint32_t)N);
  uint32_t idx = (PER_ROW ? col : row) * (uint32_t)C + c;
  float s = dropScale(seedp, depth, mod, idx, rate);
  float prod = __bfloat162float(o[i]) * gt2Load(t2[i]);
  prod = __bfloat162float(__float2bfloat16(prod));
  dstb[i] = __float2bfloat16(__bfloat162float(srcb[i]) + s * prod);
}

// THE NORMALISER IS PER (r, c) AND WAS COMPUTED PER ELEMENT.  `nz` depends on
// the mask and on (r, c) alone, so the old form recomputed it for all CZ=128
// channels of every pair element -- and identically at all 48 depths, with
// the mask loads to match.  Worse, reaching (r, c) from a flat index cost
// THREE runtime integer divisions (`i / CZ_`, `ij / N`, `ij % N`) and `bias`
// a fourth (`i % CZ_`), on 8.4M elements a call.  That is why this kernel sat
// at 251 GB/s, 51% of roof, where its neighbours run at 87-94%.
//
// `opmNzK` builds nz once over N^2 elements, and CZ is a template parameter
// now, so the channel index is a shift and a mask.  Same nz, same two bf16
// roundings, same order: bit-identical.
__global__ void opmNzK(const float* __restrict__ mask, float* __restrict__ nz,
                       int N, int S) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)N * N) return;
  int r = (int)(ij / N), c = (int)(ij % N);
  float v = 0.f;
  for (int t = 0; t < S; ++t) v += mask[(size_t)t * N + r] * mask[(size_t)t * N + c];
  nz[ij] = v;
}
template <class SRC, int CZT>
__global__ void addOpmNarrowK(float* __restrict__ dst,
                              __nv_bfloat16* __restrict__ dstb,
                              const SRC* __restrict__ src,
                              const __nv_bfloat16* __restrict__ o,
                              const float* __restrict__ bias,
                              const float* __restrict__ nzbuf,
                              float eps, size_t n,
                              bool chain_bf16 = false) {
  static_assert((CZT & (CZT - 1)) == 0, "CZ a power of two, for the shift");
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float nz = nzbuf[i / CZT];             // CZT is constant: a shift
  float v = __bfloat162float(o[i]) + bias[i & (CZT - 1)];
  v = __bfloat162float(__float2bfloat16(v));            // addBiasK's rounding
  v = __bfloat162float(__float2bfloat16(v / (eps + nz)));  // normDivK's
  float w = widen(src[i]) + v;
  if (!chain_bf16 && dst) dst[i] = w; // host-side flag, device code cannot
                                      // read the `static const bool`
  dstb[i] = __float2bfloat16(w);
}

// THE EXTRA-MSA STACK'S OUTER PRODUCT MEAN, WHICH IS A CONSTANT IN DESIGN.
// `extra_msa_mask` is all zeros there (colabdesign2/af2/features.py), so that
// stack's left and right projections are `mask * Linear(...)` = 0 and
// addOpmNarrowK above reduces to `bias / eps` with nz = 0 -- NOT to zero, which
// is the trap: at eps = 1e-3 it is a per-channel constant of rms 5.04 and
// absmax 34.2 against pair activations of rms ~63. `k` is that constant,
// precomputed per depth with addOpmNarrowK's own two bf16 roundings already
// applied, so this kernel and that one agree bit for bit on a zero mask.
template <class SRC, int CZT = 128>
__global__ void addConstNarrowK(float* __restrict__ dst,
                                __nv_bfloat16* __restrict__ dstb,
                                const SRC* __restrict__ src,
                                const float* __restrict__ k, int CZ_, size_t n,
                                bool chain_bf16 = false) {
  static_assert((CZT & (CZT - 1)) == 0, "CZ a power of two, for the mask");
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float w = widen(src[i]) + k[i & (CZT - 1)];
  if (!chain_bf16 && dst) dst[i] = w;
  dstb[i] = __float2bfloat16(w);
}

// bias -> addOpmNarrowK's value at o = 0 and nz = 0, once a depth
__global__ void opmConstK(const float* __restrict__ bias, float* __restrict__ k,
                          float eps, int CZ_) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= CZ_) return;
  float v = __bfloat162float(__float2bfloat16(bias[c]));
  k[c] = __bfloat162float(__float2bfloat16(v / eps));
}

// the same for the prefix-ladder's pass-through branches
template <class SRC>
__global__ void copyNarrowK(float* __restrict__ dst, __nv_bfloat16* __restrict__ dstb,
                            const SRC* __restrict__ src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = widen(src[i]);
  if (dst) dst[i] = v;
  dstb[i] = __float2bfloat16(v);
}

// THE GRADIENT CHAIN IS bf16, and these let one kernel serve both widths.
// dp (the pair cotangent) is bf16 because a probe showed it costs nothing --
// grad/pair 1.111/1.043/1.067e-01 against f32's 1.112/1.121/1.105, inside the
// band the dnbb atomics already impose -- and halves its traffic across seven
// touch points a block.  dm (the msa cotangent) stays f32: it is 0.46 MB, so
// there is nothing to win, and the modules that write it are shared with the
// pair track only through `transition`, which is templated on the width.
__device__ __forceinline__ float toF(float x) { return x; }
__device__ __forceinline__ float toF(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ void stF(float* p, float v) { *p = v; }
__device__ __forceinline__ void stF(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// A PROBE: round an f32 buffer through bf16 in place.  Used to price a bf16
// gradient chain without building one -- the real change would halve dp's
// traffic (~21 ms) but risks SWAMPING, where a module's contribution is lost
// because it is under 2^-8 of the running total.  Measuring beats guessing.
// round a bf16 tensor to 8-bit precision in place, at a fixed scale, so an
// int8 keep tier can be PRICED without being written -- the same trick
// SAVE_BF16 uses for the save-set.  See the TM_Q8 note in trimul.cuh.
__global__ void quant8InPlaceK(__nv_bfloat16* __restrict__ x, float s, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __bfloat162float(x[i]) / s;
  v = fminf(127.f, fmaxf(-127.f, rintf(v)));
  x[i] = __float2bfloat16(v * s);
}

__global__ void roundInPlaceK(float* __restrict__ x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __bfloat162float(__float2bfloat16(x[i]));
}

__global__ void copyF32K(float* __restrict__ dst, const float* __restrict__ src,
                         size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

// acc += d, the residual's other half: the cotangent flows BOTH into the module
// and past it, so the input's gradient is the sum.
__global__ void accK(float* __restrict__ acc, const float* __restrict__ d,
                     size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += d[i];
}

// One profiler a process.  A module reaches it without being handed anything,
// which is the only way to instrument kernels several layers down a call that
// the stack makes 48 times.
inline Prof& prof() { static Prof p; static bool once = (p.init(), true); (void)once; return p; }

#define PROF_STEP(NAME, ...)                                                   \
  do { if (!prof().at(2)) { __VA_ARGS__; } else {                              \
         int _s = prof().slot(NAME); prof().begin(_s, st);                      \
         __VA_ARGS__; prof().end(_s, st); } } while (0)
