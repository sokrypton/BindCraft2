#include <set>
// The CUDA Evoformer stack as an XLA custom call, so JAX can run it.
//
// Two handlers, one a VJP of the other:
//
//   native_evoformer_fwd(pair, msa, pair_mask, msa_mask)
//       -> (pair, msa, psave, msave)
//   native_evoformer_bwd(ct_pair, ct_msa, pair_mask, msa_mask, psave, msave)
//       -> (dpair, dmsa)
//
// WEIGHTS ARE NOT ARGUMENTS.  Design freezes them, so they are loaded once from
// the manifest `weights.py` writes and kept on the device for the process's
// life.  Passing 85 parameters x 48 depths through the FFI would mean 4080
// buffers, or a packing convention on both sides, to express something that
// never changes during a design run.
//
// BOTH CALLS ARE PURE.  The remat save-set -- each depth's input, ~830 MB at
// N=256 and depth 48 -- crosses as a real output of the forward and a real
// input to the backward, written in place by pointing the Stack's save
// buffers at XLA's own allocation.  Holding it in the Stack instead and
// having the backward reach for it was the first design, and JAX's async
// dispatch broke it within a minute: a second forward overtook the first
// backward, which is a wrong gradient rather than an error.  Carrying only
// the inputs across would be pure too, but would cost the backward an extra
// full forward (~400 ms of 1500).
//
//   nvcc -O3 -std=c++17 -arch=sm_86 --compiler-options -fPIC -shared \
//        -I. -I$(python -c 'import jax.ffi;print(jax.ffi.include_dir())') \
//        ffi.cu -lcublas -lcublasLt -o libnative_evoformer.so
#include <mutex>

#include "stack.cuh"
#include "tplstack.cuh"
#include "tplembed.cuh"
#include "seqfeat.cuh"
#include "inputfeat.cuh"
#include "confidence.cuh"
#include "kabsch.cuh"
#include "res_tables.cuh"   // RES_CHI_PI_PERIODIC: the monomer 1-D row's
                            // only table the callers do not pass
#include "design_step.cuh"
#include "prod_losses.cuh"
#include "act_const.cuh"
#include "structmod.cuh"
#include "embed.cuh"
#include "heads.cuh"
#if NE_CUPTI
#include "cupti_prof.cuh"
#endif
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

__global__ void bf16ToF32OutK(const __nv_bfloat16* __restrict__ a,
                              float* __restrict__ b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __bfloat162float(a[i]);
}

// One Stack a shape.  A design run holds one; a size sweep would hold several,
// and each costs its scratch, so they are kept rather than rebuilt.
// THE WEIGHTS DIRECTORY IS PART OF THE KEY, not a global. There are now TWO
// native stacks in one process -- the main 48-block one and the extra-MSA
// stack's 4 pair-only blocks -- with different weights, and the old global
// `g_weights_dir` aborted the process on the second one.
struct Key {
  int N, S, depth;
  bool pair_only, opm_first, dropout;
  std::string dir;
  bool operator<(const Key& o) const {
    return std::tie(N, S, depth, pair_only, opm_first, dropout, dir) <
           std::tie(o.N, o.S, o.depth, o.pair_only, o.opm_first, o.dropout,
                    o.dir);
  }
};

std::mutex g_mu;
std::map<Key, stack::Stack*> g_stacks;
std::map<std::string, Oracle*> g_weights;

// NE_TIME times the stack WHERE IT RUNS.  The standalone driver says 1526 ms at
// N=256 depth 48 and the same stack inside the model behaves like 1895, and no
// end-to-end timing can tell a slower stack from a stack that is waiting: XLA
// dispatches asynchronously, so a handler that returns in 100 us may still have
// its kernels queued behind something else.  Events are recorded ON THE
// HANDLER'S STREAM, so they bracket the stack's own kernels and nothing else.
// Reading them needs a sync, which perturbs the pipeline -- hence the env gate,
// and hence reporting the wall time around the handler too: when the two agree
// the stack owns the stream, and when wall >> events it is sharing it.
struct Timer {
  cudaEvent_t a, b;
  double dev = 0, wall = 0;
  int n = 0;
  bool made = false;
  void begin(cudaStream_t st) {
    if (!made) { CU(cudaEventCreate(&a)); CU(cudaEventCreate(&b)); made = true; }
    CU(cudaEventRecord(a, st));
  }
  void end(cudaStream_t st, double w) {
    CU(cudaEventRecord(b, st));
    CU(cudaEventSynchronize(b));
    float ms = 0; CU(cudaEventElapsedTime(&ms, a, b));
    dev += ms; wall += w; ++n;
  }
  void report(const char* tag) {
    fprintf(stderr, "NE_TIME %-4s n=%-4d device %8.1f ms  wall %8.1f ms\n",
            tag, n, dev / n, wall / n);
  }
};
Timer g_tf, g_tb;

double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

stack::Stack* get(int N, int S, int depth, const std::string& dir,
                  bool pair_only, bool opm_first, bool dropout,
                  bool defer_keep = false) {
  std::lock_guard<std::mutex> lk(g_mu);
  Oracle*& w = g_weights[dir];
  if (!w) w = new Oracle(dir);
  Key k{N, S, depth, pair_only, opm_first, dropout, dir};
  auto it = g_stacks.find(k);
  if (it != g_stacks.end()) return it->second;
  // A SECOND SHAPE STARVES THE FIRST, and nothing here can free it: no
  // module in this directory has a destructor, so a Stack's buffers live
  // until the process exits.  That is a deliberate choice for a design run,
  // which holds one shape -- but a LENGTH SWEEP holds several, and the cost
  // was never measured.  It is 6%: running L=175 then L=224 in one process
  // left 4.45 GB free at the second, which cut the keep tiers to flash 0/48,
  // acts 8/48 and took the step 1117 -> 1187.6 ms.  Say so rather than let
  // it look like the second length is simply slower.
  for (auto& kv : g_stacks) {
    // N ONLY, not S: the extra-MSA stack is pair-only and carries S=1 while
    // the main stack has S=2, so keying on S makes this fire on the ONE pair
    // of stacks that is meant to coexist.  It did, on the first run after it
    // was added.  A different N is a different LENGTH, which is the case
    // that actually starves.
    if (kv.first.N != N) {
      size_t fb = 0, tb = 0; cudaMemGetInfo(&fb, &tb);
      fprintf(stderr,
              "native: a stack for N=%d is already resident and cannot be "
              "freed (no destructors); building N=%d with %.2f GB left. The "
              "keep tiers will take what is there, so this length may run "
              "SLOWER than it would in its own process.\n",
              kv.first.N, N, fb / 1e9);
      break;
    }
  }
  auto* s = new stack::Stack();
  // BEFORE alloc(): pair_only decides which modules exist and whether
  // weightSlots() carries the outer product mean's constant.
  s->b.pair_only = pair_only;
  s->b.opm_first = opm_first;
  // dropout decides whether alloc() reserves the two mask scratches, so it
  // has to be set before alloc and it has to be in the key
  s->b.use_dropout = dropout;
  // a stack built at trace time takes only the mandatory buffers; the keep
  // tiers come from regrant(), once XLA has stopped growing
  s->defer_keep = defer_keep;
  if (pair_only) s->b.do_msa = false;
  s->alloc(N, S, depth);
  s->setWeights(*w);
  g_stacks.emplace(k, s);
  return s;
}

// XLA hands us bf16 buffers; the stack's own layouts are the same.
// FLOAT32 IN AND OUT.  AF2 casts only the activations entering its first
// module to bf16: its weights stay f32, so every module returns f32 and the
// carry between blocks is f32 (see common.cuh:addF32K).  The caller hands this
// the model's bf16 activations widened to f32, which is exact, and gets back
// what the model's own scan returns.
ffi::Error FwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> pair,
                   ffi::Buffer<ffi::F32> msa, ffi::Buffer<ffi::F32> pair_mask,
                   ffi::Buffer<ffi::F32> msa_mask,
                   ffi::Buffer<ffi::U32> seed, int64_t depth,
                   int64_t save, int64_t pair_only, int64_t own_save,
                   int64_t opm_first, int64_t dropout,
                   std::string_view weights_dir,
                   ffi::Result<ffi::Buffer<ffi::F32>> pair_out,
                   ffi::Result<ffi::Buffer<ffi::F32>> msa_out,
                   ffi::Result<ffi::Buffer<ffi::BF16>> psave,
                   ffi::Result<ffi::Buffer<ffi::F32>> msave) {
  // BLK_MEM: WHAT XLA HANDED US, ONCE PER SHAPE.  The 7.59 GB that appears
  // before the N=522 main stack allocates is mostly on this side of the FFI,
  // and these are the buffers it consists of -- measuring them beats guessing
  // at them, which is what the note in CHUNK_REPLAY.md asks for.
  if (envOn("BLK_MEM")) {
    static std::set<std::pair<long long, long long>> seen_shape;
    auto key = std::make_pair((long long)pair.size_bytes(),
                              (long long)psave->size_bytes());
    if (seen_shape.insert(key).second) {
      size_t fb = 0, tb = 0; cudaMemGetInfo(&fb, &tb);
      auto gb = [](size_t b) { return b / 1e9; };
      fprintf(stderr,
              "ffi fwd buffers: pair %.2f  msa %.2f  masks %.2f  pair_out %.2f"
              "  msa_out %.2f  PSAVE %.2f  msave %.2f  = %.2f GB"
              "   (free %.2f of %.2f)\n",
              gb(pair.size_bytes()), gb(msa.size_bytes()),
              gb(pair_mask.size_bytes() + msa_mask.size_bytes()),
              gb(pair_out->size_bytes()), gb(msa_out->size_bytes()),
              gb(psave->size_bytes()), gb(msave->size_bytes()),
              gb(pair.size_bytes() + msa.size_bytes() + pair_mask.size_bytes()
                 + msa_mask.size_bytes() + pair_out->size_bytes()
                 + msa_out->size_bytes() + psave->size_bytes()
                 + msave->size_bytes()),
              gb(fb), gb(tb));
    }
  }
  auto pd = pair.dimensions(), md = msa.dimensions();
  if (pd.size() != 3 || md.size() != 3)
    return ffi::Error::InvalidArgument("pair must be [N,N,Cz], msa [S,N,Cm]");
  int N = (int)pd[0], S = (int)md[0];
  if (pd[2] != block::CZ || md[2] != block::CM)
    return ffi::Error::InvalidArgument("unexpected channel widths");

  stack::Stack* s = get(N, S, (int)depth, std::string(weights_dir), pair_only != 0,
                               opm_first != 0, dropout != 0);
  // POINT AT XLA'S BUFFER, do not copy into ours.  The seed is read by the
  // kernels on the device, so a per-step reseed is just a different value in
  // an input array -- no attribute change, so no recompile.
  // [0] the seed, [1] the enable flag -- both runtime, see common.cuh
  if (dropout) s->b.drop_seed = const_cast<uint32_t*>(seed.typed_data());
  s->setStream(stream);
  const float* pb = pair.typed_data();
  const float* mb = msa.typed_data();
  // `save` 0 is the detached pass: psave/msave come in empty and the stack
  // reuses one block's worth of its own, so XLA never allocates the 811 MB
  // save-set for a forward whose gradient is not taken.
  s->save_full = (save != 0);   // picks which forward graph to capture/replay
#if NE_CUPTI
  // THE WHOLE STEP, NOT JUST THIS STACK.  CUPTI records every kernel on the
  // device, so switching it on here profiles XLA's share of a design step as
  // well as ours -- the ~18% this directory has only ever sized by
  // subtraction.  The window is counted in SAVE forwards, which is one a
  // step, and it opens late enough that compilation and the first-touch
  // faults are behind us.
  if (save && envOn("NE_CUPTI")) {
    static int steps = 0;
    const int warm = getenv("NE_CUPTI_WARM") ? atoi(getenv("NE_CUPTI_WARM")) : 6;
    const int span = getenv("NE_CUPTI_SPAN") ? atoi(getenv("NE_CUPTI_SPAN")) : 2;
    ++steps;
    if (steps == warm) { cuprof::init(); cuprof::start(); }
    else if (steps == warm + span) cuprof::stop(getenv("NE_CUPTI_TOP") ? atoi(getenv("NE_CUPTI_TOP")) : 40);
  }
#endif
  // THE SECOND KEEP GRANT, once XLA has stopped growing -- see regrant().
  // Here and nowhere else: a save forward is the one moment when no kept
  // buffer is live, because the backward that reads them has not run yet.
  if (save) s->regrant();
  if (save) {
    // under a graph the save set has to be ours, not XLA's -- see
    // useOwnSaveSet() in stack.cuh for why
    if (own_save) s->useOwnSaveSet();
    // XLA's BF16 native type is unsigned short; the stack wants __nv_bfloat16
    else s->setSaveBuffers(reinterpret_cast<__nv_bfloat16*>(psave->typed_data()),
                           msave->typed_data());
  } else {
    s->allocOneSave();
    s->setSaveBuffers(s->p1save, s->m1save, /*single*/true);
  }
  // NE_PTRS prices the CUDA-graph hypothesis: a captured graph bakes in the
  // pointers it was captured with, and every one of these is XLA's, not
  // ours.  If any of them moves between iterations, a graph captured on call
  // two is reading someone else's memory by call three hundred -- which is
  // exactly the shape of the open divergence (single shot fine, long design
  // finite but wrong).
  if (envOn("NE_PTRS")) {
    static int call = 0;
    static const void* was[6] = {nullptr};
    const void* now_[6] = {pb, mb, pair_mask.typed_data(), msa_mask.typed_data(),
                           save ? (const void*)psave->typed_data() : nullptr,
                           save ? (const void*)msave->typed_data() : nullptr};
    static const char* nm[6] = {"pair", "msa", "pair_mask", "msa_mask",
                                "psave", "msave"};
    for (int i = 0; i < 6; ++i)
      if (call && now_[i] != was[i])
        fprintf(stderr, "NE_PTRS fwd call %d: %s MOVED %p -> %p\n", call, nm[i],
                was[i], now_[i]);
    for (int i = 0; i < 6; ++i) was[i] = now_[i];
    if (call < 3)
      fprintf(stderr, "NE_PTRS fwd call %d: pair %p mask %p psave %p\n", call,
              now_[0], now_[2], now_[4]);
    ++call;
  }
  bool timing = envOn("NE_TIME");
  double w0 = 0;
  if (timing) { w0 = now_ms(); g_tf.begin(stream); }
  // THE STACK WRITES XLA'S OUTPUT BUFFERS DIRECTLY.  This used to let the
  // stack allocate its own pout/mout and then copy them here -- 139 MB and a
  // 139 MB device-to-device copy at N=522, for a value that was already in
  // the right shape.  See stack.cuh's note on why the widen that produces it
  // cannot alias its source.
  //
  // SET EVERY CALL, not once: the Stack is cached across calls (g_stacks) and
  // XLA's buffers move.
  s->pout = pair_out->typed_data();
  s->mout = msa_out->typed_data();
  s->forward(pb, mb, pair_mask.typed_data(), msa_mask.typed_data());
  size_t np = (size_t)N * N * block::CZ;
  if (timing) { g_tf.end(stream, now_ms() - w0); g_tf.report("fwd");
                prof().report("fwd"); prof().reset(); }
  if (getenv("NE_DEBUG")) {
    CU(cudaStreamSynchronize(stream));
    auto peek = [&](const char* tag, const void* p) {
      float h[4];
      CU(cudaMemcpy(h, p, 16, cudaMemcpyDeviceToHost));
      fprintf(stderr, "  %-12s %9.4f %9.4f %9.4f %9.4f\n", tag,
              h[0], h[1], h[2], h[3]);
    };
    fprintf(stderr, "NE_DEBUG N=%d S=%d depth=%d np=%zu\n", N, S, (int)depth, np);
    peek("in pair", pb);
    // s->pout IS pair_out now -- the stack wrote it in place -- so these are
    // one buffer, not two, and there is no copy between them to check.
    peek("pair_out", pair_out->typed_data());
    peek("msa_out", msa_out->typed_data());
    fprintf(stderr, "  pair_out bytes=%zu  expected=%zu\n",
            pair_out->size_bytes(), np * 4);
  }
  return ffi::Error::Success();
}

ffi::Error BwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> ct_pair,
                   ffi::Buffer<ffi::F32> ct_msa, ffi::Buffer<ffi::F32> pair_mask,
                   ffi::Buffer<ffi::F32> msa_mask,
                   ffi::Buffer<ffi::BF16> psave, ffi::Buffer<ffi::F32> msave,
                   ffi::Buffer<ffi::U32> seed,
                   int64_t depth, int64_t pair_only, int64_t own_save,
                   int64_t opm_first, int64_t dropout,
                   std::string_view weights_dir,
                   ffi::Result<ffi::Buffer<ffi::F32>> dpair,
                   ffi::Result<ffi::Buffer<ffi::F32>> dmsa) {
  auto pd = ct_pair.dimensions(), md = ct_msa.dimensions();
  int N = (int)pd[0], S = (int)md[0];
  stack::Stack* s = get(N, S, (int)depth, std::string(weights_dir), pair_only != 0,
                               opm_first != 0, dropout != 0);
  // POINT AT XLA'S BUFFER, do not copy into ours.  The seed is read by the
  // kernels on the device, so a per-step reseed is just a different value in
  // an input array -- no attribute change, so no recompile.
  // [0] the seed, [1] the enable flag -- both runtime, see common.cuh
  if (dropout) s->b.drop_seed = const_cast<uint32_t*>(seed.typed_data());
  s->setStream(stream);
  // the forward's save-set, handed back as data rather than remembered
  if (own_save) s->useOwnSaveSet();
  else s->setSaveBuffers(
      reinterpret_cast<__nv_bfloat16*>(const_cast<uint16_t*>(psave.typed_data())),
      const_cast<float*>(msave.typed_data()));
  bool timing = envOn("NE_TIME");
  double w0 = 0;
  if (timing) { w0 = now_ms(); g_tb.begin(stream); }
  s->backward(ct_pair.typed_data(), ct_msa.typed_data(),
              pair_mask.typed_data(), msa_mask.typed_data());
  size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
  // the stack's pair cotangent is bf16 (see common.cuh:toF); the FFI's is f32
  bf16ToF32OutK<<<(np + 255) / 256, 256, 0, stream>>>(s->b.dp, dpair->typed_data(), np);
  CU(cudaMemcpyAsync(dmsa->typed_data(), s->b.dm, nm * 4,
                     cudaMemcpyDeviceToDevice, stream));
  if (timing) { g_tb.end(stream, now_ms() - w0); g_tb.report("bwd");
                prof().report("bwd"); prof().reset(); }
  return ffi::Error::Success();
}


// ---------------------------------------------------------------- templates
//
// AF2's TEMPLATE PAIR STACK, the two 64-channel iterations.  A separate entry
// point rather than another mode of the Evoformer one: different width,
// different depth, no MSA side and no keep tiers, and its backward replays
// from the saved input instead of taking a save-set apart.
struct TKey {
  int N, depth;
  std::string dir;
  bool operator<(const TKey& o) const {
    return std::tie(N, depth, dir) < std::tie(o.N, o.depth, o.dir);
  }
};
std::map<TKey, tplstack::TplStack*> g_tpl;

tplstack::TplStack* tpl_get(int N, int depth, const std::string& dir) {
  std::lock_guard<std::mutex> lk(g_mu);
  TKey k{N, depth, dir};
  auto it = g_tpl.find(k);
  if (it != g_tpl.end()) return it->second;
  Oracle*& w = g_weights[dir];
  if (!w) w = new Oracle(dir);
  auto* t = new tplstack::TplStack();
  t->alloc(N, depth);
  t->setWeights(*w);
  g_tpl[k] = t;
  return t;
}

ffi::Error TplFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> pair,
                      ffi::Buffer<ffi::F32> pair_mask,
                      ffi::Buffer<ffi::U32> seed, int64_t depth,
                      int64_t dropout, std::string_view weights_dir,
                      ffi::Result<ffi::Buffer<ffi::F32>> out,
                      ffi::Result<ffi::Buffer<ffi::BF16>> save) {
  auto d = pair.dimensions();
  if (d.size() != 3 || d[2] != tplstack::CZ)
    return ffi::Error::InvalidArgument("template pair must be [N,N,64]");
  int N = (int)d[0];
  auto* t = tpl_get(N, (int)depth, std::string(weights_dir));
  t->setStream(stream);
  t->setDropout(dropout != 0, seed.typed_data());
  size_t np = t->pairElems();
  int g = (int)((np + 255) / 256);
  // the stack is bf16 throughout, as every other one here is
  static __nv_bfloat16* in = nullptr;
  static size_t in_n = 0;
  if (in_n < np) {
    if (in) CU(cudaFree(in));
    CU(cudaMalloc(&in, np * 2)); in_n = np;
  }
  toBf16K<<<g, 256, 0, stream>>>(in, pair.typed_data(), np);
  t->forward(in, pair_mask.typed_data(),
             reinterpret_cast<__nv_bfloat16*>(save->typed_data()));
  bf16ToF32OutK<<<g, 256, 0, stream>>>(t->out(), out->typed_data(), np);
  return ffi::Error::Success();
}

ffi::Error TplBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> ct,
                      ffi::Buffer<ffi::F32> pair_mask,
                      ffi::Buffer<ffi::BF16> save,
                      ffi::Buffer<ffi::U32> seed, int64_t depth,
                      int64_t dropout, std::string_view weights_dir,
                      ffi::Result<ffi::Buffer<ffi::F32>> dpair) {
  auto d = ct.dimensions();
  int N = (int)d[0];
  auto* t = tpl_get(N, (int)depth, std::string(weights_dir));
  t->setStream(stream);
  t->setDropout(dropout != 0, seed.typed_data());
  size_t np = t->pairElems();
  int g = (int)((np + 255) / 256);
  static __nv_bfloat16* ctb = nullptr;
  static size_t ct_n = 0;
  if (ct_n < np) {
    if (ctb) CU(cudaFree(ctb));
    CU(cudaMalloc(&ctb, np * 2)); ct_n = np;
  }
  toBf16K<<<g, 256, 0, stream>>>(ctb, ct.typed_data(), np);
  t->backward(ctb, pair_mask.typed_data(),
              reinterpret_cast<const __nv_bfloat16*>(save.typed_data()));
  bf16ToF32OutK<<<g, 256, 0, stream>>>(t->dpair(), dpair->typed_data(), np);
  return ffi::Error::Success();
}


// ---------------------------------------------------------------------------
// The post-trunk step: structure module + all four heads, in one call.
//
// This is the boundary that lets a design step keep its losses in Python
// while XLA compiles none of the model.  Everything between the trunk's
// (single, pair) and the tensors a loss actually reads -- atom14 positions
// and the four heads' logits -- happens here.
//
// THE BACKWARD REPLAYS THE FORWARD.  StructureModule keeps its forward state
// in member buffers (foldstack deliberately holds one iteration's save-set
// and remats the other seven), so a second forward on the same object
// overwrites what the first one's backward still needs.  BindCraft runs
// several prediction states through one model, and nothing stops XLA from
// issuing fwd(A), fwd(B), bwd(B), bwd(A) -- which would silently hand A's
// backward B's activations.  Replaying costs one structure-module forward
// and removes the ordering assumption entirely.
// ---------------------------------------------------------------------------
__global__ void addPlainK(float* __restrict__ dst, const float* __restrict__ src,
                          size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// NOT KEYED ON THE STREAM any more.  It used to be: foldstack owns a tree of
// cuBLAS handles and had no setStream, so the object was built on the stream
// it would run on.  That made the structure module impossible to preallocate,
// because at TRACE time there is no XLA stream to build it on and a null-
// stream copy is one the real call never finds.  structmod::setStream now
// walks the tree and rebinds every handle, so the key is the shape alone and
// the stream is applied on the way out of sm_get.
//
// WHAT THIS GIVES UP: two streams running the SAME shape at the same time
// used to get an object each and now share one, scratch buffers included.
// XLA hands these calls one stream and always has, so nothing exercises it
// today -- but it is a real hazard and not a theoretical one, so if a
// second stream ever appears, the key needs the stream back and the
// prealloc needs a different answer (a shape-keyed pool handed out per
// stream, most likely).
struct SKey {
  int N, depth;
  std::string dir;
  bool operator<(const SKey& o) const {
    return std::tie(N, depth, dir) < std::tie(o.N, o.depth, o.dir);
  }
};
struct SmHeads {
  structmod::StructureModule sm;
  heads::Heads hd;
};
std::map<SKey, SmHeads*> g_sm;

SmHeads* sm_get(int N, int depth, const std::string& dir,
                cudaStream_t stream) {
  std::lock_guard<std::mutex> lk(g_mu);
  SKey k{N, depth, dir};
  auto it = g_sm.find(k);
  if (it != g_sm.end()) { it->second->sm.setStream(stream);
                          it->second->hd.setStream(stream);
                          return it->second; }
  Oracle*& w = g_weights[dir];
  if (!w) w = new Oracle(dir);
  auto* m = new SmHeads();
  m->sm.do_atoms = true;          // bc2 reads final_atom_positions
  m->sm.alloc(N, depth, stream);
  m->sm.setWeights(*w);
  m->hd.alloc(N, stream);
  m->hd.setWeights(*w);
  g_sm[k] = m;
  return m;
}

ffi::Error SmFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> single,
                     ffi::Buffer<ffi::F32> pair, ffi::Buffer<ffi::F32> mask,
                     ffi::Buffer<ffi::F32> aatype,
                     ffi::Buffer<ffi::U32> seed, int64_t depth,
                     std::string_view weights_dir,
                     ffi::Result<ffi::Buffer<ffi::F32>> atom,
                     ffi::Result<ffi::Buffer<ffi::F32>> dgram,
                     ffi::Result<ffi::Buffer<ffi::F32>> pae,
                     ffi::Result<ffi::Buffer<ffi::F32>> plddt,
                     ffi::Result<ffi::Buffer<ffi::F32>> expres) {
  auto d = single.dimensions();
  if (d.size() != 2 || d[1] != structmod::C)
    return ffi::Error::InvalidArgument("single must be [N,384]");
  int N = (int)d[0];
  auto* m = sm_get(N, (int)depth, std::string(weights_dir), stream);
  m->sm.setDropout(true, seed.typed_data());
  m->sm.forward(single.typed_data(), pair.typed_data(), mask.typed_data(),
                aatype.typed_data());
  // pLDDT reads the fold module's act; experimentally_resolved reads the
  // trunk single it was handed
  m->hd.forward(pair.typed_data(), m->sm.outAct());
  m->hd.expResFwd(single.typed_data());
  size_t na = (size_t)N * atom14::NA * 3, np = (size_t)N * N;
  CU(cudaMemcpyAsync(atom->typed_data(), m->sm.outAtom(), na * 4,
                     cudaMemcpyDeviceToDevice, stream));
  CU(cudaMemcpyAsync(dgram->typed_data(), m->hd.dgram,
                     np * heads::NDGRAM * 4, cudaMemcpyDeviceToDevice, stream));
  CU(cudaMemcpyAsync(pae->typed_data(), m->hd.pae, np * heads::NPAE * 4,
                     cudaMemcpyDeviceToDevice, stream));
  CU(cudaMemcpyAsync(plddt->typed_data(), m->hd.plddt,
                     (size_t)N * heads::PL_BINS * 4, cudaMemcpyDeviceToDevice,
                     stream));
  CU(cudaMemcpyAsync(expres->typed_data(), m->hd.expres,
                     (size_t)N * heads::NER * 4, cudaMemcpyDeviceToDevice,
                     stream));
  return ffi::Error::Success();
}

ffi::Error SmBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> single,
                     ffi::Buffer<ffi::F32> pair, ffi::Buffer<ffi::F32> mask,
                     ffi::Buffer<ffi::F32> aatype,
                     ffi::Buffer<ffi::F32> datom, ffi::Buffer<ffi::F32> ddgram,
                     ffi::Buffer<ffi::F32> dpae, ffi::Buffer<ffi::F32> dplddt,
                     ffi::Buffer<ffi::F32> dexpres,
                     ffi::Buffer<ffi::U32> seed, int64_t depth,
                     std::string_view weights_dir,
                     ffi::Result<ffi::Buffer<ffi::F32>> dsingle,
                     ffi::Result<ffi::Buffer<ffi::F32>> dpair) {
  int N = (int)single.dimensions()[0];
  auto* m = sm_get(N, (int)depth, std::string(weights_dir), stream);
  // THE SAME SEED, so the replay draws the SAME MASKS.  dropScale is a pure
  // hash of (seed, depth, module, index) -- that is what lets a replayed
  // forward reproduce a stochastic one, and what would silently break if the
  // seed were drawn per call instead of handed in.
  m->sm.setDropout(true, seed.typed_data());
  // the replay -- see the note above
  m->sm.forward(single.typed_data(), pair.typed_data(), mask.typed_data(),
                aatype.typed_data());
  m->hd.forward(pair.typed_data(), m->sm.outAct());

  // the heads first: they produce the structure module's act cotangent and
  // write dpair, which the structure module then accumulates onto
  static float* dact = nullptr;
  static size_t dact_n = 0;
  size_t nc = (size_t)N * structmod::C;
  if (dact_n < nc) {
    if (dact) CU(cudaFree(dact));
    CU(cudaMalloc(&dact, nc * 4)); dact_n = nc;
  }
  m->hd.backward(ddgram.typed_data(), dpae.typed_data(), dplddt.typed_data(),
                 m->sm.outAct(), dpair->typed_data(), dact);
  // the trunk single's own head cotangent starts dsingle (beta=0); the
  // structure module's backward overwrites it, so this has to go after --
  // instead it lands in a scratch and is added below.
  static float* dse = nullptr;
  static size_t dse_n = 0;
  if (dse_n < nc) {
    if (dse) CU(cudaFree(dse));
    CU(cudaMalloc(&dse, nc * 4)); dse_n = nc;
  }
  m->hd.expResBwd(dexpres.typed_data(), dse, /*beta*/0.f);

  // the rotation/translation cotangents are zero: bc2 reads positions, not
  // frames, so nothing downstream differentiates the output frame directly
  static float *drot = nullptr, *dtrans = nullptr;
  static int frame_n = 0;
  if (frame_n < N) {
    if (drot) { CU(cudaFree(drot)); CU(cudaFree(dtrans)); }
    CU(cudaMalloc(&drot, (size_t)N * 9 * 4));
    CU(cudaMalloc(&dtrans, (size_t)N * 3 * 4));
    frame_n = N;
  }
  CU(cudaMemsetAsync(drot, 0, (size_t)N * 9 * 4, stream));
  CU(cudaMemsetAsync(dtrans, 0, (size_t)N * 3 * 4, stream));

  // THE HEADS' SHARE OF dpair STAYS WHERE IT IS.  structmod::backward used to
  // OVERWRITE dpair, so this staged the whole thing aside in f32 and added it
  // back -- a 139 MB buffer at N=522 plus a 139 MB copy and a 139 MB
  // read-modify-write, for an answer that is just a sum.  `acc_pair` makes
  // its final LayerNorm backward accumulate instead, which is the same two
  // lines the trimul and transition backwards already use for exactly this.
  // dsingle keeps its staging: it is N*C, not N*N*C, and not worth a second
  // flag.
  m->sm.backward(dact, drot, dtrans, single.typed_data(),
                 dsingle->typed_data(), dpair->typed_data(),
                 datom.typed_data(), aatype.typed_data(), /*acc_pair*/true);
  addPlainK<<<(int)((nc + 255) / 256), 256, 0, stream>>>(
      dsingle->typed_data(), dse, nc);
  return ffi::Error::Success();
}


// ---------------------------------------------------------------------------
// The embedder: sequence features -> the msa and pair the trunk consumes.
//
// This is the other end of the design step.  With the trunk and the
// post-trunk step already native, it is most of what XLA still compiles that
// is not a loss -- and it is where the gradient REACHES THE SEQUENCE, since
// everything downstream hands its cotangent back through these four Linears.
//
// NO REPLAY AND NO SAVE SET, unlike both stacks and the post-trunk step: this
// backward is a pure function of the cotangents, target_feat and the weights.
// Nothing the forward computed is read again, so both calls are re-entrant
// with nothing carried between them.  The relative-position term contributes
// no gradient at all -- it is a gather from integer features.
// ---------------------------------------------------------------------------
struct EKey {
  int N, S;
  std::string dir;
  cudaStream_t stream;
  bool operator<(const EKey& o) const {
    return std::tie(N, S, dir, stream) <
           std::tie(o.N, o.S, o.dir, o.stream);
  }
};
std::map<EKey, embed::Embed*> g_emb;

embed::Embed* emb_get(int N, int S, const std::string& dir,
                      cudaStream_t stream) {
  std::lock_guard<std::mutex> lk(g_mu);
  EKey k{N, S, dir, stream};
  auto it = g_emb.find(k);
  if (it != g_emb.end()) return it->second;
  Oracle*& w = g_weights[dir];
  if (!w) w = new Oracle(dir);
  auto* m = new embed::Embed();
  m->alloc(N, S, stream);
  m->setWeights(*w);
  g_emb[k] = m;
  return m;
}

ffi::Error EmbFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> tf,
                      ffi::Buffer<ffi::F32> mf,
                      ffi::Buffer<ffi::F32> residue_index,
                      ffi::Buffer<ffi::F32> asym_id,
                      ffi::Buffer<ffi::F32> entity_id,
                      ffi::Buffer<ffi::F32> sym_id,
                      std::string_view weights_dir,
                      ffi::Result<ffi::Buffer<ffi::F32>> msa,
                      ffi::Result<ffi::Buffer<ffi::F32>> pair) {
  auto d = tf.dimensions();
  if (d.size() != 2 || d[1] != embed::NAA)
    return ffi::Error::InvalidArgument("target_feat must be [N,21]");
  auto dm = mf.dimensions();
  if (dm.size() != 3 || dm[2] != embed::NMSA_F)
    return ffi::Error::InvalidArgument("msa_feat must be [S,N,49]");
  int N = (int)d[0], S = (int)dm[0];
  auto* m = emb_get(N, S, std::string(weights_dir), stream);
  m->forward(tf.typed_data(), mf.typed_data(), residue_index.typed_data(),
             asym_id.typed_data(), entity_id.typed_data(),
             sym_id.typed_data());
  CU(cudaMemcpyAsync(msa->typed_data(), m->msa,
                     (size_t)S * N * embed::CM * 4, cudaMemcpyDeviceToDevice,
                     stream));
  CU(cudaMemcpyAsync(pair->typed_data(), m->pair,
                     (size_t)N * N * embed::CZ * 4, cudaMemcpyDeviceToDevice,
                     stream));
  return ffi::Error::Success();
}

ffi::Error EmbBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> dmsa,
                      ffi::Buffer<ffi::F32> dpair, ffi::Buffer<ffi::F32> tf,
                      std::string_view weights_dir,
                      ffi::Result<ffi::Buffer<ffi::F32>> dtf,
                      ffi::Result<ffi::Buffer<ffi::F32>> dmf) {
  auto dm = dmsa.dimensions();
  int S = (int)dm[0], N = (int)dm[1];
  auto* m = emb_get(N, S, std::string(weights_dir), stream);
  m->backward(dmsa.typed_data(), dpair.typed_data(), nullptr, tf.typed_data(),
              dtf->typed_data(), dmf->typed_data());
  return ffi::Error::Success();
}


// ---------------------------------------------------------------------------
// The whole template embedding: the query projection, the native pair stack
// and the aggregation, in one call.  With the eight constant projections
// hoisted out by the caller, this is everything the template path still costs
// XLA.
// ---------------------------------------------------------------------------
// OFF THE STREAM KEY, for the same reason as SKey above and with far less
// work: the whole template tree already had setStream (TplEmbed -> TplStack
// -> TplBlock), so only the key and the rebind below were missing.
//
// THIS IS THE TEMPLATE STACK THAT ACTUALLY RUNS.  `tpl_get` builds a
// standalone one, and in the normal path it never executes -- the template
// EMBEDDING "supersedes the stack runner WITHOUT disabling it: this one
// returns before SingleTemplateEmbedding runs", in native.py's own words.
// So native_tplstack_prealloc was preallocating an object nobody calls,
// which is exactly how it behaved: it cost memory and bought nothing.  The
// embedder owns the stack that runs, and it is the one worth preallocating.
struct TEKey {
  int N, depth;
  std::string dir;
  bool operator<(const TEKey& o) const {
    return std::tie(N, depth, dir) < std::tie(o.N, o.depth, o.dir);
  }
};
std::map<TEKey, tplembed::TplEmbed*> g_te;

tplembed::TplEmbed* te_get(int N, int depth, const std::string& dir,
                           cudaStream_t stream) {
  std::lock_guard<std::mutex> lk(g_mu);
  TEKey k{N, depth, dir};
  auto it = g_te.find(k);
  if (it != g_te.end()) { it->second->setStream(stream); return it->second; }
  Oracle*& w = g_weights[dir];
  if (!w) w = new Oracle(dir);
  auto* m = new tplembed::TplEmbed();
  m->alloc(N, depth, stream);
  m->setWeights(*w);
  g_te[k] = m;
  return m;
}

ffi::Error TeFwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> query,
                     ffi::Buffer<ffi::F32> act_const,
                     ffi::Buffer<ffi::F32> pair_mask,
                     ffi::Buffer<ffi::U32> seed, int64_t depth,
                     int64_t dropout, std::string_view weights_dir,
                     ffi::Result<ffi::Buffer<ffi::F32>> out,
                     ffi::Result<ffi::Buffer<ffi::BF16>> save) {
  auto d = query.dimensions();
  if (d.size() != 3 || d[2] != tplembed::CQ)
    return ffi::Error::InvalidArgument("query must be [N,N,128]");
  int N = (int)d[0];
  auto* m = te_get(N, (int)depth, std::string(weights_dir), stream);
  m->setDropout(dropout != 0, seed.typed_data());
  m->forward(query.typed_data(), act_const.typed_data(),
             pair_mask.typed_data(), out->typed_data(),
             reinterpret_cast<__nv_bfloat16*>(save->typed_data()));
  return ffi::Error::Success();
}

ffi::Error TeBwdImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> ct,
                     ffi::Buffer<ffi::F32> query,
                     ffi::Buffer<ffi::F32> act_const,
                     ffi::Buffer<ffi::F32> pair_mask,
                     ffi::Buffer<ffi::BF16> save, ffi::Buffer<ffi::U32> seed,
                     int64_t depth, int64_t dropout,
                     std::string_view weights_dir,
                     ffi::Result<ffi::Buffer<ffi::F32>> dquery) {
  int N = (int)ct.dimensions()[0];
  auto* m = te_get(N, (int)depth, std::string(weights_dir), stream);
  m->setDropout(dropout != 0, seed.typed_data());
  // query and act_const come in so the backward can REPLAY the forward; see
  // the note on TemplateEmbedding::backward for why it must.
  m->backward(ct.typed_data(), query.typed_data(), act_const.typed_data(),
              pair_mask.typed_data(),
              reinterpret_cast<const __nv_bfloat16*>(save.typed_data()),
              dquery->typed_data());
  return ffi::Error::Success();
}

}  // namespace

// BUILD THE STACK AT TRACE TIME, BEFORE XLA HAS ALLOCATED ANYTHING.
//
// The block scratch is 4-5 GB at these lengths, and whether it fits is
// decided by WHO ALLOCATES FIRST.  Measured on the A10 with BLK_MEM, two
// complexes that differ only in length:
//
//   552  the two binder_alone stacks run first, XLA then grows to 17.83 GB
//        in use, and the complex stack takes its 4.30 GB out of the 5.86
//        that is left.  The structure module gets 1.47 GB and dies.
//   587  the COMPLEX stack runs first and takes 4.79 GB with 13.64 GB free.
//        Everything after it fits, at a LONGER length.
//
// Nothing about 552 is bigger.  The order of the two predictions in the
// step decided it, and that order is not ours to choose from in here.  So
// the caller says, at TRACE time -- when XLA has allocated nothing at all
// and the whole card is free -- which stacks the step is going to use, and
// they are built then.  `get()` caches on the same key the FFI call will
// use, so the call that follows finds this one rather than making a second.
//
// This is NOT the trace-time ballast that was refuted last session.  That
// held memory nobody would use and XLA grew into whatever was left; this
// allocates the exact buffers the step is about to run on.  There is
// nothing to release and nothing to get the release point wrong.
//
// It returns 0 on success and prints what it did, because the whole family
// of defects here is silent: see the five in CHUNK_REPLAY.md.  It is also
// OUTSIDE the anonymous namespace above, which is where one of those five
// died -- `extern "C"` inside it exports nothing and `nm -D` says so.
extern "C" int native_stack_prealloc(int N, int S, int depth,
                                     const char* weights_dir, int pair_only,
                                     int opm_first, int dropout) {
  size_t fb0 = 0, tb = 0;
  if (cudaMemGetInfo(&fb0, &tb) != cudaSuccess) {
    fprintf(stderr, "native: prealloc N=%d could not read the device\n", N);
    return 1;
  }
  get(N, S, depth, std::string(weights_dir), pair_only != 0, opm_first != 0,
      dropout != 0, /*defer_keep*/true);
  size_t fb1 = 0;
  cudaMemGetInfo(&fb1, &tb);
  fprintf(stderr,
          "native: prealloc N=%d S=%d depth=%d pair_only=%d took %.2f GB "
          "(%.2f of %.2f GB free after)\n",
          N, S, depth, pair_only, (fb0 - fb1) / 1e9, fb1 / 1e9, tb / 1e9);
  return 0;
}

// THE OTHER TWO ALLOCATORS, AND WHY THEY COME AS A PAIR.
//
// Once the evoformer stacks stopped racing, 617 and 647 started dying in
// tplblock/tplstack, so the obvious next move was to build the template
// stack at trace time too.  ON ITS OWN THAT IS A REGRESSION, measured: it
// takes 2.97 GB at N=608 and 3.58 at N=672 before XLA has allocated, and
//
//   617  moved from tplstack to trimul_body -- still out of memory
//   647  still out of memory
//   600, 608  structure module, as before
//   577  structmod.cuh:68 -- WAS PASSING
//   587  ipa.cuh:534      -- WAS PASSING
//
// Read the failure sites: taking the template stack early does not lose the
// memory, it MOVES THE SQUEEZE ONTO THE STRUCTURE MODULE, which allocates
// last and has no keep tiers to give back.  So the half-measure is worse
// than nothing and the whole measure is the pair -- which is why the
// structure module had to come off the stream key first (see SKey above).
//
// SO THE PAIR WAS MEASURED, AND THE PAIR IS REFUTED TOO: with both on, 577,
// 587, 600, 608 AND 617 all fail -- worse than either alone, and worse than
// neither.  The template stack takes its 2.97 GB at the padded bucket before
// XLA has allocated, the structure module another 1.57-1.71, and ~8 GB is
// not enough for the rest of the step.
//
// The structure module alone is NEUTRAL: 577, 587 and 608 run at the same
// speed with the IDENTICAL gradient, and 617/632 still die -- the failure
// only moves, from tplblock to the template stack's own triattn.
//
// WHY THERE IS NOTHING LEFT HERE.  At the 640 bucket the step wants 6.99 GB
// of evoformer stacks (the two share one pair scratch -- see "adopted the
// other stack's pair scratch" under BLK_MEM), 3.25 for the template stack
// and 1.71 for the structure module, and XLA wants the remaining twelve.
// That is a capacity wall on a 23.68 GB card, not a race, and no ordering
// wins it.  608 is the ceiling until something gets genuinely smaller.
//
// Both entry points stay, OFF unless NB_PREALLOC_TPL / NB_PREALLOC_SM asks
// for them, so the next person re-tests on a bigger card with an env var
// rather than by rebuilding this reasoning.
extern "C" int native_tplstack_prealloc(int N, int depth,
                                        const char* weights_dir) {
  size_t fb0 = 0, tb = 0;
  if (cudaMemGetInfo(&fb0, &tb) != cudaSuccess) return 1;
  tpl_get(N, depth, std::string(weights_dir));
  size_t fb1 = 0;
  cudaMemGetInfo(&fb1, &tb);
  fprintf(stderr,
          "native: prealloc tplstack N=%d depth=%d took %.2f GB "
          "(%.2f of %.2f GB free after)\n",
          N, depth, (fb0 - fb1) / 1e9, fb1 / 1e9, tb / 1e9);
  return 0;
}

// The null stream here is NOT the stream it will run on: sm_get rebinds the
// whole handle tree on the way out, every call.  Nothing in the structure
// module is stream-ordered -- see the note on structmod::setStream.
extern "C" int native_sm_prealloc(int N, int depth, const char* weights_dir) {
  size_t fb0 = 0, tb = 0;
  if (cudaMemGetInfo(&fb0, &tb) != cudaSuccess) return 1;
  sm_get(N, depth, std::string(weights_dir), /*stream*/nullptr);
  size_t fb1 = 0;
  cudaMemGetInfo(&fb1, &tb);
  fprintf(stderr,
          "native: prealloc structmod N=%d depth=%d took %.2f GB "
          "(%.2f of %.2f GB free after)\n",
          N, depth, (fb0 - fb1) / 1e9, fb1 / 1e9, tb / 1e9);
  return 0;
}

// THE TEMPLATE EMBEDDER, which owns the template stack that actually runs
// and is where 617 dies once everything else has been made smaller.
extern "C" int native_tplembed_prealloc(int N, int depth,
                                        const char* weights_dir) {
  size_t fb0 = 0, tb = 0;
  if (cudaMemGetInfo(&fb0, &tb) != cudaSuccess) return 1;
  te_get(N, depth, std::string(weights_dir), /*stream*/nullptr);
  size_t fb1 = 0;
  cudaMemGetInfo(&fb1, &tb);
  fprintf(stderr,
          "native: prealloc tplembed N=%d depth=%d took %.2f GB "
          "(%.2f of %.2f GB free after)\n",
          N, depth, (fb0 - fb1) / 1e9, fb1 / 1e9, tb / 1e9);
  return 0;
}

// `emb_get` is still keyed on the stream and is still not preallocated: it
// is small, and it has never been a failure site.

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NativeEvoformerFwd, FwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U32>>()    // seed: runtime, not an attribute
        .Attr<int64_t>("depth")
        .Attr<int64_t>("save")
        .Attr<int64_t>("pair_only")
        .Attr<int64_t>("own_save")
        .Attr<int64_t>("opm_first")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::BF16>>()   // psave: the pair save-set is narrow
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NativeEvoformerBwd, BwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // psave: the pair save-set is narrow
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U32>>()    // seed: runtime, not an attribute
        .Attr<int64_t>("depth")
        .Attr<int64_t>("pair_only")
        .Attr<int64_t>("own_save")
        .Attr<int64_t>("opm_first")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TemplatePairStackFwd, TplFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // pair  [N,N,64]
        .Arg<ffi::Buffer<ffi::F32>>()      // pair_mask [N,N]
        .Arg<ffi::Buffer<ffi::U32>>()      // [seed, enabled], runtime
        .Attr<int64_t>("depth")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()      // out   [N,N,64]
        .Ret<ffi::Buffer<ffi::BF16>>());   // save: the input, for the replay

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TemplatePairStackBwd, TplBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // ct    [N,N,64]
        .Arg<ffi::Buffer<ffi::F32>>()      // pair_mask
        .Arg<ffi::Buffer<ffi::BF16>>()     // save
        .Arg<ffi::Buffer<ffi::U32>>()      // [seed, enabled]
        .Attr<int64_t>("depth")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>());    // dpair [N,N,64]

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StructureHeadsFwd, SmFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // single [N,384]
        .Arg<ffi::Buffer<ffi::F32>>()      // pair   [N,N,128]
        .Arg<ffi::Buffer<ffi::F32>>()      // mask   [N]
        .Arg<ffi::Buffer<ffi::F32>>()      // aatype [N]
        .Arg<ffi::Buffer<ffi::U32>>()      // [seed, enabled], runtime
        .Attr<int64_t>("depth")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()      // atom14 [N,14,3]
        .Ret<ffi::Buffer<ffi::F32>>()      // dgram  [N,N,64]
        .Ret<ffi::Buffer<ffi::F32>>()      // pae    [N,N,64]
        .Ret<ffi::Buffer<ffi::F32>>()      // plddt  [N,50]
        .Ret<ffi::Buffer<ffi::F32>>());    // expres [N,37]

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    StructureHeadsBwd, SmBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // single
        .Arg<ffi::Buffer<ffi::F32>>()      // pair
        .Arg<ffi::Buffer<ffi::F32>>()      // mask
        .Arg<ffi::Buffer<ffi::F32>>()      // aatype
        .Arg<ffi::Buffer<ffi::F32>>()      // d atom14
        .Arg<ffi::Buffer<ffi::F32>>()      // d dgram
        .Arg<ffi::Buffer<ffi::F32>>()      // d pae
        .Arg<ffi::Buffer<ffi::F32>>()      // d plddt
        .Arg<ffi::Buffer<ffi::F32>>()      // d expres
        .Arg<ffi::Buffer<ffi::U32>>()      // [seed, enabled], same as the fwd
        .Attr<int64_t>("depth")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()      // dsingle [N,384]
        .Ret<ffi::Buffer<ffi::F32>>());    // dpair   [N,N,128]

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EmbedderFwd, EmbFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // target_feat   [N,21]
        .Arg<ffi::Buffer<ffi::F32>>()      // msa_feat      [S,N,49]
        .Arg<ffi::Buffer<ffi::F32>>()      // residue_index [N]
        .Arg<ffi::Buffer<ffi::F32>>()      // asym_id       [N]
        .Arg<ffi::Buffer<ffi::F32>>()      // entity_id     [N]
        .Arg<ffi::Buffer<ffi::F32>>()      // sym_id        [N]
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()      // msa  [S,N,256]
        .Ret<ffi::Buffer<ffi::F32>>());    // pair [N,N,128]

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    EmbedderBwd, EmbBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // d msa
        .Arg<ffi::Buffer<ffi::F32>>()      // d pair
        .Arg<ffi::Buffer<ffi::F32>>()      // target_feat, the one input the
                                           // backward still needs
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()      // d target_feat [N,21]
        .Ret<ffi::Buffer<ffi::F32>>());    // d msa_feat    [S,N,49]

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TemplateEmbeddingFwd, TeFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // query     [N,N,128]
        .Arg<ffi::Buffer<ffi::F32>>()      // act_const [N,N,64]
        .Arg<ffi::Buffer<ffi::F32>>()      // pair_mask [N,N]
        .Arg<ffi::Buffer<ffi::U32>>()      // [seed, enabled]
        .Attr<int64_t>("depth")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>()      // out  [N,N,128]
        .Ret<ffi::Buffer<ffi::BF16>>());   // save, for the stack's replay

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TemplateEmbeddingBwd, TeBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // d out
        .Arg<ffi::Buffer<ffi::F32>>()      // query     [N,N,128]
        .Arg<ffi::Buffer<ffi::F32>>()      // act_const [N,N,64]
        .Arg<ffi::Buffer<ffi::F32>>()      // pair_mask
        .Arg<ffi::Buffer<ffi::BF16>>()     // save
        .Arg<ffi::Buffer<ffi::U32>>()      // [seed, enabled]
        .Attr<int64_t>("depth")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("weights_dir")
        .Ret<ffi::Buffer<ffi::F32>>());    // d query [N,N,128]

// ===================== the glue, gated one piece at a time ==============
// Each of these is checked as a FORMULA in numpy first (seqfeat_formula.py
// and friends) and only then compiled, because the template embedding
// proved the maths is where the error lives.

ffi::Error SeqFeatImpl(cudaStream_t st, ffi::Buffer<ffi::F32> seq,
                       ffi::Buffer<ffi::U8> flags, ffi::Buffer<ffi::F32> bias,
                       float sw, float ohw, float temp, float ls,
                       float omitted_half,
                       ffi::Result<ffi::Buffer<ffi::F32>> feat,
                       ffi::Result<ffi::Buffer<ffi::F32>> prof) {
  const int N = (int)seq.dimensions()[0];
  seqfeat::Scalars sc{sw, ohw, temp, ls};
  const float* b = bias.element_count() ? bias.typed_data() : nullptr;
  seqfeat::seqFeatK<<<(N + 127) / 128, 128, 0, st>>>(
      seq.typed_data(), flags.typed_data(), b, sc, omitted_half,
      feat->typed_data(), prof->typed_data(), N);
  return ffi::Error::Success();
}

ffi::Error ConfidenceImpl(cudaStream_t st, ffi::Buffer<ffi::F32> lddt,
                          ffi::Buffer<ffi::F32> pae_l,
                          ffi::Buffer<ffi::F32> centres,
                          ffi::Buffer<ffi::F32> tm_bin,
                          ffi::Buffer<ffi::F32> expres,
                          ffi::Buffer<ffi::F32> smask,
                          ffi::Buffer<ffi::S32> asym,
                          ffi::Buffer<ffi::S32> n,
                          ffi::Result<ffi::Buffer<ffi::F32>> plddt,
                          ffi::Result<ffi::Buffer<ffi::F32>> pae,
                          ffi::Result<ffi::Buffer<ffi::F32>> ptm,
                          ffi::Result<ffi::Buffer<ffi::F32>> iptm,
                          ffi::Result<ffi::Buffer<ffi::F32>> iptm_pr,
                          ffi::Result<ffi::Buffer<ffi::F32>> expres_ca,
                          ffi::Result<ffi::Buffer<ffi::F32>> scratch) {
  const int NMAX = (int)pae_l.dimensions()[0];
  const int PB = (int)pae_l.dimensions()[2];
  const size_t np = (size_t)NMAX * NMAX;
  float* err = scratch->typed_data();          // [NMAX*NMAX] x2 + [NMAX]
  float* tm = err + np;
  float* per = tm + np;
  confidence::plddtK<<<(NMAX + 127) / 128, 128, 0, st>>>(
      lddt.typed_data(), n.typed_data(), plddt->typed_data(), NMAX);
  confidence::paeTermK<<<(int)((np + 127) / 128), 128, 0, st>>>(
      pae_l.typed_data(), centres.typed_data(), tm_bin.typed_data(),
      n.typed_data(), err, tm, NMAX, PB);
  confidence::symmetriseK<<<(int)((np + 127) / 128), 128, 0, st>>>(
      err, n.typed_data(), pae->typed_data(), NMAX);
  confidence::perAlignmentK<<<NMAX, 32, 0, st>>>(
      tm, smask.typed_data(), asym.typed_data(), 0, n.typed_data(), per, NMAX);
  confidence::weightedMaxK<<<1, 32, 0, st>>>(
      per, smask.typed_data(), n.typed_data(), ptm->typed_data(), NMAX);
  confidence::perAlignmentK<<<NMAX, 32, 0, st>>>(
      tm, smask.typed_data(), asym.typed_data(), 1, n.typed_data(),
      iptm_pr->typed_data(), NMAX);
  confidence::weightedMaxK<<<1, 32, 0, st>>>(
      iptm_pr->typed_data(), smask.typed_data(), n.typed_data(),
      iptm->typed_data(), NMAX);
  confidence::expresCaK<<<(NMAX + 127) / 128, 128, 0, st>>>(
      expres.typed_data(), n.typed_data(), expres_ca->typed_data(), NMAX);
  return ffi::Error::Success();
}

ffi::Error KabschImpl(cudaStream_t st, ffi::Buffer<ffi::F32> pred,
                      ffi::Buffer<ffi::F32> pmask, ffi::Buffer<ffi::F32> tmpl,
                      ffi::Buffer<ffi::F32> tmask, ffi::Buffer<ffi::U8> flags,
                      ffi::Buffer<ffi::S32> n,
                      ffi::Result<ffi::Buffer<ffi::F32>> out,
                      ffi::Result<ffi::Buffer<ffi::F32>> scratch) {
  const int NMAX = (int)pred.dimensions()[0];
  float* stats = scratch->typed_data();        // [15] then [9]
  float* R = stats + 15;
  kabsch::covK<<<1, 32, 0, st>>>(pred.typed_data(), tmpl.typed_data(),
                                 tmask.typed_data(), flags.typed_data(),
                                 n.typed_data(), stats, NMAX);
  kabsch::rotationK<<<1, 32, 0, st>>>(stats, R);
  kabsch::applyK<<<(int)(((size_t)NMAX * 37 + 127) / 128), 128, 0, st>>>(
      pred.typed_data(), pmask.typed_data(), stats, R, n.typed_data(),
      out->typed_data(), NMAX);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GlueSeqFeat, SeqFeatImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("sw").Attr<float>("ohw").Attr<float>("temp")
        .Attr<float>("ls").Attr<float>("omitted_half")
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GlueConfidence, ConfidenceImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GlueKabsch, KabschImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::U8>>().Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());

ffi::Error GlueBwdImpl(cudaStream_t st, ffi::Buffer<ffi::F32> seq,
                       ffi::Buffer<ffi::U8> flags, ffi::Buffer<ffi::F32> bias,
                       ffi::Buffer<ffi::F32> d_feat,
                       ffi::Buffer<ffi::F32> d_prof,
                       float sw, float ohw, float temp, float ls,
                       float omitted_half,
                       ffi::Result<ffi::Buffer<ffi::F32>> d_seq) {
  const int N = (int)seq.dimensions()[0];
  seqfeat::Scalars sc{sw, ohw, temp, ls};
  const float* b = bias.element_count() ? bias.typed_data() : nullptr;
  seqfeat::seqFeatBwdK<<<(N + 127) / 128, 128, 0, st>>>(
      seq.typed_data(), flags.typed_data(), b, sc, omitted_half,
      d_feat.typed_data(), d_prof.typed_data(), d_seq->typed_data(), N);
  return ffi::Error::Success();
}

ffi::Error ConfBwdImpl(cudaStream_t st, ffi::Buffer<ffi::F32> lddt,
                       ffi::Buffer<ffi::F32> pae_l,
                       ffi::Buffer<ffi::F32> centres,
                       ffi::Buffer<ffi::F32> expres,
                       ffi::Buffer<ffi::F32> g_plddt,
                       ffi::Buffer<ffi::F32> g_pae,
                       ffi::Buffer<ffi::F32> g_expres,
                       ffi::Buffer<ffi::S32> n,
                       ffi::Result<ffi::Buffer<ffi::F32>> d_lddt,
                       ffi::Result<ffi::Buffer<ffi::F32>> d_pae,
                       ffi::Result<ffi::Buffer<ffi::F32>> d_expres) {
  const int NMAX = (int)pae_l.dimensions()[0];
  const int PB = (int)pae_l.dimensions()[2];
  const size_t np = (size_t)NMAX * NMAX;
  confidence::plddtBwdK<<<(NMAX + 127) / 128, 128, 0, st>>>(
      lddt.typed_data(), g_plddt.typed_data(), n.typed_data(),
      d_lddt->typed_data(), NMAX);
  // pae WRITES; ptm/iptm would accumulate after it -- see confidence.cuh
  confidence::paeBwdK<<<(int)((np + 127) / 128), 128, 0, st>>>(
      pae_l.typed_data(), centres.typed_data(), g_pae.typed_data(),
      n.typed_data(), d_pae->typed_data(), NMAX, PB);
  confidence::expresCaBwdK<<<(int)(((size_t)NMAX * 37 + 127) / 128), 128, 0, st>>>(
      expres.typed_data(), g_expres.typed_data(), n.typed_data(),
      d_expres->typed_data(), NMAX);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GlueSeqFeatBwd, GlueBwdImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Attr<float>("sw").Attr<float>("ohw").Attr<float>("temp")
        .Attr<float>("ls").Attr<float>("omitted_half")
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GlueConfBwd, ConfBwdImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());

// ===================== the whole design step in one call ===============
// OFF THE STREAM KEY, so the whole step can be built at TRACE time.
// design_step::setStream rebinds every member; nothing in the tree is
// stream-ordered.  Without this the fused step builds on first CALL, by
// which point XLA has grown -- measured at complex 487, "before any stack
// allocation, 22.54 GB of 23.68 is already in use", and the step then dies
// with 0.77 GB to work in.  That is the SAME allocation race
// native_stack_prealloc fixed for the staged path, not a larger footprint.
// ONE OBJECT FOR EVERY LENGTH, instead of one per 32-bucket.
//
// The states of a design step are not the same length -- a binder alone is
// 220 where its complex is 607 -- and bc2 pads each to its own bucket, so
// each arrives here with a different NMAX and gets its own DesignStep.  The
// second one is 4.34 GB beside the first's 15.67, and that is precisely what
// puts complex 617 out of memory: XLA then asks for 2.15 GB with 2.33 free.
// Padding every state up to the biggest instead fits, and makes the short
// state run at the long state's length: 25.8 s against 14.2.
//
// DesignStep::setN re-derives the object at a shorter length against the
// buffers it already has, so neither is necessary.  With NB_ONE_OBJECT the
// key drops `nmax` and the longest allocation serves every state.
//
// ON BY DEFAULT since 2026-10-14, and the evidence is exactness rather than
// speed: |grad| against the per-state path is BIT-IDENTICAL at every length
// the two can both run -- complex 487, 547, 587 and 607, 7.54957 / 9.24292
// / 8.75168 / 9.02069 in both arms.  That is the gate that matters, because
// the failure this has to be trusted not to make is silent: a stride that
// kept the long length while the rest moved reads real numbers from the
// wrong place.  Above 608 there is nothing to compare against -- per-state
// does not run there, which is the whole point of this.
//
// NB_ONE_OBJECT=0 goes back to one DesignStep per 32-bucket, and with it
// BC2_NATIVE_STEP_SHARE_N's threshold.  Both paths are KEPT, and not out of
// caution about this one: per-state is the ORACLE the exactness gate
// compares against, and deleting it would delete the gate.
static bool oneObject() {
  static bool on = envOnDef("NB_ONE_OBJECT", true);
  return on;
}

struct DsKey {
  int nmax, recycles, msa_rows;
  // CAN this step drop, not does it drop now: it decides whether the two
  // Evoformer stacks reserve their mask scratches, which happens inside
  // alloc.  Whether a mask is drawn is seed[1], per call.
  int dropout;
  // NO BACKWARD, EVER -- see Plan::lean.  In the key because calling
  // backward() on a lean object dereferences a null, so a design must never
  // be handed one.  `nmax` is dropped under NB_ONE_OBJECT and this is not:
  // sharing a lean object with a design is the failure it prevents.
  //
  // AFTER `dropout`, and the order is not cosmetic: every site builds this
  // with an aggregate initialiser, so a member inserted ABOVE one that is
  // already being passed silently swaps the two.  It did -- `lean` arrived
  // as the dropout flag and dropout as `lean` -- and nothing failed: the
  // key was still consistent with itself, the prealloc still printed LEAN
  // because that print reads its own argument, and the only symptom was a
  // lean allocation that was not lean.  Appending is what makes a new
  // field's absence a compile error at every site instead.
  int lean = 0;
  std::string trunk, extra, tpl, post, emb;
  bool operator<(const DsKey& o) const {
    // `nmax` is deliberately absent under NB_ONE_OBJECT -- see oneObject().
    const int a = oneObject() ? 0 : nmax, b = oneObject() ? 0 : o.nmax;
    return std::tie(a, recycles, msa_rows, dropout, lean, trunk, extra, tpl,
                    post, emb)
         < std::tie(b, o.recycles, o.msa_rows, o.dropout, o.lean, o.trunk,
                    o.extra, o.tpl, o.post, o.emb);
  }
};
static std::map<DsKey, design_step::DesignStep*> g_ds;
design_step::DesignStep* ds_get(const DsKey& k, inputfeat::Tables t,
                                tpl1d::Tables t1, cudaStream_t st,
                                bool defer_keep = false) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto W = [&](const std::string& d) -> Oracle& {
    Oracle*& w = g_weights[d];
    if (!w) w = new Oracle(d);
    return *w;
  };
  // THE REGIME IS READ OFF THE WEIGHTS, not passed as an attribute.
  //
  // alphafold2_ptm pads target_feat on BOTH sides, so `preprocess_1d` is
  // [22, 256] where multimer_v3 is [21, 256] -- the same shape test
  // ckpt.cuh::looksMonomer and alphafold3/af2/convert.py both use, and the
  // one that makes a conversion idempotent.  A new FFI attribute would have
  // to be threaded through every caller in two repositories and could then
  // DISAGREE with the directory it was passed beside; a property of the
  // checkpoint cannot.  It needs no place in DsKey either: the five
  // directories are already in the key, and two regimes never share them.
  bool mono = false;
  {
    Oracle& e = W(k.emb);
    const std::string nm = "param/preprocess_1d/weights";
    mono = e.has(nm) && !e[nm].shape.empty() && e[nm].shape[0] == 22;
  }
  // AND THE MONOMER'S 1-D TEMPLATE ROW NEEDS A TABLE THE CALLERS DO NOT
  // PASS.  `chi_pi_periodic` is what the alternative torsion angles mirror;
  // it exists only on this path, and a fourth buffer argument through every
  // caller to carry 88 floats that are AF2's own constants would be the same
  // churn the regime flag avoids.  BEFORE the cache lookup, because the hit
  // path re-stores `t1` every call -- a fill applied only on construction
  // would hold for one call and be nulled on the next.
  if (mono && !t1.chi_pi_periodic) {
    static float* g_chipp = nullptr;
    if (!g_chipp) {
      CU(cudaMalloc(&g_chipp, sizeof(RES_CHI_PI_PERIODIC)));
      CU(cudaMemcpy(g_chipp, RES_CHI_PI_PERIODIC,
                    sizeof(RES_CHI_PI_PERIODIC), cudaMemcpyHostToDevice));
    }
    t1.chi_pi_periodic = g_chipp;
  }
  auto it = g_ds.find(k);
  if (it != g_ds.end()) {
    it->second->setStream(st);
    // RE-SET THE TABLE POINTERS EVERY CALL.  setWeights STORES them
    // (`tables = t`, `t1d_tab = t1`) and they are device pointers into
    // XLA's buffers, which the trace-time prealloc does not have and which
    // XLA may move between calls anyway.  Host-side and free.  This is the
    // cached-bias-pointer bug from the README in a new place: there, a
    // cuBLASLt descriptor baked a bias POINTER and blocks 1..47 silently
    // used block 0's.
    it->second->tables = t;
    it->second->t1d_tab = t1;
    // The length this CALL is for.  A no-op unless the object is already at
    // it, which is the common case within one state.
    if (oneObject()) it->second->setN(k.nmax);
    return it->second;
  }
  auto* m = new design_step::DesignStep();
  design_step::Plan p{};
  p.nmax = k.nmax; p.recycles = k.recycles; p.msa_rows = k.msa_rows;
  p.trunk_depth = 48; p.extra_depth = 4; p.tpl_depth = 2; p.sm_depth = 8;
  p.defer_keep = defer_keep;
  p.dropout = k.dropout != 0;
  p.lean = k.lean != 0;
  p.monomer = mono;
  m->alloc(p, st);
  m->setWeights(W(k.trunk), W(k.extra), W(k.tpl), W(k.post), W(k.emb), t, t1);
  g_ds[k] = m;
  return m;
}

// THE FUSED STEP, AT TRACE TIME.  Same fix as native_stack_prealloc and for
// the same reason: built on first CALL, the fused step finds XLA already
// holding 22.54 of 23.68 GB at complex 487 and dies with 0.77 GB to work
// in.  It is not a bigger footprint -- it is losing the same allocation
// race, and it loses it harder because ONE call allocates everything where
// the staged path spreads it over several the prealloc already covers.
// THE TWO SAVE-SLOT POLICIES, EXPOSED SO THEY CAN BE COMPARED.
//
// native.py's _save_slots duplicates Stack::saveSlots, and the file says
// what a disagreement costs: XLA allocates the save-set from the shape
// Python declares and this code slices it, so if Python says fewer slots
// than this slices, the backward reads past the end of somebody else's
// buffer.  Not a compile error, not a crash, just a wrong gradient.  Now
// that the policy is a budget calculation rather than one threshold, the
// two are far easier to drift apart -- so there is a way to ask this side
// directly and diff it against the other.
// THE PRODUCTION LOSSES, exposed so loss_gate.py can diff them against the
// registered python function each one claims to be -- value AND gradient,
// because a value-only check passed binder_pae reading the wrong state.
extern "C" int native_loss_plddt_probe(const float* plddt, const float* mask,
                                       int n, float dL,
                                       float* out_value, float* out_dplddt) {
  float *dp = nullptr, *dm = nullptr, *dout = nullptr, *ddp = nullptr;
  CU(::cudaMalloc(&dp, (size_t)n * 4));
  CU(::cudaMalloc(&dm, (size_t)n * 4));
  CU(::cudaMalloc(&dout, 2 * 4));
  CU(::cudaMalloc(&ddp, (size_t)n * 4));
  CU(cudaMemcpy(dp, plddt, (size_t)n * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dm, mask, (size_t)n * 4, cudaMemcpyHostToDevice));
  prodloss::plddtLossK<<<1, 256>>>(dp, dm, n, dout);
  float host[2] = {0, 0};
  CU(cudaMemcpy(host, dout, 2 * 4, cudaMemcpyDeviceToHost));
  prodloss::plddtLossBwdK<<<(n + 255) / 256, 256>>>(dm, host[1], dL, n, ddp);
  CU(cudaMemcpy(out_dplddt, ddp, (size_t)n * 4, cudaMemcpyDeviceToHost));
  *out_value = host[0];
  CU(::cudaFree(dp)); CU(::cudaFree(dm)); CU(::cudaFree(dout)); CU(::cudaFree(ddp));
  return 0;
}

// binder_helicity: _masked_mean(distogram_pair_loss(dg, bins < 6.0, BINARY),
// helical_pair_mask).  The masks are passed IN rather than rebuilt here --
// the gate is for the numerics, which is where a port goes wrong, not for
// python's mask construction, which the driver will call separately.
extern "C" int native_loss_helicity_probe(const float* dg, const char* pairmask,
                                          int n, int B, float cutoff, float dL,
                                          float* out_value, float* out_ddg) {
  const size_t np = (size_t)n * n, nb = np * (size_t)B;
  float *ddg_, *dpl, *dout, *dddg; char *dsel, *dpm;
  CU(::cudaMalloc(&ddg_, nb * 4)); CU(::cudaMalloc(&dpl, np * 4));
  CU(::cudaMalloc(&dout, 2 * 4)); CU(::cudaMalloc(&dddg, nb * 4));
  CU(::cudaMalloc(&dsel, (size_t)B)); CU(::cudaMalloc(&dpm, np));
  CU(cudaMemcpy(ddg_, dg, nb * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dpm, pairmask, np, cudaMemcpyHostToDevice));
  prodloss::binSelectK<<<(B + 63) / 64, 64>>>(B, cutoff, dsel);
  prodloss::pairLossK<<<(int)((np + 255) / 256), 256>>>(ddg_, dsel, n, n, B, 1, dpl);
  prodloss::maskedMean2dK<<<1, 256>>>(dpl, dpm, np, dout);
  float host[2] = {0, 0};
  CU(cudaMemcpy(host, dout, 2 * 4, cudaMemcpyDeviceToHost));
  float* dpl_g; CU(::cudaMalloc(&dpl_g, np * 4));
  prodloss::maskedMean2dBwdK<<<(int)((np + 255) / 256), 256>>>(dpm, host[1], dL, np, dpl_g);
  prodloss::pairLossBwdK<<<(int)((np + 255) / 256), 256>>>(ddg_, dsel, dpl_g, n, n, B, 1, dddg);
  CU(cudaMemcpy(out_ddg, dddg, nb * 4, cudaMemcpyDeviceToHost));
  *out_value = host[0];
  for (void* q : {(void*)ddg_, (void*)dpl, (void*)dout, (void*)dddg,
                  (void*)dsel, (void*)dpm, (void*)dpl_g}) CU(::cudaFree(q));
  return 0;
}

// distogram_contact_loss: pair_loss (NON-binary branch) -> best_contact_mean
// along pairs with contacts_per_residue -> best_contact_mean along residues
// with contact_residue_count.  binder_contacts and interface_contacts are
// both this with different masks and constants, which is the point of
// building the primitive rather than the loss.
//
// `crc < 0` means float('inf') -- every masked residue counts.
extern "C" int native_loss_contacts_probe(const float* dg, const char* pairmask,
                                          const char* rowmask, int n, int B,
                                          float cutoff, int cpr, int crc,
                                          float dL, float* out_value,
                                          float* out_ddg) {
  const size_t np = (size_t)n * n, nb = np * (size_t)B;
  float *ddg_, *dpl, *drow, *dcnt1, *dcnt2, *dout, *dddg, *dpl_g, *drow_g;
  char *dsel, *dpm, *drm, *dsel1, *dsel2;
  CU(::cudaMalloc(&ddg_, nb * 4));   CU(::cudaMalloc(&dpl, np * 4));
  CU(::cudaMalloc(&drow, (size_t)n * 4));
  CU(::cudaMalloc(&dcnt1, (size_t)n * 4)); CU(::cudaMalloc(&dcnt2, 4));
  CU(::cudaMalloc(&dout, 4));        CU(::cudaMalloc(&dddg, nb * 4));
  CU(::cudaMalloc(&dpl_g, np * 4));  CU(::cudaMalloc(&drow_g, (size_t)n * 4));
  CU(::cudaMalloc(&dsel, (size_t)B));
  CU(::cudaMalloc(&dpm, np));        CU(::cudaMalloc(&drm, (size_t)n));
  CU(::cudaMalloc(&dsel1, np));      CU(::cudaMalloc(&dsel2, (size_t)n));
  CU(cudaMemcpy(ddg_, dg, nb * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dpm, pairmask, np, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(drm, rowmask, (size_t)n, cudaMemcpyHostToDevice));
  const int kc = cpr < 0 ? n : cpr, kr = crc < 0 ? n : crc;

  prodloss::binSelectK<<<(B + 63) / 64, 64>>>(B, cutoff, dsel);
  prodloss::pairLossK<<<(int)((np + 255) / 256), 256>>>(ddg_, dsel, n, n, B, 0, dpl);
  prodloss::bestContactMeanK<<<n, 1>>>(dpl, dpm, n, n, kc, drow, dsel1, dcnt1);
  prodloss::bestContactMeanK<<<1, 1>>>(drow, drm, 1, n, kr, dout, dsel2, dcnt2);
  CU(cudaMemcpy(out_value, dout, 4, cudaMemcpyDeviceToHost));

  float dLh = dL;
  float* dLd; CU(::cudaMalloc(&dLd, 4));
  CU(cudaMemcpy(dLd, &dLh, 4, cudaMemcpyHostToDevice));
  prodloss::bestContactMeanBwdK<<<(n + 255) / 256, 256>>>(dsel2, dcnt2, dLd, 1, n, drow_g);
  prodloss::bestContactMeanBwdK<<<(int)((np + 255) / 256), 256>>>(dsel1, dcnt1, drow_g, n, n, dpl_g);
  prodloss::pairLossBwdK<<<(int)((np + 255) / 256), 256>>>(ddg_, dsel, dpl_g, n, n, B, 0, dddg);
  CU(cudaMemcpy(out_ddg, dddg, nb * 4, cudaMemcpyDeviceToHost));

  for (void* q : {(void*)ddg_, (void*)dpl, (void*)drow, (void*)dcnt1, (void*)dcnt2,
                  (void*)dout, (void*)dddg, (void*)dpl_g, (void*)drow_g, (void*)dsel,
                  (void*)dpm, (void*)drm, (void*)dsel1, (void*)dsel2, (void*)dLd})
    CU(::cudaFree(q));
  return 0;
}

// chain_pair_pae_loss, which is both binder_pae and interface_pae.  `w` is
// the outer product of the two chain-group weight vectors, built by the
// caller: rows x cols need not be square (interface_pae is binder x target).
extern "C" int native_loss_pae_probe(const float* pae, const float* w,
                                     int rows, int cols, float dL,
                                     float* out_value, float* out_dpae) {
  const size_t n = (size_t)rows * cols;
  float *dp, *dw, *dout, *dd;
  CU(::cudaMalloc(&dp, n * 4)); CU(::cudaMalloc(&dw, n * 4));
  CU(::cudaMalloc(&dout, 2 * 4)); CU(::cudaMalloc(&dd, n * 4));
  CU(cudaMemcpy(dp, pae, n * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dw, w, n * 4, cudaMemcpyHostToDevice));
  const float scale = 1.f / 31.f;
  prodloss::weightedMaskedMeanK<<<1, 256>>>(dp, dw, scale, n, dout);
  float host[2] = {0, 0};
  CU(cudaMemcpy(host, dout, 2 * 4, cudaMemcpyDeviceToHost));
  prodloss::weightedMaskedMeanBwdK<<<(int)((n + 255) / 256), 256>>>(
      dw, host[1], dL, scale, n, dd);
  CU(cudaMemcpy(out_dpae, dd, n * 4, cudaMemcpyDeviceToHost));
  *out_value = host[0];
  for (void* q : {(void*)dp, (void*)dw, (void*)dout, (void*)dd}) CU(::cudaFree(q));
  return 0;
}

// compactness (radius_of_gyration_loss).  `gr` is 2.38 * n_real^0.365 and
// `count` the designed-chain weight sum, both computed by the caller.
extern "C" int native_loss_compactness_probe(const float* xyz, const float* m,
                                             int n, float gr, float count,
                                             float dL, float* out_value,
                                             float* out_dxyz) {
  float *dx, *dm, *dfw, *dsum, *dd;
  CU(::cudaMalloc(&dx, (size_t)n * 12)); CU(::cudaMalloc(&dm, (size_t)n * 4));
  CU(::cudaMalloc(&dfw, 8 * 4)); CU(::cudaMalloc(&dsum, 3 * 4));
  CU(::cudaMalloc(&dd, (size_t)n * 12));
  CU(cudaMemset(dd, 0, (size_t)n * 12));
  CU(cudaMemcpy(dx, xyz, (size_t)n * 12, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dm, m, (size_t)n * 4, cudaMemcpyHostToDevice));
  prodloss::radiusK<<<1, 256>>>(dx, dm, n, gr, dfw);
  float fw[8];
  CU(cudaMemcpy(fw, dfw, 8 * 4, cudaMemcpyDeviceToHost));
  prodloss::radiusSumK<<<1, 256>>>(dx, dm, dfw, n, dsum);
  prodloss::radiusBwdK<<<(n + 255) / 256, 256>>>(dx, dm, dfw, n, dL / count, dsum, dd);
  CU(cudaMemcpy(out_dxyz, dd, (size_t)n * 12, cudaMemcpyDeviceToHost));
  *out_value = fw[0] / count;
  for (void* q : {(void*)dx, (void*)dm, (void*)dfw, (void*)dsum, (void*)dd})
    CU(::cudaFree(q));
  return 0;
}

// act_const, gated against bc2's own __const_only__ harvest.  The weights
// are loaded BY NAME from the template dump so the eight Linears cannot be
// bound to each other -- the failure bc2's docstring warns about.
extern "C" int native_act_const_probe(const char* tpl_dir,
                                      const float* pb, const float* pbm,
                                      const float* atoms, const float* amask,
                                      const int32_t* aatype, const int32_t* asym,
                                      int n, int nmax, float* out) {
  Oracle o(tpl_dir);
  auto dev = [&](const std::string& k) { o[k].to_device(); return o[k].dev; };
  const float* W[8];
  std::vector<float> bsum(actconst::CH, 0.f);
  for (int i = 0; i < 8; ++i) {
    const std::string base = "param/template_pair_embedding_" + std::to_string(i);
    W[i] = dev(base + "/weights");
    Tensor& bt = o[base + "/bias"];
    for (int c = 0; c < actconst::CH; ++c) bsum[c] += bt.host[c];
  }
  float *dpb, *dpbm, *datoms, *damask, *dB, *dout;
  int32_t *daa, *dasym;
  const size_t nn = (size_t)nmax * nmax * actconst::CH;
  CU(::cudaMalloc(&dpb, (size_t)n * 3 * 4));   CU(::cudaMalloc(&dpbm, (size_t)n * 4));
  CU(::cudaMalloc(&datoms, (size_t)n * 37 * 3 * 4));
  CU(::cudaMalloc(&damask, (size_t)n * 37 * 4));
  CU(::cudaMalloc(&daa, (size_t)n * 4));       CU(::cudaMalloc(&dasym, (size_t)n * 4));
  CU(::cudaMalloc(&dB, actconst::CH * 4));     CU(::cudaMalloc(&dout, nn * 4));
  CU(cudaMemset(dout, 0, nn * 4));
  CU(cudaMemcpy(dpb, pb, (size_t)n * 3 * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dpbm, pbm, (size_t)n * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(datoms, atoms, (size_t)n * 37 * 3 * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(damask, amask, (size_t)n * 37 * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(daa, aatype, (size_t)n * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dasym, asym, (size_t)n * 4, cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dB, bsum.data(), actconst::CH * 4, cudaMemcpyHostToDevice));
  dim3 g(n, n);
  actconst::actConstK<<<g, actconst::CH>>>(dpb, dpbm, datoms, damask, daa, dasym,
                                           n, nmax, W[0], W[1], W[2], W[3],
                                           W[4], W[5], W[6], W[7], dB, dout);
  CU(cudaDeviceSynchronize());
  CU(cudaMemcpy(out, dout, nn * 4, cudaMemcpyDeviceToHost));
  for (void* q : {(void*)dpb,(void*)dpbm,(void*)datoms,(void*)damask,
                  (void*)daa,(void*)dasym,(void*)dB,(void*)dout}) CU(::cudaFree(q));
  return 0;
}

extern "C" int native_save_slots_probe(int depth, int n) {
  stack::Stack s;
  s.N = n;
  s.depth = depth;
  return s.saveSlots();
}

extern "C" int native_design_step_prealloc(int nmax, int recycles,
                                           int msa_rows, int dropout,
                                           const char* trunk_dir,
                                           const char* extra_dir,
                                           const char* tpl_dir,
                                           const char* post_dir,
                                           const char* emb_dir) {
  size_t fb0 = 0, tb = 0;
  if (cudaMemGetInfo(&fb0, &tb) != cudaSuccess) return 1;
  // HOW LONG, not just how much.  The harness reports ONE "compile" number
  // and it is XLA's tracing and compilation PLUS this prealloc PLUS the
  // weight conversion inside it.  Those are different problems with
  // different fixes, and at complex 987 on an A100 that number is 29.3 s
  // with nothing to say which part to look at.
  const auto t_pre0 = std::chrono::steady_clock::now();
  DsKey k{nmax, recycles, msa_rows, dropout, /*lean*/0, std::string(trunk_dir),
          std::string(extra_dir), std::string(tpl_dir), std::string(post_dir),
          std::string(emb_dir)};
  ds_get(k, inputfeat::Tables{}, tpl1d::Tables{}, /*stream*/nullptr,
         /*defer_keep*/true);
  size_t fb1 = 0;
  cudaMemGetInfo(&fb1, &tb);
  fprintf(stderr, "native: prealloc design_step NMAX=%d built in %.2f s\n",
          nmax, std::chrono::duration<double>(
              std::chrono::steady_clock::now() - t_pre0).count());
  fprintf(stderr,
          "native: prealloc design_step NMAX=%d took %.2f GB "
          "(%.2f of %.2f GB free after)\n",
          nmax, (fb0 - fb1) / 1e9, fb1 / 1e9, tb / 1e9);
  return 0;
}


// THE PREDICT-ONLY STEP.  Same operands as the fused design step, same
// eight outputs, and NO carry: the save set and pout/mout that the design
// forward hands to its backward are not produced, because there is no
// backward to hand them to.
//
// This is a MEMORY feature, not a speed one, and the distinction cost a
// wrong claim.  A forward is ~1/3 of a design step (176 ms against 589 at
// complex 175), but that third is what the BACKWARD costs -- the saving
// here is the four carry buffers XLA no longer allocates, the biggest of
// which is slots * N^2 * 128 bf16, plus the keep tiers, which exist for a
// backward that is not coming (no regrant() below).
//
// The NUMBERS must not move: predictBuffers() keeps the bf16 narrowing at
// the head of every block, which is the approximation the design forward
// makes, and only collapses the 48 kept inputs into one reused slot.
// ds_predict_gate.py is what holds that: the eight outputs of this handler
// against the eight of NativeDesignStep on identical inputs.
// ...AND THE PREDICT PATH'S OWN BUFFERS, at trace time for the same reason
// everything else here is: the one reused save slot and the trunk's own
// pout/mout are a ~285 MB cudaMalloc at complex 608, and asked for on the
// first CALL it is asked for with XLA already holding the card.
extern "C" int native_design_step_predict_prealloc(
    int nmax, int recycles, int msa_rows, int dropout, int lean,
    const char* trunk_dir,
    const char* extra_dir, const char* tpl_dir, const char* post_dir,
    const char* emb_dir) {
  // SAY WHAT IT COST.  The design prealloc reports and this did not, so
  // under NB_PREDICT_LEAN -- the one path that does not go through the
  // design prealloc at all -- nothing printed and the footprint the flag
  // exists to change was invisible.  Measuring it was how the first lean
  // number came back as "saves 0.13 GB", which was the reporting, not the
  // allocation.
  size_t fb0 = 0, tb = 0;
  if (cudaMemGetInfo(&fb0, &tb) != cudaSuccess) return 1;
  DsKey k{nmax, recycles, msa_rows, dropout, lean, std::string(trunk_dir),
          std::string(extra_dir), std::string(tpl_dir), std::string(post_dir),
          std::string(emb_dir)};
  auto* m = ds_get(k, inputfeat::Tables{}, tpl1d::Tables{}, /*stream*/nullptr,
                   /*defer_keep*/true);
  m->predictBuffers();
  size_t fb1 = 0;
  cudaMemGetInfo(&fb1, &tb);
  fprintf(stderr,
          "native: prealloc predict%s NMAX=%d took %.2f GB "
          "(%.2f of %.2f GB free after)\n", lean ? " LEAN" : "", nmax,
          (double)(fb0 - fb1) / 1e9, fb1 / 1e9, tb / 1e9);
  return 0;
}

ffi::Error DesignStepPredictImpl(
    cudaStream_t st,
    ffi::Buffer<ffi::F32> sequence, ffi::Buffer<ffi::F32> atoms,
    ffi::Buffer<ffi::F32> atom_mask, ffi::Buffer<ffi::U8> flags,
    ffi::Buffer<ffi::S32> residue_index, ffi::Buffer<ffi::S32> asym_id,
    ffi::Buffer<ffi::S32> entity_id, ffi::Buffer<ffi::S32> interface_asym,
    ffi::Buffer<ffi::F32> seq_mask, ffi::Buffer<ffi::F32> act_const,
    ffi::Buffer<ffi::S32> n, ffi::Buffer<ffi::U32> seed,
    ffi::Buffer<ffi::F32> t14m, ffi::Buffer<ffi::F32> t37m,
    ffi::Buffer<ffi::S32> t1437, ffi::Buffer<ffi::S32> t3714,
    ffi::Buffer<ffi::S32> chi_idx, ffi::Buffer<ffi::F32> chi_mask,
    ffi::Buffer<ffi::F32> scalars,
    int64_t recycles, int64_t msa_rows, int64_t dropout, int64_t lean,
    std::string_view trunk_dir, std::string_view extra_dir,
    std::string_view tpl_dir, std::string_view post_dir,
    std::string_view emb_dir,
    int64_t cyclic_mode, float flexibility,
    ffi::Result<ffi::Buffer<ffi::F32>> out_atoms,
    ffi::Result<ffi::Buffer<ffi::F32>> out_plddt,
    ffi::Result<ffi::Buffer<ffi::F32>> out_pae,
    ffi::Result<ffi::Buffer<ffi::F32>> out_ptm,
    ffi::Result<ffi::Buffer<ffi::F32>> out_iptm,
    ffi::Result<ffi::Buffer<ffi::F32>> out_iptm_pr,
    ffi::Result<ffi::Buffer<ffi::F32>> out_expres,
    ffi::Result<ffi::Buffer<ffi::F32>> out_dgram) {
  const int NMAX = (int)sequence.dimensions()[0];
  inputfeat::Tables t{t14m.typed_data(), t37m.typed_data(),
                      t1437.typed_data(), t3714.typed_data()};
  tpl1d::Tables t1{chi_idx.typed_data(), chi_mask.typed_data()};
  DsKey k{NMAX, (int)recycles, (int)msa_rows, (int)dropout, /*lean*/lean,
          std::string(trunk_dir),
          std::string(extra_dir), std::string(tpl_dir), std::string(post_dir),
          std::string(emb_dir)};
  auto* m = ds_get(k, t, t1, st);
  m->st = st;
  // bc2's cyclic_offset_mode and target_flexibility, this call's -- one
  // object serves every state, so they are set per call, not per object
  m->plan.cyclic_mode = (int)cyclic_mode;
  m->plan.flexibility = flexibility;

  design_step::Inputs in{};
  in.sequence = sequence.typed_data();
  in.atoms = atoms.typed_data();
  in.atom_mask = atom_mask.typed_data();
  in.flags = flags.typed_data();
  in.residue_index = residue_index.typed_data();
  in.asym_id = asym_id.typed_data();
  in.entity_id = entity_id.typed_data();
  in.interface_asym = interface_asym.typed_data();
  in.seq_mask = seq_mask.typed_data();
  in.act_const = act_const.typed_data();
  in.n = n.typed_data();
  in.seed = seed.typed_data();
  in.scalars = scalars.typed_data();

  design_step::Outputs out{};
  out.atoms = out_atoms->typed_data();
  out.plddt = out_plddt->typed_data();
  out.pae = out_pae->typed_data();
  out.ptm = out_ptm->typed_data();
  out.iptm = out_iptm->typed_data();
  out.iptm_per_residue = out_iptm_pr->typed_data();
  out.expres_ca = out_expres->typed_data();
  out.distogram = out_dgram->typed_data();

  const bool tm = envOn("NB_DS_TIME");
  cudaEvent_t f0, f1;
  if (tm) {
    CU(cudaEventCreate(&f0)); CU(cudaEventCreate(&f1));
    CU(cudaEventRecord(f0, st));
  }
  m->forward(in, out, /*want_save*/false);
  if (tm) {
    float a = 0;
    CU(cudaEventRecord(f1, st));
    CU(cudaEventSynchronize(f1));
    CU(cudaEventElapsedTime(&a, f0, f1));
    printf("NB_DS_TIME N=%d predict %7.2f ms\n", NMAX, a);
    fflush(stdout);
    CU(cudaEventDestroy(f0)); CU(cudaEventDestroy(f1));
  }
  return ffi::Error::Success();
}

ffi::Error DesignStepImpl(
    cudaStream_t st,
    ffi::Buffer<ffi::F32> sequence, ffi::Buffer<ffi::F32> atoms,
    ffi::Buffer<ffi::F32> atom_mask, ffi::Buffer<ffi::U8> flags,
    ffi::Buffer<ffi::S32> residue_index, ffi::Buffer<ffi::S32> asym_id,
    ffi::Buffer<ffi::S32> entity_id, ffi::Buffer<ffi::S32> interface_asym,
    ffi::Buffer<ffi::F32> seq_mask, ffi::Buffer<ffi::F32> act_const,
    ffi::Buffer<ffi::S32> n, ffi::Buffer<ffi::U32> seed,
    ffi::Buffer<ffi::F32> t14m, ffi::Buffer<ffi::F32> t37m,
    ffi::Buffer<ffi::S32> t1437, ffi::Buffer<ffi::S32> t3714,
    ffi::Buffer<ffi::S32> chi_idx, ffi::Buffer<ffi::F32> chi_mask,
    ffi::Buffer<ffi::F32> scalars,
    int64_t recycles, int64_t msa_rows, int64_t dropout,
    std::string_view trunk_dir, std::string_view extra_dir,
    std::string_view tpl_dir, std::string_view post_dir,
    std::string_view emb_dir,
    int64_t cyclic_mode, float flexibility,
    ffi::Result<ffi::Buffer<ffi::F32>> out_atoms,
    ffi::Result<ffi::Buffer<ffi::F32>> out_plddt,
    ffi::Result<ffi::Buffer<ffi::F32>> out_pae,
    ffi::Result<ffi::Buffer<ffi::F32>> out_ptm,
    ffi::Result<ffi::Buffer<ffi::F32>> out_iptm,
    ffi::Result<ffi::Buffer<ffi::F32>> out_iptm_pr,
    ffi::Result<ffi::Buffer<ffi::F32>> out_expres,
    ffi::Result<ffi::Buffer<ffi::F32>> out_dgram,
    // the trunk, carried to the backward: save set + the two outputs
    ffi::Result<ffi::Buffer<ffi::BF16>> carry_psave,
    ffi::Result<ffi::Buffer<ffi::F32>> carry_msave,
    ffi::Result<ffi::Buffer<ffi::F32>> carry_pout,
    ffi::Result<ffi::Buffer<ffi::F32>> carry_mout) {
  const int NMAX = (int)sequence.dimensions()[0];
  inputfeat::Tables t{t14m.typed_data(), t37m.typed_data(),
                      t1437.typed_data(), t3714.typed_data()};
  tpl1d::Tables t1{chi_idx.typed_data(), chi_mask.typed_data()};
  DsKey k{NMAX, (int)recycles, (int)msa_rows, (int)dropout, /*lean*/0,
          std::string(trunk_dir),
          std::string(extra_dir), std::string(tpl_dir), std::string(post_dir),
          std::string(emb_dir)};
  auto* m = ds_get(k, t, t1, st);
  m->st = st;
  // bc2's cyclic_offset_mode and target_flexibility, this call's -- one
  // object serves every state, so they are set per call, not per object
  m->plan.cyclic_mode = (int)cyclic_mode;
  m->plan.flexibility = flexibility;

  design_step::Inputs in{};
  in.sequence = sequence.typed_data();
  in.atoms = atoms.typed_data();
  in.atom_mask = atom_mask.typed_data();
  in.flags = flags.typed_data();
  in.residue_index = residue_index.typed_data();
  in.asym_id = asym_id.typed_data();
  in.entity_id = entity_id.typed_data();
  in.interface_asym = interface_asym.typed_data();
  in.seq_mask = seq_mask.typed_data();
  in.act_const = act_const.typed_data();
  in.n = n.typed_data();                 // RUNTIME length, read on device
  in.seed = seed.typed_data();
  in.scalars = scalars.typed_data();   // [4], on the device, NOT attributes

  design_step::Outputs out{};
  out.atoms = out_atoms->typed_data();
  out.plddt = out_plddt->typed_data();
  out.pae = out_pae->typed_data();
  out.ptm = out_ptm->typed_data();
  out.iptm = out_iptm->typed_data();
  out.iptm_per_residue = out_iptm_pr->typed_data();
  out.expres_ca = out_expres->typed_data();
  out.distogram = out_dgram->typed_data();

  // THE TRUNK'S SAVE SET AND OUTPUTS GO TO XLA, not to members, so the
  // backward can use them instead of recomputing 48 blocks.  See
  // design_step.cuh's setTrunkCarry.
  m->setTrunkCarry(reinterpret_cast<__nv_bfloat16*>(carry_psave->typed_data()),
                   carry_msave->typed_data(), carry_pout->typed_data(),
                   carry_mout->typed_data());
  // Hand the two stacks their keep tiers now that XLA has stopped growing.
  // The stack handler does this for the staged path; nothing did it here.
  m->regrant();
  m->skip_trunk = false;
#if NE_CUPTI
  // The fused step never reaches the stack handler's CUPTI window, so it had
  // no kernel breakdown at all -- which is how the +15% at complex 215 stayed
  // a guess.  Same window, counted in fused forwards: one a state a step.
  if (envOn("NE_CUPTI")) {
    static int steps = 0;
    const int warm = getenv("NE_CUPTI_WARM") ? atoi(getenv("NE_CUPTI_WARM")) : 6;
    const int span = getenv("NE_CUPTI_SPAN") ? atoi(getenv("NE_CUPTI_SPAN")) : 2;
    ++steps;
    if (steps == warm) { cuprof::init(); cuprof::start(); }
    else if (steps == warm + span) cuprof::stop(getenv("NE_CUPTI_TOP") ? atoi(getenv("NE_CUPTI_TOP")) : 40);
  }
#endif
  const bool tm = envOn("NB_DS_TIME");
  cudaEvent_t f0, f1;
  if (tm) {
    CU(cudaEventCreate(&f0)); CU(cudaEventCreate(&f1));
    CU(cudaEventRecord(f0, st));
  }
  m->forward(in, out, /*want_save*/true);
  if (tm) {
    float a = 0;
    CU(cudaEventRecord(f1, st));
    CU(cudaEventSynchronize(f1));
    CU(cudaEventElapsedTime(&a, f0, f1));
    printf("NB_DS_TIME N=%d forward %7.2f ms\n", NMAX, a);
    fflush(stdout);
    CU(cudaEventDestroy(f0)); CU(cudaEventDestroy(f1));
  }
  return ffi::Error::Success();
}

// THE BACKWARD IS A FORWARD TOO.  design_step's backward reads what the
// forward left -- the head logits, the per-alignment vector, the Kabsch
// rotation -- so a backward-only custom call has to run the forward first,
// into the scratch outputs the step carries for exactly this.  It is not a
// replay-free shortcut: the components inside still replay, which is what
// makes recycling safe (see the Cotangents note in design_step.cuh).
ffi::Error DesignStepBwdImpl(
    cudaStream_t st,
    ffi::Buffer<ffi::F32> sequence, ffi::Buffer<ffi::F32> atoms,
    ffi::Buffer<ffi::F32> atom_mask, ffi::Buffer<ffi::U8> flags,
    ffi::Buffer<ffi::S32> residue_index, ffi::Buffer<ffi::S32> asym_id,
    ffi::Buffer<ffi::S32> entity_id, ffi::Buffer<ffi::S32> interface_asym,
    ffi::Buffer<ffi::F32> seq_mask, ffi::Buffer<ffi::F32> act_const,
    ffi::Buffer<ffi::S32> n, ffi::Buffer<ffi::U32> seed,
    ffi::Buffer<ffi::F32> t14m, ffi::Buffer<ffi::F32> t37m,
    ffi::Buffer<ffi::S32> t1437, ffi::Buffer<ffi::S32> t3714,
    ffi::Buffer<ffi::S32> chi_idx, ffi::Buffer<ffi::F32> chi_mask,
    ffi::Buffer<ffi::F32> ct_atoms, ffi::Buffer<ffi::F32> ct_plddt,
    ffi::Buffer<ffi::F32> ct_pae, ffi::Buffer<ffi::F32> ct_ptm,
    ffi::Buffer<ffi::F32> ct_iptm, ffi::Buffer<ffi::F32> ct_iptm_pr,
    ffi::Buffer<ffi::F32> ct_expres, ffi::Buffer<ffi::F32> ct_dgram,
    ffi::Buffer<ffi::F32> scalars,
    // ALL Args BEFORE ALL Attrs: the binder's order is the parameter
    // order, and getting it wrong is a static_assert inside api.h that
    // prints the whole signature twice and names nothing.
    ffi::Buffer<ffi::BF16> carry_psave, ffi::Buffer<ffi::F32> carry_msave,
    ffi::Buffer<ffi::F32> carry_pout, ffi::Buffer<ffi::F32> carry_mout,
    int64_t recycles, int64_t msa_rows, int64_t dropout,
    std::string_view trunk_dir, std::string_view extra_dir,
    std::string_view tpl_dir, std::string_view post_dir,
    std::string_view emb_dir,
    int64_t cyclic_mode, float flexibility,
    ffi::Result<ffi::Buffer<ffi::F32>> d_sequence) {
  const int NMAX = (int)sequence.dimensions()[0];
  inputfeat::Tables t{t14m.typed_data(), t37m.typed_data(),
                      t1437.typed_data(), t3714.typed_data()};
  tpl1d::Tables t1{chi_idx.typed_data(), chi_mask.typed_data()};
  DsKey k{NMAX, (int)recycles, (int)msa_rows, (int)dropout, /*lean*/0,
          std::string(trunk_dir),
          std::string(extra_dir), std::string(tpl_dir), std::string(post_dir),
          std::string(emb_dir)};
  auto* m = ds_get(k, t, t1, st);
  m->st = st;
  // bc2's cyclic_offset_mode and target_flexibility, this call's -- one
  // object serves every state, so they are set per call, not per object
  m->plan.cyclic_mode = (int)cyclic_mode;
  m->plan.flexibility = flexibility;

  design_step::Inputs in{};
  in.sequence = sequence.typed_data();
  in.atoms = atoms.typed_data();
  in.atom_mask = atom_mask.typed_data();
  in.flags = flags.typed_data();
  in.residue_index = residue_index.typed_data();
  in.asym_id = asym_id.typed_data();
  in.entity_id = entity_id.typed_data();
  in.interface_asym = interface_asym.typed_data();
  in.seq_mask = seq_mask.typed_data();
  in.act_const = act_const.typed_data();
  in.n = n.typed_data();
  in.seed = seed.typed_data();
  in.scalars = scalars.typed_data();

  // THE REPLAY, MINUS THE TRUNK.  The caller hands back the save set and
  // pout/mout the forward wrote, so emb/tpl/extra/sm/hd rerun (cheap and
  // stateless) while the 48 blocks do not.  NB_NO_TRUNK_CARRY falls back
  // to the full replay, which is what this did before and is the thing to
  // compare against.
  const bool carry = !getenv("NB_NO_TRUNK_CARRY");
  if (carry)
    m->setTrunkCarry(
        reinterpret_cast<__nv_bfloat16*>(carry_psave.typed_data()),
        carry_msave.typed_data(), carry_pout.typed_data(),
        carry_mout.typed_data());
  m->skip_trunk = carry;
  // NB_DS_TIME: what the replay still costs, against the backward it
  // serves.  The carry took 48 of 54 blocks out of it; whether the six
  // that remain are worth carrying too is a question only this answers,
  // and it is small enough at the lengths that matter (2% at 487) that
  // guessing would be how the next day gets spent.
  const bool tm = envOn("NB_DS_TIME");
  cudaEvent_t e0, e1, e2;
  if (tm) {
    CU(cudaEventCreate(&e0)); CU(cudaEventCreate(&e1));
    CU(cudaEventCreate(&e2));
    CU(cudaEventRecord(e0, st));
  }
  m->forward(in, m->scratch, /*want_save*/true);
  m->skip_trunk = false;
  if (tm) CU(cudaEventRecord(e1, st));

  design_step::Cotangents ct{};
  ct.atoms = ct_atoms.typed_data();
  ct.plddt = ct_plddt.typed_data();
  ct.pae = ct_pae.typed_data();
  ct.ptm = ct_ptm.typed_data();
  ct.iptm = ct_iptm.typed_data();
  ct.iptm_per_residue = ct_iptm_pr.typed_data();
  ct.expres_ca = ct_expres.typed_data();
  ct.distogram = ct_dgram.typed_data();
  m->backward(in, ct, d_sequence->typed_data());
  if (tm) {
    float a = 0, b = 0;
    CU(cudaEventRecord(e2, st));
    CU(cudaEventSynchronize(e2));
    CU(cudaEventElapsedTime(&a, e0, e1));
    CU(cudaEventElapsedTime(&b, e1, e2));
    printf("NB_DS_TIME N=%d replay %7.2f ms  backward %7.2f ms  "
           "replay is %4.1f%% of the two\n", NMAX, a, b,
           100.0 * a / (a + b));
    fflush(stdout);
    CU(cudaEventDestroy(e0)); CU(cudaEventDestroy(e1));
    CU(cudaEventDestroy(e2));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NativeDesignStepBwd, DesignStepBwdImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // the four design scalars [4]
        // the trunk carried from the forward: save set + pout/mout
        .Arg<ffi::Buffer<ffi::BF16>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Attr<int64_t>("recycles").Attr<int64_t>("msa_rows")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("trunk_dir")
        .Attr<std::string_view>("extra_dir")
        .Attr<std::string_view>("tpl_dir")
        .Attr<std::string_view>("post_dir")
        .Attr<std::string_view>("emb_dir")
        .Attr<int64_t>("cyclic_mode").Attr<float>("flexibility")
        .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NativeDesignStepPredict, DesignStepPredictImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // the four design scalars [4]
        .Attr<int64_t>("recycles").Attr<int64_t>("msa_rows")
        .Attr<int64_t>("dropout").Attr<int64_t>("lean")
        .Attr<std::string_view>("trunk_dir")
        .Attr<std::string_view>("extra_dir")
        .Attr<std::string_view>("tpl_dir")
        .Attr<std::string_view>("post_dir")
        .Attr<std::string_view>("emb_dir")
        .Attr<int64_t>("cyclic_mode").Attr<float>("flexibility")
        // EIGHT results, not twelve: no carry.
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NativeDesignStep, DesignStepImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::U8>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::F32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::S32>>()
        .Arg<ffi::Buffer<ffi::S32>>().Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()      // the four design scalars [4]
        .Attr<int64_t>("recycles").Attr<int64_t>("msa_rows")
        .Attr<int64_t>("dropout")
        .Attr<std::string_view>("trunk_dir")
        .Attr<std::string_view>("extra_dir")
        .Attr<std::string_view>("tpl_dir")
        .Attr<std::string_view>("post_dir")
        .Attr<std::string_view>("emb_dir")
        .Attr<int64_t>("cyclic_mode").Attr<float>("flexibility")
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::BF16>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());
