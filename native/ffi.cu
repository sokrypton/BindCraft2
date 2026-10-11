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
#include "structmod.cuh"
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
                  bool pair_only, bool opm_first, bool dropout) {
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
  s->forward(pb, mb, pair_mask.typed_data(), msa_mask.typed_data());
  size_t np = (size_t)N * N * block::CZ, nm = (size_t)S * N * block::CM;
  CU(cudaMemcpyAsync(pair_out->typed_data(), s->pout, np * 4,
                     cudaMemcpyDeviceToDevice, stream));
  CU(cudaMemcpyAsync(msa_out->typed_data(), s->mout, nm * 4,
                     cudaMemcpyDeviceToDevice, stream));
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
    peek("s->pout", s->pout);
    peek("pair_out", pair_out->typed_data());
    peek("s->mout", s->mout);
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

// KEYED ON THE STREAM.  foldstack owns a tree of cuBLAS handles and has no
// setStream, so rather than thread a setter through it the object is built on
// the stream it will run on.  XLA hands these calls one stream, so this is one
// object in practice; a second stream would simply get its own.
struct SKey {
  int N, depth;
  std::string dir;
  cudaStream_t stream;
  bool operator<(const SKey& o) const {
    return std::tie(N, depth, dir, stream) <
           std::tie(o.N, o.depth, o.dir, o.stream);
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
  SKey k{N, depth, dir, stream};
  auto it = g_sm.find(k);
  if (it != g_sm.end()) return it->second;
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

  // structmod::backward WRITES dpair rather than accumulating, so the heads'
  // share -- already sitting in the output buffer -- is staged aside and
  // added back afterwards.  Same for dsingle and the expres head's share.
  static float* dpz = nullptr;
  static size_t dpz_n = 0;
  size_t npz = (size_t)N * N * structmod::CZ;
  if (dpz_n < npz) {
    if (dpz) CU(cudaFree(dpz));
    CU(cudaMalloc(&dpz, npz * 4)); dpz_n = npz;
  }
  CU(cudaMemcpyAsync(dpz, dpair->typed_data(), npz * 4,
                     cudaMemcpyDeviceToDevice, stream));
  m->sm.backward(dact, drot, dtrans, single.typed_data(),
                 dsingle->typed_data(), dpair->typed_data(),
                 datom.typed_data(), aatype.typed_data());
  addPlainK<<<(int)((npz + 255) / 256), 256, 0, stream>>>(
      dpair->typed_data(), dpz, npz);
  addPlainK<<<(int)((nc + 255) / 256), 256, 0, stream>>>(
      dsingle->typed_data(), dse, nc);
  return ffi::Error::Success();
}

}  // namespace

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
