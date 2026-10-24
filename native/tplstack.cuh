// AF2's TEMPLATE PAIR STACK: the two TplBlock iterations, as one call.
//
// WHY A WRAPPER AND NOT stack.cuh.  stack.cuh exists for the 48-block
// Evoformer and is mostly the machinery that makes 48 blocks fit -- keep
// tiers, regrant, replay, a donated save-set.  This is two blocks of a
// 64-channel pair, 6.4 MB a buffer at N=224, and tplblock.cuh already keeps
// every module input, so its backward needs no replay at all.  Chaining two
// of them is the whole job.
//
// THE SAVE-SET IS THE STACK'S INPUT, AND THE BACKWARD REPLAYS.  JAX dispatches
// asynchronously, so a second forward can overtake the first backward; if the
// blocks simply kept their own buffers that forward would overwrite them and
// the backward would differentiate the wrong input -- a wrong gradient, not
// an error.
//
// Keeping each block's five module INPUTS is not enough, and this was
// measured, not assumed: with the five inputs restored but the modules'
// own forward state (attention statistics, gates) left as the intervening
// forward wrote it, the gradient came back at cosine 0.025 against a clean
// one.  The modules reuse what their forward left behind, which is why
// block.cuh replays before its backward too.
//
// So the save-set is just the input pair, and backward() re-runs the forward
// from it.  One extra forward of two 64-channel blocks, against a save-set
// ten times smaller (6.4 MB at N=224, not 64), and correct by construction
// rather than by an inventory of what each module happens to cache.
#pragma once
#include "tplblock.cuh"
#include <vector>

namespace tplstack {

static const int CZ = tplblock::CZ;        // 64
static const int NSAVE = 5;                // p1..p5 of each block

struct TplStack {
  int N = 0, depth = 0;
  cudaStream_t st = nullptr;
  // ONE BLOCK FOR EVERY DEPTH, as stack.cuh already does for its 48.  A
  // block per depth meant a MODULE SCRATCH SET per depth -- tmo/tmi/tas/tae/
  // ptr, 2.7 GB each at N=448 -- and two of them is why the template
  // embedding was the largest single allocation in the library (5.98 GB of
  // 21.1).  The weights differ per depth, so they move instead: weightSlots()
  // hands out the pointers and use(b) re-points them.
  tplblock::TplBlock blk;
  std::vector<void**> slots;              // the block's weight pointers
  std::vector<std::vector<void*>> wsets;  // what they should hold, per depth
  __nv_bfloat16* ctmp = nullptr;          // the cotangent between blocks
  // where a restored p1 lives: the block borrows its input pointer, and on the
  // backward there is no caller to borrow from
  std::vector<__nv_bfloat16*> p1store;

  size_t pairElems() const { return (size_t)N * N * CZ; }
  size_t saveElems() const { return pairElems(); }   // the input pair

  template <class EvoBlock = void>
  void alloc(int N_, int depth_, cudaStream_t s = nullptr, EvoBlock* donor = nullptr) {
    N = N_; depth = depth_; st = s;
    blk.alloc(N, st, donor);
    slots = blk.weightSlots();
    // NOT UNCONDITIONALLY.  These are the per-depth WEIGHT pointers, which
    // have nothing to do with N, and alloc() runs a second time when the
    // step is re-derived at a shorter length (see nbrealloc in common.cuh).
    // Wiping them there left `use(b)` writing nulls into every slot, and the
    // next matmul went in with B=(nil).
    if (wsets.size() != (size_t)depth ||
        (!wsets.empty() && wsets[0].size() != slots.size()))
      wsets.assign(depth, std::vector<void*>(slots.size(), nullptr));
    // p1store[b] is block b's INPUT.  The chain cannot feed a block from its
    // own p6b now that there is one block: forward() only POINTS p1b at what
    // it is given, so input and output would alias.  stack.cuh solves it the
    // same way, with pin[i].
    p1store.resize(depth, nullptr);
    for (int b = nblean::on() ? 1 : 0; b < depth; ++b)
      CU(cudaMalloc(&p1store[b], pairElems() * 2));
    if constexpr (!std::is_void<EvoBlock>::value) {
      if (donor) adoptScratch(*donor);
    }
  }

  // the depth's weights, and its own dropout masks
  void use(int b) {
    blk.drop_depth = (uint32_t)b;
    for (size_t k = 0; k < slots.size(); ++k) *slots[k] = wsets[b][k];
  }

  template <class EvoBlock>
  void adoptScratch(EvoBlock& b) {
    blk.adoptScratch(b);
    const size_t half_cb = b.chainBytes() / 2;
    if (half_cb >= pairElems() * 2) {
      if (p1store.size() > 0 && p1store[0] && b.p2b)
        nbrealloc::adoptPtr((void**)&p1store[0], (char*)b.p2b + half_cb, half_cb);
      if (p1store.size() > 1 && p1store[1] && b.p3b)
        nbrealloc::adoptPtr((void**)&p1store[1], (char*)b.p3b + half_cb, half_cb);
    }
  }

  void setStream(cudaStream_t s) {
    st = s;
    blk.setStream(s);
  }

  // every block gets its own drop_depth, so the two do not draw one mask
  void setDropout(bool on, const uint32_t* seed) {
    blk.use_dropout = on;
    blk.drop_seed = seed;      // drop_depth is set per depth by use()
  }

  // the manifest names blocks the way weights.py writes them for every other
  // stack here: param/block{b}/<module>/<param>
  void setWeights(Oracle& o) {
    for (int b = 0; b < depth; ++b) {
      blk.setWeights(o, "param/block" + std::to_string(b) + "/");
      for (size_t k = 0; k < slots.size(); ++k) wsets[b][k] = *slots[k];
    }
  }

  // EVERY BLOCK FROM ONE FLAT PREFIX.  The oracle's template dump is a single
  // iteration (param/<module>/...), which is what tplblock.cu gates against.
  // The stack gate reuses it: giving both blocks the same weights still
  // exercises the chaining and the save-set, which is what a stack adds.
  void setWeightsFlat(Oracle& o, const std::string& P = "param/") {
    for (int b = 0; b < depth; ++b) {
      blk.setWeights(o, P);
      for (size_t k = 0; k < slots.size(); ++k) wsets[b][k] = *slots[k];
    }
  }

  __nv_bfloat16* out() { return blk.p6b; }
  __nv_bfloat16* dpair() { return blk.dp; }   // block 0's dp

  // save may be null, for a forward whose gradient is not wanted
  void forward(const __nv_bfloat16* pair, const float* pair_mask,
               __nv_bfloat16* save, bool for_bwd = false) {
    const size_t np = pairElems();
    if (save)
      CU(cudaMemcpyAsync(save, pair, np * 2, cudaMemcpyDeviceToDevice, st));
    if (for_bwd)
      CU(cudaMemcpyAsync(p1store[0], pair, np * 2, cudaMemcpyDeviceToDevice, st));
    // CHAIN THROUGH p1store, NOT THROUGH p6b.  With one block, p6b is both
    // this depth's output and the next depth's input, and forward() only
    // points p1b at what it is handed -- so the next block would overwrite
    // its own input.  Writing each non-final block's output directly into
    // p1store[b + 1] breaks that without a device-to-device copy.
    __nv_bfloat16* const p6b_orig = blk.p6b;
    for (int b = 0; b < depth; ++b) {
      const __nv_bfloat16* in = pair;
      if (b) {
        if (blk.p6b != p1store[b])
          CU(cudaMemcpyAsync(p1store[b], blk.p6b, np * 2,
                             cudaMemcpyDeviceToDevice, st));
        in = p1store[b];
      } else if (for_bwd) {
        in = p1store[0];
      }
      blk.p6b = (b + 1 < depth && p1store[b + 1] != in) ? p1store[b + 1] : p6b_orig;
      use(b);
      const bool last_bwd = for_bwd && (b == depth - 1);
      blk.forward(in, pair_mask, /*keepBwd=*/last_bwd, /*tail=*/true, /*haveMaskT=*/(b > 0));
    }
    blk.p6b = p6b_orig;
  }

  void backward(const __nv_bfloat16* ct, const float* pair_mask,
                const __nv_bfloat16* save, bool already_forwarded = false) {
    const size_t np = pairElems();
    if (!already_forwarded) {
      // REPLAY FIRST, from the input this backward belongs to.  p1store[0] is
      // where it lands, because the first block borrows its input pointer and
      // `save` is the caller's buffer, which must not be aliased by the chain.
      CU(cudaMemcpyAsync(p1store[0], save, np * 2, cudaMemcpyDeviceToDevice, st));
      // PASS 1 recovers every depth's INPUT.  It used to leave every depth's
      // intermediates live at once, which is exactly what cost a module scratch
      // set per depth; here only the last block's survive, and p1store holds
      // what is needed to regenerate the rest.
      __nv_bfloat16* const p6b_orig = blk.p6b;
      for (int b = 0; b < depth; ++b) {
        blk.p6b = (b + 1 < depth && p1store[b + 1] != p1store[b]) ? p1store[b + 1] : p6b_orig;
        use(b);
        const bool last = (b == depth - 1);
        blk.forward(p1store[b], pair_mask, /*keepBwd=*/last, /*tail=*/!last, /*haveMaskT=*/(b > 0));
        if (b + 1 < depth && blk.p6b != p1store[b + 1])
          CU(cudaMemcpyAsync(p1store[b + 1], blk.p6b, np * 2,
                             cudaMemcpyDeviceToDevice, st));
      }
      blk.p6b = p6b_orig;
    }
    // PASS 2 walks back, replaying each block immediately before its own
    // backward -- stack.cuh's pattern.  The block that pass 1 finished on is
    // already current, so it is not replayed twice.
    const __nv_bfloat16* c = ct;
    for (int b = depth - 1; b >= 0; --b) {
      if (b != depth - 1) {
        use(b);
        blk.forward(p1store[b], pair_mask, /*keepBwd=*/true, /*tail=*/false, /*haveMaskT=*/true);
      }
      blk.backward(c, pair_mask);
      c = blk.dp;
    }
  }
};

}  // namespace tplstack
