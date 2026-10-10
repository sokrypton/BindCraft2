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
  std::vector<tplblock::TplBlock> blk;
  // where a restored p1 lives: the block borrows its input pointer, and on the
  // backward there is no caller to borrow from
  std::vector<__nv_bfloat16*> p1store;

  size_t pairElems() const { return (size_t)N * N * CZ; }
  size_t saveElems() const { return pairElems(); }   // the input pair

  void alloc(int N_, int depth_, cudaStream_t s = nullptr) {
    N = N_; depth = depth_; st = s;
    blk.resize(depth);
    p1store.resize(depth, nullptr);
    for (int b = 0; b < depth; ++b) {
      blk[b].alloc(N, st);
      CU(cudaMalloc(&p1store[b], pairElems() * 2));
    }
  }

  void setStream(cudaStream_t s) {
    st = s;
    for (auto& b : blk) b.setStream(s);
  }

  // every block gets its own drop_depth, so the two do not draw one mask
  void setDropout(bool on, const uint32_t* seed) {
    for (int b = 0; b < depth; ++b) {
      blk[b].use_dropout = on;
      blk[b].drop_seed = seed;
      blk[b].drop_depth = (uint32_t)b;
    }
  }

  // the manifest names blocks the way weights.py writes them for every other
  // stack here: param/block{b}/<module>/<param>
  void setWeights(Oracle& o) {
    for (int b = 0; b < depth; ++b)
      blk[b].setWeights(o, "param/block" + std::to_string(b) + "/");
  }

  // EVERY BLOCK FROM ONE FLAT PREFIX.  The oracle's template dump is a single
  // iteration (param/<module>/...), which is what tplblock.cu gates against.
  // The stack gate reuses it: giving both blocks the same weights still
  // exercises the chaining and the save-set, which is what a stack adds.
  void setWeightsFlat(Oracle& o, const std::string& P = "param/") {
    for (int b = 0; b < depth; ++b) blk[b].setWeights(o, P);
  }

  __nv_bfloat16* out() { return blk[depth - 1].p6b; }
  __nv_bfloat16* dpair() { return blk[0].dp; }

  // save may be null, for a forward whose gradient is not wanted
  void forward(const __nv_bfloat16* pair, const float* pair_mask,
               __nv_bfloat16* save) {
    const size_t np = pairElems();
    if (save)
      CU(cudaMemcpyAsync(save, pair, np * 2, cudaMemcpyDeviceToDevice, st));
    const __nv_bfloat16* in = pair;
    for (int b = 0; b < depth; ++b) {
      blk[b].forward(in, pair_mask);
      in = blk[b].p6b;
    }
  }

  void backward(const __nv_bfloat16* ct, const float* pair_mask,
                const __nv_bfloat16* save) {
    const size_t np = pairElems();
    // REPLAY FIRST, from the input this backward belongs to.  p1store[0] is
    // where it lands, because the first block borrows its input pointer and
    // `save` is the caller's buffer, which must not be aliased by the chain.
    CU(cudaMemcpyAsync(p1store[0], save, np * 2, cudaMemcpyDeviceToDevice, st));
    const __nv_bfloat16* in = p1store[0];
    for (int b = 0; b < depth; ++b) {
      blk[b].forward(in, pair_mask);
      in = blk[b].p6b;
    }
    const __nv_bfloat16* c = ct;
    for (int b = depth - 1; b >= 0; --b) {
      blk[b].backward(c, pair_mask);
      c = blk[b].dp;
    }
  }
};

}  // namespace tplstack
