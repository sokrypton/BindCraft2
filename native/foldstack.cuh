// AF2's structure module as a stack: eight FoldIterations over one frame.
//
// WHAT MAKES THIS DIFFERENT FROM THE EVOFORMER'S STACK is what is shared and
// what is not.  There, 48 blocks each have their own weights and the stack's
// problem is memory -- it keeps one block's save-set and remats the rest.
// Here the eight iterations share ONE set of 46 arrays, and the save-set is
// small enough to keep in full: an iteration's state is N-shaped except for
// the attention itself, which is N^2 x 12 floats -- 56 MB over eight at
// N = 384, against the Evoformer's 100 MB for a single block at N = 256.
// So this stack stores and does not replay.
//
// THE PAIR IS SHARED TOO, and in the other direction: `static_feat_2d` is the
// same tensor for every iteration, so its gradient is a SUM over all eight.
// Only the first backward may zero it, which is what `accPair` is for.
#pragma once
#include "foldit.cuh"

namespace foldstack {

static const int C = foldit::C;

struct FoldStack {
  int N = 0, depth = 0;
  cudaStream_t st = nullptr;
  std::vector<foldit::FoldIteration> it;
  // the frame after each iteration: rot[0]/trans[0] is the stack's input, so
  // there are depth + 1 of them.  The ACT chain is each iteration's own a4,
  // which it already keeps, so only the stack's input act is held here.
  std::vector<float*> rot, trans;
  float* dact = nullptr;
  float *drotS = nullptr, *dtransS = nullptr;

  // ONLY THE LAST ITERATION BUILDS ATOMS.  The real module computes a
  // sidechain every iteration and the auxiliary FAPE losses read all eight;
  // `final_atom_positions`, which is what a design loss consumes, is the
  // last one's.  Wiring just that keeps the gradient exactly what the loss
  // asks for -- when the earlier ones are needed they become seven more
  // do_atoms flags and seven more datom inputs, nothing structural.
  bool do_atoms = false;

  void alloc(int N_, int depth_, cudaStream_t s = nullptr) {
    N = N_; depth = depth_; st = s;
    it.resize(depth);
    // the flag has to be set BEFORE alloc: it is what decides whether the
    // sidechain and the atom builder get buffers at all
    if (do_atoms) it[depth - 1].do_atoms = true;
    for (auto& f : it) f.alloc(N, s);
    rot.resize(depth + 1); trans.resize(depth + 1);
    // rot[0]/trans[0] are the caller's and are never written
    for (int i = 1; i <= depth; ++i) {
      CU(cudaMalloc(&rot[i], (size_t)N * 9 * 4));
      CU(cudaMalloc(&trans[i], (size_t)N * 3 * 4));
    }
    CU(cudaMalloc(&dact, (size_t)N * C * 4));
    CU(cudaMalloc(&drotS, (size_t)N * 9 * 4));
    CU(cudaMalloc(&dtransS, (size_t)N * 3 * 4));
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    it[0].setWeights(o, P);
    for (int i = 1; i < depth; ++i) it[i].shareWeights(it[0]);
    // THE ATOM-BUILDING ITERATION LOADS ITS OWN sidechain weights and tables.
    // It cannot share them from it[0]: only the LAST iteration has do_atoms,
    // so it[0] never allocated or loaded a sidechain and sharing from it
    // copied null pointers (cuBLAS 7, INVALID_VALUE, on the first
    // input_projection).
    if (do_atoms) {
      it[depth - 1].sc.setWeights(o, P);
      it[depth - 1].at.setTables(o);
    }
  }

  // EVERY ITERATION GETS ITS OWN drop_depth, so the eight do not draw one
  // mask.  AF2 splits a fresh key per iteration and twice within it; here the
  // iteration index is the hash's depth and the two sites are its module id,
  // which gives the same independence without carrying keys.
  void setDropout(bool on, const uint32_t* seed) {
    for (int i = 0; i < depth; ++i) {
      it[i].drop_seed = on ? seed : nullptr;
      it[i].drop_depth = (uint32_t)i;
    }
  }

  const float* actIn(int i, const float* act0) const {
    return i == 0 ? act0 : it[i - 1].a4;
  }

  void forward(const float* act, const float* pair, const float* mask,
               const float* rot0, const float* trans0,
               const float* initial = nullptr, const float* aatype = nullptr) {
    rot[0] = const_cast<float*>(rot0);
    trans[0] = const_cast<float*>(trans0);
    for (int i = 0; i < depth; ++i)
      it[i].forward(actIn(i, act), pair, mask, rot[i], trans[i], rot[i + 1],
                    trans[i + 1], initial, aatype);
  }

  float* outAtom() { return it[depth - 1].at.out; }

  float* outAct() { return it[depth - 1].a4; }
  float* outRot() { return rot[depth]; }
  float* outTrans() { return trans[depth]; }

  // `dactIn`/`drout`/`dtout` are the cotangents on the stack's outputs.
  void backward(const float* dactIn, const float* drout, const float* dtout,
                const float* act, const float* pair, float* dsingle,
                float* dpair, float* drot, float* dtrans,
                const float* datom = nullptr, const float* initial = nullptr,
                const float* aatype = nullptr, float* dinitial = nullptr) {
    size_t n = (size_t)N * C;
    int g = (int)((n + 255) / 256);
    CU(cudaMemcpyAsync(dact, dactIn, n * 4, cudaMemcpyDeviceToDevice, st));
    CU(cudaMemcpyAsync(drot, drout, (size_t)N * 9 * 4, cudaMemcpyDeviceToDevice, st));
    CU(cudaMemcpyAsync(dtrans, dtout, (size_t)N * 3 * 4, cudaMemcpyDeviceToDevice, st));
    for (int i = depth - 1; i >= 0; --i) {
      // drot/dtrans carry the frame gradient from iteration i+1 down to i; the
      // iteration overwrites them with its own input's, which is exactly the
      // chain.  dsingle likewise becomes the gradient on THIS iteration's act,
      // which is the previous one's output.
      it[i].backward(dact, drot, dtrans, actIn(i, act), pair, rot[i], trans[i],
                     dsingle, dpair, drotS, dtransS,
                     /*accPair*/ i != depth - 1,
                     i == depth - 1 ? datom : nullptr, initial, aatype,
                     dinitial, /*accInitial*/ false);
      CU(cudaMemcpyAsync(drot, drotS, (size_t)N * 9 * 4, cudaMemcpyDeviceToDevice, st));
      CU(cudaMemcpyAsync(dtrans, dtransS, (size_t)N * 3 * 4,
                         cudaMemcpyDeviceToDevice, st));
      if (i) CU(cudaMemcpyAsync(dact, dsingle, n * 4, cudaMemcpyDeviceToDevice, st));
    }
    (void)g;
  }
};

}  // namespace foldstack
