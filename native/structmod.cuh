// AF2's StructureModule: the wrapper around the eight fold iterations.
//
//   act     = LayerNorm(single)        <- this IS initial_act, 384 wide
//   act     = Linear(act)              <- initial_projection
//   act_2d  = LayerNorm(pair)
//   rigid   = IDENTITY
//   8 x FoldIteration(act, act_2d, rigid)
//
// THE FRAME STARTS AS THE IDENTITY.  Nothing learned goes in: the module
// builds a structure from the representations alone, and every rotation in
// the output came from eight QuatRigid updates composed onto I.  (AF2 also
// accepts `initial_atom_pos` and derives a frame from N/CA/C instead; that is
// a separate entry point and not this one.)
//
// initial_act IS THE LAYER-NORMED SINGLE, BEFORE the initial projection.  It
// is handed to every iteration's sidechain, unchanged -- so like the pair, its
// gradient is a sum over all eight.  Taking the post-projection act for it
// would be a plausible reading and a wrong one.
//
// THE PAIR IS LAYER-NORMED ONCE, HERE, not inside the iterations.  What the
// stack calls `static_feat_2d` is already normalised.
#pragma once
#include "foldstack.cuh"

namespace structmod {

static const int C = foldit::C;     // 384
static const int CZ = ipa::CZ;      // 128

__global__ void identityFrameK(float* __restrict__ rot, float* __restrict__ trans,
                               int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  float* R = rot + (size_t)n * 9;
#pragma unroll
  for (int i = 0; i < 9; ++i) R[i] = (i % 4 == 0) ? 1.f : 0.f;
  float* T = trans + (size_t)n * 3;
  T[0] = T[1] = T[2] = 0.f;
}

struct StructureModule {
  int N = 0, depth = 8;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  foldstack::FoldStack stack;
  bool do_atoms = false;

  float *lnS = nullptr, *lnSo = nullptr;     // single_layer_norm
  float *lnP = nullptr, *lnPo = nullptr;     // pair_layer_norm
  float *wip = nullptr, *bip = nullptr;      // initial_projection

  float *initial = nullptr, *act0 = nullptr, *pairN = nullptr;
  float *muS = nullptr, *rsS = nullptr, *muP = nullptr, *rsP = nullptr;
  float *rot0 = nullptr, *trans0 = nullptr;
  float *dact0 = nullptr, *dinitial = nullptr, *dpairN = nullptr;
  float *drot0 = nullptr, *dtrans0 = nullptr;
  const float* pairIn = nullptr;   // borrowed, for the backward's LayerNorm

  void alloc(int N_, int depth_ = 8, cudaStream_t s = nullptr) {
    N = N_; depth = depth_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    CUB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));
    stack.do_atoms = do_atoms;
    stack.alloc(N, depth, s);
    size_t nc = (size_t)N * C, np = (size_t)N * N * CZ;
    for (auto q : {&initial, &act0, &dact0, &dinitial}) CU(cudaMalloc(q, nc * 4));
    CU(cudaMalloc(&pairN, np * 4));
    CU(cudaMalloc(&dpairN, np * 4));
    for (auto q : {&muS, &rsS}) CU(cudaMalloc(q, (size_t)N * 4));
    for (auto q : {&muP, &rsP}) CU(cudaMalloc(q, (size_t)N * N * 4));
    CU(cudaMalloc(&rot0, (size_t)N * 9 * 4));
    CU(cudaMalloc(&trans0, (size_t)N * 3 * 4));
    CU(cudaMalloc(&drot0, (size_t)N * 9 * 4));
    CU(cudaMalloc(&dtrans0, (size_t)N * 3 * 4));
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    // the iterations live under fold_iteration/ when the dump carries the
    // whole module; see the same probe in foldstack.cu
    const std::string F = o.has(P + "fold_iteration/transition/weights")
                              ? P + "fold_iteration/" : P;
    stack.setWeights(o, F);
    auto get = [&](const char* nm, float** dst, size_t cnt) {
      Tensor& t = o[P + nm];
      CU(cudaMalloc(dst, cnt * 4));
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("single_layer_norm/scale", &lnS, C);
    get("single_layer_norm/offset", &lnSo, C);
    get("pair_layer_norm/scale", &lnP, CZ);
    get("pair_layer_norm/offset", &lnPo, CZ);
    get("initial_projection/weights", &wip, (size_t)C * C);
    get("initial_projection/bias", &bip, C);
  }

  // dropout lives in the fold iterations; the two LayerNorms and the initial
  // projection out here have none
  void setDropout(bool on, const uint32_t* seed) { stack.setDropout(on, seed); }

  void forward(const float* single, const float* pair, const float* mask,
               const float* aatype = nullptr) {
    size_t nc = (size_t)N * C, np = (size_t)N * N * CZ;
    int gc = (int)((nc + 255) / 256), gp = (int)((np + 255) / 256);
    foldit::lnFwdK<><<<(N + 7) / 8, 256, 0, st>>>(single, lnS, lnSo, initial,
                                                  muS, rsS, N);
    // the pair's LayerNorm is over CZ, not C -- a different width, so it gets
    // its own instantiation rather than the single's kernel
    pairIn = pair;
    foldit::lnFwdK<CZ><<<((size_t)N * N + 7) / 8, 256, 0, st>>>(
        pair, lnP, lnPo, pairN, muP, rsP, (int)((size_t)N * N));
    const float alpha = 1.f, beta = 0.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, C, N, C, &alpha, wip, C,
                    initial, C, &beta, act0, C));
    addRowBiasK<<<gc, 256, 0, st>>>(act0, bip, C, nc);
    identityFrameK<<<(N + 255) / 256, 256, 0, st>>>(rot0, trans0, N);
    stack.forward(act0, pairN, mask, rot0, trans0, initial, aatype);
    (void)gp;
  }

  float* outAct() { return stack.outAct(); }
  float* outRot() { return stack.outRot(); }
  float* outTrans() { return stack.outTrans(); }
  float* outAtom() { return stack.outAtom(); }

  void backward(const float* dact, const float* drout, const float* dtout,
                const float* single, float* dsingle, float* dpair,
                const float* datom = nullptr, const float* aatype = nullptr,
                bool acc_pair = false) {
    size_t nc = (size_t)N * C;
    int gc = (int)((nc + 255) / 256);
    stack.backward(dact, drout, dtout, act0, pairN, dact0, dpairN, drot0,
                   dtrans0, datom, initial, aatype, dinitial);
    // THE IDENTITY FRAME IS A CONSTANT, so drot0/dtrans0 stop here -- they are
    // computed and discarded, which is the one place in this port where a
    // gradient is deliberately dropped rather than propagated.
    // initial_act IS READ TWICE: by the initial projection and by the
    // sidechain.  The stack's backward has already put the sidechain's share
    // in dinitial (or left it untouched when the atoms are off), so the
    // projection accumulates onto it with beta = 1 rather than overwriting.
    const float alpha = 1.f;
    const float beta = (do_atoms && datom) ? 1.f : 0.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, C, N, C, &alpha, wip, C,
                    dact0, C, &beta, dinitial, C));
    foldit::lnBwdK<><<<(N + 7) / 8, 256, 0, st>>>(dinitial, single, lnS, muS,
                                                  rsS, dsingle, N);
    // `acc_pair`: add into dpair rather than overwrite it, so a caller that
    // already has another module's share sitting there does not have to copy
    // it aside.  ffi.cu's post-trunk backward is the one that does.
    if (acc_pair)
      foldit::lnBwdK<CZ, true><<<((size_t)N * N + 7) / 8, 256, 0, st>>>(
          dpairN, pairIn, lnP, muP, rsP, dpair, (int)((size_t)N * N));
    else
      foldit::lnBwdK<CZ, false><<<((size_t)N * N + 7) / 8, 256, 0, st>>>(
          dpairN, pairIn, lnP, muP, rsP, dpair, (int)((size_t)N * N));
    (void)gc;
  }
};

}  // namespace structmod
