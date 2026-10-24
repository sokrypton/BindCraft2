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

// `initial_atom_pos` -> (rot0, trans0):
//   multimer (pos_scale == 20): all_atom_multimer.make_transform_from_reference(
//       a_xyz=N, b_xyz=CA, c_xyz=C).scale_translation(1 / position_scale)
//   monomer  (pos_scale == 10): quat_affine.QuatAffine(rot_to_quat(
//       make_transform_from_reference(N, CA, C)[0]), CA).scale_translation(...)
// The two produce the same Gram-Schmidt rotation matrix on any non-zero
// N/CA/C triple; on an all-zero row (a de novo binder residue on pass 0, or a
// padded row past *np) multimer produces the 3x3 zero matrix while monomer's
// eigh(0) returns last eigenvector (0,0,0,1) -> QUAT_TO_ROT[3,3] = diag(-1,-1,1).
__global__ void refFrameK(const float* __restrict__ pos,
                          const int32_t* __restrict__ np,
                          float* __restrict__ rot,
                          float* __restrict__ trans,
                          float pos_scale, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float* R = rot + (size_t)i * 9;
  float* T = trans + (size_t)i * 3;
  const float z[3] = {0.f, 0.f, 0.f};
  const float* p = (np && i >= *np) ? nullptr : pos + (size_t)i * 37 * 3;
  const float* n  = p ? p + 0 * 3 : z;
  const float* ca = p ? p + 1 * 3 : z;
  const float* c  = p ? p + 2 * 3 : z;
  float e0[3] = {c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]};
  float e1[3] = {n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]};
  float l0 = sqrtf(e0[0]*e0[0] + e0[1]*e0[1] + e0[2]*e0[2]) + 1e-20f;
  for (int k = 0; k < 3; ++k) e0[k] /= l0;
  const float d = e1[0]*e0[0] + e1[1]*e0[1] + e1[2]*e0[2];
  for (int k = 0; k < 3; ++k) e1[k] -= d * e0[k];
  float l1 = sqrtf(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]) + 1e-20f;
  for (int k = 0; k < 3; ++k) e1[k] /= l1;
  const float e2[3] = {e0[1]*e1[2] - e0[2]*e1[1],
                       e0[2]*e1[0] - e0[0]*e1[2],
                       e0[0]*e1[1] - e0[1]*e1[0]};
  R[0] = e0[0]; R[1] = e1[0]; R[2] = e2[0];
  R[3] = e0[1]; R[4] = e1[1]; R[5] = e2[1];
  R[6] = e0[2]; R[7] = e1[2]; R[8] = e2[2];
  if (pos_scale == 10.f && R[0]*R[0] + R[4]*R[4] + R[8]*R[8] == 0.f) {
    R[0] = -1.f; R[4] = -1.f; R[8] = 1.f;
  }
  const float inv = 1.f / pos_scale;
  T[0] = ca[0] * inv; T[1] = ca[1] * inv; T[2] = ca[2] * inv;
}

struct StructureModule {
  int N = 0, depth = 8;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  foldstack::FoldStack stack;
  bool do_atoms = false;
  float pos_scale = foldit::POSITION_SCALE;   // 10 under alphafold2_ptm

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
    stack.pos_scale = pos_scale;
    stack.alloc(N, depth, s);
    size_t nc = (size_t)N * C, np = (size_t)N * N * CZ;
    for (auto q : {&initial, &act0}) CU(cudaMalloc(q, nc * 4));
    CU(cudaMalloc(&pairN, np * 4));
    for (auto q : {&muS, &rsS}) CU(cudaMalloc(q, (size_t)N * 4));
    for (auto q : {&muP, &rsP}) CU(cudaMalloc(q, (size_t)N * N * 4));
    CU(cudaMalloc(&rot0, (size_t)N * 9 * 4));
    CU(cudaMalloc(&trans0, (size_t)N * 3 * 4));
    if (!nblean::on()) {
      for (auto q : {&dact0, &dinitial}) CU(cudaMalloc(q, nc * 4));
      CU(cudaMalloc(&dpairN, np * 4));
      CU(cudaMalloc(&drot0, (size_t)N * 9 * 4));
      CU(cudaMalloc(&dtrans0, (size_t)N * 3 * 4));
    } else {
      dact0 = dinitial = dpairN = drot0 = dtrans0 = nullptr;
    }
  }

  void adoptEvoScratch(void* p1b, void* p5b, size_t cb_max, int nmax_alloc) {
    if (!p1b || !p5b || depth < 1) return;
    size_t slot = ((size_t)nmax_alloc * nmax_alloc * ipa::H * 4 + 255) & ~size_t(255);
    if (5 * slot > cb_max) return;
    if (nblean::on()) {
      nbrealloc::adoptPtr((void**)&stack.it[0].attn.attn, p1b, slot);
      for (int i = 1; i < depth; ++i) stack.it[i].attn.attn = stack.it[0].attn.attn;
    } else {
      for (int i = 0; i < 5 && i < depth; ++i)
        nbrealloc::adoptPtr((void**)&stack.it[i].attn.attn, (char*)p1b + i * slot, slot);
      for (int i = 5; i < 8 && i < depth; ++i)
        nbrealloc::adoptPtr((void**)&stack.it[i].attn.attn, (char*)p5b + (i - 5) * slot, slot);
    }
    nbrealloc::adoptPtr((void**)&stack.it[0].attn.a2d, (char*)p5b + 3 * slot, slot);
    if (!nblean::on())
      nbrealloc::adoptPtr((void**)&stack.it[0].attn.dattn, (char*)p5b + 4 * slot, slot);
    for (int i = 1; i < depth; ++i) {
      stack.it[i].attn.a2d = stack.it[0].attn.a2d;
      stack.it[i].attn.dattn = stack.it[0].attn.dattn;
    }
  }

  // REBIND THE STREAM.  Every buffer in here comes from plain cudaMalloc,
  // which is not stream-ordered, and the stream is used for nothing but
  // kernel launches and cublasSetStream.  So a module allocated on one
  // stream runs correctly on another once this has been called -- which is
  // what lets the module be BUILT AT TRACE TIME, before XLA's stream exists.
  // See the prealloc note in ffi.cu.
  void setStream(cudaStream_t s) {
    st = s; CUB(cublasSetStream(cub, st));
    stack.setStream(s);
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
      recordWeightPack(*dst);
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
               const float* aatype = nullptr,
               const float* init_pos = nullptr,
               const int32_t* np_dev = nullptr) {
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
    if (init_pos)
      refFrameK<<<(N + 255) / 256, 256, 0, st>>>(init_pos, np_dev, rot0, trans0,
                                                 pos_scale, N);
    else
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
    // THE STARTING FRAME IS A CONSTANT, so drot0/dtrans0 stop here -- they are
    // computed and discarded, which is the one place in this port where a
    // gradient is deliberately dropped rather than propagated.  That holds
    // for refFrameK's frames too: pass 0 reads the input atoms, and later
    // passes read prev_pos, which bc2 stop-gradients between recycles.
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
