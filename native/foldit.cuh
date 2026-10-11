// One FoldIteration of AF2's structure module, forward and backward.
//
//   a1 = act + IPA(act, pair, mask, rigid)
//   a2 = LayerNorm_attention(a1)
//   a3 = a2 + Linear3(relu(Linear2(relu(Linear1(a2)))))
//   a4 = LayerNorm_transition(a3)
//   rigid' = rigid @ QuatRigid(a4)
//   -> act = a4, rigid = rigid'
//
// THE EIGHT ITERATIONS SHARE ONE SET OF WEIGHTS.  Unlike the Evoformer, whose
// checkpoint carries a leading axis of 48, the structure module stores 46
// arrays and runs them eight times -- so a stack over this is a loop, with no
// per-depth weight swap to arrange.
//
// THE OUTPUT ACTIVATION IS READ TWICE: it is the iteration's `act` output and
// it is also what the rigid update projects.  So on the way back, da4 is the
// sum of the incoming cotangent and QuatRigid's own -- which is the one place
// the backward is not a straight reversal of the forward.
#pragma once
#include "ipa.cuh"
#include "quatrigid.cuh"
#include "sidechain.cuh"
#include "atom14.cuh"

namespace foldit {

static const int C = 384;
// the template default below cannot name `C` while a template parameter of
// the same name is in scope, so the width has a second spelling
static const int foldit_C = 384;
static const float LN_EPS = 1e-5f;
// nm -> angstrom for the sidechain's frame; multimer's config says 20
static const float POSITION_SCALE = 20.f;

// f32 LayerNorm over the last axis.  One warp a row: C = 384 is 12 elements a
// lane, which fits in registers, but the rows are re-read in the backward
// anyway so this keeps only mu and rstd.
template <int C = foldit_C>
__global__ void lnFwdK(const float* __restrict__ x, const float* __restrict__ scale,
                       const float* __restrict__ offset, float* __restrict__ y,
                       float* __restrict__ mu, float* __restrict__ rstd, int N) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= N) return;
  int lane = threadIdx.x % 32;
  const float* X = x + (size_t)row * C;
  float s = 0.f, s2 = 0.f;
  for (int c = lane; c < C; c += 32) { float v = X[c]; s += v; s2 += v * v; }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s += __shfl_xor_sync(0xffffffffu, s, o);
    s2 += __shfl_xor_sync(0xffffffffu, s2, o);
  }
  float m = s / C;
  // THE TWO-PASS FORM, not E[x^2] - E[x]^2.  The one-pass variance cancels two
  // large numbers and has cost this repo a gate before (see the af2 bf16
  // note); here it is f32 and cheap to do right.
  float v = 0.f;
  for (int c = lane; c < C; c += 32) { float d = X[c] - m; v += d * d; }
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  float r = rsqrtf(v / C + LN_EPS);
  if (lane == 0) { mu[row] = m; rstd[row] = r; }
  float* Y = y + (size_t)row * C;
  for (int c = lane; c < C; c += 32) Y[c] = (X[c] - m) * r * scale[c] + offset[c];
}

template <int C = foldit_C>
__global__ void lnBwdK(const float* __restrict__ dy, const float* __restrict__ x,
                       const float* __restrict__ scale, const float* __restrict__ mu,
                       const float* __restrict__ rstd, float* __restrict__ dx,
                       int N) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= N) return;
  int lane = threadIdx.x % 32;
  const float* X = x + (size_t)row * C;
  const float* DY = dy + (size_t)row * C;
  float m = mu[row], r = rstd[row];
  float s1 = 0.f, s2 = 0.f;
  for (int c = lane; c < C; c += 32) {
    float g = DY[c] * scale[c];
    s1 += g;
    s2 += g * (X[c] - m) * r;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s1 += __shfl_xor_sync(0xffffffffu, s1, o);
    s2 += __shfl_xor_sync(0xffffffffu, s2, o);
  }
  s1 /= C; s2 /= C;
  float* DX = dx + (size_t)row * C;
  for (int c = lane; c < C; c += 32) {
    float g = DY[c] * scale[c];
    DX[c] = r * (g - s1 - (X[c] - m) * r * s2);
  }
}

__global__ void reluK(float* __restrict__ x, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = fmaxf(x[i], 0.f);
}

// dx *= (h > 0), with h the POST-relu value -- which carries the sign, so the
// pre-activation need not be kept.
__global__ void reluBwdK(float* __restrict__ dx, const float* __restrict__ h,
                         size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] = h[i] > 0.f ? dx[i] : 0.f;
}

__global__ void addK(float* __restrict__ d, const float* __restrict__ a,
                     const float* __restrict__ b, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = a[i] + b[i];
}

__global__ void scaleToK(float* __restrict__ d, const float* __restrict__ a,
                         float f, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = a[i] * f;
}

// d = a + f * b, which is how the frame's two cotangents meet
__global__ void addScaledK(float* __restrict__ d, const float* __restrict__ a,
                           const float* __restrict__ b, float f, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = a[i] + f * b[i];
}

__global__ void accK(float* __restrict__ d, const float* __restrict__ a, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] += a[i];
}

struct FoldIteration {
  int N = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  ipa::IPA attn;
  qrigid::QuatRigid rig;
  // THE SIDECHAIN AND THE ATOM BUILDER ARE OPTIONAL, because the act/rigid
  // chain can be gated without them and the stack's own gates do.  When they
  // are on, this iteration's OUTPUT FRAME gains a second consumer: the next
  // iteration and this iteration's atoms both read it, so its cotangent is a
  // sum -- which is the only structural change they make to the backward.
  // DROPOUT.  AF2 applies it inside the fold iteration at two points, and
  // NOT the way the Evoformer does: folding_multimer.FoldIteration drops the
  // WHOLE act after each residual add (rate 0.1, broadcast_dim None), where
  // every Evoformer site scales a module delta before the residual and shares
  // the mask across a row or a column.  So this uses dropFlatK, elementwise
  // over [N, C], and the mask index is the flat one.
  //
  // The seed is a DEVICE pointer, two words: [seed, enabled].  Both are
  // runtime values, so a stage that turns dropout off costs no recompile --
  // which is the whole reason the Evoformer's dropout is built this way, and
  // the reason the structure module could not simply bake the rate in.
  const uint32_t* drop_seed = nullptr;
  uint32_t drop_depth = 0;        // the iteration index: 8 iterations, 8 masks
  enum { DM_SM_ATTN = 21, DM_SM_TRANS = 22 };
  static constexpr float SM_DROP = 0.1f;

  bool do_atoms = false;
  sidechain::Sidechain sc;
  atom14::Atom14 at;
  float* tscaled = nullptr;     // the frame's translation x position_scale
  float* dang = nullptr;
  float *drotA = nullptr, *dtransA = nullptr;
  float *drotSum = nullptr, *dtransSum = nullptr;
  float* dscAct = nullptr;
  const float* rout_ = nullptr;   // the output frame, borrowed for the backward

  float *ln1s = nullptr, *ln1o = nullptr, *ln2s = nullptr, *ln2o = nullptr;
  float *w1 = nullptr, *b1 = nullptr, *w2 = nullptr, *b2 = nullptr;
  float *w3 = nullptr, *b3 = nullptr;

  float *a1 = nullptr, *a2 = nullptr, *a3 = nullptr, *a4 = nullptr;
  float *h1 = nullptr, *h2 = nullptr, *tr = nullptr;
  float *mu1 = nullptr, *rs1 = nullptr, *mu2 = nullptr, *rs2 = nullptr;
  float *d1 = nullptr, *d2 = nullptr, *d3 = nullptr, *dh = nullptr;

  void alloc(int N_, cudaStream_t s = nullptr) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    CUB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));
    attn.alloc(N, s);
    rig.alloc(N, s);
    if (do_atoms) {
      sc.alloc(N, s);
      at.alloc(N, s);
      CU(cudaMalloc(&tscaled, (size_t)N * 3 * 4));
      CU(cudaMalloc(&dang, (size_t)N * sidechain::NANG * 4));
      CU(cudaMalloc(&drotA, (size_t)N * 9 * 4));
      CU(cudaMalloc(&dtransA, (size_t)N * 3 * 4));
      CU(cudaMalloc(&drotSum, (size_t)N * 9 * 4));
      CU(cudaMalloc(&dtransSum, (size_t)N * 3 * 4));
      CU(cudaMalloc(&dscAct, (size_t)N * C * 4));
    }
    size_t n = (size_t)N * C;
    for (auto q : {&a1, &a2, &a3, &a4, &h1, &h2, &tr, &d1, &d2, &d3, &dh})
      CU(cudaMalloc(q, n * 4));
    for (auto q : {&mu1, &rs1, &mu2, &rs2}) CU(cudaMalloc(q, (size_t)N * 4));
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    attn.setWeights(o, P);
    rig.setWeights(o, P);
    if (do_atoms) { sc.setWeights(o, P); at.setTables(o); }
    auto get = [&](const char* name, float** dst, size_t cnt) {
      Tensor& t = o[P + name];
      if (t.n() != cnt) {
        fprintf(stderr, "foldit: %s has %zu, expected %zu\n", name, t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("attention_layer_norm/scale", &ln1s, C);
    get("attention_layer_norm/offset", &ln1o, C);
    get("transition_layer_norm/scale", &ln2s, C);
    get("transition_layer_norm/offset", &ln2o, C);
    get("transition/weights", &w1, (size_t)C * C);
    get("transition/bias", &b1, C);
    get("transition_1/weights", &w2, (size_t)C * C);
    get("transition_1/bias", &b2, C);
    get("transition_2/weights", &w3, (size_t)C * C);
    get("transition_2/bias", &b3, C);
  }

  void shareWeights(const FoldIteration& o) {
    attn.shareWeights(o.attn);
    rig.shareWeights(o.rig);
    // only when the DONOR has them; see the note in FoldStack::setWeights
    if (do_atoms && o.do_atoms) { sc.shareWeights(o.sc); at.shareTables(o.at); }
    ln1s = o.ln1s; ln1o = o.ln1o; ln2s = o.ln2s; ln2o = o.ln2o;
    w1 = o.w1; b1 = o.b1; w2 = o.w2; b2 = o.b2; w3 = o.w3; b3 = o.b3;
  }

  void gemm(int M, int Nn, int K, const float* A, const float* B, float* D,
            float beta = 0.f) {
    const float alpha = 1.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, Nn, M, K, &alpha, B, Nn, A,
                    K, &beta, D, Nn));
  }
  void gemmNT(int M, int K, int Nn, const float* A, const float* B, float* D,
              float beta = 0.f) {
    const float alpha = 1.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, K, M, Nn, &alpha, B, Nn, A,
                    Nn, &beta, D, K));
  }

  void forward(const float* act, const float* pair, const float* mask,
               const float* rot, const float* trans, float* rout, float* tout,
               const float* initial = nullptr, const float* aatype = nullptr) {
    size_t n = (size_t)N * C;
    int g = (int)((n + 255) / 256), gw = (N + 7) / 8;
    attn.forward(act, pair, mask, rot, trans);
    addK<<<g, 256, 0, st>>>(a1, act, attn.out, n);
    // IN PLACE, so a1 holds what the LayerNorm actually saw -- the backward
    // reads a1 as the LayerNorm's saved input and must get the dropped value
    if (drop_seed)
      dropFlatK<<<g, 256, 0, st>>>(a1, a1, drop_seed, drop_depth, DM_SM_ATTN,
                                   SM_DROP, n);
    lnFwdK<><<<gw, 256, 0, st>>>(a1, ln1s, ln1o, a2, mu1, rs1, N);
    gemm(N, C, C, a2, w1, h1);
    addRowBiasK<<<g, 256, 0, st>>>(h1, b1, C, n);
    reluK<<<g, 256, 0, st>>>(h1, n);
    gemm(N, C, C, h1, w2, h2);
    addRowBiasK<<<g, 256, 0, st>>>(h2, b2, C, n);
    reluK<<<g, 256, 0, st>>>(h2, n);
    gemm(N, C, C, h2, w3, tr);
    addRowBiasK<<<g, 256, 0, st>>>(tr, b3, C, n);
    addK<<<g, 256, 0, st>>>(a3, tr, a2, n);
    if (drop_seed)
      dropFlatK<<<g, 256, 0, st>>>(a3, a3, drop_seed, drop_depth, DM_SM_TRANS,
                                   SM_DROP, n);
    lnFwdK<><<<gw, 256, 0, st>>>(a3, ln2s, ln2o, a4, mu2, rs2, N);
    rig.forward(a4, rot, trans, rout, tout);
    rout_ = rout;
    if (do_atoms) {
      // the sidechain reads the frame AFTER the update, with its translation
      // in angstroms
      scaleToK<<<(N * 3 + 255) / 256, 256, 0, st>>>(tscaled, tout,
                                                    POSITION_SCALE, (size_t)N * 3);
      sc.forward(a4, initial);
      at.forward(sc.ang, rout, tscaled, aatype);
    }
  }

  float* outAct() { return a4; }

  // THE MIDDLE OF THE ITERATION ON ITS OWN: a1 -> LN -> transition -> LN.
  // forward() calls these after the attention; the gate calls them with a1
  // supplied directly, which is the only way to ask whether this half is
  // where the assembled backward loses its precision.
  void midForward(const float* a1in) {
    size_t n = (size_t)N * C;
    int g = (int)((n + 255) / 256), gw = (N + 7) / 8;
    if (a1in != a1) CU(cudaMemcpyAsync(a1, a1in, n * 4, cudaMemcpyDeviceToDevice, st));
    lnFwdK<><<<gw, 256, 0, st>>>(a1, ln1s, ln1o, a2, mu1, rs1, N);
    gemm(N, C, C, a2, w1, h1);
    addRowBiasK<<<g, 256, 0, st>>>(h1, b1, C, n);
    reluK<<<g, 256, 0, st>>>(h1, n);
    gemm(N, C, C, h1, w2, h2);
    addRowBiasK<<<g, 256, 0, st>>>(h2, b2, C, n);
    reluK<<<g, 256, 0, st>>>(h2, n);
    gemm(N, C, C, h2, w3, tr);
    addRowBiasK<<<g, 256, 0, st>>>(tr, b3, C, n);
    addK<<<g, 256, 0, st>>>(a3, tr, a2, n);
    lnFwdK<><<<gw, 256, 0, st>>>(a3, ln2s, ln2o, a4, mu2, rs2, N);
  }

  void midBackward(const float* da4, float* da1) {
    size_t n = (size_t)N * C;
    int g = (int)((n + 255) / 256), gw = (N + 7) / 8;
    lnBwdK<><<<gw, 256, 0, st>>>(da4, a3, ln2s, mu2, rs2, d2, N);
    gemmNT(N, C, C, d2, w3, dh);
    reluBwdK<<<g, 256, 0, st>>>(dh, h2, n);
    gemmNT(N, C, C, dh, w2, d3);
    reluBwdK<<<g, 256, 0, st>>>(d3, h1, n);
    gemmNT(N, C, C, d3, w1, dh);
    accK<<<g, 256, 0, st>>>(dh, d2, n);
    lnBwdK<><<<gw, 256, 0, st>>>(dh, a1, ln1s, mu1, rs1, da1, N);
  }

  // `dact` is the cotangent on the output act, `drout`/`dtout` on the output
  // frame.  dsingle/dpair/drot/dtrans are written.
  void backward(const float* dact, const float* drout, const float* dtout,
                const float* act, const float* pair, const float* rot,
                const float* trans, float* dsingle, float* dpair, float* drot,
                float* dtrans, bool accPair = false,
                const float* datom = nullptr, const float* initial = nullptr,
                const float* aatype = nullptr, float* dinitial = nullptr,
                bool accInitial = false) {
    size_t n = (size_t)N * C;
    int g = (int)((n + 255) / 256), gw = (N + 7) / 8;
    // THE OUTPUT FRAME HAS TWO CONSUMERS when the atoms are on: the next
    // iteration (drout/dtout) and this iteration's own atom builder.  Their
    // cotangents add, and the atom builder's translation gradient carries the
    // position_scale with it.
    const float* drUse = drout;
    const float* dtUse = dtout;
    if (do_atoms && datom) {
      at.backward(datom, sc.ang, rout_, aatype, dang, drotA, dtransA);
      addScaledK<<<(N * 9 + 255) / 256, 256, 0, st>>>(drotSum, drout, drotA, 1.f,
                                                      (size_t)N * 9);
      addScaledK<<<(N * 3 + 255) / 256, 256, 0, st>>>(
          dtransSum, dtout, dtransA, POSITION_SCALE, (size_t)N * 3);
      drUse = drotSum; dtUse = dtransSum;
      sc.backward(dang, a4, initial, dscAct, dinitial, accInitial);
    }
    // THE OUTPUT ACT IS READ TWICE -- as the iteration's output and as the
    // rigid update's input -- so its two cotangents add before anything else
    // happens.  d1 ends up holding da4.
    rig.backward(drUse, dtUse, rot, d1, drot, dtrans);
    accK<<<g, 256, 0, st>>>(d1, dact, n);
    // a4 is read by the rigid update, by the iteration's output, AND by the
    // sidechain
    if (do_atoms && datom) accK<<<g, 256, 0, st>>>(d1, dscAct, n);
    lnBwdK<><<<gw, 256, 0, st>>>(d1, a3, ln2s, mu2, rs2, d2, N);   // da3
    // dropout is a multiply by a 0-or-1/keep mask, so its backward is the
    // SAME mask.  dropScale is a pure hash of (seed, depth, module, index),
    // which is why the backward can regenerate it rather than store it.
    if (drop_seed)
      dropFlatK<<<g, 256, 0, st>>>(d2, d2, drop_seed, drop_depth, DM_SM_TRANS,
                                   SM_DROP, n);
    // the transition, in reverse; the residual means da2 starts as da3
    gemmNT(N, C, C, d2, w3, dh);
    reluBwdK<<<g, 256, 0, st>>>(dh, h2, n);
    gemmNT(N, C, C, dh, w2, d3);
    reluBwdK<<<g, 256, 0, st>>>(d3, h1, n);
    gemmNT(N, C, C, d3, w1, dh);
    accK<<<g, 256, 0, st>>>(dh, d2, n);                          // da2
    lnBwdK<><<<gw, 256, 0, st>>>(dh, a1, ln1s, mu1, rs1, d2, N);   // da1
    if (drop_seed)
      dropFlatK<<<g, 256, 0, st>>>(d2, d2, drop_seed, drop_depth, DM_SM_ATTN,
                                   SM_DROP, n);
    // IPA, whose frame gradients land ON TOP of the composition's: the same
    // rigid was handed to both, so drot/dtrans accumulate rather than being
    // overwritten.
    attn.backward(d2, act, pair, rot, trans, d3, dpair, attn.drotScratch(),
                  attn.dtransScratch(), accPair);
    accK<<<(N * 9 + 255) / 256, 256, 0, st>>>(drot, attn.drotScratch(), (size_t)N * 9);
    accK<<<(N * 3 + 255) / 256, 256, 0, st>>>(dtrans, attn.dtransScratch(), (size_t)N * 3);
    // and the residual around IPA
    addK<<<g, 256, 0, st>>>(dsingle, d3, d2, n);
  }
};

}  // namespace foldit
