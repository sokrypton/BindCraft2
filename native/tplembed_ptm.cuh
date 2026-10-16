// AF2-ptm's TEMPLATE EMBEDDING: the whole thing, forward only, computed ONCE.
//
// The monomer's template embedder is not the multimer's, and the difference
// is not only which Linears it has -- it is that THE QUERY NEVER REACHES THE
// OUTPUT.
//
//   act  = Linear(concat of 88 template-only features)   embedding2d, 88 -> 64
//   act  = TplStack(act)                     the two native 64-channel blocks
//   act  = LayerNorm(act)                    output_layer_norm (no relu here)
//   out  = act @ W + output_b                the pointwise attention, folded
//
// `SingleTemplateEmbedding` (modules.py, monomer) reads `query_embedding`
// for its DTYPE and for nothing else -- every one of its 88 features comes
// from the template's aatype, atoms and masks.  The query enters only as the
// QUERY of `TemplatePointwiseAttention`, which softmaxes over the template
// axis; bc2's target_template_features emits `template_mask = jnp.ones((1,))`,
// so that axis has length ONE and the weight is 1 whatever the logit is.
//
// Both facts are MEASURED, not argued, by tplembed_ptm_oracle.py against bc2's
// own module: running it on a query seven times larger moves the output by
// max|d| 0.000e+00, and the folded Linear
//
//     W[c,d] = sum_h sum_a value_w[c,h,a] output_w[h,a,d]
//
// reproduces the attention to max|d| 2.1e-06 (4.5e-07 of max|out|, float32
// round-off).
//
// SO THERE IS NO BACKWARD HERE, AND NOTHING TO RECOMPUTE PER STEP.  The
// multimer's ninth feature IS the query (template_pair_embedding_8 on
// query_embedding_norm), which is why tplembed.cuh has a backward and hoists
// only the other eight into act_const.  Under ptm the whole contribution is a
// constant [N, N, 128] for a given target and length: compute it when the
// weights are installed, add it to the pair, and never touch it again.
//
// WHAT THIS DOES NOT DO: more than one template.  With T > 1 the attention is
// a real softmax over the template axis and the query does reach the output.
// No BindCraft campaign is in that regime -- the mask is literally ones((1,))
// -- and a caller that gets there must not use this path.
#pragma once
// tplembed.cuh for bf16OutK (and tplstack / foldit through it): the two
// embedders share the stack, the LayerNorm kernels and the narrowing pass,
// and a second copy of bf16OutK is a second thing to keep in step.
#include "tplembed.cuh"

namespace tpeptm {

static const int CQ = 128;                 // the pair channels it adds to
static const int CT = tplstack::CZ;        // 64, the template stack's width
static const int NFEAT = 88;               // 39 + 1 + 22 + 22 + 3 + 1
static const int NBINS = 39, NAATYPE = 22;
static constexpr float MIN_BIN = 3.25f, MAX_BIN = 50.75f;
static const int GLY = 7;     // 'G' in AF2's restype order, ARNDCQEGHILKMFPSTWYV

// THE FEATURES ARE bf16 BEFORE THE LINEAR, not just after it.  modules.py
// casts the dgram, both one-hots and both masks to `dtype` and multiplies the
// assembled act by the backbone mask in that dtype, so an honest f32
// restatement is a different function -- the same 1.5e-02 the multimer's
// act_const paid before its cast was found.
__device__ inline float bf16q(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// One (i, j) pair, all 64 output channels.  The 88-wide feature vector is
// never materialised: at N=768 that is 208 MB for something read once.
//
// `W` is embedding2d/weights [88, 64] in its checkpoint layout, so the slices
// below are rows of it -- 0..38 the dgram, 39 the pseudo-beta mask, 40..61
// the column one-hot, 62..83 the row one-hot, 84..86 the unit vector,
// 87 the backbone mask.  THE ORDER IS THE CONCATENATION'S, and getting it
// wrong is a port that is plausible everywhere.
// pseudo_beta_fn: CB for everything but glycine, CA for glycine.  Derived
// here rather than taken as an argument because the step has no template
// pseudo-beta to hand over -- the multimer path never needed one, its eight
// constant features being computed by the caller.
__device__ __forceinline__ void pseudoBeta(const float* __restrict__ atoms,
                                           const float* __restrict__ amask,
                                           int a, int i, float* p, float* m) {
  const int CA = 1, CB = 3;
  const int at = (a == GLY) ? CA : CB;
  const float* q = atoms + ((size_t)i * 37 + at) * 3;
  p[0] = q[0]; p[1] = q[1]; p[2] = q[2];
  *m = amask[(size_t)i * 37 + at];
}

__global__ void embed2dK(const float* __restrict__ atoms, // [N,37,3]
                         const float* __restrict__ amask,  // [N,37]
                         const int32_t* __restrict__ aatype,
                         const int32_t* __restrict__ asym,
                         int N, int nmax,
                         const float* __restrict__ W,
                         const float* __restrict__ B,
                         float* __restrict__ out) {
  const int i = blockIdx.x, j = blockIdx.y;
  if (i >= N || j >= N) return;
  const int c = threadIdx.x;
  if (c >= CT) return;

  const float mc = (asym[i] == asym[j]) ? 1.f : 0.f;   // multichain_mask_2d
  float pbi[3], pbj[3], mi, mj;
  pseudoBeta(atoms, amask, aatype[i], i, pbi, &mi);
  pseudoBeta(atoms, amask, aatype[j], j, pbj, &mj);
  // (A) the PSEUDO-BETA mask, which gates the distogram and is feature 39
  const float pbm2 = bf16q(mi * mj * mc);
  // (B) the BACKBONE mask -- N, CA and C all present.  A PLAIN PRODUCT: the
  // multimer takes sqrt(bmi * bmj) and the monomer does not, so sharing
  // act_const's kernel would be wrong here by exactly that square root on
  // every partially resolved residue.
  const float bmi = amask[(size_t)i * 37 + 0] * amask[(size_t)i * 37 + 1] *
                    amask[(size_t)i * 37 + 2];
  const float bmj = amask[(size_t)j * 37 + 0] * amask[(size_t)j * 37 + 1] *
                    amask[(size_t)j * 37 + 2];
  const float bm2 = bf16q(bmi * bmj * mc);

  float acc = 0.f;
  // 0..38: the template distogram.  dgram_from_positions' bins are
  // linspace(min, max, 39) SQUARED with the last upper break at 1e8, and the
  // one-hot is multiplied by (A) before the cast.
  if (pbm2 > 0.f) {
    float d2 = 0.f;
    for (int k = 0; k < 3; ++k) {
      const float t = pbi[k] - pbj[k];
      d2 += t * t;
    }
    const float step = (MAX_BIN - MIN_BIN) / (float)(NBINS - 1);
    for (int b = 0; b < NBINS; ++b) {
      const float lo = MIN_BIN + step * (float)b, lo2 = lo * lo;
      const float hi = lo + step, hi2 = (b == NBINS - 1) ? 1e8f : hi * hi;
      if (d2 > lo2 && d2 < hi2) { acc += W[(size_t)b * CT + c] * pbm2; break; }
    }
  }
  // 39: the pseudo-beta mask itself
  acc += W[(size_t)NBINS * CT + c] * pbm2;
  // 40..61 and 62..83: the aatype one-hots.  aatype[None, :, :] is tiled over
  // ROWS, so it varies along the COLUMN and comes FIRST in the concatenation.
  const int ai = aatype[i] < NAATYPE ? aatype[i] : NAATYPE - 1;
  const int aj = aatype[j] < NAATYPE ? aatype[j] : NAATYPE - 1;
  acc += W[(size_t)(NBINS + 1 + aj) * CT + c];
  acc += W[(size_t)(NBINS + 1 + NAATYPE + ai) * CT + c];
  // 84..86: the unit vector.  use_template_unit_vector is FALSE in both ptm
  // configs, so these three columns are zeros -- not computed and dropped,
  // genuinely zero, which is why no backbone frame appears in this kernel.
  // 87: the backbone mask
  acc += W[(size_t)(NFEAT - 1) * CT + c] * bm2;

  // ...and the WHOLE assembled vector is multiplied by the backbone mask
  // before the Linear, so the product distributes over every term above --
  // but NOT over the bias, which the Linear adds afterwards.
  out[((size_t)i * nmax + j) * CT + c] = acc * bm2 + B[c];
}

// out[i,j,:] = act[i,j,:] @ W + b, with `scale` 0 when no template exists
// (modules.py: `embedding *= (jnp.sum(template_mask) > 0.)`).
struct TplEmbedPtm {
  int N = 0, depth = 2;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  tplstack::TplStack stack;

  template <class EvoBlock>
  void adoptScratch(EvoBlock& b) { stack.adoptScratch(b); }

  float *w2d = nullptr, *b2d = nullptr;     // embedding2d
  float *oS = nullptr, *oO = nullptr;       // output_layer_norm
  float *wfold = nullptr, *bo = nullptr;    // the folded pointwise attention

  float *act = nullptr, *pre = nullptr, *emb = nullptr;
  float *mu = nullptr, *rs = nullptr;
  __nv_bfloat16 *actb = nullptr;

  size_t pairElems() const { return (size_t)N * N * CT; }
  size_t outElems() const { return (size_t)N * N * CQ; }

  void alloc(int N_, int depth_ = 2, cudaStream_t s = nullptr) {
    N = N_; depth = depth_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    // THE MONOMER ORDER, set before alloc for the same reason block.cuh sets
    // its own: the flag decides what the block does and must not change
    // under a built object.
    stack.blk.attn_first = true;
    stack.alloc(N, depth, s);
    for (auto q : {&act, &pre, &emb}) CU(cudaMalloc(q, pairElems() * 4));
    CU(cudaMalloc(&actb, pairElems() * 2));
    CU(cudaMalloc(&mu, (size_t)N * N * 4));
    CU(cudaMalloc(&rs, (size_t)N * N * 4));
  }

  // A NEVER-ALLOCATED OBJECT STILL GETS ITS STREAM SET.  DesignStep holds
  // both template embedders as members and allocates exactly one of them, and
  // setStream runs over everything it holds -- so under a MULTIMER plan this
  // object's `cub` is null and cublasSetStream returns NOT_INITIALIZED.  The
  // fused-step gate found it as `tplembed_ptm.cuh:188 cublas: 1`, which is a
  // hard abort in a path that has nothing to do with the monomer.  Guarding
  // here rather than at the caller: the caller would have to know which of
  // the two was built, which is the thing this flag exists to hide.
  void setStream(cudaStream_t s) {
    st = s;
    if (cub) CUB(cublasSetStream(cub, s));
    if (stack.blk.N) stack.setStream(s);
  }
  void setDropout(bool on, const uint32_t* seed) {
    if (stack.blk.N) stack.setDropout(on, seed);
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    stack.setWeights(o);
    auto get = [&](const std::string& nm, float** dst, size_t cnt) {
      Tensor& t = o[P + nm];
      if (t.n() != cnt) {
        fprintf(stderr, "tplembed_ptm: %s has %zu, expected %zu\n",
                nm.c_str(), t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      recordWeightPack(*dst);
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("single_template_embedding/embedding2d/weights", &w2d,
        (size_t)NFEAT * CT);
    get("single_template_embedding/embedding2d/bias", &b2d, CT);
    get("single_template_embedding/output_layer_norm/scale", &oS, CT);
    get("single_template_embedding/output_layer_norm/offset", &oO, CT);
    get("attention/output_b", &bo, CQ);
    // THE FOLD IS BUILT HERE, from value_w and output_w, rather than loaded:
    // a dump carrying a precomputed [64, 128] would be a second source of
    // truth for a product of two tensors the checkpoint already has, and
    // nothing would check it.  Gating=False in the template attention's
    // config, so there is no gate to fold in as well.
    Tensor& vw = o[P + "attention/value_w"];    // [CT, H, A]
    Tensor& ow = o[P + "attention/output_w"];   // [H, A, CQ]
    if (vw.shape.size() != 3 || ow.shape.size() != 3 ||
        vw.shape[0] != CT || ow.shape[2] != CQ ||
        vw.shape[1] != ow.shape[0] || vw.shape[2] != ow.shape[1]) {
      fprintf(stderr, "tplembed_ptm: attention value_w/output_w do not "
                      "compose\n");
      abort();
    }
    const int H = vw.shape[1], A = vw.shape[2];
    std::vector<float> f((size_t)CT * CQ, 0.f);
    for (int c = 0; c < CT; ++c)
      for (int d = 0; d < CQ; ++d) {
        double s = 0.0;
        for (int h = 0; h < H; ++h)
          for (int a = 0; a < A; ++a)
            s += (double)vw.host[((size_t)c * H + h) * A + a] *
                 (double)ow.host[((size_t)h * A + a) * CQ + d];
        f[(size_t)c * CQ + d] = (float)s;
      }
    CU(cudaMalloc(&wfold, f.size() * 4));
    recordWeightPack(wfold);
    CU(cudaMemcpy(wfold, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
  }

  void gemm(int M, int Nn, int K, const float* A, const float* B, float* D) {
    const float alpha = 1.f, beta = 0.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, Nn, M, K, &alpha, B, Nn, A,
                    K, &beta, D, Nn));
  }

  // `out` is [N, N, 128], the contribution to add to the pair.  `n` is the
  // real length; the kernel writes with stride `N` so a shorter run lands
  // where the rest of the step expects it.
  void forward(const float* atoms, const float* amask,
               const int32_t* aatype, const int32_t* asym,
               const float* pair_mask, float* out, int n = 0) {
    if (!n) n = N;
    const size_t np = pairElems(), no = outElems();
    const int gp = (int)((np + 255) / 256), go = (int)((no + 255) / 256);
    const int rows = (int)((size_t)N * N);
    dim3 grid(N, N);
    embed2dK<<<grid, 64, 0, st>>>(atoms, amask, aatype, asym, n, N, w2d, b2d,
                                  act);
    toBf16K<<<gp, 256, 0, st>>>(actb, act, np);
    stack.forward(actb, pair_mask, nullptr);
    tplembed::bf16OutK<<<gp, 256, 0, st>>>(pre, stack.out(), np);
    foldit::lnFwdK<CT><<<(rows + 7) / 8, 256, 0, st>>>(pre, oS, oO, emb, mu, rs,
                                                       rows);
    // NO RELU AND NO DIVISION BY num_templates: those belong to the MULTIMER's
    // TemplateEmbedding.  SingleTemplateEmbedding ends at the LayerNorm, and
    // the pointwise attention below is linear in what it is handed.
    gemm(rows, CQ, CT, emb, wfold, out);
    addRowBiasK<<<go, 256, 0, st>>>(out, bo, CQ, no);
  }
};

}  // namespace tpeptm
