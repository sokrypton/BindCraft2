// AF2's TEMPLATE EMBEDDING, end to end, around the pair stack that is already
// native.
//
//   q    = LayerNorm(query_embedding)          query_embedding_norm, 128 wide
//   act  = act_const + Linear(q)               template_pair_embedding_8
//   act  = TplStack(act)                       the two native 64-channel blocks
//   act  = LayerNorm(act)                      output_layer_norm, 64 wide
//   emb  = relu(act / num_templates)
//   out  = Linear(emb)                         output_linear, back to 128
//
// WHY THE CONSTANT COMES IN RATHER THAN BEING BUILT: eight of the nine
// features construct_input assembles depend only on the template, so their
// summed projection cannot change while a trajectory runs.  The caller
// harvests it once and hands it over; see template_act_const in bc2.  That is
// why this module takes act_const and never touches a distogram, a backbone
// frame or an atom position.
//
// num_templates is 1 for every BindCraft design -- target_template_features
// emits template_mask jnp.ones((1,)) -- so the mean and the scan around it are
// an identity this does not reproduce.  A caller with more than one template
// must not use this path.
#pragma once
#include "tplstack.cuh"
#include "foldit.cuh"   // lnFwdK/lnBwdK, reluK/reluBwdK, accK

namespace tplembed {

// ffi.cu keeps its own copy of this next to the custom calls; tplembed is a
// header and cannot reach into it, so it carries its own.
__global__ void bf16OutK(float* __restrict__ dst,
                         const __nv_bfloat16* __restrict__ src, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

static const int CQ = 128;                 // query / output pair channels
static const int CT = tplstack::CZ;        // 64, the template stack's width

struct TplEmbed {
  int N = 0, depth = 2;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  tplstack::TplStack stack;

  float *lnS = nullptr, *lnO = nullptr;     // query_embedding_norm
  float *w8 = nullptr, *b8 = nullptr;       // template_pair_embedding_8
  float *oS = nullptr, *oO = nullptr;       // output_layer_norm
  float *wo = nullptr, *bo = nullptr;       // output_linear

  float *qn = nullptr, *mu = nullptr, *rs = nullptr;
  float *mu2 = nullptr, *rs2 = nullptr;     // output_layer_norm's statistics
  float *act = nullptr, *emb = nullptr, *pre = nullptr;
  float *dact = nullptr, *dqn = nullptr;
  __nv_bfloat16 *actb = nullptr, *dactb = nullptr;
  float* outScratch = nullptr;              // the replay's discarded output

  size_t pairElems() const { return (size_t)N * N * CT; }
  size_t queryElems() const { return (size_t)N * N * CQ; }
  size_t saveElems() const { return stack.saveElems(); }

  void alloc(int N_, int depth_ = 2, cudaStream_t s = nullptr) {
    N = N_; depth = depth_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    stack.alloc(N, depth, s);
    for (auto q : {&qn, &dqn}) CU(cudaMalloc(q, queryElems() * 4));
    for (auto q : {&act, &emb, &pre, &dact}) CU(cudaMalloc(q, pairElems() * 4));
    CU(cudaMalloc(&actb, pairElems() * 2));
    CU(cudaMalloc(&dactb, pairElems() * 2));
    CU(cudaMalloc(&mu, (size_t)N * N * 4));
    CU(cudaMalloc(&rs, (size_t)N * N * 4));
    CU(cudaMalloc(&mu2, (size_t)N * N * 4));
    CU(cudaMalloc(&rs2, (size_t)N * N * 4));
    CU(cudaMalloc(&outScratch, queryElems() * 4));
  }

  void setStream(cudaStream_t s) { st = s; CUB(cublasSetStream(cub, s));
                                   stack.setStream(s); }
  void setDropout(bool on, const uint32_t* seed) { stack.setDropout(on, seed); }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    // setWeights, NOT setWeightsFlat: the flat one deliberately gives every
    // block the SAME weights, which is a convenience for the stack's own gate
    // and wrong for a real model.  Wiring it here made the output 1.84x the
    // reference, which is what the gate caught.  Blocks come in as
    // param/block0/..., param/block1/..., so the caller's dumper has to split
    // layer_stack's leading axis.
    stack.setWeights(o);
    auto get = [&](const char* nm, float** dst, size_t cnt) {
      Tensor& t = o[P + nm];
      if (t.n() != cnt) {
        fprintf(stderr, "tplembed: %s has %zu, expected %zu\n", nm, t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("query_embedding_norm/scale", &lnS, CQ);
    get("query_embedding_norm/offset", &lnO, CQ);
    get("template_pair_embedding_8/weights", &w8, (size_t)CQ * CT);
    get("template_pair_embedding_8/bias", &b8, CT);
    get("output_layer_norm/scale", &oS, CT);
    get("output_layer_norm/offset", &oO, CT);
    get("output_linear/weights", &wo, (size_t)CT * CQ);
    get("output_linear/bias", &bo, CQ);
  }

  void gemm(int M, int Nn, int K, const float* A, const float* B, float* D) {
    const float alpha = 1.f, beta = 0.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, Nn, M, K, &alpha, B, Nn, A,
                    K, &beta, D, Nn));
  }
  void gemmNT(int M, int K, int Nn, const float* A, const float* B, float* D,
              float beta = 0.f) {
    const float alpha = 1.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, K, M, Nn, &alpha, B, Nn, A,
                    Nn, &beta, D, K));
  }

  // `act_const` is the precomputed sum of the eight constant projections
  void forward(const float* query, const float* act_const,
               const float* pair_mask, float* out, __nv_bfloat16* save) {
    const size_t nq = queryElems(), np = pairElems();
    const int gq = (int)((nq + 255) / 256), gp = (int)((np + 255) / 256);
    const int rows = (int)((size_t)N * N);
    foldit::lnFwdK<CQ><<<(rows + 7) / 8, 256, 0, st>>>(query, lnS, lnO, qn, mu,
                                                       rs, rows);
    gemm(rows, CT, CQ, qn, w8, act);
    addRowBiasK<<<gp, 256, 0, st>>>(act, b8, CT, np);
    foldit::accK<<<gp, 256, 0, st>>>(act, act_const, np);
    toBf16K<<<gp, 256, 0, st>>>(actb, act, np);
    stack.forward(actb, pair_mask, save);
    bf16OutK<<<gp, 256, 0, st>>>(pre, stack.out(), np);
    // SingleTemplateEmbedding ends with this LayerNorm, and skipping it was
    // the gate's 1.75-vs-0.95: it is what puts the stack's output on the
    // scale output_linear's weights were trained against.
    foldit::lnFwdK<CT><<<(rows + 7) / 8, 256, 0, st>>>(pre, oS, oO, emb, mu2,
                                                       rs2, rows);
    // num_templates is 1, so the mean is the identity; the relu is not
    foldit::reluK<<<gp, 256, 0, st>>>(emb, np);
    gemm(rows, CQ, CT, emb, wo, out);
    addRowBiasK<<<gq, 256, 0, st>>>(out, bo, CQ, nq);
  }

  // cotangent on the embedding -> the query.  act_const is a constant, so no
  // gradient flows back to it, which is the whole point of hoisting it.
  // THE BACKWARD REPLAYS THE FORWARD FIRST, and that is not an optimisation
  // to be removed.  Everything below reads emb, pre, mu2, rs2, mu, rs -- all
  // of it member state.  A gate runs one forward and then its backward, so
  // the state is still the right one and it measures green.  A DESIGN STEP
  // DOES NOT: it runs forwards for every state and every recycle before any
  // backward, so by the time a backward runs, another forward has overwritten
  // all of it.  Replaying from (query, act_const) makes the op stateless and
  // is what the pair stack already does with its own save set.
  void backward(const float* dout, const float* query, const float* act_const,
                const float* pair_mask, const __nv_bfloat16* save,
                float* dquery) {
    const size_t nq = queryElems(), np = pairElems();
    const int gp = (int)((np + 255) / 256);
    const int rows = (int)((size_t)N * N);
    forward(query, act_const, pair_mask, outScratch, const_cast<__nv_bfloat16*>(save));
    gemmNT(rows, CT, CQ, dout, wo, dact);
    foldit::reluBwdK<<<gp, 256, 0, st>>>(dact, emb, np);
    foldit::lnBwdK<CT><<<(rows + 7) / 8, 256, 0, st>>>(dact, pre, oS, mu2, rs2,
                                                      act, rows);
    toBf16K<<<gp, 256, 0, st>>>(dactb, act, np);
    stack.backward(dactb, pair_mask, save);
    bf16OutK<<<gp, 256, 0, st>>>(dact, stack.dpair(), np);
    gemmNT(rows, CQ, CT, dact, w8, dqn);
    foldit::lnBwdK<CQ><<<(rows + 7) / 8, 256, 0, st>>>(dqn, query, lnS, mu,
                                                       rs, dquery, rows);
    (void)nq;
  }
};

}  // namespace tplembed
