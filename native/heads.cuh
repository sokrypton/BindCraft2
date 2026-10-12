// AF2's three design-relevant heads, forward and backward.
//
//   distogram   Linear(pair) -> 64, then SYMMETRISED: logits = h + h^T
//   PAE         Linear(pair) -> 64, NOT symmetrised
//   pLDDT       LayerNorm(single) -> Linear -> relu -> Linear -> relu
//               -> Linear -> 50
//
// THE SYMMETRY IS THE WHOLE DIFFERENCE between the two pair heads, and it is
// easy to apply to both by reflex.  The distogram predicts a DISTANCE, which
// is symmetric; the predicted aligned error is the error of j in i's frame,
// which is not.  Symmetrising the PAE would make i_pae look better-behaved
// than it is.
//
// These are what the design losses read: con and i_con come off the distogram
// logits, the plddt term off the pLDDT head, the pae terms off the PAE head.
// So this is the first place the loss gradient enters the model, and the last
// thing that has to be exact before a loss can be written against it.
//
// f32 throughout, like the structure module.
#pragma once
#include "foldit.cuh"

namespace heads {

static const int CZ = 128;     // pair channels
static const int CS = 384;     // single channels
static const int NDGRAM = 64;  // distogram bins
static const int NPAE = 64;    // predicted aligned error bins
static const int PL_C = 128;   // pLDDT hidden
static const int PL_BINS = 50;
static const int NER = 37;     // experimentally-resolved: one logit per atom37 slot

// logits = h + h^T over the two residue axes, channels untouched
__global__ void symmetriseK(const float* __restrict__ h, float* __restrict__ o,
                            int N) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * N * NDGRAM) return;
  int c = (int)(i % NDGRAM);
  size_t ij = i / NDGRAM;
  int j = (int)(ij % N), q = (int)(ij / N);
  o[i] = h[i] + h[((size_t)j * N + q) * NDGRAM + c];
}

// and its backward, which is the same operation: d/dh of (h + h^T) sends a
// cotangent to BOTH (i,j) and (j,i).
__global__ void symmetriseBwdK(const float* __restrict__ d,
                               float* __restrict__ o, int N) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * N * NDGRAM) return;
  int c = (int)(i % NDGRAM);
  size_t ij = i / NDGRAM;
  int j = (int)(ij % N), q = (int)(ij / N);
  o[i] = d[i] + d[((size_t)j * N + q) * NDGRAM + c];
}

struct Heads {
  int N = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;

  float *wdg = nullptr, *bdg = nullptr;     // distogram half_logits
  float *wpae = nullptr, *bpae = nullptr;   // pae logits
  float *lnS = nullptr, *lnO = nullptr;     // plddt input_layer_norm
  float *w0 = nullptr, *b0 = nullptr, *w1 = nullptr, *b1 = nullptr;
  float *wl = nullptr, *bl = nullptr;       // plddt logits
  float *wer = nullptr, *ber = nullptr;     // experimentally_resolved logits
  bool has_er = false;

  float *half = nullptr, *dgram = nullptr, *pae = nullptr;
  float *pln = nullptr, *h0 = nullptr, *h1 = nullptr, *plddt = nullptr;
  float *expres = nullptr;
  float *muS = nullptr, *rsS = nullptr;
  float *dhalf = nullptr, *dtmp = nullptr, *dtmp2 = nullptr;

  void alloc(int N_, cudaStream_t s = nullptr) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    CUB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));
    size_t np = (size_t)N * N;
    CU(cudaMalloc(&half, np * NDGRAM * 4));
    CU(cudaMalloc(&dgram, np * NDGRAM * 4));
    CU(cudaMalloc(&pae, np * NPAE * 4));
    // dhalf ALIASES half.  `half` is the distogram's pre-symmetrisation
    // logits: written by the gemm + addRowBiasK and consumed by symmetriseK,
    // all three in the forward, dead immediately after.  `dhalf` is its
    // cotangent: written by symmetriseBwdK and consumed by the gemmNT, both
    // in the backward.  Same extent (np * NDGRAM * 4), live ranges that
    // cannot overlap -- the forward rewrites `half` from scratch every call.
    // 67 MB at N=512.  It does NOT by itself lift the length ceiling -- a
    // 487-residue complex dies two lines up, in `pae`, with 6 MB free, and
    // freeing something allocated AFTER the failing malloc cannot help that.
    // It is kept because the memory is real and the margin is thin: of the
    // 21.1 GB a 427-residue design holds, ~15 GB is this library's (XLA's
    // pool is 5.4), so every buffer that need not exist should not.
    dhalf = half;
    CU(cudaMalloc(&pln, (size_t)N * CS * 4));
    CU(cudaMalloc(&h0, (size_t)N * PL_C * 4));
    CU(cudaMalloc(&h1, (size_t)N * PL_C * 4));
    CU(cudaMalloc(&plddt, (size_t)N * PL_BINS * 4));
    CU(cudaMalloc(&muS, (size_t)N * 4));
    CU(cudaMalloc(&rsS, (size_t)N * 4));
    CU(cudaMalloc(&dtmp, (size_t)N * PL_C * 4));
    CU(cudaMalloc(&dtmp2, (size_t)N * CS * 4));
    CU(cudaMalloc(&expres, (size_t)N * NER * 4));
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

  void setStream(cudaStream_t s) {
    st = s; CUB(cublasSetStream(cub, st));
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    auto get = [&](const char* nm, float** dst, size_t cnt) {
      Tensor& t = o[P + nm];
      if (t.n() != cnt) {
        fprintf(stderr, "heads: %s has %zu, expected %zu\n", nm, t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("distogram_head/half_logits/weights", &wdg, (size_t)CZ * NDGRAM);
    get("distogram_head/half_logits/bias", &bdg, NDGRAM);
    get("predicted_aligned_error_head/logits/weights", &wpae, (size_t)CZ * NPAE);
    get("predicted_aligned_error_head/logits/bias", &bpae, NPAE);
    get("predicted_lddt_head/input_layer_norm/scale", &lnS, CS);
    get("predicted_lddt_head/input_layer_norm/offset", &lnO, CS);
    get("predicted_lddt_head/act_0/weights", &w0, (size_t)CS * PL_C);
    get("predicted_lddt_head/act_0/bias", &b0, PL_C);
    get("predicted_lddt_head/act_1/weights", &w1, (size_t)PL_C * PL_C);
    get("predicted_lddt_head/act_1/bias", &b1, PL_C);
    get("predicted_lddt_head/logits/weights", &wl, (size_t)PL_C * PL_BINS);
    get("predicted_lddt_head/logits/bias", &bl, PL_BINS);
    // optional: design chains that never read this head do not dump it
    if (o.t.count(P + "experimentally_resolved_head/logits/weights")) {
      get("experimentally_resolved_head/logits/weights", &wer, (size_t)CS * NER);
      get("experimentally_resolved_head/logits/bias", &ber, NER);
      has_er = true;
    }
  }

  // The experimentally-resolved head reads the EVOFORMER single, not the
  // structure module's act -- see AlphaFoldIteration, which hands the heads
  // representations['single'] while pLDDT gets representations['structure_module'].
  void expResFwd(const float* single_evo) {
    if (!has_er) return;
    size_t n = (size_t)N * NER;
    gemm(N, NER, CS, single_evo, wer, expres);
    addRowBiasK<<<(int)((n + 255) / 256), 256, 0, st>>>(expres, ber, NER, n);
  }

  // beta=1 accumulates onto dsingle_evo (the trunk single also feeds the
  // structure module, so its cotangent arrives from two places)
  void expResBwd(const float* dexpres, float* dsingle_evo, float beta = 1.f) {
    if (!has_er) return;
    gemmNT(N, CS, NER, dexpres, wer, dsingle_evo, beta);
  }

  // `pair` is the trunk's pair, `single` the structure module's act output
  void forward(const float* pair, const float* single) {
    size_t np = (size_t)N * N;
    int gd = (int)((np * NDGRAM + 255) / 256);
    gemm((int)np, NDGRAM, CZ, pair, wdg, half);
    addRowBiasK<<<gd, 256, 0, st>>>(half, bdg, NDGRAM, np * NDGRAM);
    symmetriseK<<<gd, 256, 0, st>>>(half, dgram, N);
    gemm((int)np, NPAE, CZ, pair, wpae, pae);
    addRowBiasK<<<(int)((np * NPAE + 255) / 256), 256, 0, st>>>(
        pae, bpae, NPAE, np * NPAE);

    foldit::lnFwdK<CS><<<(N + 7) / 8, 256, 0, st>>>(single, lnS, lnO, pln, muS,
                                                    rsS, N);
    size_t nc = (size_t)N * PL_C;
    int gc = (int)((nc + 255) / 256);
    gemm(N, PL_C, CS, pln, w0, h0);
    addRowBiasK<<<gc, 256, 0, st>>>(h0, b0, PL_C, nc);
    foldit::reluK<<<gc, 256, 0, st>>>(h0, nc);
    gemm(N, PL_C, PL_C, h0, w1, h1);
    addRowBiasK<<<gc, 256, 0, st>>>(h1, b1, PL_C, nc);
    foldit::reluK<<<gc, 256, 0, st>>>(h1, nc);
    gemm(N, PL_BINS, PL_C, h1, wl, plddt);
    addRowBiasK<<<(int)(((size_t)N * PL_BINS + 255) / 256), 256, 0, st>>>(
        plddt, bl, PL_BINS, (size_t)N * PL_BINS);
  }

  // cotangents on the three logit tensors -> the pair and the single.
  // dpair is WRITTEN (it takes both pair heads); dsingle is written too.
  void backward(const float* ddgram, const float* dpae, const float* dplddt,
                const float* single, float* dpair, float* dsingle) {
    size_t np = (size_t)N * N, nc = (size_t)N * PL_C;
    int gd = (int)((np * NDGRAM + 255) / 256), gc = (int)((nc + 255) / 256);
    // the distogram's symmetrisation sends each cotangent to both (i,j) and
    // (j,i) before the Linear
    symmetriseBwdK<<<gd, 256, 0, st>>>(ddgram, dhalf, N);
    gemmNT((int)np, CZ, NDGRAM, dhalf, wdg, dpair);
    // the PAE's Linear ACCUMULATES onto it -- both heads read the same pair
    gemmNT((int)np, CZ, NPAE, dpae, wpae, dpair, /*beta*/1.f);

    gemmNT(N, PL_C, PL_BINS, dplddt, wl, dtmp);
    foldit::reluBwdK<<<gc, 256, 0, st>>>(dtmp, h1, nc);
    gemmNT(N, PL_C, PL_C, dtmp, w1, dtmp);
    foldit::reluBwdK<<<gc, 256, 0, st>>>(dtmp, h0, nc);
    gemmNT(N, CS, PL_C, dtmp, w0, dtmp2);
    foldit::lnBwdK<CS><<<(N + 7) / 8, 256, 0, st>>>(dtmp2, single, lnS, muS,
                                                    rsS, dsingle, N);
  }
};

}  // namespace heads
