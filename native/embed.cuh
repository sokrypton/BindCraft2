// AF2's input embedder: sequence features -> the msa, pair and single
// representations the trunk consumes.
//
//   tf          = pad(target_feat, 1)                 [N, 21]
//   msa_act     = Linear(tf)[None] + Linear(msa_feat) [S, N, 256]
//   pair        = Linear(tf)[:,None] + Linear(tf)[None,:]   (left + right)
//   pair       += position_activations(relative features)
//   single      = Linear(msa_act[0])                  [N, 384]
//
// WHY THIS ONE MATTERS MOST FOR DESIGN: it is where the gradient reaches the
// SEQUENCE.  Everything downstream -- trunk, structure module, heads -- hands
// its cotangent back through these four Linears onto target_feat and msa_feat,
// and a design step's whole job is to turn that into a sequence update.
//
// THE RELATIVE-POSITION TERM IS A GATHER, NOT A MATMUL.  rel_feat is a
// concatenation of three ONE-HOTS (a 66-wide residue offset, a 1-wide entity
// match, a 6-wide chain offset), so multiplying it by [73, 128] selects rows:
//
//   pair[i,j] += W[off] + same_entity * W[66] + W[67 + chain] + b
//
// Materialising the [N, N, 73] one-hot to feed a GEMM would be 2.4 MB a
// residue pair at N=256 and arithmetically identical.
//
// THE 1D PREPROCESS IS MASKED BY THE FEATURE, not by seq_mask: rows whose
// target_feat sums to zero are zeroed after the Linear, which is how a padded
// residue stays padded through the bias.
#pragma once
#include "foldit.cuh"

namespace embed {

static const int NAA = 21;      // target_feat padded (20 + 1)
static const int NMSA_F = 49;   // msa_feat channels
static const int CM = 256;      // msa_channel
static const int CZ = 128;      // pair_channel
static const int CS = 384;      // single
static const int MAX_REL = 32;
static const int REL_BINS = 2 * MAX_REL + 2;     // 66
static const int MAX_CHAIN = 2;
static const int CHAIN_BINS = 2 * MAX_CHAIN + 2; // 6
static const int REL_TOT = REL_BINS + 1 + CHAIN_BINS;  // 73

// pair[i,j] = left[i] + right[j] + relpos(i,j), the three one-hot rows
// gathered rather than multiplied
__global__ void pairInitK(const float* __restrict__ left,
                          const float* __restrict__ right,
                          const float* __restrict__ wpos,
                          const float* __restrict__ bpos,
                          const float* __restrict__ residue_index,
                          const float* __restrict__ asym_id,
                          const float* __restrict__ entity_id,
                          const float* __restrict__ sym_id,
                          // OPTIONAL [N,N] precomputed offsets.  bc2 does not
                          // use residue_index[i]-residue_index[j]: it passes
                          // its own `offset` feature (cyclic_sequence_offsets),
                          // and modules_multimer prefers batch['offset']
                          // whenever it is present -- which, in bc2, is
                          // always.  Computing the difference here instead
                          // gives a DIFFERENT relative encoding on any cyclic
                          // design and matches everywhere else, so it would
                          // gate green and be wrong exactly where it matters.
                          const int32_t* __restrict__ off_mat,
                          float* __restrict__ pair, int N) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)N * N * CZ) return;
  int c = (int)(idx % CZ);
  size_t ij = idx / CZ;
  int j = (int)(ij % N), i = (int)(ij / N);

  int off = off_mat ? off_mat[ij]
                    : (int)residue_index[i] - (int)residue_index[j];
  bool same_asym = asym_id[i] == asym_id[j];
  int clipped = off + MAX_REL;
  clipped = clipped < 0 ? 0 : (clipped > 2 * MAX_REL ? 2 * MAX_REL : clipped);
  // a residue pair in DIFFERENT chains lands in the one bin past the clip
  int rel_bin = same_asym ? clipped : (2 * MAX_REL + 1);

  bool same_entity = entity_id[i] == entity_id[j];
  int rsym = (int)sym_id[i] - (int)sym_id[j];
  int cc = rsym + MAX_CHAIN;
  cc = cc < 0 ? 0 : (cc > 2 * MAX_CHAIN ? 2 * MAX_CHAIN : cc);
  int chain_bin = same_entity ? cc : (2 * MAX_CHAIN + 1);

  float v = left[(size_t)i * CZ + c] + right[(size_t)j * CZ + c];
  v += wpos[(size_t)rel_bin * CZ + c];
  if (same_entity) v += wpos[(size_t)REL_BINS * CZ + c];
  v += wpos[(size_t)(REL_BINS + 1 + chain_bin) * CZ + c];
  v += bpos[c];
  pair[idx] = v;
}

// dleft[i] = sum_j dpair[i,j,:] ; dright[j] = sum_i dpair[i,j,:]
// One block a (row, channel-chunk); the relpos rows take no gradient because
// their inputs are INDICES.
__global__ void pairInitBwdK(const float* __restrict__ dpair,
                             float* __restrict__ dleft,
                             float* __restrict__ dright, int N) {
  int c = blockIdx.x % CZ, i = blockIdx.x / CZ;
  float sl = 0.f, sr = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    sl += dpair[((size_t)i * N + j) * CZ + c];   // row i, this is dleft[i]
    sr += dpair[((size_t)j * N + i) * CZ + c];   // column i, this is dright[i]
  }
  __shared__ float red[2][32];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    sl += __shfl_down_sync(0xffffffffu, sl, o);
    sr += __shfl_down_sync(0xffffffffu, sr, o);
  }
  if (lane == 0) { red[0][warp] = sl; red[1][warp] = sr; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f, b = 0.f;
    for (int w = 0; w < blockDim.x / 32; ++w) { a += red[0][w]; b += red[1][w]; }
    dleft[(size_t)i * CZ + c] = a;
    dright[(size_t)i * CZ + c] = b;
  }
}

// msa_act[s, n, :] = pre1d[n, :] + premsa[s, n, :]
__global__ void msaAddK(const float* __restrict__ pre1d,
                        const float* __restrict__ premsa,
                        float* __restrict__ out, int S, int N) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)S * N * CM) return;
  size_t nc = i % ((size_t)N * CM);
  out[i] = pre1d[nc] + premsa[i];
}

// and back: pre1d's cotangent is the sum over the S rows that read it
__global__ void msaAddBwdK(const float* __restrict__ dout,
                           float* __restrict__ dpre1d, int S, int N) {
  size_t nc = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (nc >= (size_t)N * CM) return;
  float s = 0.f;
  for (int t = 0; t < S; ++t) s += dout[(size_t)t * N * CM + nc];
  dpre1d[nc] = s;
}

// rows whose target_feat is all zero are zeroed AFTER the Linear
__global__ void featMaskK(float* __restrict__ x, const float* __restrict__ tf,
                          int N, int width) {
  int n = blockIdx.x, c = threadIdx.x;
  float s = 0.f;
  for (int k = 0; k < NAA; ++k) s += tf[(size_t)n * NAA + k];
  for (int cc = c; cc < width; cc += blockDim.x)
    if (s == 0.f) x[(size_t)n * width + cc] = 0.f;
}

struct Embed {
  int N = 0, S = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;

  float *w1d = nullptr, *b1d = nullptr, *wmsa = nullptr, *bmsa = nullptr;
  float *wl = nullptr, *bl = nullptr, *wr = nullptr, *br = nullptr;
  float *wpos = nullptr, *bpos = nullptr, *wsa = nullptr, *bsa = nullptr;

  float *pre1d = nullptr, *premsa = nullptr, *msa = nullptr;
  float *left = nullptr, *right = nullptr, *pair = nullptr, *single = nullptr;
  float *dpre1d = nullptr, *dleft = nullptr, *dright = nullptr, *dmsaact = nullptr;

  void alloc(int N_, int S_, cudaStream_t s = nullptr) {
    N = N_; S = S_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    CUB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));
    CU(cudaMalloc(&pre1d, (size_t)N * CM * 4));
    CU(cudaMalloc(&premsa, (size_t)S * N * CM * 4));
    CU(cudaMalloc(&msa, (size_t)S * N * CM * 4));
    CU(cudaMalloc(&dmsaact, (size_t)S * N * CM * 4));
    CU(cudaMalloc(&left, (size_t)N * CZ * 4));
    CU(cudaMalloc(&right, (size_t)N * CZ * 4));
    CU(cudaMalloc(&dleft, (size_t)N * CZ * 4));
    CU(cudaMalloc(&dright, (size_t)N * CZ * 4));
    CU(cudaMalloc(&dpre1d, (size_t)N * CM * 4));
    CU(cudaMalloc(&pair, (size_t)N * N * CZ * 4));
    CU(cudaMalloc(&single, (size_t)N * CS * 4));
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
        fprintf(stderr, "embed: %s has %zu, expected %zu\n", nm, t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      recordWeightPack(*dst);
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("preprocess_1d/weights", &w1d, (size_t)NAA * CM);
    get("preprocess_1d/bias", &b1d, CM);
    get("preprocess_msa/weights", &wmsa, (size_t)NMSA_F * CM);
    get("preprocess_msa/bias", &bmsa, CM);
    get("left_single/weights", &wl, (size_t)NAA * CZ);
    get("left_single/bias", &bl, CZ);
    get("right_single/weights", &wr, (size_t)NAA * CZ);
    get("right_single/bias", &br, CZ);
    // the dump keeps haiku's method scope, as the checkpoint does; see the
    // naming note in oracle_ipa.py
    const std::string RE = "evoformer/~_relative_encoding/";
    auto getr = [&](const std::string& nm, float** dst, size_t cnt) {
      const std::string k = o.has(P + RE + nm) ? P + RE + nm : P + nm;
      Tensor& t = o[k];
      CU(cudaMalloc(dst, cnt * 4));
      recordWeightPack(*dst);
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    getr("position_activations/weights", &wpos, (size_t)REL_TOT * CZ);
    getr("position_activations/bias", &bpos, CZ);
    // OPTIONAL: single_activations belongs to the post-trunk step, so an
    // embedder-only dump has no reason to carry it.  The driver loads a dump
    // that does.
    if (o.has(P + "single_activations/weights")) {
      get("single_activations/weights", &wsa, (size_t)CM * CS);
      get("single_activations/bias", &bsa, CS);
    }
  }

  // `tf` is target_feat ALREADY PADDED to NAA; `mf` is msa_feat [S, N, 49]
  void forward(const float* tf, const float* mf, const float* residue_index,
               const float* asym_id, const float* entity_id,
               const float* sym_id, const int32_t* off_mat = nullptr) {
    size_t nm = (size_t)N * CM, sm = (size_t)S * N * CM;
    gemm(N, CM, NAA, tf, w1d, pre1d);
    addRowBiasK<<<(int)((nm + 255) / 256), 256, 0, st>>>(pre1d, b1d, CM, nm);
    featMaskK<<<N, 256, 0, st>>>(pre1d, tf, N, CM);
    gemm(S * N, CM, NMSA_F, mf, wmsa, premsa);
    addRowBiasK<<<(int)((sm + 255) / 256), 256, 0, st>>>(premsa, bmsa, CM, sm);
    msaAddK<<<(int)((sm + 255) / 256), 256, 0, st>>>(pre1d, premsa, msa, S, N);

    gemm(N, CZ, NAA, tf, wl, left);
    addRowBiasK<<<(int)(((size_t)N * CZ + 255) / 256), 256, 0, st>>>(
        left, bl, CZ, (size_t)N * CZ);
    gemm(N, CZ, NAA, tf, wr, right);
    addRowBiasK<<<(int)(((size_t)N * CZ + 255) / 256), 256, 0, st>>>(
        right, br, CZ, (size_t)N * CZ);
    pairInitK<<<(int)(((size_t)N * N * CZ + 255) / 256), 256, 0, st>>>(
        left, right, wpos, bpos, residue_index, asym_id, entity_id, sym_id,
        off_mat, pair, N);

  }

  // THE SINGLE REPRESENTATION IS NOT PART OF THE EMBEDDER.  AF2 applies
  // single_activations to the EVOFORMER'S OUTPUT msa[0], not to the msa the
  // embedder builds -- so this is a separate call the driver makes AFTER the
  // trunk.  It lived inside forward() until a read of modules.py showed the
  // ordering, and the gate had not caught it because the oracle was written
  // from the same wrong assumption.
  void singleFromTrunk(const float* msa_out, float* out) {
    gemm(N, CS, CM, msa_out, wsa, out);
    addRowBiasK<<<(int)(((size_t)N * CS + 255) / 256), 256, 0, st>>>(
        out, bsa, CS, (size_t)N * CS);
  }

  // and its backward, onto the trunk's msa row 0
  void singleFromTrunkBwd(const float* dsingle, float* dmsa0, float beta = 0.f) {
    gemmNT(N, CM, CS, dsingle, wsa, dmsa0, beta);
  }

  // cotangents on msa, pair and single -> target_feat and msa_feat
  void backward(const float* dmsa, const float* dpair, const float* dsingle,
                const float* tf, float* dtf, float* dmf) {
    size_t sm = (size_t)S * N * CM;
    CU(cudaMemcpyAsync(dmsaact, dmsa, sm * 4, cudaMemcpyDeviceToDevice, st));
    // `dsingle` is accepted but belongs to singleFromTrunkBwd now; a caller
    // that still passes one is adding the trunk's gradient, not the
    // embedder's, and that is its business to get right.
    (void)dsingle;
    gemmNT(S * N, NMSA_F, CM, dmsaact, wmsa, dmf);
    msaAddBwdK<<<(int)(((size_t)N * CM + 255) / 256), 256, 0, st>>>(
        dmsaact, dpre1d, S, N);
    featMaskK<<<N, 256, 0, st>>>(dpre1d, tf, N, CM);
    gemmNT(N, NAA, CM, dpre1d, w1d, dtf);

    pairInitBwdK<<<N * CZ, 256, 0, st>>>(dpair, dleft, dright, N);
    gemmNT(N, NAA, CZ, dleft, wl, dtf, /*beta*/1.f);
    gemmNT(N, NAA, CZ, dright, wr, dtf, /*beta*/1.f);
  }
};

}  // namespace embed
