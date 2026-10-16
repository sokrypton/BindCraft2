// THE THREE RECYCLE TERMS, which are not zero at zero recycles.
//
// AF2's embedder adds three things to what the input Linears produce:
//
//   pair += prev_pos_linear(dgram_from_positions(pseudo_beta(prev_pos)))
//   msa[0] += prev_msa_first_row_norm(prev_msa_first_row)
//   pair += prev_pair_norm(prev_pair)
//
// EVERY ONE OF THEM IS NONZERO ON THE FIRST PASS, when `prev` is all
// zeros.  A LayerNorm of an all-zero vector is its OFFSET, not zero, and
// the distogram of coordinates that are all at the origin is ALL ZEROS --
// not one-hot in bin 0, as this comment said until it was checked against
// the kernel below: the first bin's lower break is 3.25^2 = 10.56 and the
// test is `d2 > lo2`, so a distance of zero sets no bin at all.  AF2's
// dgram_from_positions does the same.  The Linear therefore contributes its
// BIAS and nothing else -- still nonzero, which is the point, but by a
// different route than the sentence it replaces claimed.  Dropping these
// terms because "the first pass has no previous state" is a model change
// that no shape check and no finiteness check can see.
//
// They also carry the recycling itself: runOnce writes the three buffers
// at the end of every pass, so the next pass reads them.
#pragma once
#include "common.cuh"

namespace recycle {

constexpr int DBINS = 15;      // config.prev_pos.num_bins
constexpr float MIN_BIN = 3.25f, MAX_BIN = 20.75f;
constexpr int CA = 1, CB = 3;  // atom_order
constexpr int GLY = 7;         // restype_order['G']
constexpr int ATOM = 37;

// ZERO RECYCLES: a LayerNorm of an all-zero row IS ITS OFFSET.
//
// The note at the top of this file says the three terms are nonzero on the
// first pass; this is the other half of that fact.  (0 - 0) * rstd is 0 for
// any eps, so lnFwd over a zeroed buffer writes `offset[c]` into every one
// of its N*N rows -- a broadcast constant.  At recycles = 0 that is the
// whole of the two norm terms, every pass, because forward() zeroes the
// previous state before each one.
//
// So the buffer being normalised need not exist and need not be read.  That
// is `prev_pair`, N*N*128*4 -- 277 MB at N=736, held for the life of a step
// to have zeros read out of it.
//
// AND THE SAME IS TRUE OF THE THIRD TERM, for the reason the header now
// gives: the distogram of an all-zero `prev_pos` is all zeros, so the
// gemm contributes nothing and the Linear's BIAS is the whole of it --
// another broadcast constant.  With all three broadcasts, `dg`, `lnbuf`,
// `pb`, `prev_pos`, `mu` and `rstd` are unnecessary as well, which is a
// further 314 MB at N=736.
//
// The rounding is kept where it was: lnFwd's output was rounded to the bf16
// grid before the accumulate, so the offset is rounded before it is added,
// which is the same number.
__global__ void accBroadcastRoundedK(float* __restrict__ d,
                                     const float* __restrict__ off,
                                     int C, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    d[i] += __bfloat162float(__float2bfloat16(off[i % (size_t)C]));
}

// pseudo_beta_fn with all_atom_mask None: CB, or CA for glycine
__global__ void pseudoBetaK(const int32_t* __restrict__ aatype,
                            const float* __restrict__ pos,   // [NMAX,37,3]
                            float* __restrict__ out,         // [NMAX,3]
                            int NMAX) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NMAX) return;
  const int a = (aatype[i] == GLY) ? CA : CB;
  for (int c = 0; c < 3; ++c)
    out[(size_t)i * 3 + c] = pos[((size_t)i * ATOM + a) * 3 + c];
}

// dgram_from_positions: the breaks are on the SQUARED distance, and the
// comparison is strict both ways -- a distance exactly on a break lands in
// no bin at all, which is AF2's own behaviour and not a bug to fix here.
__global__ void dgramK(const float* __restrict__ pb,   // [NMAX,3]
                       const int32_t* __restrict__ np,
                       float* __restrict__ out,        // [NMAX,NMAX,15]
                       int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * NMAX) return;
  const int i = (int)(p / NMAX), j = (int)(p % NMAX);
  float* o = out + p * DBINS;
  if (i >= *np || j >= *np) {
    for (int b = 0; b < DBINS; ++b) o[b] = 0.f;
    return;
  }
  float d2 = 0.f;
  for (int c = 0; c < 3; ++c) {
    const float d = pb[(size_t)i * 3 + c] - pb[(size_t)j * 3 + c];
    d2 += d * d;
  }
  const float step = (MAX_BIN - MIN_BIN) / (float)(DBINS - 1);
  for (int b = 0; b < DBINS; ++b) {
    const float lo = MIN_BIN + step * (float)b;
    const float lo2 = lo * lo;
    const float hi2 = (b == DBINS - 1) ? 1e8f
                                       : (lo + step) * (lo + step);
    o[b] = (d2 > lo2 && d2 < hi2) ? 1.f : 0.f;
  }
}

struct Recycle {
  int N = 0, S = 1;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  float *wpp = nullptr, *bpp = nullptr;                // prev_pos_linear
  float *s_msa = nullptr, *o_msa = nullptr;            // prev_msa_first_row_norm
  float *s_pair = nullptr, *o_pair = nullptr;          // prev_pair_norm
  float *pb = nullptr, *dg = nullptr, *lnbuf = nullptr;
  // lnFwdK writes mu and rstd UNCONDITIONALLY -- `mu[row] = m` with no null
  // check -- so a caller that does not want them still has to give it
  // somewhere to put them
  float *mu = nullptr, *rstd = nullptr;
  // the state that recycling carries
  float *prev_pos = nullptr, *prev_pair = nullptr, *prev_msa0 = nullptr;
  // Set BEFORE alloc().  At zero recycles the previous state is zero at
  // every pass, so the two normed buffers are not allocated at all; see
  // accBroadcastRoundedK.  `prev_pos` IS still allocated -- the distogram
  // term reads it, and its zeros are not a broadcast constant because
  // dgramK masks on the runtime length.
  bool prev_zero = false;

  void alloc(int N_, cudaStream_t s) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, s));
    const size_t n1 = (size_t)N, nn = (size_t)N * N;
    if (!prev_zero) {
      CU(cudaMalloc(&pb, n1 * 3 * 4));
      CU(cudaMalloc(&dg, nn * DBINS * 4));
      CU(cudaMalloc(&lnbuf, nn * 128 * 4));
      CU(cudaMalloc(&prev_pos, n1 * ATOM * 3 * 4));
    }
    if (!prev_zero) {
      CU(cudaMalloc(&prev_pair, nn * 128 * 4));
      CU(cudaMalloc(&prev_msa0, n1 * 256 * 4));
    }
    if (!prev_zero) {
      CU(cudaMalloc(&mu, nn * 4));
      CU(cudaMalloc(&rstd, nn * 4));
    }
  }

  void zeroPrev() {
    const size_t n1 = (size_t)N, nn = (size_t)N * N;
    if (prev_zero) return;
    CU(cudaMemsetAsync(prev_pos, 0, n1 * ATOM * 3 * 4, st));
    if (!prev_zero) {
      CU(cudaMemsetAsync(prev_pair, 0, nn * 128 * 4, st));
      CU(cudaMemsetAsync(prev_msa0, 0, n1 * 256 * 4, st));
    }
  }

  void setStream(cudaStream_t s) {
    st = s; CUB(cublasSetStream(cub, st));
  }


  void setWeights(Oracle& o, const std::string& P = "param/") {
    auto get = [&](const std::string& nm, float** dst, size_t cnt) {
      Tensor& t = o[P + nm];
      if (t.n() != cnt) {
        fprintf(stderr, "recycle: %s has %zu, want %zu\n",
                nm.c_str(), t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      recordWeightPack(*dst);
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("prev_pos_linear/weights", &wpp, (size_t)DBINS * 128);
    get("prev_pos_linear/bias", &bpp, 128);
    get("prev_msa_first_row_norm/scale", &s_msa, 256);
    get("prev_msa_first_row_norm/offset", &o_msa, 256);
    get("prev_pair_norm/scale", &s_pair, 128);
    get("prev_pair_norm/offset", &o_pair, 128);
  }

  // adds all three terms in place
  void forward(const int32_t* aatype, const int32_t* n,
               float* pair, float* msa) {
    const size_t nn = (size_t)N * N, np = nn * 128;
    if (prev_zero) {
      // prev_pos is zero -> the distogram is zero -> the gemm is zero and
      // the Linear's bias is the whole term.  See the header.
      accBroadcastRoundedK<<<(int)((np + 255) / 256), 256, 0, st>>>(
          pair, bpp, 128, np);
      roundBf16K<<<(int)((np + 255) / 256), 256, 0, st>>>(pair, np);
    } else {
    pseudoBetaK<<<(N + 127) / 128, 128, 0, st>>>(aatype, prev_pos, pb, N);
    dgramK<<<(int)((nn + 127) / 128), 128, 0, st>>>(pb, n, dg, N);
    const float alpha = 1.f, beta = 0.f, one = 1.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, 128, (int)nn, DBINS,
                    &alpha, wpp, 128, dg, DBINS, &beta, lnbuf, 128));
    addRowBiasK<<<(int)((np + 255) / 256), 256, 0, st>>>(lnbuf, bpp, 128, np);
    // EVERY STEP OF THIS LANDS ON THE bf16 GRID in AF2 -- the whole
    // embedder runs inside utils.bfloat16_context().  See roundBf16K.
    roundBf16K<<<(int)((np + 255) / 256), 256, 0, st>>>(lnbuf, np);
    foldit::accK<<<(int)((np + 255) / 256), 256, 0, st>>>(pair, lnbuf, np);
    roundBf16K<<<(int)((np + 255) / 256), 256, 0, st>>>(pair, np);
    (void)one;
    }

    if (prev_zero) {
      accBroadcastRoundedK<<<(int)((np + 255) / 256), 256, 0, st>>>(
          pair, o_pair, 128, np);
    } else {
      foldit::lnFwdK<128><<<(int)((nn + 7) / 8), 256, 0, st>>>(
          prev_pair, s_pair, o_pair, lnbuf, mu, rstd, (int)nn);
      roundBf16K<<<(int)((np + 255) / 256), 256, 0, st>>>(lnbuf, np);
      foldit::accK<<<(int)((np + 255) / 256), 256, 0, st>>>(pair, lnbuf, np);
    }
    roundBf16K<<<(int)((np + 255) / 256), 256, 0, st>>>(pair, np);

    const size_t nm = (size_t)N * 256;
    if (prev_zero) {
      accBroadcastRoundedK<<<(int)((nm + 255) / 256), 256, 0, st>>>(
          msa, o_msa, 256, nm);
    } else {
      foldit::lnFwdK<256><<<(N + 7) / 8, 256, 0, st>>>(
          prev_msa0, s_msa, o_msa, lnbuf, mu, rstd, N);
      roundBf16K<<<(int)((nm + 255) / 256), 256, 0, st>>>(lnbuf, nm);
      foldit::accK<<<(int)((nm + 255) / 256), 256, 0, st>>>(msa, lnbuf, nm);
    }
    roundBf16K<<<(int)((nm + 255) / 256), 256, 0, st>>>(msa, nm);
  }

  // what the next pass reads
  void save(const float* atom37, const float* pair_out, const float* msa_out) {
    const size_t n1 = (size_t)N, nn = (size_t)N * N;
    if (prev_zero) return;
    CU(cudaMemcpyAsync(prev_pos, atom37, n1 * ATOM * 3 * 4,
                       cudaMemcpyDeviceToDevice, st));
    // At zero recycles nothing reads these -- forward() zeroes the previous
    // state before every pass -- and they are not allocated.
    if (!prev_zero) {
      CU(cudaMemcpyAsync(prev_pair, pair_out, nn * 128 * 4,
                         cudaMemcpyDeviceToDevice, st));
      CU(cudaMemcpyAsync(prev_msa0, msa_out, n1 * 256 * 4,
                         cudaMemcpyDeviceToDevice, st));
    }
  }
};

}  // namespace recycle
