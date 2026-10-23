// AF2's MSA row attention with pair bias, as a reusable module.
//
// Follows the block interface set in msacol.cuh: bf16 activations in AF2's
// layouts, float32 gradients across the boundary, forward() writes its DELTA
// and the caller owns the residual.
//
// This is the first module that touches BOTH activation streams -- it reads the
// msa and the pair, and its gradient goes back into both.  The attention itself
// is trivial (two alignment rows); what costs is the bias, a LayerNorm over the
// whole pair and a [N*N, 128] x [128, 8] projection, and a gradient that lands
// on the pair rather than the msa.  So it is gated on grad/pair as much as on
// grad/msa.
//
// Both kernels are reused unchanged: localfold's flash forward and
// flash_bwd.cuh take the head count at run time and are compiled for D = 32,
// which is what this needs (8 heads of 32 against triangle attention's 4).
#pragma once
#include "common.cuh"
#include "lt.cuh"
#include "lf_flash.cuh"
#include "flash_bwd.cuh"

namespace mrow {

static const int CM = 256;
static const int CZ = 128;
static const int H = 8;
static const int D = 32;
static const int W = H * D;
static const int TOT = 4 * W;
static const float LN_EPS = 1e-5f;

template <int NW>
__device__ __forceinline__ float blockSum(float v, float* s, int tid) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if ((tid & 31) == 0) s[tid >> 5] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < NW; ++i) t += s[i];
  __syncthreads();
  return t;
}

// ONE WARP A ROW, no shared memory and no __syncthreads -- see trimul.cuh.
// It matters here mostly for the PAIR norm, which runs over all N^2 rows (the
// msa norm's are only S*N).
template <int C>
__global__ void lnFwdK(const __nv_bfloat16* __restrict__ x, const float* __restrict__ scale,
                       const float* __restrict__ offset, __nv_bfloat16* __restrict__ xn,
                       float* __restrict__ mu, float* __restrict__ rstd, int M) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  int lane = threadIdx.x & 31;
  const int K = C / 32;
  const size_t base = (size_t)row * C;
  float v[K], s = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    v[k] = __bfloat162float(x[base + lane + k * 32]);
    s += v[k];
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  float m = s / C, s2 = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) { v[k] -= m; s2 += v[k] * v[k]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
  float r = rsqrtf(s2 / C + LN_EPS);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = lane + k * 32;
    xn[base + c] = __float2bfloat16(v[k] * r * scale[c] + offset[c]);
  }
  if (lane == 0) { mu[row] = m; rstd[row] = r; }
}

// ACC adds into `dx` rather than overwriting it, so the caller can hand its
// running gradient straight in.  The five PAIR modules got this earlier; the
// MSA row attention is the one that was left, because its pair gradient went
// out through the block's `sp` scratch and an accK -- an extra 25.7 MB write
// and 77 MB of accumulate a block.
template <int C, bool ACC = false, class DT = float>
__global__ void lnBwdK(const __nv_bfloat16* __restrict__ dxn,
                       const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ scale, const float* __restrict__ mu,
                       const float* __restrict__ rstd, DT* __restrict__ dx, int M) {
  int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= M) return;
  int lane = threadIdx.x & 31;
  const int K = C / 32;
  const size_t base = (size_t)row * C;
  float r = rstd[row], mu_ = mu[row];
  float xh[K], dxh[K], s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    int c = lane + k * 32;
    xh[k] = (__bfloat162float(x[base + c]) - mu_) * r;
    dxh[k] = __bfloat162float(dxn[base + c]) * scale[c];
    s1 += dxh[k];
    s2 += dxh[k] * xh[k];
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s1 += __shfl_xor_sync(0xffffffff, s1, o);
    s2 += __shfl_xor_sync(0xffffffff, s2, o);
  }
  float mean_d = s1 / C, mean_dx = s2 / C;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    size_t o = base + lane + k * 32;
    float v = r * (dxh[k] - mean_d - xh[k] * mean_dx);
    if (ACC) v += toF(dx[o]);
    stF(dx + o, v);
  }
}

// [M, H] -> [H, M], and into the log2 domain localfold's kernel reads its bias in
__global__ void nbbToHalfK(const __nv_bfloat16* __restrict__ src, half* __restrict__ dst,
                           size_t M) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= M) return;
#pragma unroll
  for (int h = 0; h < H; ++h)
    dst[(size_t)h * M + ij] =
        __float2half(fmaxf(__bfloat162float(src[ij * H + h]) * LOG2E, -6e4f));
}
// ...and back: layout only, NO log2 factor.  The kernel scores in the log2
// domain, so the forward hands it log2(e)*nbb -- but the backward's
// P * (dP - D) is already dL/dS in the NATURAL domain, because the ln2 from
// differentiating exp2 cancels the log2(e) in u = log2(e)*S.  Applying the
// factor here as well put grad/pair at relL2 3.1e-01.
__global__ void dnbbToRowK(const float* __restrict__ dnbbF,
                           const long long* __restrict__ dnbbI,
                           const float* __restrict__ lscale,
                           __nv_bfloat16* __restrict__ dst,
                           size_t M) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= M) return;
  const float sc = lscale[1];
  if (dnbbI) {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      size_t idx = (size_t)h * M + ij;
      float v = (float)dnbbI[idx] * (1.0f / 1099511627776.0f);
      dst[ij * H + h] = __float2bfloat16(v * sc);
    }
  } else {
#pragma unroll
    for (int h = 0; h < H; ++h) {
      size_t idx = (size_t)h * M + ij;
      dst[ij * H + h] = __float2bfloat16(dnbbF[idx] * sc);
    }
  }
}

__global__ void combToHalfK(const __nv_bfloat16* __restrict__ comb, half* __restrict__ q,
                            size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) q[i] = __float2half(__bfloat162float(comb[i]));
}
__global__ void f32ToBf16K(const float* __restrict__ a, __nv_bfloat16* __restrict__ b,
                           size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __float2bfloat16(a[i]);
}
__global__ void transposeWK(const float* __restrict__ a, __nv_bfloat16* __restrict__ b,
                            int R, int Cc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R * Cc) b[(size_t)(i % Cc) * R + i / Cc] = __float2bfloat16(a[i]);
}
__global__ void scatterQKVK(const float* __restrict__ dQ, const float* __restrict__ dK,
                            const float* __restrict__ dV, const float* __restrict__ lscale,
                            __nv_bfloat16* __restrict__ dcomb,
                            size_t M) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * (size_t)W) return;
  size_t r = i / W; int c = i % W;
  const float sc = lscale[1];
  dcomb[r * TOT + c] = __float2bfloat16(dQ[i] * sc);
  dcomb[r * TOT + W + c] = __float2bfloat16(dK[i] * sc);
  dcomb[r * TOT + 2 * W + c] = __float2bfloat16(dV[i] * sc);
}
__global__ void gateK(const half* __restrict__ av, const __nv_bfloat16* __restrict__ comb,
                      __nv_bfloat16* __restrict__ ag, size_t M) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * (size_t)W) return;
  size_t r = i / W; int c = i % W;
  float g = 1.f / (1.f + __expf(-__bfloat162float(comb[r * TOT + 3 * W + c])));
  ag[i] = __float2bfloat16(__half2float(av[i]) * g);
}
__global__ void dGateK(const __nv_bfloat16* __restrict__ dag, const half* __restrict__ av,
                       const __nv_bfloat16* __restrict__ comb, half* __restrict__ dav,
                       __nv_bfloat16* __restrict__ dcomb, size_t M) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * (size_t)W) return;
  size_t r = i / W; int c = i % W;
  float g = 1.f / (1.f + __expf(-__bfloat162float(comb[r * TOT + 3 * W + c])));
  float d = __bfloat162float(dag[i]);
  dav[i] = __float2half(d * g);
  dcomb[r * TOT + 3 * W + c] = __float2bfloat16(d * __half2float(av[i]) * g * (1.f - g));
}


struct MsaRow {
  Lt lt;
  int N = 0, S = 0;
  size_t Mm = 0, Mp = 0;
  cudaStream_t st = nullptr;
  __nv_bfloat16 *wcat, *wcatT, *bcat, *wo, *woT, *bo, *w2d, *w2dT;
  float *qn_s, *qn_o, *f2_s, *f2_o;
  __nv_bfloat16 *x, *pn, *comb, *nbb, *ag, *out, *dcomb, *dag, *dnbbB, *dx, *dpn;
  half *qkvgH, *biasH, *avH, *davH, *outH;
  float *mu, *rstd, *pmu, *prstd, *lseF, *Dv, *dQf, *dKf, *dVf, *dnbbF;
  // dnbb's int64 fixed-point twin -- the float atomics are order-dependent
  // and that is what stopped a design trajectory reproducing.  See
  // nbbFixed in flash_bwd_body.cuh; NE_NONDET_NBB=1 restores them.
  long long* dnbbI = nullptr;
  // dynamic loss scaling for the f16 attention backward: [0] scales the
  // cotangent down, [1] scales the three gradients back up.  See common.cuh.
  float* lscale = nullptr;
  unsigned* lmax = nullptr;

  // Every pointer setWeights() assigns, and nothing else.  A 48-deep stack
  // needs 48 sets of WEIGHTS but only one set of scratch, so it calls
  // setWeights() once a depth, snapshots these slots, and re-points them per
  // block -- which is why the split between the two groups above is load-bearing
  // rather than tidiness.
  std::vector<void**> weightSlots() {
    return {(void**)&wcat, (void**)&wcatT, (void**)&bcat, (void**)&wo, (void**)&woT, (void**)&bo, (void**)&w2d, (void**)&w2dT, (void**)&qn_s, (void**)&qn_o, (void**)&f2_s, (void**)&f2_o};
  }

  // EXACTLY WHAT backward() READS, which is not everything forward() writes.
  // `x`, `pn`, `nbb`, `ag` and `out` are forward-only and are NOT here: the
  // backward narrows the incoming gradient into `dag` and goes straight to
  // the projections.  Listing them anyway would nearly double the tier for
  // nothing, and `pn` alone is N^2*CZ -- the pair LayerNorm -- which would
  // take this tier from 34 ms/GB to about 8.
  std::vector<KeepSlot> keepSlots() {
    return {{(void**)&comb,   Mm * TOT * 2,              KEEP_MSA},
            {(void**)&qkvgH,  Mm * TOT * sizeof(half),   KEEP_MSA},
            {(void**)&biasH,  (size_t)H * Mp * sizeof(half), KEEP_MSA},
            {(void**)&avH,    Mm * W * sizeof(half),     KEEP_MSA},
            {(void**)&lseF,   (size_t)S * H * N * 4,     KEEP_MSA},
            {(void**)&Dv,     (size_t)S * H * N * 4,     KEEP_MSA},
            {(void**)&mu,     Mm * 4,                    KEEP_MSA},
            {(void**)&rstd,   Mm * 4,                    KEEP_MSA},
            {(void**)&pmu,    Mp * 4,                    KEEP_MSA},
            {(void**)&prstd,  Mp * 4,                    KEEP_MSA}};
  }

  void alloc(int N_, int S_, cudaStream_t s = nullptr) {
    N = N_; S = S_; Mm = (size_t)S * N; Mp = (size_t)N * N; st = s;
    lt.init(); LF_STREAM = st;
    auto B = [&](__nv_bfloat16** p, size_t n) { CU(cudaMalloc(p, n * 2)); };
    B(&x, Mm * CM); B(&pn, Mp * CZ); B(&comb, Mm * TOT); B(&nbb, (size_t)H * Mp);
    B(&ag, Mm * W); B(&out, Mm * CM);
    CU(cudaMalloc(&qkvgH, Mm * TOT * sizeof(half)));
    CU(cudaMalloc(&biasH, (size_t)H * Mp * sizeof(half)));
    CU(cudaMalloc(&avH, Mm * W * sizeof(half)));
    CU(cudaMalloc(&outH, Mm * W * sizeof(half)));
    CU(cudaMalloc(&mu, Mm * 4)); CU(cudaMalloc(&rstd, Mm * 4));
    CU(cudaMalloc(&pmu, Mp * 4)); CU(cudaMalloc(&prstd, Mp * 4));
    CU(cudaMalloc(&lseF, (size_t)S * H * N * 4));
    CU(cudaMalloc(&Dv, (size_t)S * H * N * 4));
    if (!nblean::on()) {
      B(&dcomb, Mm * TOT); B(&dag, Mm * W);
      B(&dnbbB, Mp * H); B(&dx, Mm * CM); B(&dpn, Mp * CZ);
      CU(cudaMalloc(&davH, Mm * W * sizeof(half)));
      CU(cudaMalloc(&dQf, Mm * W * 4)); CU(cudaMalloc(&dKf, Mm * W * 4));
      CU(cudaMalloc(&dVf, Mm * W * 4)); CU(cudaMalloc(&dnbbF, (size_t)H * Mp * 4));
      if (!envOn("NE_NONDET_NBB"))
        CU(cudaMalloc(&dnbbI, (size_t)H * Mp * 8));
      CU(cudaMalloc(&lscale, 2 * 4)); CU(cudaMalloc(&lmax, 4));
    } else {
      dcomb = dag = dnbbB = dx = dpn = nullptr;
      davH = nullptr;
      dQf = dKf = dVf = dnbbF = nullptr;
      dnbbI = nullptr;
      lscale = nullptr; lmax = nullptr;
    }
  }


  void setWeights(Oracle& o, const std::string& P) {
    auto dev = [&](const std::string& k) { o[k].to_device(); return o[k].dev; };
    auto& qw = o[P + "attention/query_w"]; auto& kw = o[P + "attention/key_w"];
    auto& vw = o[P + "attention/value_w"]; auto& gw = o[P + "attention/gating_w"];
    std::vector<float> host((size_t)CM * TOT, 0.f);
    float sc = 1.f / sqrtf((float)D);
    for (int r = 0; r < CM; ++r)
      for (int c = 0; c < W; ++c) {
        host[(size_t)r * TOT + c] = qw.host[(size_t)r * W + c] * sc;
        host[(size_t)r * TOT + W + c] = kw.host[(size_t)r * W + c];
        host[(size_t)r * TOT + 2 * W + c] = vw.host[(size_t)r * W + c];
        host[(size_t)r * TOT + 3 * W + c] = gw.host[(size_t)r * W + c];
      }
    float* d; CU(cudaMalloc(&d, host.size() * 4));
    CU(cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&wcat, host.size() * 2));
      recordWeightPack(wcat); CU(cudaMalloc(&wcatT, host.size() * 2));
      recordWeightPack(wcatT);
    f32ToBf16K<<<(host.size() + 255) / 256, 256>>>(d, wcat, host.size());
    transposeWK<<<(host.size() + 255) / 256, 256>>>(d, wcatT, CM, TOT);
    CU(cudaFree(d));
    std::vector<float> hb(TOT, 0.f);
    auto& gb = o[P + "attention/gating_b"];
    for (int c = 0; c < W; ++c) hb[3 * W + c] = gb.host[c];
    float* db; CU(cudaMalloc(&db, TOT * 4));
    CU(cudaMemcpy(db, hb.data(), TOT * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&bcat, TOT * 2));
      recordWeightPack(bcat); f32ToBf16K<<<(TOT + 255) / 256, 256>>>(db, bcat, TOT);
    CU(cudaFree(db));
    float* wof = dev(P + "attention/output_w");
    CU(cudaMalloc(&wo, (size_t)W * CM * 2));
      recordWeightPack(wo); CU(cudaMalloc(&woT, (size_t)W * CM * 2));
      recordWeightPack(woT);
    f32ToBf16K<<<((size_t)W * CM + 255) / 256, 256>>>(wof, wo, (size_t)W * CM);
    transposeWK<<<((size_t)W * CM + 255) / 256, 256>>>(wof, woT, W, CM);
    CU(cudaMalloc(&bo, CM * 2));
      recordWeightPack(bo);
    f32ToBf16K<<<(CM + 255) / 256, 256>>>(dev(P + "attention/output_b"), bo, CM);
    float* w2f = dev(P + "feat_2d_weights");
    CU(cudaMalloc(&w2d, (size_t)CZ * H * 2));
      recordWeightPack(w2d); CU(cudaMalloc(&w2dT, (size_t)CZ * H * 2));
      recordWeightPack(w2dT);
    f32ToBf16K<<<((size_t)CZ * H + 255) / 256, 256>>>(w2f, w2d, (size_t)CZ * H);
    transposeWK<<<((size_t)CZ * H + 255) / 256, 256>>>(w2f, w2dT, CZ, H);
    qn_s = dev(P + "query_norm/scale");   qn_o = dev(P + "query_norm/offset");
    f2_s = dev(P + "feat_2d_norm/scale"); f2_o = dev(P + "feat_2d_norm/offset");
  }

  // msa [S][N][Cm], pair [N][N][Cz] -> out [S][N][Cm], the delta
  void forward(const __nv_bfloat16* msa, const __nv_bfloat16* pair, const float* mask) {
    LF_STREAM = st;
    lnFwdK<CM><<<(Mm + 7) / 8, 256, 0, st>>>(msa, qn_s, qn_o, x, mu, rstd, Mm);
    lnFwdK<CZ><<<(Mp + 7) / 8, 256, 0, st>>>(pair, f2_s, f2_o, pn, pmu, prstd, Mp);
    lt.matmul(Mm, TOT, CM, x, wcatT, comb, CUDA_R_16BF, CUBLASLT_EPILOGUE_BIAS,
              bcat, nullptr, st);
    lt.matmul(Mp, H, CZ, pn, w2dT, nbb, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st);
    nbbToHalfK<<<(Mp + 255) / 256, 256, 0, st>>>(nbb, biasH, Mp);
    combToHalfK<<<(Mm * TOT + 255) / 256, 256, 0, st>>>(comb, qkvgH, Mm * TOT);
    // `out` is the kernel's GATED output; avOut is the ungated one the backward needs
    flashGridHalfAt<32, 4, 48>(qkvgH, biasH, N, mask, outH, N, H, 0, (size_t)S,
                               false, 1.f, nullptr, avH, lseF);
    gateK<<<(Mm * W + 255) / 256, 256, 0, st>>>(avH, comb, ag, Mm);
    lt.matmul(Mm, CM, W, ag, woT, out, CUDA_R_16BF, CUBLASLT_EPILOGUE_BIAS, bo,
              nullptr, st);
  }

  // dout f32 [S][N][Cm] -> dmsa f32 and dpair f32, both without the residual
  // DP is the PAIR cotangent's width (bf16 on the pair track); the msa
  // cotangent stays f32, which is why only one of the two is templated.
  template <class DP>
  void backward(const float* dout, const __nv_bfloat16* msa, const __nv_bfloat16* pair,
                const float* mask, float* dmsa, DP* dpair,
                bool accPair = false) {
    LF_STREAM = st;
    __nv_bfloat16* doutB = dag;           // narrow the incoming gradient once
    f32ToBf16K<<<(Mm * CM + 255) / 256, 256, 0, st>>>(dout, dx, Mm * CM);
    lt.matmul(Mm, W, CM, dx, wo, doutB, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st);
    dGateK<<<(Mm * W + 255) / 256, 256, 0, st>>>(doutB, avH, comb, davH, dcomb, Mm);
    // SCALE THE COTANGENT INTO f16's RANGE, and only it: dGateK also wrote the
    // gate gradient into `dcomb`, which must not move.  Dv is computed from
    // the scaled davH below, so it scales with it automatically and the
    // softmax correction stays consistent.
    const size_t nmw = (size_t)Mm * W;
    CU(cudaMemsetAsync(lmax, 0, 4, st));
    maxAbsHalfK<<<256, 256, 0, st>>>(davH, lmax, nmw);
    pickScaleK<<<1, 1, 0, st>>>(lmax, lscale, 1024.f);
    fb::rowDotK<<<((size_t)S * N * H + 255) / 256, 256, 0, st>>>(
        davH, avH, Dv, N, H, D, S, lscale);
    CU(cudaMemsetAsync(dKf, 0, Mm * W * 4, st));
    CU(cudaMemsetAsync(dVf, 0, Mm * W * 4, st));
    if (dnbbI) CU(cudaMemsetAsync(dnbbI, 0, (size_t)H * Mp * 8, st));
    else       CU(cudaMemsetAsync(dnbbF, 0, (size_t)H * Mp * 4, st));
    static bool once = false;
    if (!once) {
      CU(cudaFuncSetAttribute((const void*)fb::flashBwdK<true, true, false, true>,
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              (int)fb::bwdSmem()));
      once = true;
    }
    dim3 g((N + fb::BM - 1) / fb::BM, H, S);
    fb::flashBwdK<true, true, false, true><<<g, 32 * fb::WARPS, fb::bwdSmem(), st>>>(
        qkvgH, TOT, biasH, davH, lseF, Dv, mask, nullptr, dQf, dKf, dVf, N, H,
        dnbbF, dnbbI);
    scatterQKVK<<<(Mm * W + 255) / 256, 256, 0, st>>>(dQf, dKf, dVf, lscale, dcomb, Mm);
    lt.matmul(Mm, CM, TOT, dcomb, wcat, dx, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st);
    lnBwdK<CM><<<(Mm + 7) / 8, 256, 0, st>>>(dx, msa, qn_s, mu, rstd, dmsa, Mm);
    // the bias gradient belongs to the PAIR, through feat_2d_weights and its norm
    dnbbToRowK<<<(Mp + 255) / 256, 256, 0, st>>>(dnbbF, dnbbI, lscale, dnbbB, Mp);
    lt.matmul(Mp, CZ, H, dnbbB, w2d, dpn, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st);
    if (accPair) lnBwdK<CZ, true, DP><<<(Mp + 7) / 8, 256, 0, st>>>(dpn, pair, f2_s, pmu, prstd, dpair, Mp);
    else         lnBwdK<CZ, false, DP><<<(Mp + 7) / 8, 256, 0, st>>>(dpn, pair, f2_s, pmu, prstd, dpair, Mp);
  }
};

}  // namespace mrow
