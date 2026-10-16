// AF2's MSA column attention as a reusable module.
//
// THE BLOCK INTERFACE, set here and followed by every module after it:
//
//   * ACTIVATIONS are bf16 in the layouts AF2 uses -- msa [S][N][Cm], pair
//     [N][N][Cz] -- because that is what one module hands the next, and what
//     AF2 itself carries under global_config.bfloat16.  The standalone gates
//     convert the oracle's float32 at the boundary, so what they measure is the
//     precision a real chain would carry rather than a friendlier one.
//   * GRADIENTS cross a module boundary in float32.  The GEMMs still consume
//     bf16 -- that is what the tensor cores take -- but a block adds a residual
//     to the gradient at every one of nine steps, forty-eight times over, and
//     those adds are cheap to do properly.
//
//     MEASURED, because I first wrote the opposite here: making them float32
//     changes this module's backward by nothing at all, 7.552e-02 against
//     7.551e-02.  The step from the standalone 4.97e-02 to 7.55e-02 is the
//     bf16 ACTIVATION input, not the gradient -- the LayerNorm reads a bf16 msa
//     and its backward recomputes xhat from it.  Which is faithful: AF2 under
//     global_config.bfloat16 carries bf16 activations too, so this is what the
//     model costs, not what the port costs.  The float32 gradient is a choice
//     about ACCUMULATION that a single module cannot test, and it stays because
//     it is nearly free, not because this measured it.
//   * a module OWNS its scratch and its save-set, allocated once in alloc().
//   * forward() reads the activation and writes its own `out` delta; the
//     caller adds the residual.  backward() reads the delta's gradient and
//     writes the activation's, likewise without the residual.  Keeping the
//     residual outside is what lets a block chain them without each module
//     knowing where it sits.
//   * the LayerNorm backward recomputes from the module's INPUT, which is the
//     bf16 activation, plus the mean and inverse it saved -- so nothing needs
//     the original float32 anywhere.
//
// It attends over the alignment depth, which is two in design: 256 independent
// 2x2 problems a head, not one N x N problem.  Flash is the wrong tool and
// localfold's kernels would refuse the shape -- this is what
// flash_attention_min_len guards.  One warp takes one (position, head) and
// carries the whole S x S problem in registers, which is also why there is no
// separate backward kernel: recomputing the scores beats storing them.
//
// AF2 writes swapaxes(msa, -2, -3) on the way in and out; here the LayerNorms
// index [n][s][c] straight out of [s][n][c], so the transpose is an index.
#pragma once
#include "common.cuh"
#include "lt.cuh"

namespace mcol {

static const int CM = 256;
static const int H = 8;
static const int D = 32;
static const int W = H * D;
static const int TOT = 4 * W;
static const int MAXS = 8;
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

__global__ void lnFwdTK(const __nv_bfloat16* __restrict__ msa,
                        const float* __restrict__ scale, const float* __restrict__ offset,
                        __nv_bfloat16* __restrict__ xn, float* __restrict__ mu,
                        float* __restrict__ rstd, int N, int S) {
  int row = blockIdx.x, c = threadIdx.x;      // row = n*S + s
  int n = row / S, s = row % S;
  float v = __bfloat162float(msa[((size_t)s * N + n) * CM + c]);
  __shared__ float red[CM / 32];
  float m = blockSum<CM / 32>(v, red, c) / CM;
  float d = v - m;
  float r = rsqrtf(blockSum<CM / 32>(d * d, red, c) / CM + LN_EPS);
  xn[(size_t)row * CM + c] = __float2bfloat16(d * r * scale[c] + offset[c]);
  if (c == 0) { mu[row] = m; rstd[row] = r; }
}

__global__ void lnBwdTK(const __nv_bfloat16* __restrict__ dxn,
                        const __nv_bfloat16* __restrict__ msa,
                        const float* __restrict__ scale, const float* __restrict__ mu,
                        const float* __restrict__ rstd, float* __restrict__ dmsa,
                        int N, int S) {
  int row = blockIdx.x, c = threadIdx.x;
  int n = row / S, s = row % S;
  size_t src = ((size_t)s * N + n) * CM + c;
  size_t o = (size_t)row * CM + c;
  float r = rstd[row];
  float xhat = (__bfloat162float(msa[src]) - mu[row]) * r;
  float dxhat = __bfloat162float(dxn[o]) * scale[c];
  __shared__ float red[CM / 32];
  float mean_d = blockSum<CM / 32>(dxhat, red, c) / CM;
  float mean_dx = blockSum<CM / 32>(dxhat * xhat, red, c) / CM;
  dmsa[src] = r * (dxhat - mean_d - xhat * mean_dx);
}

__global__ void attnFwdK(const __nv_bfloat16* __restrict__ comb,
                         const float* __restrict__ mask, __nv_bfloat16* __restrict__ ag,
                         __nv_bfloat16* __restrict__ av, int N, int S) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  int n = w / H, h = w % H;
  if (n >= N) return;
  const int e = lane;
  float q[MAXS], k[MAXS], v[MAXS], g[MAXS];
#pragma unroll
  for (int s = 0; s < MAXS; ++s) {
    if (s >= S) { q[s] = k[s] = v[s] = g[s] = 0.f; continue; }
    size_t r = (size_t)n * S + s;
    q[s] = __bfloat162float(comb[r * TOT + h * D + e]);
    k[s] = __bfloat162float(comb[r * TOT + W + h * D + e]);
    v[s] = __bfloat162float(comb[r * TOT + 2 * W + h * D + e]);
    g[s] = __bfloat162float(comb[r * TOT + 3 * W + h * D + e]);
  }
  for (int i = 0; i < S; ++i) {
    float m = -1e30f, sc[MAXS];
#pragma unroll
    for (int j = 0; j < MAXS; ++j) {
      if (j >= S) { sc[j] = -1e30f; continue; }
      float dot = q[i] * k[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
      sc[j] = dot + 1e9f * (mask[(size_t)j * N + n] - 1.f);
      m = fmaxf(m, sc[j]);
    }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < MAXS; ++j) { sc[j] = __expf(sc[j] - m); sum += sc[j]; }
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < MAXS; ++j) acc += sc[j] * v[j];
    acc /= sum;
    size_t r = (size_t)n * S + i;
    float sig = 1.f / (1.f + __expf(-g[i]));
    av[r * W + h * D + e] = __float2bfloat16(acc);
    ag[r * W + h * D + e] = __float2bfloat16(acc * sig);
  }
}

__global__ void attnBwdK(const __nv_bfloat16* __restrict__ comb,
                         const __nv_bfloat16* __restrict__ av,
                         const __nv_bfloat16* __restrict__ dag,
                         const float* __restrict__ mask, __nv_bfloat16* __restrict__ dcomb,
                         int N, int S) {
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  int n = w / H, h = w % H;
  if (n >= N) return;
  const int e = lane;
  float q[MAXS], k[MAXS], v[MAXS], g[MAXS], dq[MAXS], dk[MAXS], dv[MAXS];
#pragma unroll
  for (int s = 0; s < MAXS; ++s) {
    dq[s] = dk[s] = dv[s] = 0.f;
    if (s >= S) { q[s] = k[s] = v[s] = g[s] = 0.f; continue; }
    size_t r = (size_t)n * S + s;
    q[s] = __bfloat162float(comb[r * TOT + h * D + e]);
    k[s] = __bfloat162float(comb[r * TOT + W + h * D + e]);
    v[s] = __bfloat162float(comb[r * TOT + 2 * W + h * D + e]);
    g[s] = __bfloat162float(comb[r * TOT + 3 * W + h * D + e]);
  }
  for (int i = 0; i < S; ++i) {
    size_t r = (size_t)n * S + i;
    float sig = 1.f / (1.f + __expf(-g[i]));
    float dagv = __bfloat162float(dag[r * W + h * D + e]);
    float accv = __bfloat162float(av[r * W + h * D + e]);
    dcomb[r * TOT + 3 * W + h * D + e] =
        __float2bfloat16(dagv * accv * sig * (1.f - sig));
    float dacc = dagv * sig;
    float p[MAXS], m = -1e30f;
#pragma unroll
    for (int j = 0; j < MAXS; ++j) {
      if (j >= S) { p[j] = -1e30f; continue; }
      float dot = q[i] * k[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
      p[j] = dot + 1e9f * (mask[(size_t)j * N + n] - 1.f);
      m = fmaxf(m, p[j]);
    }
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < MAXS; ++j) { p[j] = __expf(p[j] - m); sum += p[j]; }
#pragma unroll
    for (int j = 0; j < MAXS; ++j) p[j] /= sum;
    float dP[MAXS], dot = 0.f;
#pragma unroll
    for (int j = 0; j < MAXS; ++j) {
      float t = dacc * v[j];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
      dP[j] = t;
    }
#pragma unroll
    for (int j = 0; j < MAXS; ++j) dot += p[j] * dP[j];
#pragma unroll
    for (int j = 0; j < MAXS; ++j) {
      if (j >= S) continue;
      dv[j] += p[j] * dacc;
      float dS = p[j] * (dP[j] - dot);
      dq[i] += dS * k[j];
      dk[j] += dS * q[i];
    }
  }
#pragma unroll
  for (int s = 0; s < MAXS; ++s) {
    if (s >= S) continue;
    size_t r = (size_t)n * S + s;
    dcomb[r * TOT + h * D + e] = __float2bfloat16(dq[s]);
    dcomb[r * TOT + W + h * D + e] = __float2bfloat16(dk[s]);
    dcomb[r * TOT + 2 * W + h * D + e] = __float2bfloat16(dv[s]);
  }
}

// [n][s][Cm] -> [s][n][Cm]: the one place the notional transpose is real
__global__ void outScatterK(const __nv_bfloat16* __restrict__ o,
                            __nv_bfloat16* __restrict__ dst, int N, int S) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * S * CM) return;
  size_t row = i / CM; int c = i % CM;
  dst[(((size_t)(row % S) * N) + row / S) * CM + c] = o[i];
}
// the incoming gradient is float32 and is narrowed here, once, for the GEMM
__global__ void dOutGatherK(const float* __restrict__ src,
                            __nv_bfloat16* __restrict__ o, int N, int S) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * S * CM) return;
  size_t row = i / CM; int c = i % CM;
  o[i] = __float2bfloat16(src[(((size_t)(row % S) * N) + row / S) * CM + c]);
}

__global__ void toBf16K(const float* __restrict__ a, __nv_bfloat16* __restrict__ b,
                        size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __float2bfloat16(a[i]);
}
__global__ void transWK(const float* __restrict__ a, __nv_bfloat16* __restrict__ b,
                        int R, int Cc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < R * Cc) b[(size_t)(i % Cc) * R + i / Cc] = __float2bfloat16(a[i]);
}

struct MsaCol {
  Lt lt;
  int N = 0, S = 0;
  size_t M = 0;
  cudaStream_t st = nullptr;
  __nv_bfloat16 *wcat = nullptr, *wcatT, *bcat, *wo, *woT, *bo;
  float *qn_s, *qn_o;
  __nv_bfloat16 *x, *comb, *av, *ag, *ot, *out, *dcomb, *dag, *dx, *dot_;
  float *mu, *rstd;
  int blocks = 0, thr = 128;

  // Every pointer setWeights() assigns, and nothing else.  A 48-deep stack
  // needs 48 sets of WEIGHTS but only one set of scratch, so it calls
  // setWeights() once a depth, snapshots these slots, and re-points them per
  // block -- which is why the split between the two groups above is load-bearing
  // rather than tidiness.
  std::vector<void**> weightSlots() {
    return {(void**)&wcat, (void**)&wcatT, (void**)&bcat, (void**)&wo, (void**)&woT, (void**)&bo, (void**)&qn_s, (void**)&qn_o};
  }

  // backward() reads `comb`, `av`, `mu` and `rstd` and nothing else forward()
  // wrote.  `av` was MISSED on the first pass -- the inventory grep looked
  // for `avH`, which is msa_row's name for it -- and the stack gate caught
  // it at grad/pair 0.9716 against a band of 0.9781-0.9785.  See KEEP_MSA in
  // common.cuh.
  std::vector<KeepSlot> keepSlots() {
    return {{(void**)&comb, M * TOT * 2, KEEP_MSA},
            {(void**)&av,   M * W * 2,   KEEP_MSA},
            {(void**)&mu,   M * 4,       KEEP_MSA},
            {(void**)&rstd, M * 4,       KEEP_MSA}};
  }

  void alloc(int N_, int S_, cudaStream_t s = nullptr) {
    N = N_; S = S_; M = (size_t)N * S; st = s;
    lt.init();
    auto B = [&](__nv_bfloat16** p, size_t n) { CU(cudaMalloc(p, n * 2)); };
    B(&x, M * CM); B(&comb, M * TOT); B(&av, M * W); B(&ag, M * W);
    B(&ot, M * CM); B(&out, M * CM); B(&dcomb, M * TOT); B(&dag, M * W);
    B(&dx, M * CM); B(&dot_, M * CM);
    CU(cudaMalloc(&mu, M * 4)); CU(cudaMalloc(&rstd, M * 4));
    blocks = (N * H * 32 + thr - 1) / thr;
  }

  // weights straight from an oracle dump; a block will hand these in instead

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
    toBf16K<<<(host.size() + 255) / 256, 256>>>(d, wcat, host.size());
    transWK<<<(host.size() + 255) / 256, 256>>>(d, wcatT, CM, TOT);
    CU(cudaFree(d));
    std::vector<float> hb(TOT, 0.f);
    auto& gb = o[P + "attention/gating_b"];
    for (int c = 0; c < W; ++c) hb[3 * W + c] = gb.host[c];
    float* db; CU(cudaMalloc(&db, TOT * 4));
    CU(cudaMemcpy(db, hb.data(), TOT * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&bcat, TOT * 2));
      recordWeightPack(bcat); toBf16K<<<(TOT + 255) / 256, 256>>>(db, bcat, TOT);
    CU(cudaFree(db));
    float* wof = dev(P + "attention/output_w");
    CU(cudaMalloc(&wo, (size_t)W * CM * 2));
      recordWeightPack(wo); CU(cudaMalloc(&woT, (size_t)W * CM * 2));
      recordWeightPack(woT);
    toBf16K<<<((size_t)W * CM + 255) / 256, 256>>>(wof, wo, (size_t)W * CM);
    transWK<<<((size_t)W * CM + 255) / 256, 256>>>(wof, woT, W, CM);
    CU(cudaMalloc(&bo, CM * 2));
      recordWeightPack(bo);
    toBf16K<<<(CM + 255) / 256, 256>>>(dev(P + "attention/output_b"), bo, CM);
    qn_s = dev(P + "query_norm/scale");
    qn_o = dev(P + "query_norm/offset");
  }

  // msa [S][N][Cm] bf16 -> out [S][N][Cm] bf16, the DELTA (no residual)
  void forward(const __nv_bfloat16* msa, const float* mask) {
    lnFwdTK<<<M, CM, 0, st>>>(msa, qn_s, qn_o, x, mu, rstd, N, S);
    lt.matmul(M, TOT, CM, x, wcatT, comb, CUDA_R_16BF, CUBLASLT_EPILOGUE_BIAS,
              bcat, nullptr, st);
    attnFwdK<<<blocks, thr, 0, st>>>(comb, mask, ag, av, N, S);
    lt.matmul(M, CM, W, ag, woT, ot, CUDA_R_16BF, CUBLASLT_EPILOGUE_BIAS, bo,
              nullptr, st);
    outScatterK<<<(M * CM + 255) / 256, 256, 0, st>>>(ot, out, N, S);
  }

  // dout [S][N][Cm] -> dmsa [S][N][Cm], likewise without the residual
  void backward(const float* dout, const __nv_bfloat16* msa, const float* mask,
                float* dmsa) {
    dOutGatherK<<<(M * CM + 255) / 256, 256, 0, st>>>(dout, dot_, N, S);
    lt.matmul(M, W, CM, dot_, wo, dag, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st);
    attnBwdK<<<blocks, thr, 0, st>>>(comb, av, dag, mask, dcomb, N, S);
    lt.matmul(M, CM, TOT, dcomb, wcat, dx, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st);
    lnBwdTK<<<M, CM, 0, st>>>(dx, msa, qn_s, mu, rstd, dmsa, N, S);
  }
};

}  // namespace mcol
