// AF2's outer product mean, as a reusable module.
//
// Follows the block interface set in msacol.cuh: bf16 activations, float32
// gradients across the boundary, forward() writes its DELTA.  This one reads
// the msa and writes a PAIR delta, so it is the mirror of msarow.
//
// Reassociated: contracting output_w into `right` first makes the shared index
// (s,c) of extent S*O rather than (c,e) of extent O*O, which is cheaper
// whenever the alignment is shallower than the outer channel (32).  Design runs
// one or two rows, so it always is, and the [N, O, O, N] intermediate -- 134 MB
// at N=256 -- never exists.
#pragma once
#include "common.cuh"
#include <type_traits>

#include "lt.cuh"

namespace opm {

static const int CM = 256;
static const int CZ = 128;
static const int O = 32;
static const float LN_EPS = 1e-5f;
static const float OPM_EPS = 1e-3f;

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

__global__ void lnFwdK(const __nv_bfloat16* __restrict__ x, const float* __restrict__ scale,
                       const float* __restrict__ offset, __nv_bfloat16* __restrict__ xn,
                       float* __restrict__ mu, float* __restrict__ rstd, int M) {
  int row = blockIdx.x, c = threadIdx.x;
  float v = __bfloat162float(x[(size_t)row * CM + c]);
  __shared__ float red[CM / 32];
  float m = blockSum<CM / 32>(v, red, c) / CM;
  float d = v - m;
  float r = rsqrtf(blockSum<CM / 32>(d * d, red, c) / CM + LN_EPS);
  xn[(size_t)row * CM + c] = __float2bfloat16(d * r * scale[c] + offset[c]);
  if (c == 0) { mu[row] = m; rstd[row] = r; }
}

__global__ void lnBwdK(const __nv_bfloat16* __restrict__ dxn,
                       const __nv_bfloat16* __restrict__ x,
                       const float* __restrict__ scale, const float* __restrict__ mu,
                       const float* __restrict__ rstd, float* __restrict__ dx, int M) {
  int row = blockIdx.x, c = threadIdx.x;
  size_t o = (size_t)row * CM + c;
  float r = rstd[row];
  float xhat = (__bfloat162float(x[o]) - mu[row]) * r;
  float dxhat = __bfloat162float(dxn[o]) * scale[c];
  __shared__ float red[CM / 32];
  float mean_d = blockSum<CM / 32>(dxhat, red, c) / CM;
  float mean_dx = blockSum<CM / 32>(dxhat * xhat, red, c) / CM;
  dx[o] = r * (dxhat - mean_d - xhat * mean_dx);
}

// lr = [left | right] masked; `right` goes out as [s][N][O] for the first GEMM
// and `L` as [N][s*O] -- the layout the second one contracts over.
__global__ void splitK(const __nv_bfloat16* __restrict__ lr, const float* __restrict__ mask,
                       __nv_bfloat16* __restrict__ left, __nv_bfloat16* __restrict__ right,
                       __nv_bfloat16* __restrict__ L, int N, int S) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)S * N * O;
  if (t >= n) return;
  size_t row = t / O; int c = t % O;              // row = s*N + i
  int s = (int)(row / N), i = (int)(row % N);
  float mk = mask[(size_t)s * N + i];
  float l = mk * __bfloat162float(lr[row * (2 * O) + c]);
  float r = mk * __bfloat162float(lr[row * (2 * O) + O + c]);
  left[t] = __float2bfloat16(l);
  right[t] = __float2bfloat16(r);
  L[(size_t)i * (S * O) + s * O + c] = __float2bfloat16(l);   // [N][S*O]
}

// the backward of the same: dL and dright -> dlr, through the mask
__global__ void unsplitK(const float* __restrict__ dL, const __nv_bfloat16* __restrict__ dright,
                         const float* __restrict__ mask, __nv_bfloat16* __restrict__ dlr,
                         int N, int S) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)S * N * O;
  if (t >= n) return;
  size_t row = t / O; int c = t % O;
  int s = (int)(row / N), i = (int)(row % N);
  float mk = mask[(size_t)s * N + i];
  dlr[row * (2 * O) + c] = __float2bfloat16(dL[(size_t)i * (S * O) + s * O + c] * mk);
  dlr[row * (2 * O) + O + c] = __float2bfloat16(__bfloat162float(dright[t]) * mk);
}

// norm[i,j] = sum_s mask[s,i] mask[s,j]; the output is divided by eps + norm
__global__ void normDivK(__nv_bfloat16* __restrict__ out, const float* __restrict__ mask,
                         int N, int S, bool fwd, const __nv_bfloat16* __restrict__ src,
                         const float* __restrict__ nzbuf = nullptr) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= (size_t)N * N * CZ) return;
  size_t ij = t / CZ;
  float nz = 0.f;
  if (nzbuf) {
    nz = nzbuf[ij];
  } else {
    int i = (int)(ij / N), j = (int)(ij % N);
    for (int s = 0; s < S; ++s) nz += mask[(size_t)s * N + i] * mask[(size_t)s * N + j];
  }
  float v = __bfloat162float(fwd ? out[t] : src[t]) / (OPM_EPS + nz);
  out[t] = __float2bfloat16(v);
}

__global__ void addBiasK(__nv_bfloat16* __restrict__ out, const float* __restrict__ b,
                         size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float2bfloat16(__bfloat162float(out[i]) + b[i % CZ]);
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
// dT is [(s,c)][(j,f)]; the single GEMM that replaces 64 tiny ones wants
// [(s,j)][(c,f)].  2.1M elements, so the transpose costs far less than the
// launches it retires.
__global__ void dTtransK(const __nv_bfloat16* __restrict__ dT,
                         __nv_bfloat16* __restrict__ dTt, int N, int S) {
  size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t n = (size_t)S * O * N * CZ;
  if (t >= n) return;
  size_t f = t % CZ, rest = t / CZ;
  size_t j = rest % N, rest2 = rest / N;
  size_t c = rest2 % O, sIdx = rest2 / O;
  dTt[((sIdx * N + j) * O + c) * CZ + f] = dT[t];
}

__global__ void bf16ToF32K(const __nv_bfloat16* __restrict__ a, float* __restrict__ b,
                           size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = __bfloat162float(a[i]);
}

// row-major batched C[Mo,No] = opA(A) @ opB(B)
static void bmm(cublasHandle_t h, bool ta, bool tb, int Mo, int No, int K,
                const __nv_bfloat16* A, int lda, long long sA,
                const __nv_bfloat16* B, int ldb, long long sB,
                __nv_bfloat16* C, int ldc, long long sC, int batch, float be = 0.f) {
  float al = 1.f;
  CUB(cublasGemmStridedBatchedEx(
      h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, No, Mo, K,
      &al, B, CUDA_R_16BF, ldb, sB, A, CUDA_R_16BF, lda, sA, &be, C, CUDA_R_16BF,
      ldc, sC, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}


struct Opm {
  Lt lt;
  cublasHandle_t cub;
  int N = 0, S = 0;
  size_t Mm = 0, Mp = 0;
  cudaStream_t st = nullptr;
  __nv_bfloat16 *wlr, *wlrT, *blr, *ow, *owT;
  float *ln_s, *ln_o, *ob;
  __nv_bfloat16 *a, *lr, *left, *right, *L, *T, *out, *dT, *dTt, *dright, *dlr, *da, *dLb;
  float *mu, *rstd, *dL;

  // Every pointer setWeights() assigns, and nothing else.  A 48-deep stack
  // needs 48 sets of WEIGHTS but only one set of scratch, so it calls
  // setWeights() once a depth, snapshots these slots, and re-points them per
  // block -- which is why the split between the two groups above is load-bearing
  // rather than tidiness.
  std::vector<void**> weightSlots() {
    return {(void**)&wlr, (void**)&wlrT, (void**)&blr, (void**)&ow, (void**)&owT, (void**)&ln_s, (void**)&ln_o, (void**)&ob};
  }

  void alloc(int N_, int S_, cudaStream_t s = nullptr) {
    N = N_; S = S_; Mm = (size_t)S * N; Mp = (size_t)N * N; st = s;
    lt.init(); CUB(cublasCreate(&cub)); CUB(cublasSetStream(cub, st));
    auto B = [&](__nv_bfloat16** p, size_t n) { CU(cudaMalloc(p, n * 2)); };
    B(&a, Mm * CM); B(&lr, Mm * 2 * O); B(&left, Mm * O); B(&right, Mm * O);
    B(&L, (size_t)N * S * O); B(&T, (size_t)S * O * N * CZ); B(&out, Mp * CZ);
    if (!nblean::on()) {
      // dT doubles as the narrowed incoming gradient, which is [N][N][Cz] --
      // bigger than the [S*O][N*Cz] it holds later, so size it for the larger
      B(&dT, Mp * CZ); B(&dTt, (size_t)S * O * N * CZ);
      B(&dright, Mm * O); B(&dlr, Mm * 2 * O); B(&da, Mm * CM);
      B(&dLb, (size_t)N * S * O);
      CU(cudaMalloc(&dL, (size_t)N * S * O * 4));
    } else {
      dT = dTt = dright = dlr = da = dLb = nullptr;
      dL = nullptr;
    }
    CU(cudaMalloc(&mu, Mm * 4)); CU(cudaMalloc(&rstd, Mm * 4));
  }

  template <class Tmul>
  void adoptBlockScratch(const Tmul& m) {
    const size_t need = Mp * CZ * 2;
    const size_t mcap = nbrealloc::capacityOf((void* const*)&m.da, (size_t)m.M * 128 * 2);
    if (m.da && need <= mcap) {
      nbrealloc::adoptPtr((void**)&out, m.da, mcap);
      if (dT) nbrealloc::adoptPtr((void**)&dT, m.da, mcap);
    } else {
      const size_t acap = nbrealloc::capacityOf((void* const*)&m.a, (size_t)m.M * 128 * 2);
      if (m.a && need <= acap)
        nbrealloc::adoptPtr((void**)&out, m.a, acap);
    }
  }


  void setWeights(Oracle& o, const std::string& P) {
    auto dev = [&](const std::string& k) { o[k].to_device(); return o[k].dev; };
    auto& lw = o[P + "left_projection/weights"];
    auto& rw = o[P + "right_projection/weights"];
    std::vector<float> host((size_t)CM * 2 * O);
    for (int r = 0; r < CM; ++r)
      for (int c = 0; c < O; ++c) {
        host[(size_t)r * 2 * O + c] = lw.host[(size_t)r * O + c];
        host[(size_t)r * 2 * O + O + c] = rw.host[(size_t)r * O + c];
      }
    float* d; CU(cudaMalloc(&d, host.size() * 4));
    CU(cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&wlr, host.size() * 2));
      recordWeightPack(wlr); CU(cudaMalloc(&wlrT, host.size() * 2));
      recordWeightPack(wlrT);
    f32ToBf16K<<<(host.size() + 255) / 256, 256>>>(d, wlr, host.size());
    transposeWK<<<(host.size() + 255) / 256, 256>>>(d, wlrT, CM, 2 * O);
    CU(cudaFree(d));
    std::vector<float> hb(2 * O);
    auto& lb = o[P + "left_projection/bias"]; auto& rb = o[P + "right_projection/bias"];
    for (int c = 0; c < O; ++c) { hb[c] = lb.host[c]; hb[O + c] = rb.host[c]; }
    float* db; CU(cudaMalloc(&db, 2 * O * 4));
    CU(cudaMemcpy(db, hb.data(), 2 * O * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&blr, 2 * O * 2));
      recordWeightPack(blr); f32ToBf16K<<<1, 2 * O>>>(db, blr, 2 * O);
    CU(cudaFree(db));
    float* owf = dev(P + "output_w");
    CU(cudaMalloc(&ow, (size_t)O * O * CZ * 2));
      recordWeightPack(ow);
    f32ToBf16K<<<((size_t)O * O * CZ + 255) / 256, 256>>>(owf, ow, (size_t)O * O * CZ);
    // output_w is [c][e][f]; the backward's single GEMM contracts over (c,f)
    auto& owh = o[P + "output_w"];
    std::vector<float> hw((size_t)O * CZ * O);
    for (int c = 0; c < O; ++c)
      for (int e = 0; e < O; ++e)
        for (int f = 0; f < CZ; ++f)
          hw[((size_t)c * CZ + f) * O + e] = owh.host[((size_t)c * O + e) * CZ + f];
    float* dw; CU(cudaMalloc(&dw, hw.size() * 4));
    CU(cudaMemcpy(dw, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&owT, hw.size() * 2));
      recordWeightPack(owT);
    f32ToBf16K<<<(hw.size() + 255) / 256, 256>>>(dw, owT, hw.size());
    CU(cudaFree(dw));
    ln_s = dev(P + "layer_norm_input/scale");
    ln_o = dev(P + "layer_norm_input/offset");
    ob = dev(P + "output_b");
  }

  // msa [S][N][Cm] -> out [N][N][Cz], the PAIR delta
  // `fusedOut` means the caller applies the bias and the normalise itself, as
  // part of its residual add -- so `out` is left as the raw bmm result.
  void forward(const __nv_bfloat16* msa, const float* mask, bool fusedOut = false) {
    lnFwdK<<<Mm, CM, 0, st>>>(msa, ln_s, ln_o, a, mu, rstd, Mm);
    lt.matmul(Mm, 2 * O, CM, a, wlrT, lr, CUDA_R_16BF, CUBLASLT_EPILOGUE_BIAS,
              blr, nullptr, st);
    splitK<<<(Mm * O + 255) / 256, 256, 0, st>>>(lr, mask, left, right, L, N, S);
    for (int s = 0; s < S; ++s)
      bmm(cub, false, false, N, CZ, O, right + (size_t)s * N * O, O, 0,
          ow, CZ, (long long)O * CZ, T + (size_t)s * O * N * CZ, CZ,
          (long long)N * CZ, O);
    bmm(cub, false, false, N, (int)((size_t)N * CZ), S * O, L, S * O, 0,
        T, (int)((size_t)N * CZ), 0, out, (int)((size_t)N * CZ), 0, 1);
    if (!fusedOut) {
      addBiasK<<<(Mp * CZ + 255) / 256, 256, 0, st>>>(out, ob, Mp * CZ);
      normDivK<<<(Mp * CZ + 255) / 256, 256, 0, st>>>(out, mask, N, S, true, nullptr);
    }
  }

  // dout f32 [N][N][Cz] -> dmsa f32 [S][N][Cm]
  // DT is the PAIR cotangent's width: this module reads it (the outer product
  // mean's own output is the pair) and writes the msa gradient, which is f32.
  template <class DT>
  void backward(const DT* dout, const __nv_bfloat16* msa, const float* mask,
                float* dmsa, const float* nzbuf = nullptr) {
    // normDivK already takes a separate source for the backward direction, so a
    // bf16 cotangent goes straight in -- no copy and no narrowing pass, just
    // the divide reading dp and writing dT.
    if constexpr (std::is_same<DT, __nv_bfloat16>::value) {
      normDivK<<<(Mp * CZ + 255) / 256, 256, 0, st>>>(dT, mask, N, S, false, dout, nzbuf);
    } else {
      f32ToBf16K<<<(Mp * CZ + 255) / 256, 256, 0, st>>>(dout, dT, Mp * CZ);
      normDivK<<<(Mp * CZ + 255) / 256, 256, 0, st>>>(dT, mask, N, S, false, dT, nzbuf);
    }
    bmm(cub, false, true, N, S * O, (int)((size_t)N * CZ), dT,
        (int)((size_t)N * CZ), 0, T, (int)((size_t)N * CZ), 0, dLb, S * O, 0, 1);
    bf16ToF32K<<<((size_t)N * S * O + 255) / 256, 256, 0, st>>>(dLb, dL,
                                                               (size_t)N * S * O);
    bmm(cub, true, false, S * O, (int)((size_t)N * CZ), N, L, S * O, 0,
        dT, (int)((size_t)N * CZ), 0, dTt, (int)((size_t)N * CZ), 0, 1);
    dTtransK<<<((size_t)S * O * N * CZ + 255) / 256, 256, 0, st>>>(dTt, dT, N, S);
    bmm(cub, false, false, (int)Mm, O, O * CZ, dT, O * CZ, 0, owT, O, 0,
        dright, O, 0, 1);
    unsplitK<<<(Mm * O + 255) / 256, 256, 0, st>>>(dL, dright, mask, dlr, N, S);
    lt.matmul(Mm, CM, 2 * O, dlr, wlr, da, CUDA_R_16BF, CUBLASLT_EPILOGUE_DEFAULT,
              nullptr, nullptr, st);
    lnBwdK<<<Mm, CM, 0, st>>>(da, msa, ln_s, mu, rstd, dmsa, Mm);
  }
};

}  // namespace opm
