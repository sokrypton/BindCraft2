// The design step's input and update sides: soft_seq, norm_seq_grad, Adam.
//
//   logits = x * alpha + bias
//   pssm   = softmax(logits)
//   soft   = softmax(logits / temp)
//   hard   = one_hot(argmax(soft))           STRAIGHT-THROUGH
//   pseudo = h * hard + (1 - h) * (s * soft + (1 - s) * x)
//
// THE STRAIGHT-THROUGH ESTIMATOR IS THE WHOLE TRICK: forward is a discrete
// one-hot, backward is the softmax's jacobian.  Without it a hard sequence
// receives no gradient at all and design stops.  So in the backward `hard`
// and `soft` are the SAME path -- which is why the two coefficients collapse
// into one below.
//
//   d pseudo / d x = [h + (1-h) s] * (alpha/temp) * J_soft + (1-h)(1-s) * I
//
// and J_soft^T g = soft * (g - <soft, g>), the usual softmax backward.
//
// ALL FOUR KNOBS ARE RUNTIME VALUES, not compile-time: a schedule moves
// alpha, temp, soft and hard between steps, and in JAX they are traced so
// that costs no recompile.  Here they are kernel arguments, which costs
// nothing at all.
#pragma once
#include "common.cuh"

namespace seqx {

static const int NAA = 20;

__global__ void softSeqK(const float* __restrict__ x, const float* __restrict__ bias,
                         float alpha, float temp, float sw, float hw,
                         float* __restrict__ soft, float* __restrict__ pseudo,
                         int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* X = x + (size_t)n * NAA;
  float lg[NAA];
  float m = -1e30f, mt = -1e30f;
#pragma unroll
  for (int a = 0; a < NAA; ++a) {
    lg[a] = X[a] * alpha + (bias ? bias[(size_t)n * NAA + a] : 0.f);
    m = fmaxf(m, lg[a]);
  }
#pragma unroll
  for (int a = 0; a < NAA; ++a) mt = fmaxf(mt, lg[a] / temp);
  float s = 0.f;
#pragma unroll
  for (int a = 0; a < NAA; ++a) s += __expf(lg[a] / temp - mt);
  float* S = soft + (size_t)n * NAA;
  int best = 0;
  float bv = -1e30f;
#pragma unroll
  for (int a = 0; a < NAA; ++a) {
    S[a] = __expf(lg[a] / temp - mt) / s;
    if (S[a] > bv) { bv = S[a]; best = a; }
  }
  float* P = pseudo + (size_t)n * NAA;
#pragma unroll
  for (int a = 0; a < NAA; ++a) {
    float hard = (a == best) ? 1.f : 0.f;
    float v = sw * S[a] + (1.f - sw) * X[a];
    P[a] = hw * hard + (1.f - hw) * v;
  }
}

__global__ void softSeqBwdK(const float* __restrict__ dpseudo,
                            const float* __restrict__ soft, float alpha,
                            float temp, float sw, float hw,
                            float* __restrict__ dx, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* D = dpseudo + (size_t)n * NAA;
  const float* S = soft + (size_t)n * NAA;
  // hard and soft share a path under the straight-through estimator
  const float csoft = hw + (1.f - hw) * sw;
  const float cid = (1.f - hw) * (1.f - sw);
  float dot = 0.f;
#pragma unroll
  for (int a = 0; a < NAA; ++a) dot += S[a] * D[a];
  float* G = dx + (size_t)n * NAA;
#pragma unroll
  for (int a = 0; a < NAA; ++a)
    G[a] = csoft * (alpha / temp) * S[a] * (D[a] - dot) + cid * D[a];
}

// v1's _norm_seq_grad: rescale to norm sqrt(effective length), where eff_L
// counts positions whose gradient is not identically zero.
//
// NOT A NICETY.  AF2's gradient with respect to a soft sequence has no
// natural scale and moves by orders of magnitude as the sequence sharpens,
// so a fixed learning rate against an unnormalised gradient is arbitrary.
// Normalising to sqrt(eff_L) rather than 1 keeps the per-position step size
// independent of the design's length.
__global__ void normSeqGradK(float* __restrict__ g, int N) {
  __shared__ float ssum, scount;
  float local = 0.f;
  int eff = 0;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    float row = 0.f;
    for (int a = 0; a < NAA; ++a) {
      float v = g[(size_t)n * NAA + a];
      row += v * v;
    }
    local += row;
    eff += row > 0.f;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    local += __shfl_xor_sync(0xffffffffu, local, o);
    eff += __shfl_xor_sync(0xffffffffu, eff, o);
  }
  if (threadIdx.x == 0) { ssum = local; scount = (float)eff; }
  __syncthreads();
  float scale = sqrtf(scount) / (sqrtf(ssum) + 1e-7f);
  for (int n = threadIdx.x; n < N; n += blockDim.x)
    for (int a = 0; a < NAA; ++a) g[(size_t)n * NAA + a] *= scale;
}

// Adam, with the bias correction written out: a design run is a few hundred
// steps, so the first few matter and the correction is not negligible.
__global__ void adamK(float* __restrict__ x, const float* __restrict__ g,
                      float* __restrict__ m, float* __restrict__ v, float lr,
                      float b1, float b2, float eps, int step, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float gi = g[i];
  float mi = b1 * m[i] + (1.f - b1) * gi;
  float vi = b2 * v[i] + (1.f - b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  float mh = mi / (1.f - __powf(b1, (float)step));
  float vh = vi / (1.f - __powf(b2, (float)step));
  x[i] -= lr * mh / (sqrtf(vh) + eps);
}

// hold chosen positions at the wildtype identity.  Keeping the parameters
// full length and overwriting (rather than parameterising only the free
// positions) is what lets seq_fixed change between steps for free.
__global__ void pinFixedK(float* __restrict__ seq, const float* __restrict__ wt,
                          const float* __restrict__ fixed, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N || fixed[n] <= 0.f) return;
  for (int a = 0; a < NAA; ++a)
    seq[(size_t)n * NAA + a] = (wt[n] == (float)a) ? 1.f : 0.f;
}

}  // namespace seqx
