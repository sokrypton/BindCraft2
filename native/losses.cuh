// The design losses: i_con, plddt and pae, forward and backward.
//
// This is the end of the chain. Everything upstream exists to put logits in
// front of these three, and the gradient a design step applies to the
// sequence starts here.
//
//   i_con  = min_k( min_k(contact_entropy, num) , num_pos )
//   plddt  = 1 - masked_mean( sum_b softmax(logits)[b] * bin_centre[b] )
//   pae    = masked_mean( sum_b softmax(logits)[b] * bin_centre[b] )
//
// WHY min_k AND NOT A MEAN.  i_con asks each designed residue for its `num`
// best contacts and then takes the `num_pos` best residues -- so the loss is
// driven by a handful of entries and its gradient is ZERO everywhere else.
// A mean would spread the same signal over N^2 pairs and design something
// different.
//
// THE min_k TIE-BREAK IS LOAD-BEARING: a row with fewer than k ELIGIBLE
// entries averages over however many it has, not over k. colabdesign2 gets
// this by sorting with NaN for the excluded and counting the non-NaN below
// k; the selection here reproduces that by counting what it actually found.
//
// f32, and the bin edges are compile-time: the distogram's 64 bins are
// [0, linspace(2.3125, 21.6875, 63)] and nothing in a design run changes them.
#pragma once
#include "common.cuh"

namespace losses {

static const int NDG = 64;      // distogram bins
static const int NPAE = 64;
static const int NPL = 50;
static const int MAXK = 64;     // the largest min_k this supports

__device__ __forceinline__ float dgramBin(int b) {
  // [0, linspace(2.3125, 21.6875, 63)]
  return b == 0 ? 0.f : 2.3125f + (21.6875f - 2.3125f) * (b - 1) / 62.f;
}

// the contact entropy of every pair, and the two softmaxes its backward needs
__global__ void conPerPairK(const float* __restrict__ dg, float cutoff,
                            int binary, float* __restrict__ p,
                            float* __restrict__ pxs, float* __restrict__ pxms,
                            int N) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)N * N) return;
  const float* L = dg + ij * NDG;
  float m = -1e30f;
#pragma unroll
  for (int b = 0; b < NDG; ++b) m = fmaxf(m, L[b]);
  float s = 0.f;
#pragma unroll
  for (int b = 0; b < NDG; ++b) s += __expf(L[b] - m);
  float lse = m + __logf(s);
  // px_ is the softmax of the logits with the OUT-OF-CUTOFF bins driven to
  // -inf, which is what `- 1e7 * (1 - bins)` does
  float m2 = -1e30f;
#pragma unroll
  for (int b = 0; b < NDG; ++b)
    if (dgramBin(b) < cutoff) m2 = fmaxf(m2, L[b]);
  float s2 = 0.f;
#pragma unroll
  for (int b = 0; b < NDG; ++b)
    if (dgramBin(b) < cutoff) s2 += __expf(L[b] - m2);
  float cat = 0.f, binsum = 0.f;
#pragma unroll
  for (int b = 0; b < NDG; ++b) {
    float px = __expf(L[b] - lse);
    float pxm = dgramBin(b) < cutoff ? __expf(L[b] - m2) / s2 : 0.f;
    pxs[ij * NDG + b] = px;
    pxms[ij * NDG + b] = pxm;
    cat += pxm * (L[b] - lse);
    if (dgramBin(b) < cutoff) binsum += px;
  }
  p[ij] = binary ? -__logf(binsum + 1e-8f) : -cat;
}

// min_k over a row: the k smallest ELIGIBLE entries, their mean, and which
// they were.  k passes of a masked argmin -- k is 2 or 9 here, so the simple
// form costs nothing and reads like the thing it implements.
// min_k over a row: the k smallest ELIGIBLE entries, their mean, and which
// they were.  ONE WARP a row, so the whole selection reduces with shuffles
// and needs no shared critical section; k is 2 or 9 here, so k passes of a
// masked argmin costs nothing and reads like the thing it implements.
//
// `found` counts what was ACTUALLY selected, which is how a row with fewer
// than k eligible entries ends up averaging over however many it has -- the
// NaN semantics of colabdesign2's sort-based version.
template <bool ROWWISE>
__global__ void minKK(const float* __restrict__ x, const float* __restrict__ mask,
                      int k, float* __restrict__ out, int* __restrict__ pick,
                      int* __restrict__ cnt, int N, int stride) {
  const int row = blockIdx.x, lane = threadIdx.x;
  extern __shared__ char taken[];
  for (int j = lane; j < N; j += 32) taken[j] = 0;
  __syncwarp();
  float acc = 0.f;
  int found = 0;
  for (int it = 0; it < k; ++it) {
    float lb = 1e30f;
    int li = -1;
    for (int j = lane; j < N; j += 32) {
      size_t midx = ROWWISE ? ((size_t)row * stride + j) : (size_t)j;
      if (mask[midx] <= 0.f || taken[j]) continue;
      float v = x[ROWWISE ? ((size_t)row * N + j) : (size_t)j];
      if (v < lb) { lb = v; li = j; }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, lb, o);
      int oi = __shfl_xor_sync(0xffffffffu, li, o);
      // the index decides ties, so every lane agrees on the same winner
      if (ov < lb || (ov == lb && oi >= 0 && (li < 0 || oi < li))) { lb = ov; li = oi; }
    }
    if (li < 0) break;                   // fewer eligible entries than k
    if (lane == 0) taken[li] = 1;
    __syncwarp();
    acc += lb;
    ++found;
    if (lane == 0 && pick) pick[(size_t)row * MAXK + it] = li;
  }
  if (lane == 0) {
    out[row] = acc / (found + 1e-8f);
    if (cnt) cnt[row] = found;
  }
}

// bin-centre expectation, shared by plddt and pae
template <int NB>
__global__ void binExpectK(const float* __restrict__ logits, float first,
                           float step, float* __restrict__ out,
                           float* __restrict__ probs, size_t M) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  const float* L = logits + i * NB;
  float m = -1e30f;
#pragma unroll
  for (int b = 0; b < NB; ++b) m = fmaxf(m, L[b]);
  float s = 0.f;
#pragma unroll
  for (int b = 0; b < NB; ++b) s += __expf(L[b] - m);
  float e = 0.f;
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float p = __expf(L[b] - m) / s;
    probs[i * NB + b] = p;
    e += p * (first + step * b);
  }
  out[i] = e;
}

// d(expectation)/d(logit) = p_b * (centre_b - expectation), times the
// cotangent the masked mean hands down
template <int NB>
__global__ void binExpectBwdK(const float* __restrict__ probs,
                              const float* __restrict__ val,
                              const float* __restrict__ dval, float first,
                              float step, float* __restrict__ dlogits,
                              size_t M) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  float e = val[i], d = dval[i];
#pragma unroll
  for (int b = 0; b < NB; ++b) {
    float p = probs[i * NB + b];
    dlogits[i * NB + b] = d * p * ((first + step * b) - e);
  }
}

// the contact entropy's backward, for the pairs min_k actually selected
__global__ void conPerPairBwdK(const float* __restrict__ dp,
                               const float* __restrict__ pxs,
                               const float* __restrict__ pxms,
                               const float* __restrict__ p, float cutoff,
                               int binary, float* __restrict__ ddg, int N) {
  size_t ij = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (ij >= (size_t)N * N) return;
  float d = dp[ij];
  float* D = ddg + ij * NDG;
  if (d == 0.f) {
#pragma unroll
    for (int b = 0; b < NDG; ++b) D[b] = 0.f;
    return;
  }
  const float* px = pxs + ij * NDG;
  const float* pxm = pxms + ij * NDG;
  if (binary) {
    // p = -log(S + eps), S = sum of the in-cutoff probabilities
    float S = 0.f;
#pragma unroll
    for (int b = 0; b < NDG; ++b) if (dgramBin(b) < cutoff) S += px[b];
    float inv = 1.f / (S + 1e-8f);
#pragma unroll
    for (int b = 0; b < NDG; ++b) {
      float in = dgramBin(b) < cutoff ? 1.f : 0.f;
      D[b] = -d * inv * (in * px[b] - S * px[b]);
    }
  } else {
    // p = -sum_b pxm[b] * logsoftmax[b].  A = sum_b pxm[b] logsoftmax[b] = -p
    float A = -p[ij];
#pragma unroll
    for (int b = 0; b < NDG; ++b) {
      float ls = __logf(fmaxf(px[b], 1e-30f));
      D[b] = -d * (pxm[b] * (ls - A) + pxm[b] - px[b]);
    }
  }
}

// scatter a min_k's cotangent back onto the entries it chose
__global__ void minKBwdK(const int* __restrict__ pick, const int* __restrict__ cnt,
                         const float* __restrict__ dout, float* __restrict__ dx,
                         int rows, int N, int rowwise) {
  int row = blockIdx.x;
  if (row >= rows) return;
  int c = cnt[row];
  if (c == 0) return;
  float g = dout[row] / (c + 1e-8f);
  for (int t = threadIdx.x; t < c; t += blockDim.x) {
    int j = pick[(size_t)row * MAXK + t];
    if (j >= 0) dx[rowwise ? ((size_t)row * N + j) : (size_t)j] = g;
  }
}

}  // namespace losses
