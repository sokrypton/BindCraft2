// alphafold_prediction_metrics: five of the nine boundary arrays.
//
// plddt, pae, ptm, iptm and iptm_per_residue from the head logits the
// native post-trunk already produces, plus the experimentally-resolved CA
// sigmoid.  The distogram is a passthrough and needs no kernel.
//
// Verified as a formula first (conf_formula.py): 7 of 7 against the JAX,
// worst 3.8e-06 on pae, which is the float32 bin arithmetic.
//
// MEASURED BEFORE BEING WRITTEN: this is 0.75 s of the design compile, not
// the 2.45 s an isolated harness claimed -- that number was the function
// compiled cold with nothing to fuse into.  So this is worth porting for
// the single-call architecture, not for the compile saving on its own.
//
// Runtime length throughout: grids are NMAX, `n` comes from device memory.
#pragma once
#include "common.cuh"

namespace confidence {

constexpr int LBINS = 50;      // predicted_lddt
constexpr int CA = 1;          // atom_order['CA']

// ---- plddt: softmax over the bins, dot with their centres -------------
__global__ void plddtK(const float* __restrict__ logits,   // [NMAX, LBINS]
                       const int32_t* __restrict__ np,
                       float* __restrict__ out, int NMAX) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NMAX) return;
  if (i >= *np) { out[i] = 0.f; return; }
  const float* L = logits + (size_t)i * LBINS;
  float m = -INFINITY;
  for (int b = 0; b < LBINS; ++b) m = fmaxf(m, L[b]);
  float z = 0.f, acc = 0.f;
  const float bw = 1.f / LBINS;
  for (int b = 0; b < LBINS; ++b) {
    const float e = __expf(L[b] - m);
    z += e;
    acc += e * (0.5f * bw + b * bw);      // bin centre
  }
  out[i] = acc / z;
}

// ---- the two bin tables -----------------------------------------------
// The per-piece FFI takes `centres` and `tm_bin` as INPUTS, because JAX had
// already built them.  The one-call step has no JAX to build them in, and
// one of the two depends on the RUNTIME length: d0 is a function of
// residue_weights.sum(), so a table baked at compile time would be the
// wrong table for every length but one.  Both are 64 floats; one block.
//
//   breaks  = linspace(0, max_error_bin=31, num_bins-1=63), step 0.5
//   centres = breaks + step/2, with a catch-all bin appended
//           = 0.5*b + 0.25 for every b, the appended one included
//   d0      = 1.24 * (max(sum(w), 19) - 15)^(1/3) - 1.8
//   tm_bin  = 1 / (1 + centres^2 / d0^2)
constexpr int PBINS = 64;
constexpr float MAX_ERROR_BIN = 31.f;

__global__ void binTablesK(const float* __restrict__ smask,
                           const int32_t* __restrict__ np,
                           float* __restrict__ centres,   // [PB]
                           float* __restrict__ tm_bin,    // [PB]
                           int NMAX, int PB) {
  __shared__ float sh;
  const int n = *np;
  if (threadIdx.x == 0) {
    float acc = 0.f;
    for (int i = 0; i < n && i < NMAX; ++i) acc += smask[i];
    sh = acc;
  }
  __syncthreads();
  const float w = fmaxf(sh, 19.f);
  const float d0 = 1.24f * cbrtf(w - 15.f) - 1.8f;
  const float step = MAX_ERROR_BIN / (float)(PB - 2);
  for (int b = threadIdx.x; b < PB; b += blockDim.x) {
    const float c = step * (float)b + 0.5f * step;
    centres[b] = c;
    tm_bin[b] = 1.f / (1.f + (c * c) / (d0 * d0));
  }
}

// ---- the shared pass over the PAE logits ------------------------------
// probs, the expected error, and the TM term all come off ONE softmax.
// pae needs the error; ptm and iptm need the TM term; computing them in
// separate kernels would read the [NMAX, NMAX, PB] tensor three times.
__global__ void paeTermK(const float* __restrict__ logits,  // [NMAX,NMAX,PB]
                         const float* __restrict__ centres, // [PB]
                         const float* __restrict__ tm_bin,  // [PB]
                         const int32_t* __restrict__ np,
                         float* __restrict__ err,           // [NMAX, NMAX]
                         float* __restrict__ tm,            // [NMAX, NMAX]
                         int NMAX, int PB) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * NMAX) return;
  const int n = *np;
  const int i = (int)(p / NMAX), j = (int)(p % NMAX);
  if (i >= n || j >= n) { err[p] = 0.f; tm[p] = 0.f; return; }
  const float* L = logits + p * PB;
  float m = -INFINITY;
  for (int b = 0; b < PB; ++b) m = fmaxf(m, L[b]);
  float z = 0.f, e_acc = 0.f, t_acc = 0.f;
  for (int b = 0; b < PB; ++b) {
    const float e = __expf(L[b] - m);
    z += e; e_acc += e * centres[b]; t_acc += e * tm_bin[b];
  }
  const float inv = 1.f / z;
  err[p] = e_acc * inv;
  tm[p] = t_acc * inv;
}

// ---- pae is symmetrised; do it after, not inside the softmax ----------
__global__ void symmetriseK(const float* __restrict__ err,
                            const int32_t* __restrict__ np,
                            float* __restrict__ out, int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * NMAX) return;
  const int i = (int)(p / NMAX), j = (int)(p % NMAX);
  out[p] = (i < *np && j < *np)
      ? 0.5f * (err[p] + err[(size_t)j * NMAX + i]) : 0.f;
}

// ---- per-alignment (i)TM: one warp per row ----------------------------
// per_alignment[i] = sum_j tm[i,j] * mask[i,j] * normed_pair_weight[i,j]
__global__ void perAlignmentK(const float* __restrict__ tm,
                              const float* __restrict__ smask,
                              const int32_t* __restrict__ asym,
                              int use_asym,            // 0 = pTM, 1 = ipTM
                              const int32_t* __restrict__ np,
                              float* __restrict__ out, int NMAX) {
  const int i = blockIdx.x;
  if (i >= NMAX) return;
  const int n = *np;
  if (i >= n) { if (threadIdx.x == 0) out[i] = 0.f; return; }
  float num = 0.f, den = 0.f;
  for (int j = threadIdx.x; j < n; j += 32) {
    const float mask = use_asym ? (asym[i] != asym[j] ? 1.f : 0.f) : 1.f;
    const float w = mask * smask[i] * smask[j];
    num += tm[(size_t)i * NMAX + j] * mask * w;
    den += w;
  }
  for (int o = 16; o > 0; o >>= 1) {
    num += __shfl_xor_sync(0xffffffffu, num, o);
    den += __shfl_xor_sync(0xffffffffu, den, o);
  }
  if (threadIdx.x == 0) out[i] = num / (1e-8f + den);
}

// ---- the scalar: max over residues of per_alignment * seq_mask --------
__global__ void weightedMaxK(const float* __restrict__ per,
                             const float* __restrict__ smask,
                             const int32_t* __restrict__ np,
                             float* __restrict__ out, int NMAX) {
  const int n = *np;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < n; i += 32) m = fmaxf(m, per[i] * smask[i]);
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (threadIdx.x == 0) *out = m;
}

// ---- experimentally resolved, CA slot ---------------------------------
__global__ void expresCaK(const float* __restrict__ logits,  // [NMAX, 37]
                          const int32_t* __restrict__ np,
                          float* __restrict__ out, int NMAX) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NMAX) return;
  out[i] = (i < *np)
      ? 1.f / (1.f + __expf(-logits[(size_t)i * 37 + CA])) : 0.f;
}

// ---- backwards --------------------------------------------------------
// Only plddt, pae and expres_ca are reached here.  ptm and iptm go through
// a MAX over residues, so their cotangent reaches exactly one row; that is
// handled by perAlignmentBwdK below rather than pretended away.
//
// For a softmax-weighted expectation  y = sum_b p_b c_b  with p = softmax(L):
//     dL_b = p_b * (c_b - y) * g
// which is the usual softmax jacobian with the bin centres as the upstream.

__global__ void plddtBwdK(const float* __restrict__ logits,
                          const float* __restrict__ g,
                          const int32_t* __restrict__ np,
                          float* __restrict__ dlogits, int NMAX) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NMAX) return;
  float* out = dlogits + (size_t)i * LBINS;
  if (i >= *np) {
    for (int b = 0; b < LBINS; ++b) out[b] = 0.f;
    return;
  }
  const float* L = logits + (size_t)i * LBINS;
  const float bw = 1.f / LBINS;
  float m = -INFINITY;
  for (int b = 0; b < LBINS; ++b) m = fmaxf(m, L[b]);
  float z = 0.f, y = 0.f;
  for (int b = 0; b < LBINS; ++b) {
    const float e = __expf(L[b] - m);
    z += e; y += e * (0.5f * bw + b * bw);
  }
  const float inv = 1.f / z;
  y *= inv;
  for (int b = 0; b < LBINS; ++b) {
    const float p = __expf(L[b] - m) * inv;
    out[b] = p * ((0.5f * bw + b * bw) - y) * g[i];
  }
}

// pae is symmetrised AFTER the expectation, so the cotangent on entry is
// (g[i,j] + g[j,i]) / 2 -- forgetting the transpose halves the gradient on
// a symmetric loss and doubles it on an antisymmetric one.
__global__ void paeBwdK(const float* __restrict__ logits,
                        const float* __restrict__ centres,
                        const float* __restrict__ g,
                        const int32_t* __restrict__ np,
                        float* __restrict__ dlogits, int NMAX, int PB) {
  const size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (size_t)NMAX * NMAX) return;
  const int n = *np;
  const int i = (int)(q / NMAX), j = (int)(q % NMAX);
  float* out = dlogits + q * PB;
  if (i >= n || j >= n) {
    for (int b = 0; b < PB; ++b) out[b] = 0.f;
    return;
  }
  const float gs = 0.5f * (g[q] + g[(size_t)j * NMAX + i]);
  const float* L = logits + q * PB;
  float m = -INFINITY;
  for (int b = 0; b < PB; ++b) m = fmaxf(m, L[b]);
  float z = 0.f, y = 0.f;
  for (int b = 0; b < PB; ++b) {
    const float e = __expf(L[b] - m);
    z += e; y += e * centres[b];
  }
  const float inv = 1.f / z;
  y *= inv;
  for (int b = 0; b < PB; ++b) {
    const float p = __expf(L[b] - m) * inv;
    out[b] = p * (centres[b] - y) * gs;
  }
}

__global__ void expresCaBwdK(const float* __restrict__ logits,
                             const float* __restrict__ g,
                             const int32_t* __restrict__ np,
                             float* __restrict__ dlogits, int NMAX) {
  const size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (size_t)NMAX * 37) return;
  const int i = (int)(q / 37), k = (int)(q % 37);
  if (i >= *np || k != CA) { dlogits[q] = 0.f; return; }
  const float s = 1.f / (1.f + __expf(-logits[q]));
  dlogits[q] = s * (1.f - s) * g[i];
}

// ---- ptm / iptm through the max --------------------------------------
// CALL ORDER IS PART OF THE CONTRACT.  paeBwdK WRITES dlogits; this one
// ACCUMULATES into it.  pae's backward must therefore run FIRST, and ptm's
// and iptm's after, or the atomicAdds are overwritten and those two
// gradients vanish -- leaving a loss on ipTM that trains on nothing while
// every number it prints stays sensible.  The three share one buffer
// because they share one set of logits.
// ptm = max_i (per_alignment[i] * seq_mask[i]).  A max routes its whole
// cotangent to ONE row, so the gradient on the pae logits is confined to
// that row's pairs.  The normalisation depends only on the masks, not on
// the logits, so it rides through as a constant.
__global__ void tmBwdK(const float* __restrict__ logits,   // [NMAX,NMAX,PB]
                       const float* __restrict__ tm_bin,   // [PB]
                       const float* __restrict__ per,      // [NMAX]
                       const float* __restrict__ smask,
                       const int32_t* __restrict__ asym,
                       // A DEVICE POINTER, not a float.  ptm and iptm are
                       // scalars, and a scalar OUTPUT's cotangent still
                       // lives on the card -- reading it on the host means
                       // a synchronise per step, and passing a literal 0
                       // (which this did) means a loss on pTM or ipTM
                       // trains on nothing while printing fine numbers.
                       int use_asym, const float* __restrict__ gp,
                       const int32_t* __restrict__ np,
                       float* __restrict__ dlogits,        // [NMAX,NMAX,PB] +=
                       int NMAX, int PB) {
  const int n = *np;
  const float g = gp ? *gp : 0.f;
  // which row won the max?  one warp finds it, every block agrees
  int best = 0; float bv = -INFINITY;
  for (int i = 0; i < n; ++i) {
    const float v = per[i] * smask[i];
    if (v > bv) { bv = v; best = i; }
  }
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;

  const float mask = use_asym ? (asym[best] != asym[j] ? 1.f : 0.f) : 1.f;
  if (mask == 0.f) return;
  // normed[best, j] = w_j / sum_k w_k, w = mask * smask[best] * smask[k]
  float den = 0.f;
  for (int k = 0; k < n; ++k) {
    const float mk = use_asym ? (asym[best] != asym[k] ? 1.f : 0.f) : 1.f;
    den += mk * smask[best] * smask[k];
  }
  const float w = mask * smask[best] * smask[j];
  const float dtm = g * smask[best] * mask * w / (1e-8f + den);

  const size_t q = ((size_t)best * NMAX + j);
  const float* L = logits + q * PB;
  float m = -INFINITY;
  for (int b = 0; b < PB; ++b) m = fmaxf(m, L[b]);
  float z = 0.f, y = 0.f;
  for (int b = 0; b < PB; ++b) {
    const float e = __expf(L[b] - m);
    z += e; y += e * tm_bin[b];
  }
  const float inv = 1.f / z;
  y *= inv;
  for (int b = 0; b < PB; ++b) {
    const float pr = __expf(L[b] - m) * inv;
    atomicAdd(&dlogits[q * PB + b], pr * (tm_bin[b] - y) * dtm);
  }
}

// iptm_per_residue IS AN OUTPUT IN ITS OWN RIGHT, not just the vector the
// max is taken over -- boundary.py lists it as differentiable, and bc2
// already uses it to weight mutations.  tmBwdK routes the whole cotangent
// to the one row the max picked; this routes a cotangent on EVERY row.
// One block a row.
__global__ void perAlignBwdK(const float* __restrict__ logits,
                             const float* __restrict__ tm_bin,
                             const float* __restrict__ dper,   // [NMAX]
                             const float* __restrict__ smask,
                             const int32_t* __restrict__ asym,
                             int use_asym,
                             const int32_t* __restrict__ np,
                             float* __restrict__ dlogits,      // +=
                             int NMAX, int PB) {
  const int n = *np;
  const int i = blockIdx.x;
  if (i >= n) return;
  const float gi = dper[i];
  if (gi == 0.f) return;
  float den = 0.f;
  for (int k = 0; k < n; ++k) {
    const float mk = use_asym ? (asym[i] != asym[k] ? 1.f : 0.f) : 1.f;
    den += mk * smask[i] * smask[k];
  }
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const float mask = use_asym ? (asym[i] != asym[j] ? 1.f : 0.f) : 1.f;
    if (mask == 0.f) continue;
    const float w = mask * smask[i] * smask[j];
    const float dtm = gi * mask * w / (1e-8f + den);
    const size_t q = ((size_t)i * NMAX + j);
    const float* L = logits + q * PB;
    float m = -INFINITY;
    for (int b = 0; b < PB; ++b) m = fmaxf(m, L[b]);
    float z = 0.f, y = 0.f;
    for (int b = 0; b < PB; ++b) {
      const float e = __expf(L[b] - m);
      z += e; y += e * tm_bin[b];
    }
    const float inv = 1.f / z;
    y *= inv;
    for (int b = 0; b < PB; ++b) {
      const float pr = __expf(L[b] - m) * inv;
      atomicAdd(&dlogits[q * PB + b], pr * (tm_bin[b] - y) * dtm);
    }
  }
}

}  // namespace confidence
