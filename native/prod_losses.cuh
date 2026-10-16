// THE PRODUCTION DESIGN LOSSES, as bindcraft/loss.py defines them.
//
// Deliberately NOT losses.cuh.  That file's i_con, plddt and pae were
// written to gate kernels and carry the right NAMES with different
// definitions -- interface_contacts in bc2 has hotspots, chain slices,
// contacts_per_residue and contact_residue_count, none of which that i_con
// knows about.  Mixing the two would be the "same name, different thing"
// trap this port is most exposed to, so production definitions live here and
// each one is gated against the registered python function it claims to be
// (loss_gate.py).
//
// EVERY LOSS HERE HAS A BACKWARD.  The design step consumes cotangents on
// its outputs; a forward-only loss cannot drive anything, so a loss without
// one is not "partially done", it is absent.
#pragma once
#include "common.cuh"

namespace prodloss {

// bindcraft/loss.py::_masked_mean's epsilon, which is load-bearing at the
// denominator and NOT a tolerance: a state with no design positions divides
// by it rather than by zero.
static constexpr float MASK_EPS = 1e-8f;

// plddt_loss(protein_states, predictions, prediction_state='binder_alone',
//            chain='binder')
//   = _masked_mean(1 - plddt[binder rows], DESIGN flags)
//   = sum_i m_i (1 - p_i) / (sum_i m_i + eps)
//
// `prediction_state` DEFAULTS TO binder_alone, not the complex -- three of
// the eight default losses do, and the gate caught binder_pae on exactly
// that.  So `p` is the binder-alone state's plddt over its own rows and `m`
// is the DESIGN flag per binder residue; the caller does the row selection.
__global__ void plddtLossK(const float* __restrict__ p,
                           const float* __restrict__ m,
                           int n, float* __restrict__ out) {
  __shared__ float sn[256], sd[256];
  float num = 0.f, den = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float mi = m[i];
    den += mi;                       // mask.sum(), zeros included
    if (mi > 0.f) num += (1.f - p[i]) * mi;   // jnp.where(mask > 0, ...)
  }
  sn[threadIdx.x] = num; sd[threadIdx.x] = den;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { sn[threadIdx.x] += sn[threadIdx.x + s];
                           sd[threadIdx.x] += sd[threadIdx.x + s]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = sn[0] / (sd[0] + MASK_EPS);   // the loss
    out[1] = sd[0];                        // the denominator, for the backward
  }
}

// d/d plddt_i = -dL * m_i / (sum(m) + eps) where m_i > 0, else 0.
// `den` is the forward's out[1] rather than a second reduction: the two must
// be the SAME number or the gradient is scaled by a slightly different
// denominator than the value was.
__global__ void plddtLossBwdK(const float* __restrict__ m, float den,
                              float dL, int n, float* __restrict__ dp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float mi = m[i];
  dp[i] = (mi > 0.f) ? -dL * mi / (den + MASK_EPS) : 0.f;
}



// ===================== THE DISTOGRAM PRIMITIVE ==========================
//
// Eight of the 37 losses are this layer plus a thin wrapper, including three
// of the eight production defaults (interface_contacts, binder_contacts,
// binder_helicity).  bindcraft/loss.py builds them from exactly three
// pieces, reproduced here:
//
//   distogram_bin_distances(B) = [0, linspace(2.3125, 21.6875, B-1)]
//   distogram_pair_loss(d, sel, binary)
//       binary : -log((sel * softmax(d)).sum(-1) + 1e-8)
//       else   : -(softmax(d - 1e7*(1-sel)) * log_softmax(d)).sum(-1)
//   best_contact_mean(v, k, mask) -- the k SMALLEST masked values, averaged,
//       with the ranking stop_gradient'd
//
// and mean_selected_contact_loss is best_contact_mean twice: once along the
// pair axis with contacts_per_residue, then along residues with
// contact_residue_count.

// The first bin is 0 and the rest are linspace(2.3125, 21.6875, B-1), so a
// bin is "selected" when its distance is under the cutoff.  Bin 0 is
// distance 0 and is therefore selected by every positive cutoff -- which is
// the opposite of the recycle distogram's convention, where d=0 sets NO bin,
// and the two must not be confused.
// `above` picks the bins AT OR BEYOND the cutoff instead of below it, which
// is what non_contact_loss asks for: it scores the probability of being FAR
// from the off-target, where every other distogram loss scores the
// probability of being close.  Same bin centres, opposite comparison.
__global__ void binSelectK(int nbins, float cutoff, char* __restrict__ sel,
                           int above = 0) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nbins) return;
  const float d = (b == 0) ? 0.f
                : 2.3125f + (21.6875f - 2.3125f) * (float)(b - 1) / (float)(nbins - 2);
  sel[b] = (above ? (d >= cutoff) : (d < cutoff)) ? 1 : 0;
}

// x = -x, in place.  non_contact_loss takes best_contact_mean of the
// NEGATED pair loss -- the k LEAST-far partners of each off-target residue
// -- and negates the result again.
__global__ void negateK(float* __restrict__ x, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = -x[i];
}

// pair_loss[r][c] from distogram[r][c][B].  One warp-free block a pair row is
// simplest and the bin count is 64, so a thread does the whole reduction.
__global__ void pairLossK(const float* __restrict__ dg, const char* __restrict__ sel,
                          int rows, int cols, int B, int binary,
                          float* __restrict__ out) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)rows * cols) return;
  const float* d = dg + p * B;
  float mx = -1e30f;
  for (int b = 0; b < B; ++b) mx = fmaxf(mx, d[b]);
  float z = 0.f;
  for (int b = 0; b < B; ++b) z += __expf(d[b] - mx);
  const float logz = __logf(z) + mx;
  if (binary) {
    float s = 0.f;
    for (int b = 0; b < B; ++b) if (sel[b]) s += __expf(d[b] - logz);
    out[p] = -__logf(s + 1e-8f);
    return;
  }
  // softmax over the SELECTED bins only (the 1e7 penalty is exactly that),
  // weighting log_softmax over ALL bins.
  float mxs = -1e30f;
  for (int b = 0; b < B; ++b) if (sel[b]) mxs = fmaxf(mxs, d[b]);
  float zs = 0.f;
  for (int b = 0; b < B; ++b) if (sel[b]) zs += __expf(d[b] - mxs);
  float acc = 0.f;
  for (int b = 0; b < B; ++b)
    if (sel[b]) acc += (__expf(d[b] - mxs) / zs) * (d[b] - logz);
  out[p] = -acc;
}

// best_contact_mean along the last axis.  `k` is contacts_per_residue, with
// INT_MAX standing for float('inf') -- then every masked entry is selected
// and this is a plain masked mean.
//
// THE RANKING IS stop_gradient'd IN BC2, so the selection is a constant and
// the backward only needs to know WHICH entries were picked: `selmask` is
// that, and `cnt` the denominator, both written here and read by the
// backward rather than recomputed, so the two cannot disagree.
//
// Ties follow argsort's stable order -- smallest value, then smallest index.
__global__ void bestContactMeanK(const float* __restrict__ v,
                                 const char* __restrict__ mask,
                                 int rows, int cols, int k,
                                 float* __restrict__ out,
                                 char* __restrict__ selmask,
                                 float* __restrict__ cnt) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const float* vr = v + (size_t)r * cols;
  const char* mr = mask + (size_t)r * cols;
  char* sr = selmask + (size_t)r * cols;
  if (threadIdx.x != 0) return;              // one thread a row: k is 1 or 2
  for (int c = 0; c < cols; ++c) sr[c] = 0;
  int taken = 0;
  if (k >= cols) {
    for (int c = 0; c < cols; ++c) if (mr[c]) { sr[c] = 1; ++taken; }
  } else {
    for (int j = 0; j < k; ++j) {
      int best = -1;
      for (int c = 0; c < cols; ++c) {
        if (!mr[c] || sr[c]) continue;
        if (best < 0 || vr[c] < vr[best]) best = c;   // stable: < keeps the first
      }
      if (best < 0) break;
      sr[best] = 1; ++taken;
    }
  }
  float s = 0.f;
  for (int c = 0; c < cols; ++c) if (sr[c]) s += vr[c];
  cnt[r] = (float)taken;
  out[r] = s / ((float)taken + 1e-4f);        // best_contact_mean's eps
}

__global__ void bestContactMeanBwdK(const char* __restrict__ selmask,
                                    const float* __restrict__ cnt,
                                    const float* __restrict__ dout,
                                    int rows, int cols, float* __restrict__ dv) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)rows * cols) return;
  const int r = (int)(p / cols);
  dv[p] = selmask[p] ? dout[r] / (cnt[r] + 1e-4f) : 0.f;
}

// d pair_loss / d distogram, for both branches.
__global__ void pairLossBwdK(const float* __restrict__ dg, const char* __restrict__ sel,
                             const float* __restrict__ dout,
                             int rows, int cols, int B, int binary,
                             float* __restrict__ ddg) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)rows * cols) return;
  const float* d = dg + p * B;
  float* o = ddg + p * B;
  const float g = dout[p];
  float mx = -1e30f;
  for (int b = 0; b < B; ++b) mx = fmaxf(mx, d[b]);
  float z = 0.f;
  for (int b = 0; b < B; ++b) z += __expf(d[b] - mx);
  const float logz = __logf(z) + mx;
  if (binary) {
    // L = -log(S + eps), S = sum_sel softmax(d)
    float S = 0.f;
    for (int b = 0; b < B; ++b) if (sel[b]) S += __expf(d[b] - logz);
    const float c = -g / (S + 1e-8f);
    for (int b = 0; b < B; ++b) {
      const float sm = __expf(d[b] - logz);
      // d S / d d_b = sm * (sel_b - S)
      o[b] = c * sm * ((sel[b] ? 1.f : 0.f) - S);
    }
    return;
  }
  // L = -(q . logp), q = softmax over selected, logp = log_softmax over all.
  float mxs = -1e30f;
  for (int b = 0; b < B; ++b) if (sel[b]) mxs = fmaxf(mxs, d[b]);
  float zs = 0.f;
  for (int b = 0; b < B; ++b) if (sel[b]) zs += __expf(d[b] - mxs);
  float qdotlp = 0.f, sumq = 0.f;
  for (int b = 0; b < B; ++b) if (sel[b]) {
    const float q = __expf(d[b] - mxs) / zs;
    qdotlp += q * (d[b] - logz);
    sumq += q;
  }
  for (int b = 0; b < B; ++b) {
    const float sm = __expf(d[b] - logz);
    const float q = sel[b] ? __expf(d[b] - mxs) / zs : 0.f;
    // through log_softmax: -(q_b - sumq * softmax_b)
    float t = -(q - sumq * sm);
    // through q itself: -q_b * (logp_b - qdotlp)
    if (sel[b]) t += -q * ((d[b] - logz) - qdotlp);
    o[b] = g * t;
  }
}


// ---- P(close): the softmax mass in the selected bins, NOT its log ------
//
// non_helical is binder_helicity's mask and cutoff with a different
// aggregation: `(softmax(d) * (bins < cutoff)).sum(-1)` where helicity takes
// -log of the same sum.  Minimising the probability pushes the i,i+3 pairs
// APART; minimising its negative log pulls them together.  Same mask, same
// cutoff, opposite structure -- which is why this is a separate kernel and
// not a flag on the one above: a sign error here is a campaign that builds
// the helix it was asked to avoid.
__global__ void probCloseK(const float* __restrict__ dg,
                           const char* __restrict__ sel,
                           int rows, int cols, int B,
                           float* __restrict__ out) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)rows * cols) return;
  const float* d = dg + p * B;
  float mx = -1e30f;
  for (int b = 0; b < B; ++b) mx = fmaxf(mx, d[b]);
  float z = 0.f, s = 0.f;
  for (int b = 0; b < B; ++b) {
    const float e = __expf(d[b] - mx);
    z += e;
    if (sel[b]) s += e;
  }
  out[p] = s / z;
}

// dS/dd_b = softmax_b * (sel_b - S)
__global__ void probCloseBwdK(const float* __restrict__ dg,
                              const char* __restrict__ sel,
                              const float* __restrict__ dout,
                              int rows, int cols, int B,
                              float* __restrict__ ddg) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)rows * cols) return;
  const float* d = dg + p * B;
  float* o = ddg + p * B;
  const float g = dout[p];
  float mx = -1e30f;
  for (int b = 0; b < B; ++b) mx = fmaxf(mx, d[b]);
  float z = 0.f, s = 0.f;
  for (int b = 0; b < B; ++b) {
    const float e = __expf(d[b] - mx);
    z += e;
    if (sel[b]) s += e;
  }
  const float S = s / z;
  for (int b = 0; b < B; ++b) {
    const float sm = __expf(d[b] - mx) / z;
    o[b] = g * sm * ((sel[b] ? 1.f : 0.f) - S);
  }
}

// _masked_mean over a [rows][cols] array, and its backward.  Same definition
// as the plddt one above: sum(where(m>0, v*m, 0)) / (sum(m) + 1e-8).
__global__ void maskedMean2dK(const float* __restrict__ v, const char* __restrict__ m,
                              size_t n, float* __restrict__ out) {
  __shared__ float sn[256], sd[256];
  float num = 0.f, den = 0.f;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float mi = m[i] ? 1.f : 0.f;
    den += mi;
    if (mi > 0.f) num += v[i] * mi;
  }
  sn[threadIdx.x] = num; sd[threadIdx.x] = den;
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (threadIdx.x < s2) { sn[threadIdx.x] += sn[threadIdx.x + s2];
                            sd[threadIdx.x] += sd[threadIdx.x + s2]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { out[0] = sn[0] / (sd[0] + MASK_EPS); out[1] = sd[0]; }
}

__global__ void maskedMean2dBwdK(const char* __restrict__ m, float den, float dL,
                                 size_t n, float* __restrict__ dv) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dv[i] = m[i] ? dL / (den + MASK_EPS) : 0.f;
}

// ===================== THE PAE PRIMITIVE ================================
//
// chain_pair_pae_loss = _masked_mean(pae / 31.0, w_a[:,None] * w_r[None,:]).
// binder_pae (no chain, no domains) and interface_pae are both this, with
// the whole complex's weights or the binder-vs-target block respectively.
//
// THE MASK IS A FLOAT WEIGHT PRODUCT, not a boolean -- complex_residue_weights
// is real-valued -- so this cannot reuse the char-mask mean above: `v * m`
// is a weighting and `m.sum()` is a sum of weights.
__global__ void weightedMaskedMeanK(const float* __restrict__ v,
                                    const float* __restrict__ w,
                                    float scale, size_t n,
                                    float* __restrict__ out) {
  __shared__ float sn[256], sd[256];
  float num = 0.f, den = 0.f;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    const float wi = w[i];
    den += wi;
    if (wi > 0.f) num += (v[i] * scale) * wi;
  }
  sn[threadIdx.x] = num; sd[threadIdx.x] = den;
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (threadIdx.x < s2) { sn[threadIdx.x] += sn[threadIdx.x + s2];
                            sd[threadIdx.x] += sd[threadIdx.x + s2]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { out[0] = sn[0] / (sd[0] + MASK_EPS); out[1] = sd[0]; }
}

// d/d v_i = dL * scale * w_i / (sum(w) + eps) where w_i > 0.  `scale` is the
// 1/31 that chain_pair_pae_loss applies to the pae BEFORE the mean, so it
// rides through the gradient too.
__global__ void weightedMaskedMeanBwdK(const float* __restrict__ w, float den,
                                       float dL, float scale, size_t n,
                                       float* __restrict__ dv) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dv[i] = (w[i] > 0.f) ? dL * scale * w[i] / (den + MASK_EPS) : 0.f;
}

// ================== THE GEOMETRY PRIMITIVE (compactness) ================
//
// radius_of_gyration_loss, registered as `compactness`:
//   c      = (xyz * m).sum(0) / (m.sum() + 1e-8)        masked centroid
//   sq_i   = |xyz_i - c|^2
//   radius = sqrt(_masked_mean(sq, m) + 1e-8)
//   loss   = elu(radius - 2.38 * n_real^0.365) * w / count
//
// This one differentiates with respect to ATOM COORDINATES, not a metric
// head, so its cotangent lands on the step's `atoms` output.
//
// THE CENTROID DEPENDS ON THE COORDINATES, which is the term that is easy to
// drop: d sq_i / d xyz_j is 2(xyz_i - c)(delta_ij - m_j/M), and the second
// half contributes a correction through EVERY j.  It very nearly cancels --
// sum_i m_i (xyz_i - c) is zero but for the epsilon in M -- which is exactly
// why dropping it would pass a careless eyeball and fail a gradient check.
__global__ void radiusK(const float* __restrict__ xyz, const float* __restrict__ m,
                        int n, float gr, float* __restrict__ out) {
  __shared__ float sm[256], sx[256], sy[256], sz[256], sq[256];
  float am = 0.f, ax = 0.f, ay = 0.f, az = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float mi = m[i];
    am += mi; ax += xyz[3*i] * mi; ay += xyz[3*i+1] * mi; az += xyz[3*i+2] * mi;
  }
  sm[threadIdx.x] = am; sx[threadIdx.x] = ax; sy[threadIdx.x] = ay; sz[threadIdx.x] = az;
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (threadIdx.x < s2) {
      sm[threadIdx.x] += sm[threadIdx.x+s2]; sx[threadIdx.x] += sx[threadIdx.x+s2];
      sy[threadIdx.x] += sy[threadIdx.x+s2]; sz[threadIdx.x] += sz[threadIdx.x+s2];
    }
    __syncthreads();
  }
  const float M = sm[0] + MASK_EPS;
  const float cx = sx[0] / M, cy = sy[0] / M, cz = sz[0] / M;
  float acc = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float mi = m[i];
    if (mi <= 0.f) continue;
    const float dx = xyz[3*i] - cx, dy = xyz[3*i+1] - cy, dz = xyz[3*i+2] - cz;
    acc += (dx*dx + dy*dy + dz*dz) * mi;
  }
  sq[threadIdx.x] = acc;
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (threadIdx.x < s2) sq[threadIdx.x] += sq[threadIdx.x+s2];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float D = sm[0] + MASK_EPS;          // _masked_mean's denominator
    const float mean = sq[0] / D;
    const float radius = sqrtf(mean + 1e-8f);
    const float x = radius - gr;
    out[0] = (x > 0.f) ? x : (__expf(x) - 1.f);   // elu
    out[1] = radius; out[2] = D; out[3] = M;
    out[4] = cx; out[5] = cy; out[6] = cz;
    out[7] = (x > 0.f) ? 1.f : __expf(x);         // elu'
  }
}

__global__ void radiusBwdK(const float* __restrict__ xyz, const float* __restrict__ m,
                           const float* __restrict__ fw, int n, float dL,
                           float* __restrict__ sumw, float* __restrict__ dxyz) {
  // two passes in one launch is not worth it at this size: the caller runs
  // this once to accumulate S = sum_i (m_i/D)(xyz_i - c), then once to write.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float D = fw[2], M = fw[3];
  const float cx = fw[4], cy = fw[5], cz = fw[6];
  const float mi = m[i];
  const float dx = xyz[3*i] - cx, dy = xyz[3*i+1] - cy, dz = xyz[3*i+2] - cz;
  const float k = dL * fw[7] * 0.5f / fw[1];      // d loss/d mean
  // d mean / d xyz_j = (2 m_j / D)(xyz_j - c) - (2 m_j / M) * S
  const float a = (mi > 0.f) ? 2.f * mi / D : 0.f;
  const float b = 2.f * mi / M;
  atomicAdd(&dxyz[3*i],   k * (a * dx - b * sumw[0]));
  atomicAdd(&dxyz[3*i+1], k * (a * dy - b * sumw[1]));
  atomicAdd(&dxyz[3*i+2], k * (a * dz - b * sumw[2]));
}

__global__ void radiusSumK(const float* __restrict__ xyz, const float* __restrict__ m,
                           const float* __restrict__ fw, int n, float* __restrict__ sumw) {
  __shared__ float sx[256], sy[256], sz[256];
  const float D = fw[2], cx = fw[4], cy = fw[5], cz = fw[6];
  float ax = 0.f, ay = 0.f, az = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float w = (m[i] > 0.f) ? m[i] / D : 0.f;
    ax += w * (xyz[3*i] - cx); ay += w * (xyz[3*i+1] - cy); az += w * (xyz[3*i+2] - cz);
  }
  sx[threadIdx.x] = ax; sy[threadIdx.x] = ay; sz[threadIdx.x] = az;
  __syncthreads();
  for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
    if (threadIdx.x < s2) { sx[threadIdx.x]+=sx[threadIdx.x+s2];
      sy[threadIdx.x]+=sy[threadIdx.x+s2]; sz[threadIdx.x]+=sz[threadIdx.x+s2]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) { sumw[0]=sx[0]; sumw[1]=sy[0]; sumw[2]=sz[0]; }
}

// ACCUMULATE a second distogram cotangent.  Two losses read the same
// distogram, and the second writing over the first is a term silently
// dropped -- exactly the failure CPP_PORT.md says must never be silent.
__global__ void accDgramK(float* __restrict__ dst, const float* __restrict__ src,
                          size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

// ===================== distogram_cce ====================================
//
// The cross-entropy of the predicted distogram against the TEMPLATE's own
// distances, binned.  bc2:
//
//   reference_bins = (pdist(pseudo_beta) > bin_edges[1:]).sum(-1)
//   ce_ij          = -(one_hot(bin_ij) * log_softmax(distogram_ij)).sum(-1)
//   loss           = masked_mean(ce, ref_mask[:,None] * ref_mask[None,:])
//
// The bins come from the template and take no gradient; only log_softmax
// does, and its derivative is the familiar softmax - one_hot.
//
// `bins` is per PAIR and is -1 where the pair is masked out, which is both
// the mask and a guard: a bin index read from an unmasked-out pair would
// index the distogram with whatever the template left there.
__global__ void distogramCceK(const float* __restrict__ dg,
                              const short* __restrict__ bins,
                              size_t npair, int nbins,
                              float* __restrict__ ce) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npair) return;
  const int b = bins[i];
  if (b < 0) { ce[i] = 0.f; return; }
  const float* row = dg + i * (size_t)nbins;
  float m = row[0];
  for (int k = 1; k < nbins; ++k) m = fmaxf(m, row[k]);
  float sum = 0.f;
  for (int k = 0; k < nbins; ++k) sum += __expf(row[k] - m);
  // log_softmax(x)_b = x_b - m - log(sum exp(x - m))
  ce[i] = -(row[b] - m - __logf(sum));
}

// d(ce_i)/d(dg_ik) = softmax_ik - [k == b_i], scaled by the incoming dce_i.
__global__ void distogramCceBwdK(const float* __restrict__ dg,
                                 const short* __restrict__ bins,
                                 const float* __restrict__ dce,
                                 size_t npair, int nbins,
                                 float* __restrict__ ddg) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npair) return;
  float* out = ddg + i * (size_t)nbins;
  const int b = bins[i];
  if (b < 0 || dce[i] == 0.f) {
    for (int k = 0; k < nbins; ++k) out[k] = 0.f;
    return;
  }
  const float* row = dg + i * (size_t)nbins;
  float m = row[0];
  for (int k = 1; k < nbins; ++k) m = fmaxf(m, row[k]);
  float sum = 0.f;
  for (int k = 0; k < nbins; ++k) sum += __expf(row[k] - m);
  const float g = dce[i];
  for (int k = 0; k < nbins; ++k)
    out[k] = g * (__expf(row[k] - m) / sum - (k == b ? 1.f : 0.f));
}

}  // namespace prodloss
