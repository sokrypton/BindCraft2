// align_prediction_to_target_template: the last of the nine.
//
// A weighted Kabsch onto the target template's CA atoms, then the whole
// prediction moved by it.  The JAX runs the CA coordinates through
// stop_gradient, so NO GRADIENT PASSES THROUGH THE ROTATION -- which is
// what makes a 3x3 SVD affordable here: it never has to be differentiated.
//
// Verified as a formula first (kabsch_formula.py) against the JAX, which
// uses jnp.linalg.svd: max|d| 7.8e-03 on coordinates stored as float16,
// i.e. the f16 quantisation of ~5 A, and det(R) = +1 exactly.
//
// THE DETERMINANT CORRECTION IS NOT OPTIONAL.  Without the sign flip on the
// third column an SVD can hand back a REFLECTION, which superposes just as
// well by RMSD and mirrors the design.  Nothing downstream would complain.
#pragma once
#include "common.cuh"

namespace kabsch {

constexpr int ATOM = 37, CA = 1;
constexpr uint8_t F_DESIGN = 1u << 0, F_TEMPLATE = 1u << 1;

// ---- weighted centroids and the 3x3 covariance ------------------------
// One block.  N is small and this is a 15-value reduction; a grid-wide
// reduction would cost more in launches than it saves.
__global__ void covK(const float* __restrict__ pred,    // [NMAX, 37, 3]
                     const float* __restrict__ tmpl,    // [NMAX, 37, 3]
                     const float* __restrict__ tmask,   // [NMAX, 37]
                     const uint8_t* __restrict__ flags, // [NMAX]
                     const int32_t* __restrict__ np,
                     float* __restrict__ out,           // [15] pc,tc,cov
                     int NMAX) {
  const int n = *np;
  float pc[3] = {0, 0, 0}, tc[3] = {0, 0, 0}, ws = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    // TEMPLATE and not DESIGN: a designed residue must not drag the frame
    const bool t = (flags[i] & F_TEMPLATE) && !(flags[i] & F_DESIGN);
    const float w = (t && tmask[(size_t)i * ATOM + CA] > 0.f) ? 1.f : 0.f;
    ws += w;
    for (int c = 0; c < 3; ++c) {
      pc[c] += w * pred[((size_t)i * ATOM + CA) * 3 + c];
      tc[c] += w * tmpl[((size_t)i * ATOM + CA) * 3 + c];
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    ws += __shfl_xor_sync(0xffffffffu, ws, o);
    for (int c = 0; c < 3; ++c) {
      pc[c] += __shfl_xor_sync(0xffffffffu, pc[c], o);
      tc[c] += __shfl_xor_sync(0xffffffffu, tc[c], o);
    }
  }
  const float inv = 1.f / (ws + 1e-8f);
  for (int c = 0; c < 3; ++c) { pc[c] *= inv; tc[c] *= inv; }

  float cov[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const bool t = (flags[i] & F_TEMPLATE) && !(flags[i] & F_DESIGN);
    const float w = (t && tmask[(size_t)i * ATOM + CA] > 0.f) ? 1.f : 0.f;
    if (w == 0.f) continue;
    float p[3], q[3];
    for (int c = 0; c < 3; ++c) {
      p[c] = pred[((size_t)i * ATOM + CA) * 3 + c] - pc[c];
      q[c] = tmpl[((size_t)i * ATOM + CA) * 3 + c] - tc[c];
    }
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) cov[a * 3 + b] += w * p[a] * q[b];
  }
  for (int o = 16; o > 0; o >>= 1)
    for (int k = 0; k < 9; ++k)
      cov[k] += __shfl_xor_sync(0xffffffffu, cov[k], o);

  if (threadIdx.x == 0) {
    for (int c = 0; c < 3; ++c) { out[c] = pc[c]; out[3 + c] = tc[c]; }
    // no aligned residues at all: the identity, as the JAX does
    for (int k = 0; k < 9; ++k)
      out[6 + k] = (ws > 0.f) ? cov[k] : ((k % 4 == 0) ? 1.f : 0.f);
  }
}

// ---- 3x3 SVD by one-sided Jacobi, then the rotation -------------------
//
// __host__ __device__ so the HOST filters can align a binder with exactly
// this solver rather than a second copy of it.  The determinant correction
// and the descending sort below are the two places a transcription goes
// wrong silently -- the sort cost 39 A once -- and one of those is one too
// many to have twice.
__host__ __device__ inline void rotationFromCov(const float* cov,  // [9]
                                                float* R) {        // [9]
  float W[9], V[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int k = 0; k < 9; ++k) W[k] = cov[k];

  for (int sweep = 0; sweep < 12; ++sweep) {
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        float a = 0, b = 0, g = 0;
        for (int i = 0; i < 3; ++i) {
          a += W[i * 3 + p] * W[i * 3 + p];
          b += W[i * 3 + q] * W[i * 3 + q];
          g += W[i * 3 + p] * W[i * 3 + q];
        }
        if (fabsf(g) < 1e-12f) continue;
        const float zeta = (b - a) / (2.f * g);
        const float t = (zeta >= 0 ? 1.f : -1.f) /
                        (fabsf(zeta) + sqrtf(1.f + zeta * zeta));
        const float c = rsqrtf(1.f + t * t), s = c * t;
        for (int i = 0; i < 3; ++i) {
          const float wp = W[i * 3 + p], wq = W[i * 3 + q];
          W[i * 3 + p] = c * wp - s * wq;
          W[i * 3 + q] = s * wp + c * wq;
          const float vp = V[i * 3 + p], vq = V[i * 3 + q];
          V[i * 3 + p] = c * vp - s * vq;
          V[i * 3 + q] = s * vp + c * vq;
        }
      }
  }
  // U = W with its columns normalised, and S their norms
  float U[9], S[3];
  for (int j = 0; j < 3; ++j) {
    float nrm = 0;
    for (int i = 0; i < 3; ++i) nrm += W[i * 3 + j] * W[i * 3 + j];
    nrm = sqrtf(nrm);
    S[j] = nrm;
    const float inv = 1.f / (nrm + 1e-20f);
    for (int i = 0; i < 3; ++i) U[i * 3 + j] = W[i * 3 + j] * inv;
  }

  // SORT BY DESCENDING SINGULAR VALUE.  Jacobi produces them in whatever
  // order the sweeps happen to leave; numpy's svd returns them descending,
  // and the determinant correction below scales the THIRD column -- which
  // has to be the SMALLEST singular value, the least-determined axis.
  // Without this the sign flip lands on an arbitrary axis and mirrors the
  // structure: the gate caught it at 39 A, and no formula check in numpy
  // could have, because numpy sorts for you.
  for (int a = 0; a < 2; ++a)
    for (int b = a + 1; b < 3; ++b)
      if (S[b] > S[a]) {
        float t = S[a]; S[a] = S[b]; S[b] = t;
        for (int i = 0; i < 3; ++i) {
          t = U[i * 3 + a]; U[i * 3 + a] = U[i * 3 + b]; U[i * 3 + b] = t;
          t = V[i * 3 + a]; V[i * 3 + a] = V[i * 3 + b]; V[i * 3 + b] = t;
        }
      }
  // d = sign(det(V U^T)); then R = (V with column 2 scaled by d) U^T
  float M[9];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float acc = 0;
      for (int k = 0; k < 3; ++k) acc += V[i * 3 + k] * U[j * 3 + k];
      M[i * 3 + j] = acc;
    }
  const float det = M[0] * (M[4] * M[8] - M[5] * M[7])
                  - M[1] * (M[3] * M[8] - M[5] * M[6])
                  + M[2] * (M[3] * M[7] - M[4] * M[6]);
  const float d = det >= 0.f ? 1.f : -1.f;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float acc = 0;
      for (int k = 0; k < 3; ++k)
        acc += V[i * 3 + k] * (k == 2 ? d : 1.f) * U[j * 3 + k];
      R[i * 3 + j] = acc;
    }
}

__global__ void rotationK(const float* __restrict__ stats,  // [15]
                          float* __restrict__ R) {          // [9]
  if (threadIdx.x) return;
  rotationFromCov(stats + 6, R);
}

// ---- move every atom ---------------------------------------------------
__global__ void applyK(const float* __restrict__ pred,
                       const float* __restrict__ pmask,
                       const float* __restrict__ stats,
                       const float* __restrict__ R,
                       const int32_t* __restrict__ np,
                       float* __restrict__ out, int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * ATOM) return;
  const int i = (int)(p / ATOM);
  if (i >= *np || pmask[p] == 0.f) {
    for (int c = 0; c < 3; ++c) out[p * 3 + c] = 0.f;
    return;
  }
  float v[3];
  for (int c = 0; c < 3; ++c) v[c] = pred[p * 3 + c] - stats[c];
  for (int c = 0; c < 3; ++c) {
    float acc = stats[3 + c];
    for (int k = 0; k < 3; ++k) acc += v[k] * R[c * 3 + k];   // v @ R^T
    out[p * 3 + c] = acc;
  }
}

// ---- and back ----------------------------------------------------------
// NOT A NO-OP.  bc2's align_prediction_to_target_template stop_gradients the
// CA coordinates it derives the rotation from -- so R, the prediction centre
// and the template centre are constants -- but the positions it then
// transforms are NOT stop_gradient'd:
//
//   aligned = (predicted - prediction_center) @ R.T + template_center
//
// so d_predicted = d_aligned @ R, masked exactly as the forward masked.
// Treating the whole superposition as constant drops that rotation and
// hands the structure module a cotangent expressed in the wrong frame --
// which is invisible to any rotation-invariant loss and wrong for every
// other one.
__global__ void applyBwdK(const float* __restrict__ dout,
                          const float* __restrict__ pmask,
                          const float* __restrict__ R,
                          const int32_t* __restrict__ np,
                          float* __restrict__ dpred, int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * ATOM) return;
  const int i = (int)(p / ATOM);
  if (i >= *np || pmask[p] == 0.f) {
    for (int k = 0; k < 3; ++k) dpred[p * 3 + k] = 0.f;
    return;
  }
  for (int k = 0; k < 3; ++k) {
    float acc = 0.f;
    for (int c = 0; c < 3; ++c) acc += dout[p * 3 + c] * R[c * 3 + k];
    dpred[p * 3 + k] = acc;
  }
}

}  // namespace kabsch
