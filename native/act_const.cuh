// act_const: the sum of SingleTemplateEmbedding's EIGHT CONSTANT projections.
//
// Eight of construct_input's nine features are built from the template alone
// -- its aatype, atom positions and atom mask -- so their summed projection
// cannot change while a trajectory runs.  bc2 harvests them once per target
// and hands them to the step; a self-contained binary has to compute them.
//
// RESTATING THIS BINDS THE WRONG WEIGHTS IF IT IS CARELESS.  bc2's own
// docstring warns that the harvest runs inside the model's module precisely
// so the eight Linears keep their haiku scope -- so every weight here is
// loaded BY NAME (template_pair_embedding_0..7) and the result is gated
// against that harvest, which bc2 records as max|d| 0.000e+00 against the
// shipped path.
//
//   i  feature                                   shape      dims
//   0  template_dgram, 39 bins, pseudo-beta       39x64       1
//   1  pseudo_beta_mask_2d                         1x64       0
//   2  aatype one-hot (22), over ROWS             22x64       1
//   3  aatype one-hot (22), over COLS             22x64       1
//   4  unit_vector.x                               1x64       0
//   5  unit_vector.y                               1x64       0
//   6  unit_vector.z                               1x64       0
//   7  backbone_mask_2d                            1x64       0
#pragma once
#include "common.cuh"

namespace actconst {

// THE FEATURES ARE bf16 BEFORE THE LINEARS, not just the output.
// construct_input does `if gc.bfloat16: unit_vector = [x.astype(bfloat16)...]`
// and casts the dgram, the mask and the one-hots to `dtype`, so a faithful f32
// restatement is WRONG BY 1.5e-02 -- which is what this cost before the cast
// was found.  Rounding here is not an approximation of the model, it IS the
// model.
__device__ inline float bf16q(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

static constexpr int NBINS = 39, NAATYPE = 22, CH = 64;
static constexpr float MIN_BIN = 3.25f, MAX_BIN = 50.75f;

// Rot3Array::from_two_vectors, then Rigid3Array(rot, CA).  e0 lies on the
// positive x axis and e1 in the xy plane with positive y -- Gram-Schmidt,
// NOT a cross-product shortcut, because the two differ whenever N, CA and C
// are not exactly orthogonal, which is always.
__device__ inline void backboneFrame(const float* n, const float* ca,
                                     const float* c, float R[9]) {
  float e0[3] = {c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]};
  float e1[3] = {n[0] - ca[0], n[1] - ca[1], n[2] - ca[2]};
  float l = sqrtf(e0[0]*e0[0] + e0[1]*e0[1] + e0[2]*e0[2]) + 1e-20f;
  for (int k = 0; k < 3; ++k) e0[k] /= l;
  const float d = e1[0]*e0[0] + e1[1]*e0[1] + e1[2]*e0[2];
  for (int k = 0; k < 3; ++k) e1[k] -= d * e0[k];
  l = sqrtf(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]) + 1e-20f;
  for (int k = 0; k < 3; ++k) e1[k] /= l;
  const float e2[3] = {e0[1]*e1[2] - e0[2]*e1[1],
                       e0[2]*e1[0] - e0[0]*e1[2],
                       e0[0]*e1[1] - e0[1]*e1[0]};
  // columns are e0, e1, e2
  R[0]=e0[0]; R[1]=e1[0]; R[2]=e2[0];
  R[3]=e0[1]; R[4]=e1[1]; R[5]=e2[1];
  R[6]=e0[2]; R[7]=e1[2]; R[8]=e2[2];
}

// One (i, j) pair, all 64 channels.  Everything is accumulated in one pass
// so no [N][N][39] distogram is ever materialised -- at N=768 that alone
// would be 92 MB, for a quantity used once.
__global__ void actConstK(const float* __restrict__ pb,     // [N,3] pseudo-beta
                          const float* __restrict__ pbm,    // [N]   its mask
                          const float* __restrict__ atoms,  // [N,37,3]
                          const float* __restrict__ amask,  // [N,37]
                          const int32_t* __restrict__ aatype,
                          const int32_t* __restrict__ asym,
                          int N, int nmax,
                          const float* __restrict__ W0, const float* __restrict__ W1,
                          const float* __restrict__ W2, const float* __restrict__ W3,
                          const float* __restrict__ W4, const float* __restrict__ W5,
                          const float* __restrict__ W6, const float* __restrict__ W7,
                          const float* __restrict__ B,     // the eight biases, summed
                          float* __restrict__ out) {
  const int i = blockIdx.x, j = blockIdx.y;
  if (i >= N || j >= N) return;
  const int c = threadIdx.x;
  if (c >= CH) return;

  const float mc = (asym[i] == asym[j]) ? 1.f : 0.f;   // multichain_mask_2d
  const float pbm2 = pbm[i] * pbm[j] * mc;

  float acc = B[c];

  // 0: the template distogram, masked.  dgram_from_positions' bins are
  // linspace(min,max,39) SQUARED, with the last upper break at 1e8.
  if (pbm2 > 0.f) {
    float d2 = 0.f;
    for (int k = 0; k < 3; ++k) {
      const float t = pb[3*i+k] - pb[3*j+k];
      d2 += t * t;
    }
    const float step = (MAX_BIN - MIN_BIN) / (float)(NBINS - 1);
    for (int b = 0; b < NBINS; ++b) {
      const float lo = MIN_BIN + step * (float)b, lo2 = lo * lo;
      const float hi = lo + step, hi2 = (b == NBINS - 1) ? 1e8f : hi * hi;
      if (d2 > lo2 && d2 < hi2) { acc += W0[b * CH + c] * bf16q(pbm2); break; }
    }
  }

  // 1: pseudo_beta_mask_2d
  acc += W1[c] * bf16q(pbm2);

  // 2 and 3: the aatype one-hots, over rows and over columns
  const int ai = aatype[i] < NAATYPE ? aatype[i] : NAATYPE - 1;
  const int aj = aatype[j] < NAATYPE ? aatype[j] : NAATYPE - 1;
  acc += W2[aj * CH + c];     // aatype[None, :, :] varies along the COLUMN
  acc += W3[ai * CH + c];     // aatype[:, None, :] varies along the ROW

  // 4-7: the unit vector in i's local frame, and the backbone mask
  const float bmi = amask[i*37+0] * amask[i*37+1] * amask[i*37+2];
  const float bmj = amask[j*37+0] * amask[j*37+1] * amask[j*37+2];
  // bf16 BEFORE the outer product and the sqrt, as the cast order has it
  const float bm2 = bf16q(sqrtf(bf16q(bf16q(bmi) * bf16q(bmj)))) * mc;
  if (bm2 > 0.f) {
    float R[9];
    backboneFrame(&atoms[(size_t)i*37*3 + 0*3], &atoms[(size_t)i*37*3 + 1*3],
                  &atoms[(size_t)i*37*3 + 2*3], R);
    const float dx = atoms[(size_t)j*37*3 + 1*3 + 0] - atoms[(size_t)i*37*3 + 1*3 + 0];
    const float dy = atoms[(size_t)j*37*3 + 1*3 + 1] - atoms[(size_t)i*37*3 + 1*3 + 1];
    const float dz = atoms[(size_t)j*37*3 + 1*3 + 2] - atoms[(size_t)i*37*3 + 1*3 + 2];
    // R^T applied to (t_j - t_i): the inverse rotation
    float v[3] = {R[0]*dx + R[3]*dy + R[6]*dz,
                  R[1]*dx + R[4]*dy + R[7]*dz,
                  R[2]*dx + R[5]*dy + R[8]*dz};
    const float l = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) + 1e-20f;
    for (int k = 0; k < 3; ++k) v[k] /= l;
    const float bm2q = bf16q(bm2);
    acc += W4[c] * bf16q(bf16q(v[0]) * bm2q);
    acc += W5[c] * bf16q(bf16q(v[1]) * bm2q);
    acc += W6[c] * bf16q(bf16q(v[2]) * bm2q);
  }
  acc += W7[c] * bf16q(bm2);

  out[((size_t)i * nmax + j) * CH + c] = acc;
}

}  // namespace actconst
