// AF2's atom builder: torsion angles plus a backbone frame -> 14 atom
// positions, forward and backward.
//
//   default_frames[8]  = table[aatype]                    (constant)
//   rot_x[g]           = rotation about x by torsion g-1  (identity for g=0)
//   af[g]              = (R_default[g] @ rot_x[g], t_default[g])
//   chi2..chi4 CHAIN:  fb[5] = af[4] @ af[5], fb[6] = fb[5] @ af[6], ...
//   global[g]          = backbone @ fb[g]
//   atom[a]            = global[group[a]] . lit_pos[a]   * exists[a]
//
// THREE THINGS THIS SHAPE IS HIDING:
//
// 1. compose_rotation KEEPS THE DEFAULT FRAME'S TRANSLATION.  af[g] is
//    (R_d R_x, t_d), not a full rigid composition -- the torsion rotates the
//    group in place.  So t_d is a pure constant and takes no gradient, while
//    R_d does appear in the angle gradient as R_d^T.
//
// 2. THE CHAIN REPLACES SLOTS 5, 6, 7 BUT NOT 4.  chi1 IS af[4]; chi2 is
//    af[4] @ af[5], chi3 chains onto chi2, chi4 onto chi3.  Writing the chain
//    as "every group composes onto the previous" would be right for 5..7 and
//    wrong for 4, which is the sort of error that moves four atoms.
//
// 3. aatype IS AN INDEX, NOT A FEATURE.  Every table here is gathered by it
//    and nothing differentiates it.  The gradient goes to the angles and to
//    the backbone frame, and to nothing else.
//
// EVERYTHING IS PER RESIDUE: no atomics, no cross-residue reduction.  One
// thread owns a residue's whole frame tree in both directions.
#pragma once
#include "common.cuh"

namespace atom14 {

static const int NG = 8;     // rigid groups
static const int NA = 14;    // atom14
static const int NTOR = 7;

// --- small rigid helpers.  A frame is 9 floats of rotation (row-major) then
// 3 of translation, which is the layout the buffers below use.
__device__ __forceinline__ void rotMul(const float* A, const float* B, float* C) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j)
      C[3 * i + j] = A[3 * i] * B[j] + A[3 * i + 1] * B[3 + j] +
                     A[3 * i + 2] * B[6 + j];
}
__device__ __forceinline__ void rotApply(const float* R, const float* v, float* o) {
#pragma unroll
  for (int i = 0; i < 3; ++i)
    o[i] = R[3 * i] * v[0] + R[3 * i + 1] * v[1] + R[3 * i + 2] * v[2];
}
// o = R^T v
__device__ __forceinline__ void rotApplyT(const float* R, const float* v, float* o) {
#pragma unroll
  for (int j = 0; j < 3; ++j)
    o[j] = R[j] * v[0] + R[3 + j] * v[1] + R[6 + j] * v[2];
}
// C = A @ B, both rigid: R = Ra Rb, t = Ra tb + ta
__device__ __forceinline__ void rigidMul(const float* A, const float* B, float* C) {
  rotMul(A, B, C);
  rotApply(A, B + 9, C + 9);
#pragma unroll
  for (int i = 0; i < 3; ++i) C[9 + i] += A[9 + i];
}

__global__ void atomFwdK(const float* __restrict__ angles,
                         const float* __restrict__ rot,
                         const float* __restrict__ trans,
                         const float* __restrict__ aatype,
                         const float* __restrict__ defFrame,   // [21][8][4][4]
                         const float* __restrict__ a14group,   // [21][14]
                         const float* __restrict__ a14lit,     // [21][14][3]
                         const float* __restrict__ a14mask,    // [21][14]
                         float* __restrict__ fb,               // [N][8][12]
                         float* __restrict__ fg,               // [N][8][12]
                         float* __restrict__ out,              // [N][14][3]
                         int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int a = (int)aatype[n];
  float af[NG][12];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    // the default frame is a 4x4; its top-left 3x3 is the rotation and its
    // first three entries of the last COLUMN are the translation
    const float* M = defFrame + (((size_t)a * NG + g) * 16);
    float Rd[9], td[3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
      for (int j = 0; j < 3; ++j) Rd[3 * i + j] = M[4 * i + j];
      td[i] = M[4 * i + 3];
    }
    // the torsion rotation, about x.  Group 0 is the backbone and gets the
    // identity -- which is what prepending (sin, cos) = (0, 1) means.
    float s = 0.f, c = 1.f;
    if (g > 0) {
      const float* A = angles + ((size_t)n * NTOR + (g - 1)) * 2;
      s = A[0]; c = A[1];
    }
    float Rx[9] = {1.f, 0.f, 0.f, 0.f, c, -s, 0.f, s, c};
    rotMul(Rd, Rx, af[g]);
    // COMPOSE_ROTATION KEEPS THE DEFAULT TRANSLATION, see the note above
    af[g][9] = td[0]; af[g][10] = td[1]; af[g][11] = td[2];
  }
  // the chi chain: 4 stays, 5..7 compose onto the one before
  float fbl[NG][12];
#pragma unroll
  for (int g = 0; g < 5; ++g)
#pragma unroll
    for (int k = 0; k < 12; ++k) fbl[g][k] = af[g][k];
  rigidMul(fbl[4], af[5], fbl[5]);
  rigidMul(fbl[5], af[6], fbl[6]);
  rigidMul(fbl[6], af[7], fbl[7]);

  float bb[12];
#pragma unroll
  for (int k = 0; k < 9; ++k) bb[k] = rot[(size_t)n * 9 + k];
#pragma unroll
  for (int k = 0; k < 3; ++k) bb[9 + k] = trans[(size_t)n * 3 + k];

#pragma unroll
  for (int g = 0; g < NG; ++g) {
    float* FB = fb + ((size_t)n * NG + g) * 12;
    float* FG = fg + ((size_t)n * NG + g) * 12;
#pragma unroll
    for (int k = 0; k < 12; ++k) FB[k] = fbl[g][k];
    float gl[12];
    rigidMul(bb, fbl[g], gl);
#pragma unroll
    for (int k = 0; k < 12; ++k) FG[k] = gl[k];
  }

  for (int at = 0; at < NA; ++at) {
    int grp = (int)a14group[(size_t)a * NA + at];
    const float* G = fg + ((size_t)n * NG + grp) * 12;
    const float* L = a14lit + (((size_t)a * NA + at) * 3);
    float p[3];
    rotApply(G, L, p);
    float msk = a14mask[(size_t)a * NA + at];
    float* O = out + (((size_t)n * NA + at) * 3);
#pragma unroll
    for (int i = 0; i < 3; ++i) O[i] = (p[i] + G[9 + i]) * msk;
  }
}

__global__ void atomBwdK(const float* __restrict__ dout,
                         const float* __restrict__ angles,
                         const float* __restrict__ rot,
                         const float* __restrict__ aatype,
                         const float* __restrict__ defFrame,
                         const float* __restrict__ a14group,
                         const float* __restrict__ a14lit,
                         const float* __restrict__ a14mask,
                         const float* __restrict__ fb,
                         float* __restrict__ dangles,
                         float* __restrict__ drot,
                         float* __restrict__ dtrans, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int a = (int)aatype[n];
  float bb[9];
#pragma unroll
  for (int k = 0; k < 9; ++k) bb[k] = rot[(size_t)n * 9 + k];

  // --- the atoms, into their groups' global frames
  float dfg[NG][12] = {};
  for (int at = 0; at < NA; ++at) {
    int grp = (int)a14group[(size_t)a * NA + at];
    const float* L = a14lit + (((size_t)a * NA + at) * 3);
    float msk = a14mask[(size_t)a * NA + at];
    const float* D = dout + (((size_t)n * NA + at) * 3);
    float d[3] = {D[0] * msk, D[1] * msk, D[2] * msk};
#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
      for (int j = 0; j < 3; ++j) dfg[grp][3 * i + j] += d[i] * L[j];
      dfg[grp][9 + i] += d[i];
    }
  }

  // --- global[g] = backbone @ fb[g]
  float dbbR[9] = {}, dbbT[3] = {};
  float dfb[NG][12] = {};
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const float* F = fb + ((size_t)n * NG + g) * 12;
    // dR_b += dR_g F^T ; dR_f += R_b^T dR_g
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) acc += dfg[g][3 * i + j] * F[3 * k + j];
        dbbR[3 * i + k] += acc + dfg[g][9 + i] * F[9 + k];   // + the carry
      }
#pragma unroll
    for (int k = 0; k < 3; ++k)
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 3; ++i) acc += dfg[g][3 * i + j] * bb[3 * i + k];
        dfb[g][3 * k + j] = acc;
      }
    float dt[3];
    rotApplyT(bb, dfg[g] + 9, dt);
#pragma unroll
    for (int k = 0; k < 3; ++k) dfb[g][9 + k] = dt[k];
#pragma unroll
    for (int i = 0; i < 3; ++i) dbbT[i] += dfg[g][9 + i];
  }

  // --- rebuild af[] (cheap) so the chain can be walked back
  float af[NG][12];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const float* M = defFrame + (((size_t)a * NG + g) * 16);
    float Rd[9];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
      for (int j = 0; j < 3; ++j) Rd[3 * i + j] = M[4 * i + j];
    float s = 0.f, c = 1.f;
    if (g > 0) {
      const float* A = angles + ((size_t)n * NTOR + (g - 1)) * 2;
      s = A[0]; c = A[1];
    }
    float Rx[9] = {1.f, 0.f, 0.f, 0.f, c, -s, 0.f, s, c};
    rotMul(Rd, Rx, af[g]);
#pragma unroll
    for (int i = 0; i < 3; ++i) af[g][9 + i] = M[4 * i + 3];
  }
  float chi[3][12];     // chi2, chi3, chi4 as the forward built them
  rigidMul(af[4], af[5], chi[0]);
  rigidMul(chi[0], af[6], chi[1]);
  rigidMul(chi[1], af[7], chi[2]);

  // --- unchain 7, then 6, then 5.  daf starts from dfb and collects.
  float daf[NG][12] = {};
#pragma unroll
  for (int g = 0; g < 5; ++g)
#pragma unroll
    for (int k = 0; k < 12; ++k) daf[g][k] = dfb[g][k];
  // C = A @ B with dC known: dA += dR_C R_B^T + dt_C t_B^T, dt_A += dt_C,
  //                           dR_B += R_A^T dR_C, dt_B += R_A^T dt_C
  float dcarry[12];
#pragma unroll
  for (int k = 0; k < 12; ++k) dcarry[k] = dfb[7][k];
  for (int step = 2; step >= 0; --step) {
    const float* A = step ? chi[step - 1] : af[4];
    const float* B = af[5 + step];
    float dA[12] = {}, dB[12] = {};
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) acc += dcarry[3 * i + j] * B[3 * k + j];
        dA[3 * i + k] = acc + dcarry[9 + i] * B[9 + k];
      }
#pragma unroll
    for (int i = 0; i < 3; ++i) dA[9 + i] = dcarry[9 + i];
#pragma unroll
    for (int k = 0; k < 3; ++k)
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 3; ++i) acc += dcarry[3 * i + j] * A[3 * i + k];
        dB[3 * k + j] = acc;
      }
    rotApplyT(A, dcarry + 9, dB + 9);
#pragma unroll
    for (int k = 0; k < 12; ++k) daf[5 + step][k] += dB[k];
    if (step) {
      // dA belongs to chi[step-1], which has its OWN incoming cotangent from
      // the frame list -- add it and carry on
#pragma unroll
      for (int k = 0; k < 12; ++k) dcarry[k] = dA[k] + dfb[5 + step - 1][k];
    } else {
#pragma unroll
      for (int k = 0; k < 12; ++k) daf[4][k] += dA[k];
    }
  }

  // --- af[g] = (R_default[g] @ rot_x[g], t_default[g]) -> the angles.
  // The translation is a constant, so only the rotation half is read here.
  float* DA = dangles + (size_t)n * NTOR * 2;
#pragma unroll
  for (int g = 1; g < NG; ++g) {
    const float* M = defFrame + (((size_t)a * NG + g) * 16);
    // dRx = Rd^T dRaf
    float dRx[9];
#pragma unroll
    for (int k = 0; k < 3; ++k)
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 3; ++i) acc += M[4 * i + k] * daf[g][3 * i + j];
        dRx[3 * k + j] = acc;
      }
    // Rx = [[1,0,0],[0,c,-s],[0,s,c]]
    DA[(g - 1) * 2 + 0] = -dRx[5] + dRx[7];     // d/ds
    DA[(g - 1) * 2 + 1] = dRx[4] + dRx[8];      // d/dc
  }
#pragma unroll
  for (int k = 0; k < 9; ++k) drot[(size_t)n * 9 + k] = dbbR[k];
#pragma unroll
  for (int k = 0; k < 3; ++k) dtrans[(size_t)n * 3 + k] = dbbT[k];
}

struct Atom14 {
  int N = 0;
  cudaStream_t st = nullptr;
  float *defFrame = nullptr, *a14group = nullptr, *a14lit = nullptr,
        *a14mask = nullptr;
  float *fb = nullptr, *fg = nullptr, *out = nullptr;

  void alloc(int N_, cudaStream_t s = nullptr) {
    N = N_; st = s;
    CU(cudaMalloc(&fb, (size_t)N * NG * 12 * 4));
    CU(cudaMalloc(&fg, (size_t)N * NG * 12 * 4));
    CU(cudaMalloc(&out, (size_t)N * NA * 3 * 4));
  }

  // the tables are AF2's own constants and travel in the dump; see the note
  // in oracle_ipa.py about why they are not re-derived here.
  void setStream(cudaStream_t s) {
    st = s;
  }

  void setTables(Oracle& o) {
    auto get = [&](const char* k, float** dst) {
      Tensor& t = o[k];
      CU(cudaMalloc(dst, t.n() * 4));
      CU(cudaMemcpy(*dst, t.host.data(), t.n() * 4, cudaMemcpyHostToDevice));
    };
    get("const/default_frame", &defFrame);
    get("const/atom14_to_group", &a14group);
    get("const/atom14_lit_pos", &a14lit);
    get("const/atom14_mask", &a14mask);
  }

  // the residue-constant tables are the same four arrays for every iteration
  void shareTables(const Atom14& o) {
    defFrame = o.defFrame; a14group = o.a14group;
    a14lit = o.a14lit; a14mask = o.a14mask;
  }

  void forward(const float* angles, const float* rot, const float* trans,
               const float* aatype) {
    atomFwdK<<<(N + 127) / 128, 128, 0, st>>>(
        angles, rot, trans, aatype, defFrame, a14group, a14lit, a14mask, fb,
        fg, out, N);
  }

  void backward(const float* dout, const float* angles, const float* rot,
                const float* aatype, float* dangles, float* drot,
                float* dtrans) {
    atomBwdK<<<(N + 127) / 128, 128, 0, st>>>(
        dout, angles, rot, aatype, defFrame, a14group, a14lit, a14mask, fb,
        dangles, drot, dtrans, N);
  }
};

}  // namespace atom14
