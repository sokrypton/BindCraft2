// AF2's invariant point attention in CUDA, forward and backward.
//
// This is the first piece of the STRUCTURE MODULE, and the structure module is
// where the design gradient lives: probing BindCraft2's loss showed ~95% of it
// entering through here rather than through the template path.  So unlike
// localfold's forward-only structure kernels, this owes a backward.
//
// THE SHAPE.  N residues, each with a 384-channel single representation and a
// RIGID FRAME -- a rotation and a translation.  Attention is computed between
// residues from three terms that are added before the softmax:
//
//   scalar   <q_scalar[q,h,:], k_scalar[k,h,:]>          H x 16 channels
//   point    -0.5 w[h] sum_p |q_point - k_point|^2       H x 4 points in 3D
//   pair     Linear(pair[q,k,:]) -> H                    one logit a head
//
// The point term is what makes it invariant: the points are projected in each
// residue's LOCAL frame and pushed to the global one, so a global rotation of
// the whole structure moves q and k together and the distance is unchanged.
//
// EVERYTHING HERE IS f32.  The trunk is bf16 and this is not: AF2 runs the
// structure module in float32, the point projection asks for
// Precision.HIGHEST specifically, and a frame is a rotation whose
// orthogonality bf16 does not preserve well enough to be worth the bandwidth
// on a module that is N-shaped rather than N^2-shaped in all but two of its
// tensors.
//
// THE LOCAL POINT LAYOUT IS SPLIT, NOT INTERLEAVED.  PointProjection emits
// 3*P channels a head and splits them as [x(P) | y(P) | z(P)] -- not as P
// consecutive xyz triples.  Getting this backwards gives a port that is
// plausible everywhere and wrong everywhere.
#pragma once
#include "common.cuh"

namespace ipa {

static const int C = 384;    // single channels
static const int CZ = 128;   // pair channels
static const int H = 12;     // heads
static const int QK = 16;    // num_scalar_qk
static const int PQK = 4;    // num_point_qk
static const int SV = 16;    // num_scalar_v
static const int PV = 8;     // num_point_v
// the output projection's input: scalars, the local point xyz, the point
// norms, and the pair averaged under the attention.
static const int OUTF = H * SV + 3 * H * PV + H * PV + H * CZ;   // 2112
static_assert(OUTF == 2112, "output_projection weights are [2112, 384]");

// Each point pair contributes Var[0.5|q|^2 - <q,k>] = 9/2, so the point term
// is scaled by 1/sqrt(P * 9/2); each scalar pair contributes Var[q k] = 1.
static const float POINT_W = 0.23570226039551584f;   // sqrt(1 / (4 * 4.5))
static const float SCALAR_W = 0.25f;                 // sqrt(1 / 16)
static const float LOGIT_W = 0.5773502691896258f;    // sqrt(1/3), three terms
static const float DIST_EPS = 1e-8f;

// the six projection widths, in the order they are concatenated
static const int W_QS = H * QK, W_KS = H * QK, W_VS = H * SV;
static const int W_QP = H * PQK * 3, W_KP = H * PQK * 3, W_VP = H * PV * 3;
static const int PROJ_TOT = W_QS + W_KS + W_VS + W_QP + W_KP + W_VP;  // 1152

__global__ void scatterProjK(const float* __restrict__ comb,
                             const float* __restrict__ bcat,
                             float* __restrict__ qs, float* __restrict__ ks,
                             float* __restrict__ vs, float* __restrict__ qpl,
                             float* __restrict__ kpl, float* __restrict__ vpl,
                             int N) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)N * PROJ_TOT) return;
  int c = (int)(i % PROJ_TOT);
  size_t n = i / PROJ_TOT;
  float v = comb[i] + bcat[c];
  if (c < W_QS) { qs[n * W_QS + c] = v; return; }
  c -= W_QS;
  if (c < W_KS) { ks[n * W_KS + c] = v; return; }
  c -= W_KS;
  if (c < W_VS) { vs[n * W_VS + c] = v; return; }
  c -= W_VS;
  if (c < W_QP) { qpl[n * W_QP + c] = v; return; }
  c -= W_QP;
  if (c < W_KP) { kpl[n * W_KP + c] = v; return; }
  c -= W_KP;
  vpl[n * W_VP + c] = v;
}

__device__ __forceinline__ float softplusf(float x) {
  // logaddexp(x, 0), written so neither branch overflows
  return x > 0.f ? x + log1pf(__expf(-x)) : log1pf(__expf(x));
}

// ---------------------------------------------------------------- forward

// local [N][H][3P] in split layout -> global [N][H][P][3]
template <int P>
__global__ void pointToGlobalK(const float* __restrict__ local,
                               const float* __restrict__ rot,
                               const float* __restrict__ trans,
                               float* __restrict__ global_, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tot = N * H * P;
  if (idx >= tot) return;
  int p = idx % P, h = (idx / P) % H, n = idx / (P * H);
  const float* L = local + ((size_t)n * H + h) * (3 * P);
  float x = L[p], y = L[P + p], z = L[2 * P + p];   // the SPLIT layout
  const float* R = rot + (size_t)n * 9;
  const float* T = trans + (size_t)n * 3;
  float* G = global_ + (size_t)idx * 3;
  G[0] = R[0] * x + R[1] * y + R[2] * z + T[0];
  G[1] = R[3] * x + R[4] * y + R[5] * z + T[1];
  G[2] = R[6] * x + R[7] * y + R[8] * z + T[2];
}

// logits[q][k][h], all three terms, the mask, and the 1/sqrt(3).
// One block a (q, k) pair; H threads.
// THE SAME LAUNCH SHAPE attnGradK HAD: one block per (q, k) with H = 12
// threads.  A block now owns one q and a tile of KPB keys, and this q's
// scalar and point queries -- re-read by every k otherwise -- are staged once.
template <int KPB>
__global__ void logitsK(const float* __restrict__ qs, const float* __restrict__ ks,
                        const float* __restrict__ qp, const float* __restrict__ kp,
                        const float* __restrict__ a2d,
                        const float* __restrict__ raw_pw,
                        const float* __restrict__ mask,
                        float* __restrict__ logits, int N) {
  extern __shared__ float sq[];               // [H][QK] then [H][PQK*3]
  float* sqp = sq + (size_t)H * QK;
  const int q = blockIdx.x, k0 = blockIdx.y * KPB;
  const int tid = threadIdx.x, nthr = KPB * H;
  for (int i = tid; i < H * QK; i += nthr) sq[i] = qs[(size_t)q * H * QK + i];
  for (int i = tid; i < H * PQK * 3; i += nthr)
    sqp[i] = qp[(size_t)q * H * PQK * 3 + i];
  __syncthreads();
  const int kl = tid / H, h = tid % H, k = k0 + kl;
  if (k >= N) return;
  const size_t qk = (size_t)q * N + k;
  float s = 0.f;
  const float* Q = sq + h * QK;
  const float* K = ks + ((size_t)k * H + h) * QK;
#pragma unroll
  for (int c = 0; c < QK; ++c) s += Q[c] * K[c];
  s *= SCALAR_W;
  // the point term
  float w = POINT_W * softplusf(raw_pw[h]);
  float pt = 0.f;
  const float* QP = sqp + h * PQK * 3;
  const float* KP = kp + ((size_t)k * H + h) * PQK * 3;
#pragma unroll
  for (int p = 0; p < PQK; ++p) {
    float dx = QP[p * 3 + 0] - KP[p * 3 + 0];
    float dy = QP[p * 3 + 1] - KP[p * 3 + 1];
    float dz = QP[p * 3 + 2] - KP[p * 3 + 2];
    pt += dx * dx + dy * dy + dz * dz;
  }
  s += -0.5f * w * pt;
  s += a2d[qk * H + h];
  s -= 1e5f * (1.f - mask[qk]);
  logits[qk * H + h] = s * LOGIT_W;
}

// softmax over the KEY axis, which is axis -2 of [q, k, h].  One warp a
// (q, h): the stride between keys is H, so the reads are strided either way
// and a warp walking them keeps the row's running state in registers.
__global__ void softmaxK(float* __restrict__ logits, float* __restrict__ sums,
                         int N) {
  int qh = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (qh >= N * H) return;
  int h = qh % H, q = qh / H;
  int lane = threadIdx.x % 32;
  float* L = logits + (size_t)q * N * H + h;
  float m = -1e30f;
  for (int k = lane; k < N; k += 32) m = fmaxf(m, L[(size_t)k * H]);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  float s = 0.f;
  for (int k = lane; k < N; k += 32) {
    float e = __expf(L[(size_t)k * H] - m);
    L[(size_t)k * H] = e;
    s += e;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  float inv = 1.f / s;
  for (int k = lane; k < N; k += 32) L[(size_t)k * H] *= inv;
  if (lane == 0) sums[qh] = s;   // unused by the forward; kept for the backward
}

// the three contractions over keys, plus the localisation and the norms.
// One block a (q, h); 128 threads.
__global__ void gatherK(const float* __restrict__ attn,
                        const float* __restrict__ vs,
                        const float* __restrict__ vp,
                        const float* __restrict__ pair,
                        const float* __restrict__ rot,
                        const float* __restrict__ trans,
                        float* __restrict__ rpg,
                        float* __restrict__ final_, int N) {
  int h = blockIdx.x % H, q = blockIdx.x / H;
  int tid = threadIdx.x, nt = blockDim.x;
  float* F = final_ + (size_t)q * OUTF;

  // result_scalar [H][SV] -> the first H*SV columns
  for (int c = tid; c < SV; c += nt) {
    float acc = 0.f;
    for (int k = 0; k < N; ++k)
      acc += attn[((size_t)q * N + k) * H + h] * vs[((size_t)k * H + h) * SV + c];
    F[h * SV + c] = acc;
  }
  // result_point_global [H][PV][3]
  for (int e = tid; e < PV * 3; e += nt) {
    float acc = 0.f;
    for (int k = 0; k < N; ++k)
      acc += attn[((size_t)q * N + k) * H + h] *
             vp[(((size_t)k * H + h) * PV) * 3 + e];
    rpg[(((size_t)q * H + h) * PV) * 3 + e] = acc;
  }
  // result_attention_over_2d [H][CZ] -- the pair, averaged under the attention
  for (int c = tid; c < CZ; c += nt) {
    float acc = 0.f;
    for (int k = 0; k < N; ++k)
      acc += attn[((size_t)q * N + k) * H + h] * pair[((size_t)q * N + k) * CZ + c];
    F[H * SV + 3 * H * PV + H * PV + h * CZ + c] = acc;
  }
  __syncthreads();
  // the local points and their norms.  THE CONCATENATION IS BY COMPONENT:
  // output_features gets x for every (h, p), then y for every (h, p), then z
  // -- because result_point_local is reshaped to [N, -1] per component before
  // the three are appended.
  if (tid < PV) {
    int p = tid;
    const float* R = rot + (size_t)q * 9;
    const float* T = trans + (size_t)q * 3;
    const float* G = rpg + (((size_t)q * H + h) * PV + p) * 3;
    float dx = G[0] - T[0], dy = G[1] - T[1], dz = G[2] - T[2];
    float lx = R[0] * dx + R[3] * dy + R[6] * dz;   // R^T, the inverse
    float ly = R[1] * dx + R[4] * dy + R[7] * dz;
    float lz = R[2] * dx + R[5] * dy + R[8] * dz;
    int base = H * SV;
    F[base + 0 * H * PV + h * PV + p] = lx;
    F[base + 1 * H * PV + h * PV + p] = ly;
    F[base + 2 * H * PV + h * PV + p] = lz;
    F[base + 3 * H * PV + h * PV + p] = sqrtf(lx * lx + ly * ly + lz * lz + DIST_EPS);
  }
}

// ---------------------------------------------------------------- backward
//
// Four activation gradients come out of this module: the single, the pair, and
// the frame's ROTATION and TRANSLATION.  The last two are what make it
// different from every module in the trunk -- the frame enters twice, once
// pushing the q/k/v points out to the global frame and once pulling the
// attention's result back into the local one, so dR and dt each collect from
// two places and are accumulated atomically rather than written.

// dfinal -> d(result_scalar), d(result_point_global), d(result_2d), and the
// frame's share of the INVERSE apply.  One thread a (q, h, p) for the point
// half; the scalar and 2d halves are pure slicing and are read where they are
// used, so nothing is copied for them.
__global__ void localBwdK(const float* __restrict__ dfinal,
                          const float* __restrict__ rpg,
                          const float* __restrict__ rot,
                          const float* __restrict__ trans,
                          float* __restrict__ drpg,
                          float* __restrict__ drot,
                          float* __restrict__ dtrans, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * H * PV) return;
  int p = idx % PV, h = (idx / PV) % H, q = idx / (PV * H);
  const float* F = dfinal + (size_t)q * OUTF;
  const int base = H * SV;
  const float* R = rot + (size_t)q * 9;
  const float* T = trans + (size_t)q * 3;
  const float* G = rpg + (((size_t)q * H + h) * PV + p) * 3;
  float dx = G[0] - T[0], dy = G[1] - T[1], dz = G[2] - T[2];
  float lx = R[0] * dx + R[3] * dy + R[6] * dz;
  float ly = R[1] * dx + R[4] * dy + R[7] * dz;
  float lz = R[2] * dx + R[5] * dy + R[8] * dz;
  float nrm = sqrtf(lx * lx + ly * ly + lz * lz + DIST_EPS);
  // the norm feeds a fifth output column, so its gradient lands back on the
  // same three components before any of them leaves the local frame
  float dn = F[base + 3 * H * PV + h * PV + p] / nrm;
  float dlx = F[base + 0 * H * PV + h * PV + p] + dn * lx;
  float dly = F[base + 1 * H * PV + h * PV + p] + dn * ly;
  float dlz = F[base + 2 * H * PV + h * PV + p] + dn * lz;
  // l = R^T d, so d <- R dl
  float dgx = R[0] * dlx + R[1] * dly + R[2] * dlz;
  float dgy = R[3] * dlx + R[4] * dly + R[5] * dlz;
  float dgz = R[6] * dlx + R[7] * dly + R[8] * dlz;
  float* DG = drpg + (((size_t)q * H + h) * PV + p) * 3;
  DG[0] = dgx; DG[1] = dgy; DG[2] = dgz;
  // d = g - t, so t takes the negative
  float* DT = dtrans + (size_t)q * 3;
  atomicAdd(DT + 0, -dgx); atomicAdd(DT + 1, -dgy); atomicAdd(DT + 2, -dgz);
  // lx = R[0] dx + R[3] dy + R[6] dz  ->  dR[3i + j] += dl_j * d_i
  float* DR = drot + (size_t)q * 9;
  const float dl[3] = {dlx, dly, dlz}, dd[3] = {dx, dy, dz};
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) atomicAdd(DR + 3 * i + j, dl[j] * dd[i]);
}

// the attention's own gradient, collected from all three things it weights.
//
// THIS KERNEL WAS 28% OF THE WHOLE STRUCTURE MODULE and the reason was its
// launch shape, not its arithmetic: one block per (q, k) with H = 12 threads.
// Twelve threads is 37.5% of a warp, it is N^2 = 16384 blocks at N = 128, and
// all twelve re-read the SAME 128 pair channels from global.
//
// Now a block owns one q and a TILE of KPB keys, so it is KPB * H threads and
// N * N/KPB blocks, and the two things every thread in it shares are staged
// once in shared memory: this q's d_r2d slice (H x CZ, the same for every k)
// and the tile's pair rows (KPB x CZ, each read by H threads instead of H
// times).
template <int KPB>
__global__ void attnGradK(const float* __restrict__ dfinal,
                          const float* __restrict__ drpg,
                          const float* __restrict__ vs,
                          const float* __restrict__ vpg,
                          const float* __restrict__ pair,
                          float* __restrict__ dattn, int N) {
  extern __shared__ float sm[];
  float* sD2 = sm;                       // [H][CZ]   depends on q only
  float* sPair = sD2 + (size_t)H * CZ;   // [KPB][CZ] depends on (q, k)
  const int q = blockIdx.x;
  const int k0 = blockIdx.y * KPB;
  const int tid = threadIdx.x, nthr = KPB * H;
  const float* F = dfinal + (size_t)q * OUTF;
  for (int i = tid; i < H * CZ; i += nthr) sD2[i] = F[H * SV + 4 * H * PV + i];
  for (int i = tid; i < KPB * CZ; i += nthr) {
    int kl = i / CZ, c = i % CZ, k = k0 + kl;
    sPair[i] = k < N ? pair[((size_t)q * N + k) * CZ + c] : 0.f;
  }
  __syncthreads();

  const int kl = tid / H, h = tid % H, k = k0 + kl;
  if (k >= N) return;
  float acc = 0.f;
#pragma unroll
  for (int c = 0; c < SV; ++c)
    acc += F[h * SV + c] * vs[((size_t)k * H + h) * SV + c];
  const float* DG = drpg + ((size_t)q * H + h) * PV * 3;
  const float* VG = vpg + ((size_t)k * H + h) * PV * 3;
#pragma unroll
  for (int e = 0; e < PV * 3; ++e) acc += DG[e] * VG[e];
  const float* D2 = sD2 + h * CZ;
  const float* PR = sPair + kl * CZ;
#pragma unroll 8
  for (int c = 0; c < CZ; ++c) acc += D2[c] * PR[c];
  dattn[((size_t)q * N + k) * H + h] = acc;
}

// softmax backward over the key axis, in place on dattn -> dlogit, with the
// 1/sqrt(3) that scaled the logits folded in.  One warp a (q, h).
__global__ void softmaxBwdK(const float* __restrict__ attn,
                            float* __restrict__ dattn, int N) {
  int qh = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (qh >= N * H) return;
  int h = qh % H, q = qh / H;
  int lane = threadIdx.x % 32;
  const float* A = attn + (size_t)q * N * H + h;
  float* D = dattn + (size_t)q * N * H + h;
  float dot = 0.f;
  for (int k = lane; k < N; k += 32) dot += A[(size_t)k * H] * D[(size_t)k * H];
#pragma unroll
  for (int o = 16; o; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
  for (int k = lane; k < N; k += 32)
    D[(size_t)k * H] = A[(size_t)k * H] * (D[(size_t)k * H] - dot) * LOGIT_W;
}

// the pair's two gradients: the one through attention_2d's logit (a GEMM, done
// by the caller) and the one through the attention-weighted average, which is
// this.  d_pair[q,k,c] += sum_h d_r2d[q,h,c] attn[q,k,h]
__global__ void dPairFromR2dK(const float* __restrict__ dfinal,
                              const float* __restrict__ attn,
                              float* __restrict__ dpair, int N) {
  size_t qk = blockIdx.x;
  int q = (int)(qk / N);
  const float* F = dfinal + (size_t)q * OUTF + H * SV + 4 * H * PV;
  const float* A = attn + qk * H;
  float* DP = dpair + qk * CZ;
  for (int c = threadIdx.x; c < CZ; c += blockDim.x) {
    float acc = 0.f;
#pragma unroll
    for (int h = 0; h < H; ++h) acc += F[h * CZ + c] * A[h];
    DP[c] += acc;
  }
}

// the two value gradients, both contractions over the QUERY axis.
// One block a (k, h).
__global__ void dValueK(const float* __restrict__ attn,
                        const float* __restrict__ dfinal,
                        const float* __restrict__ drpg,
                        float* __restrict__ dvs, float* __restrict__ dvpg,
                        int N) {
  int h = blockIdx.x % H, k = blockIdx.x / H;
  for (int c = threadIdx.x; c < SV; c += blockDim.x) {
    float acc = 0.f;
    for (int q = 0; q < N; ++q)
      acc += attn[((size_t)q * N + k) * H + h] *
             dfinal[(size_t)q * OUTF + h * SV + c];
    dvs[((size_t)k * H + h) * SV + c] = acc;
  }
  for (int e = threadIdx.x; e < PV * 3; e += blockDim.x) {
    float acc = 0.f;
    for (int q = 0; q < N; ++q)
      acc += attn[((size_t)q * N + k) * H + h] *
             drpg[((size_t)q * H + h) * PV * 3 + e];
    dvpg[((size_t)k * H + h) * PV * 3 + e] = acc;
  }
}

// the scalar query/key gradients.  THE TWO CONTRACT OVER DIFFERENT AXES --
// dq over keys and dk over queries -- which is why they are one kernel with a
// flag rather than two launches of one loop: the gate for a transposed read
// here would be a silently symmetric wrong answer.
template <bool QUERY>
__global__ void dScalarQKK(const float* __restrict__ dlog,
                           const float* __restrict__ other,
                           float* __restrict__ dst, int N) {
  int h = blockIdx.x % H, i = blockIdx.x / H;
  for (int c = threadIdx.x; c < QK; c += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < N; ++j) {
      size_t qk = QUERY ? ((size_t)i * N + j) : ((size_t)j * N + i);
      acc += dlog[qk * H + h] * other[((size_t)j * H + h) * QK + c];
    }
    dst[((size_t)i * H + h) * QK + c] = acc * SCALAR_W;
  }
}

// the point query/key gradients.  The term is -0.5 w sum_p |q - k|^2, so each
// side gets -/+ w times the attention-weighted difference.
template <bool QUERY>
__global__ void dPointQKK(const float* __restrict__ dlog,
                          const float* __restrict__ qpg,
                          const float* __restrict__ kpg,
                          const float* __restrict__ raw_pw,
                          float* __restrict__ dst, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * H * PQK * 3) return;
  int e = idx % (PQK * 3), h = (idx / (PQK * 3)) % H, i = idx / (PQK * 3 * H);
  float w = POINT_W * softplusf(raw_pw[h]);
  float acc = 0.f;
  for (int j = 0; j < N; ++j) {
    size_t qk = QUERY ? ((size_t)i * N + j) : ((size_t)j * N + i);
    int qi = QUERY ? i : j, ki = QUERY ? j : i;
    float diff = qpg[((size_t)qi * H + h) * PQK * 3 + e] -
                 kpg[((size_t)ki * H + h) * PQK * 3 + e];
    acc += dlog[qk * H + h] * diff;
  }
  dst[idx] = (QUERY ? -w : w) * acc;
}

// global point gradient -> local, plus the frame's share of the FORWARD apply.
// g = R l + t, so dl = R^T dg, dt += dg and dR[3i+j] += dg_i l_j.
template <int P>
__global__ void pointGlobalBwdK(const float* __restrict__ dglobal,
                                const float* __restrict__ local,
                                const float* __restrict__ rot,
                                float* __restrict__ dlocal,
                                float* __restrict__ drot,
                                float* __restrict__ dtrans, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * H * P) return;
  int p = idx % P, h = (idx / P) % H, n = idx / (P * H);
  const float* DG = dglobal + (size_t)idx * 3;
  const float* R = rot + (size_t)n * 9;
  float gx = DG[0], gy = DG[1], gz = DG[2];
  float lx = R[0] * gx + R[3] * gy + R[6] * gz;   // R^T
  float ly = R[1] * gx + R[4] * gy + R[7] * gz;
  float lz = R[2] * gx + R[5] * gy + R[8] * gz;
  float* L = dlocal + ((size_t)n * H + h) * (3 * P);
  L[p] = lx; L[P + p] = ly; L[2 * P + p] = lz;    // back into the SPLIT layout
  float* DT = dtrans + (size_t)n * 3;
  atomicAdd(DT + 0, gx); atomicAdd(DT + 1, gy); atomicAdd(DT + 2, gz);
  const float* LL = local + ((size_t)n * H + h) * (3 * P);
  const float l3[3] = {LL[p], LL[P + p], LL[2 * P + p]};
  const float g3[3] = {gx, gy, gz};
  float* DR = drot + (size_t)n * 9;
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) atomicAdd(DR + 3 * i + j, g3[i] * l3[j]);
}

// ---------------------------------------------------------------- the module

struct IPA {
  int N = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;

  // weights, all f32 and all row-major [K, Nn] as the checkpoint stores them
  float *wqs = nullptr, *bqs = nullptr, *wks = nullptr, *bks = nullptr;
  float *wvs = nullptr, *bvs = nullptr;
  float *wqp = nullptr, *bqp = nullptr, *wkp = nullptr, *bkp = nullptr;
  float *wvp = nullptr, *bvp = nullptr;
  float *w2d = nullptr, *b2d = nullptr;
  float *wo = nullptr, *bo = nullptr;
  float* raw_pw = nullptr;

  // activations
  float *qs = nullptr, *ks = nullptr, *vs = nullptr;
  float *qpl = nullptr, *kpl = nullptr, *vpl = nullptr;   // local
  float *qpg = nullptr, *kpg = nullptr, *vpg = nullptr;   // global
  float *a2d = nullptr, *attn = nullptr, *sums = nullptr;
  float *rpg = nullptr, *final_ = nullptr, *out = nullptr;
  // backward scratch
  float *dfinal = nullptr, *dattn = nullptr, *drpg = nullptr;
  float *dvs = nullptr, *dqs = nullptr, *dks = nullptr;
  float *dqpg = nullptr, *dkpg = nullptr, *dvpg = nullptr;
  float *dqpl = nullptr, *dkpl = nullptr, *dvpl = nullptr;
  float *drotS = nullptr, *dtransS = nullptr;
  float *wcat = nullptr, *bcat = nullptr, *projbuf = nullptr;

  void alloc(int N_, cudaStream_t s = nullptr) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    // PRECISION IS PINNED.  PointProjection asks for Precision.HIGHEST, and
    // TF32 would silently drop the mantissa to 10 bits on exactly the matmul
    // whose output feeds a squared distance.
    CUB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));
    size_t n = N;
    CU(cudaMalloc(&qs, n * H * QK * 4));
    CU(cudaMalloc(&ks, n * H * QK * 4));
    CU(cudaMalloc(&vs, n * H * SV * 4));
    CU(cudaMalloc(&qpl, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&kpl, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&vpl, n * H * PV * 3 * 4));
    CU(cudaMalloc(&qpg, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&kpg, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&vpg, n * H * PV * 3 * 4));
    CU(cudaMalloc(&a2d, n * n * H * 4));
    CU(cudaMalloc(&attn, n * n * H * 4));
    CU(cudaMalloc(&sums, n * H * 4));
    CU(cudaMalloc(&rpg, n * H * PV * 3 * 4));
    CU(cudaMalloc(&final_, n * OUTF * 4));
    CU(cudaMalloc(&out, n * C * 4));
    CU(cudaMalloc(&dfinal, n * OUTF * 4));
    CU(cudaMalloc(&dattn, n * n * H * 4));
    CU(cudaMalloc(&drpg, n * H * PV * 3 * 4));
    CU(cudaMalloc(&dvs, n * H * SV * 4));
    CU(cudaMalloc(&dqs, n * H * QK * 4));
    CU(cudaMalloc(&dks, n * H * QK * 4));
    CU(cudaMalloc(&dqpg, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&dkpg, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&dvpg, n * H * PV * 3 * 4));
    CU(cudaMalloc(&dqpl, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&dkpl, n * H * PQK * 3 * 4));
    CU(cudaMalloc(&dvpl, n * H * PV * 3 * 4));
    // A CALLER THAT HANDS THE SAME FRAME TO MORE THAN ONE MODULE cannot let
    // this one WRITE drot/dtrans -- FoldIteration gives the same rigid to both
    // IPA and the composition.  These are its scratch, so it can take the
    // module's two gradients and accumulate them itself.
    CU(cudaMalloc(&drotS, n * 9 * 4));
    CU(cudaMalloc(&dtransS, n * 3 * 4));
    // the fused projection's output, per instance -- see buildCombined
    CU(cudaMalloc(&projbuf, n * PROJ_TOT * 4));
      recordWeightPack(projbuf);
  }

  float* drotScratch() { return drotS; }
  float* dtransScratch() { return dtransS; }

  // row-major D[M, Nn] = A[M, K] @ B[K, Nn] (+ bias broadcast over rows)
  void gemm(int M, int Nn, int K, const float* A, const float* B, float* D,
            float beta = 0.f) {
    const float alpha = 1.f;
    // cuBLAS is column-major, so compute D^T = B^T A^T with the operands
    // handed over untransposed and the roles swapped.
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, Nn, M, K, &alpha, B, Nn, A,
                    K, &beta, D, Nn));
  }

  // row-major D[M, K] = A[M, Nn] @ B[K, Nn]^T -- a gradient back through a
  // Linear whose weights are stored [K, Nn].  beta = 1 accumulates.
  void gemmNT(int M, int K, int Nn, const float* A, const float* B, float* D,
              float beta = 0.f) {
    const float alpha = 1.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, K, M, Nn, &alpha, B, Nn, A,
                    Nn, &beta, D, K));
  }

  void setStream(cudaStream_t s) {
    st = s; CUB(cublasSetStream(cub, st));
  }


  void setWeights(Oracle& o, const std::string& P = "param/") {
    auto get = [&](const char* name, float** dst, size_t n) {
      Tensor& t = o[P + name];
      if (t.n() != n) {
        fprintf(stderr, "ipa: %s has %zu elements, expected %zu\n", name, t.n(), n);
        abort();
      }
      CU(cudaMalloc(dst, n * 4));
      recordWeightPack(*dst);
      CU(cudaMemcpy(*dst, t.host.data(), n * 4, cudaMemcpyHostToDevice));
    };
    const std::string I = "invariant_point_attention/";
    get((I + "q_scalar_projection/weights").c_str(), &wqs, (size_t)C * H * QK);
    get((I + "q_scalar_projection/bias").c_str(), &bqs, (size_t)H * QK);
    get((I + "k_scalar_projection/weights").c_str(), &wks, (size_t)C * H * QK);
    get((I + "k_scalar_projection/bias").c_str(), &bks, (size_t)H * QK);
    get((I + "v_scalar_projection/weights").c_str(), &wvs, (size_t)C * H * SV);
    get((I + "v_scalar_projection/bias").c_str(), &bvs, (size_t)H * SV);
    get((I + "q_point_projection/point_projection/weights").c_str(), &wqp,
        (size_t)C * H * PQK * 3);
    get((I + "q_point_projection/point_projection/bias").c_str(), &bqp,
        (size_t)H * PQK * 3);
    get((I + "k_point_projection/point_projection/weights").c_str(), &wkp,
        (size_t)C * H * PQK * 3);
    get((I + "k_point_projection/point_projection/bias").c_str(), &bkp,
        (size_t)H * PQK * 3);
    get((I + "v_point_projection/point_projection/weights").c_str(), &wvp,
        (size_t)C * H * PV * 3);
    get((I + "v_point_projection/point_projection/bias").c_str(), &bvp,
        (size_t)H * PV * 3);
    get((I + "attention_2d/weights").c_str(), &w2d, (size_t)CZ * H);
    get((I + "attention_2d/bias").c_str(), &b2d, (size_t)H);
    get((I + "output_projection/weights").c_str(), &wo, (size_t)OUTF * C);
    get((I + "output_projection/bias").c_str(), &bo, (size_t)C);
    get((I + "trainable_point_weights").c_str(), &raw_pw, (size_t)H);
    buildCombined(o, P + I);
  }

  // THE CONCATENATION IS BY OUTPUT COLUMN, so the weights interleave per input
  // row: wcat[r, col] walks the six blocks for input row r.  Built on the
  // host from the same tensors the six separate pointers hold, so there is
  // one source of truth and NE_IPA_FUSE=0 falls back to those.
  void buildCombined(Oracle& o, const std::string& I) {
    if (!envOnDef("NE_IPA_FUSE", true)) return;
    const char* names[6] = {"q_scalar_projection/weights",
                            "k_scalar_projection/weights",
                            "v_scalar_projection/weights",
                            "q_point_projection/point_projection/weights",
                            "k_point_projection/point_projection/weights",
                            "v_point_projection/point_projection/weights"};
    const char* bn[6] = {"q_scalar_projection/bias", "k_scalar_projection/bias",
                         "v_scalar_projection/bias",
                         "q_point_projection/point_projection/bias",
                         "k_point_projection/point_projection/bias",
                         "v_point_projection/point_projection/bias"};
    const int w[6] = {W_QS, W_KS, W_VS, W_QP, W_KP, W_VP};
    std::vector<float> hw((size_t)C * PROJ_TOT), hb(PROJ_TOT);
    int off = 0;
    for (int m = 0; m < 6; ++m) {
      Tensor& t = o[I + names[m]];
      Tensor& b = o[I + bn[m]];
      for (int r = 0; r < C; ++r)
        for (int cc = 0; cc < w[m]; ++cc)
          hw[(size_t)r * PROJ_TOT + off + cc] = t.host[(size_t)r * w[m] + cc];
      for (int cc = 0; cc < w[m]; ++cc) hb[off + cc] = b.host[cc];
      off += w[m];
    }
    CU(cudaMalloc(&wcat, hw.size() * 4));
    CU(cudaMemcpy(wcat, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&bcat, hb.size() * 4));
    CU(cudaMemcpy(bcat, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  }

  // THE EIGHT ITERATIONS SHARE ONE SET OF WEIGHTS, so a stack calls
  // setWeights once and points the rest here.  Only the weight pointers are
  // copied; every activation buffer stays this instance's own, because the
  // backward reads all of them.
  void shareWeights(const IPA& o) {
    wqs = o.wqs; bqs = o.bqs; wks = o.wks; bks = o.bks;
    wvs = o.wvs; bvs = o.bvs; wqp = o.wqp; bqp = o.bqp;
    wkp = o.wkp; bkp = o.bkp; wvp = o.wvp; bvp = o.bvp;
    w2d = o.w2d; b2d = o.b2d; wo = o.wo; bo = o.bo;
    raw_pw = o.raw_pw;
    wcat = o.wcat; bcat = o.bcat;   // projbuf stays this instance's own
  }

  void forward(const float* single, const float* pair, const float* mask,
               const float* rot, const float* trans) {
    size_t n = N;
    // SIX PROJECTIONS, ONE GEMM.  They all read the same [N, 384] single and
    // differ only in their output width, so concatenating the weights turns
    // six skinny matmuls -- which cuBLAS splits, hence the splitKreduce calls
    // all over the profile -- into one [N, 384] x [384, 1152].  The scatter
    // that follows also applies the six biases, so this replaces twelve
    // launches with two.
    if (wcat) {
      gemm(N, PROJ_TOT, C, single, wcat, projbuf);
      scatterProjK<<<(int)((n * PROJ_TOT + 255) / 256), 256, 0, st>>>(
          projbuf, bcat, qs, ks, vs, qpl, kpl, vpl, N);
      goto projected;
    }
    // the six projections.  Biases are added by the broadcast kernel rather
    // than a cuBLASLt epilogue because these are f32 Sgemms, not Lt matmuls.
    gemm(N, H * QK, C, single, wqs, qs);
    addRowBiasK<<<((n * H * QK) + 255) / 256, 256, 0, st>>>(qs, bqs, H * QK, n * H * QK);
    gemm(N, H * QK, C, single, wks, ks);
    addRowBiasK<<<((n * H * QK) + 255) / 256, 256, 0, st>>>(ks, bks, H * QK, n * H * QK);
    gemm(N, H * SV, C, single, wvs, vs);
    addRowBiasK<<<((n * H * SV) + 255) / 256, 256, 0, st>>>(vs, bvs, H * SV, n * H * SV);
    gemm(N, H * PQK * 3, C, single, wqp, qpl);
    addRowBiasK<<<((n * H * PQK * 3) + 255) / 256, 256, 0, st>>>(
        qpl, bqp, H * PQK * 3, n * H * PQK * 3);
    gemm(N, H * PQK * 3, C, single, wkp, kpl);
    addRowBiasK<<<((n * H * PQK * 3) + 255) / 256, 256, 0, st>>>(
        kpl, bkp, H * PQK * 3, n * H * PQK * 3);
    gemm(N, H * PV * 3, C, single, wvp, vpl);
    addRowBiasK<<<((n * H * PV * 3) + 255) / 256, 256, 0, st>>>(
        vpl, bvp, H * PV * 3, n * H * PV * 3);
  projected:
    // local -> global
    pointToGlobalK<PQK><<<(n * H * PQK + 255) / 256, 256, 0, st>>>(qpl, rot, trans, qpg, N);
    pointToGlobalK<PQK><<<(n * H * PQK + 255) / 256, 256, 0, st>>>(kpl, rot, trans, kpg, N);
    pointToGlobalK<PV><<<(n * H * PV + 255) / 256, 256, 0, st>>>(vpl, rot, trans, vpg, N);
    // the pair's own logit
    gemm(N * N, H, CZ, pair, w2d, a2d);
    addRowBiasK<<<((n * n * H) + 255) / 256, 256, 0, st>>>(a2d, b2d, H, n * n * H);

    {
      static const int KPB = 16;
      dim3 g(N, (N + KPB - 1) / KPB);
      size_t sh = ((size_t)H * QK + (size_t)H * PQK * 3) * sizeof(float);
      logitsK<KPB><<<g, KPB * H, sh, st>>>(qs, ks, qpg, kpg, a2d, raw_pw, mask,
                                           attn, N);
    }
    softmaxK<<<(n * H + 7) / 8, 256, 0, st>>>(attn, sums, N);
    gatherK<<<n * H, 128, 0, st>>>(attn, vs, vpg, pair, rot, trans, rpg, final_, N);
    gemm(N, C, OUTF, final_, wo, out);
    addRowBiasK<<<((n * C) + 255) / 256, 256, 0, st>>>(out, bo, C, n * C);
  }

  // `dout` is the cotangent on the module's output.  dsingle, dpair, drot and
  // dtrans are WRITTEN, not accumulated -- the caller owns the residual, as
  // everywhere else in this directory.  Must follow a forward() with the same
  // inputs: attn, rpg, the point locals and the globals are all read again.
  // `accPair` is for a STACK.  The pair is `static_feat_2d`: the same tensor
  // every iteration, so its gradient is a sum over all of them and only the
  // first pass may zero it.  The two writers below already accumulate; the
  // memset is the only thing in the way.
  void backward(const float* dout, const float* single, const float* pair,
                const float* rot, const float* trans, float* dsingle,
                float* dpair, float* drot, float* dtrans, bool accPair = false) {
    size_t n = N;
    CU(cudaMemsetAsync(drot, 0, n * 9 * 4, st));      // both apply directions
    CU(cudaMemsetAsync(dtrans, 0, n * 3 * 4, st));    // accumulate into these
    // back through the output projection
    gemmNT(N, OUTF, C, dout, wo, dfinal);

    // the inverse apply, and the frame's share of it
    localBwdK<<<(n * H * PV + 255) / 256, 256, 0, st>>>(
        dfinal, rpg, rot, trans, drpg, drot, dtrans, N);
    // everything the attention weighted
    {
      static const int KPB = 16;
      dim3 g(N, (N + KPB - 1) / KPB);
      size_t sh = ((size_t)H * CZ + (size_t)KPB * CZ) * sizeof(float);
      attnGradK<KPB><<<g, KPB * H, sh, st>>>(dfinal, drpg, vs, vpg, pair,
                                             dattn, N);
    }
    dValueK<<<n * H, 64, 0, st>>>(attn, dfinal, drpg, dvs, dvpg, N);
    // the pair's two routes.  The averaged one lands first because it WRITES
    // through an accumulate, so dpair has to start at zero.
    if (!accPair) CU(cudaMemsetAsync(dpair, 0, n * n * CZ * 4, st));
    dPairFromR2dK<<<n * n, 128, 0, st>>>(dfinal, attn, dpair, N);
    // softmax, in place: dattn becomes the logit gradient
    softmaxBwdK<<<(n * H + 7) / 8, 256, 0, st>>>(attn, dattn, N);
    // the logit's three terms, in reverse
    gemmNT(N * N, CZ, H, dattn, w2d, dpair, /*beta*/1.f);
    dScalarQKK<true><<<n * H, 32, 0, st>>>(dattn, ks, dqs, N);
    dScalarQKK<false><<<n * H, 32, 0, st>>>(dattn, qs, dks, N);
    dPointQKK<true><<<(n * H * PQK * 3 + 255) / 256, 256, 0, st>>>(
        dattn, qpg, kpg, raw_pw, dqpg, N);
    dPointQKK<false><<<(n * H * PQK * 3 + 255) / 256, 256, 0, st>>>(
        dattn, qpg, kpg, raw_pw, dkpg, N);
    // the forward apply, and the frame's other share
    pointGlobalBwdK<PQK><<<(n * H * PQK + 255) / 256, 256, 0, st>>>(
        dqpg, qpl, rot, dqpl, drot, dtrans, N);
    pointGlobalBwdK<PQK><<<(n * H * PQK + 255) / 256, 256, 0, st>>>(
        dkpg, kpl, rot, dkpl, drot, dtrans, N);
    pointGlobalBwdK<PV><<<(n * H * PV + 255) / 256, 256, 0, st>>>(
        dvpg, vpl, rot, dvpl, drot, dtrans, N);
    // and the six projections back onto the single representation
    gemmNT(N, C, H * QK, dqs, wqs, dsingle);
    gemmNT(N, C, H * QK, dks, wks, dsingle, 1.f);
    gemmNT(N, C, H * SV, dvs, wvs, dsingle, 1.f);
    gemmNT(N, C, H * PQK * 3, dqpl, wqp, dsingle, 1.f);
    gemmNT(N, C, H * PQK * 3, dkpl, wkp, dsingle, 1.f);
    gemmNT(N, C, H * PV * 3, dvpl, wvp, dsingle, 1.f);
  }
};

}  // namespace ipa
