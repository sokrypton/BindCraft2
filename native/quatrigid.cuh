// AF2's rigid update: QuatRigid, and the frame composition it feeds.
//
// This is the second piece of the structure module and the one whose backward
// genuinely exists nowhere -- a quaternion normalisation and a rotation
// composition, differentiated.
//
//   rigid_flat = Linear(act) -> 6          [qx, qy, qz, tx, ty, tz]
//   q          = (1, qx, qy, qz) / |(1, qx, qy, qz)|
//   R_update   = quaternion_to_matrix(q)
//   rigid      = rigid @ Rigid3Array(R_update, t)
//
// THE REAL PART IS PINNED TO 1 AND IS NOT A VARIABLE.  Only three channels
// parametrise the rotation.  w still takes part in the NORMALISATION though,
// so its square is in the norm the other three are divided by -- which is why
// the backward sums over all four components while writing only three.
//
// f32, and the Linear asks for Precision.HIGHEST: this is the matmul whose
// three outputs become a rotation.
#pragma once
#include "common.cuh"

namespace qrigid {

static const int C = 384;       // single channels
static const float QUAT_EPS = 1e-6f;

// rigid_flat -> the update's rotation and translation
__global__ void quatToRigidK(const float* __restrict__ flat,
                             float* __restrict__ rotu,
                             float* __restrict__ transu, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* F = flat + (size_t)n * 6;
  float w = 1.f, x = F[0], y = F[1], z = F[2];
  float inv = rsqrtf(fmaxf(QUAT_EPS, w * w + x * x + y * y + z * z));
  w *= inv; x *= inv; y *= inv; z *= inv;
  float* R = rotu + (size_t)n * 9;
  R[0] = 1.f - 2.f * (y * y + z * z);
  R[1] = 2.f * (x * y - w * z);
  R[2] = 2.f * (x * z + w * y);
  R[3] = 2.f * (x * y + w * z);
  R[4] = 1.f - 2.f * (x * x + z * z);
  R[5] = 2.f * (y * z - w * x);
  R[6] = 2.f * (x * z - w * y);
  R[7] = 2.f * (y * z + w * x);
  R[8] = 1.f - 2.f * (x * x + y * y);
  float* T = transu + (size_t)n * 3;
  T[0] = F[3]; T[1] = F[4]; T[2] = F[5];
}

// rigid = rigid @ update:  R' = R Ru,  t' = R tu + t
__global__ void composeK(const float* __restrict__ rot,
                         const float* __restrict__ trans,
                         const float* __restrict__ rotu,
                         const float* __restrict__ transu,
                         float* __restrict__ rout, float* __restrict__ tout,
                         int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* R = rot + (size_t)n * 9;
  const float* U = rotu + (size_t)n * 9;
  float* O = rout + (size_t)n * 9;
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float a = 0.f;
#pragma unroll
      for (int k = 0; k < 3; ++k) a += R[3 * i + k] * U[3 * k + j];
      O[3 * i + j] = a;
    }
  const float* T = trans + (size_t)n * 3;
  const float* TU = transu + (size_t)n * 3;
  float* OT = tout + (size_t)n * 3;
#pragma unroll
  for (int i = 0; i < 3; ++i)
    OT[i] = R[3 * i + 0] * TU[0] + R[3 * i + 1] * TU[1] + R[3 * i + 2] * TU[2] + T[i];
}

// the composition's backward.  dR gets two contributions -- one from the
// rotation product and one from carrying the update's translation -- and dt
// passes straight through.
__global__ void composeBwdK(const float* __restrict__ rot,
                            const float* __restrict__ rotu,
                            const float* __restrict__ transu,
                            const float* __restrict__ drout,
                            const float* __restrict__ dtout,
                            float* __restrict__ drot, float* __restrict__ dtrans,
                            float* __restrict__ drotu,
                            float* __restrict__ dtransu, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* R = rot + (size_t)n * 9;
  const float* U = rotu + (size_t)n * 9;
  const float* TU = transu + (size_t)n * 3;
  const float* DO = drout + (size_t)n * 9;
  const float* DT = dtout + (size_t)n * 3;
  float* DR = drot + (size_t)n * 9;
  float* DU = drotu + (size_t)n * 9;
  float* DTU = dtransu + (size_t)n * 3;
  float* DTR = dtrans + (size_t)n * 3;
#pragma unroll
  for (int i = 0; i < 3; ++i)
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) a += DO[3 * i + j] * U[3 * k + j];
      DR[3 * i + k] = a + DT[i] * TU[k];      // the product, then the carry
    }
#pragma unroll
  for (int k = 0; k < 3; ++k)
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 3; ++i) a += DO[3 * i + j] * R[3 * i + k];
      DU[3 * k + j] = a;
    }
#pragma unroll
  for (int k = 0; k < 3; ++k)
    DTU[k] = DT[0] * R[0 + k] + DT[1] * R[3 + k] + DT[2] * R[6 + k];
#pragma unroll
  for (int i = 0; i < 3; ++i) DTR[i] = DT[i];
}

// the quaternion's backward: dR -> dq-hat -> dq, and the normalisation is the
// part with the cross terms.  q-hat = q / |q|, so
//   dq_i = inv dqh_i - inv^3 q_i sum_j q_j dqh_j
// with j running over ALL FOUR components even though only three are written,
// because w is in the norm.
__global__ void quatBwdK(const float* __restrict__ flat,
                         const float* __restrict__ drotu,
                         const float* __restrict__ dtransu,
                         float* __restrict__ dflat, int N) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* F = flat + (size_t)n * 6;
  float w = 1.f, x = F[0], y = F[1], z = F[2];
  float n2 = fmaxf(QUAT_EPS, w * w + x * x + y * y + z * z);
  float inv = rsqrtf(n2);
  float qw = w * inv, qx = x * inv, qy = y * inv, qz = z * inv;
  const float* D = drotu + (size_t)n * 9;
  // dR/dq-hat, written out
  float dw = 0.f, dx = 0.f, dy = 0.f, dz = 0.f;
  dy += D[0] * (-4.f * qy);  dz += D[0] * (-4.f * qz);                    // xx
  dx += D[1] * ( 2.f * qy);  dy += D[1] * ( 2.f * qx);
  dw += D[1] * (-2.f * qz);  dz += D[1] * (-2.f * qw);                    // xy
  dx += D[2] * ( 2.f * qz);  dz += D[2] * ( 2.f * qx);
  dw += D[2] * ( 2.f * qy);  dy += D[2] * ( 2.f * qw);                    // xz
  dx += D[3] * ( 2.f * qy);  dy += D[3] * ( 2.f * qx);
  dw += D[3] * ( 2.f * qz);  dz += D[3] * ( 2.f * qw);                    // yx
  dx += D[4] * (-4.f * qx);  dz += D[4] * (-4.f * qz);                    // yy
  dy += D[5] * ( 2.f * qz);  dz += D[5] * ( 2.f * qy);
  dw += D[5] * (-2.f * qx);  dx += D[5] * (-2.f * qw);                    // yz
  dx += D[6] * ( 2.f * qz);  dz += D[6] * ( 2.f * qx);
  dw += D[6] * (-2.f * qy);  dy += D[6] * (-2.f * qw);                    // zx
  dy += D[7] * ( 2.f * qz);  dz += D[7] * ( 2.f * qy);
  dw += D[7] * ( 2.f * qx);  dx += D[7] * ( 2.f * qw);                    // zy
  dx += D[8] * (-4.f * qx);  dy += D[8] * (-4.f * qy);                    // zz
  // through the normalisation
  float dot = qw * dw + qx * dx + qy * dy + qz * dz;   // = sum_j q-hat_j dqh_j
  float* DF = dflat + (size_t)n * 6;
  // q_i inv^3 sum_j q_j dqh_j  ==  q-hat_i inv dot, since q = q-hat |q|
  DF[0] = inv * (dx - qx * dot);
  DF[1] = inv * (dy - qy * dot);
  DF[2] = inv * (dz - qz * dot);
  const float* DT = dtransu + (size_t)n * 3;
  DF[3] = DT[0]; DF[4] = DT[1]; DF[5] = DT[2];
}

struct QuatRigid {
  int N = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  float *w = nullptr, *b = nullptr;
  float *flat = nullptr, *rotu = nullptr, *transu = nullptr;
  float *dflat = nullptr, *drotu = nullptr, *dtransu = nullptr;

  void alloc(int N_, cudaStream_t s = nullptr) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    CUB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));   // Precision.HIGHEST
    size_t n = N;
    CU(cudaMalloc(&flat, n * 6 * 4));
    CU(cudaMalloc(&rotu, n * 9 * 4));
    CU(cudaMalloc(&transu, n * 3 * 4));
    CU(cudaMalloc(&dflat, n * 6 * 4));
    CU(cudaMalloc(&drotu, n * 9 * 4));
    CU(cudaMalloc(&dtransu, n * 3 * 4));
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    auto get = [&](const char* name, float** dst, size_t cnt) {
      Tensor& t = o[P + name];
      CU(cudaMalloc(dst, cnt * 4));
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("quat_rigid/rigid/weights", &w, (size_t)C * 6);
    get("quat_rigid/rigid/bias", &b, 6);
  }

  void shareWeights(const QuatRigid& o) { w = o.w; b = o.b; }

  void forward(const float* act, const float* rot, const float* trans,
               float* rout, float* tout) {
    const float alpha = 1.f, beta = 0.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, 6, N, C, &alpha, w, 6, act,
                    C, &beta, flat, 6));
    addRowBiasK<<<((size_t)N * 6 + 255) / 256, 256, 0, st>>>(flat, b, 6, (size_t)N * 6);
    quatToRigidK<<<(N + 255) / 256, 256, 0, st>>>(flat, rotu, transu, N);
    composeK<<<(N + 255) / 256, 256, 0, st>>>(rot, trans, rotu, transu, rout, tout, N);
  }

  void backward(const float* drout, const float* dtout, const float* rot,
                float* dact, float* drot, float* dtrans) {
    composeBwdK<<<(N + 255) / 256, 256, 0, st>>>(
        rot, rotu, transu, drout, dtout, drot, dtrans, drotu, dtransu, N);
    quatBwdK<<<(N + 255) / 256, 256, 0, st>>>(flat, drotu, dtransu, dflat, N);
    const float alpha = 1.f, beta = 0.f;
    // dact = dflat @ w^T
    CUB(cublasSgemm(cub, CUBLAS_OP_T, CUBLAS_OP_N, C, N, 6, &alpha, w, 6, dflat,
                    6, &beta, dact, C));
  }
};

}  // namespace qrigid
