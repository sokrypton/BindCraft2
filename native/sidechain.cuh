// MultiRigidSidechain's MLP half: activations -> seven torsion angles.
//
//   act = Linear_a(relu(act_cur)) + Linear_b(relu(act_initial))   -> 128
//   twice:  act += Linear(relu(Linear(relu(act))))
//   raw = Linear(relu(act))                                       -> 14
//   angles = l2_normalize(reshape(raw, [N, 7, 2]), axis=-1)
//
// TWO REPRESENTATIONS GO IN, NOT ONE.  `representations_list` is
// [act, initial_act] and each gets its OWN input_projection -- the checkpoint
// has `input_projection` and `input_projection_1` for exactly that -- and the
// two are SUMMED, which the code calls "equivalent to concat then Conv1D".
// `initial_act` is the structure module's input single representation, the
// same tensor for all eight iterations, so its gradient accumulates.
//
// WHAT IS NOT HERE: torsion_angles_to_frames and
// frames_and_literature_positions_to_atom14_pos.  Those are gathers into the
// residue-constant tables by aatype and belong with the atom builder, which
// is its own unit.  This file stops at the angles, which is where the pure
// linear algebra stops.
//
// THE NORMALISATION IS PER (residue, torsion), over the sin/cos PAIR -- a
// 2-vector, not the 14 channels.  Normalising the wrong axis gives angles
// that still look like angles.
#pragma once
#include "common.cuh"

namespace sidechain {

static const int CIN = 384;    // the single representation
static const int C = 128;      // sidechain num_channel
static const int NTOR = 7;     // torsions, each a (sin, cos) pair
static const int NANG = NTOR * 2;
static const int NRES_BLOCK = 2;
static const float L2_EPS = 1e-12f;   // l2_normalize's epsilon

__global__ void reluToK(const float* __restrict__ x, float* __restrict__ y,
                        size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = fmaxf(x[i], 0.f);
}

// dx += dy * (x > 0), reading the PRE-activation.  The relu here is applied to
// a value that is kept anyway (it is the block's residual input), so the sign
// comes from the input rather than from the output as in foldit.cuh.
__global__ void reluToBwdK(const float* __restrict__ dy, const float* __restrict__ x,
                           float* __restrict__ dx, size_t n, bool acc) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i] > 0.f ? dy[i] : 0.f;
  dx[i] = acc ? dx[i] + v : v;
}

// angles = raw / |raw| over each (sin, cos) pair.  One thread a (residue,
// torsion).
__global__ void normAnglesK(const float* __restrict__ raw,
                            float* __restrict__ ang, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * NTOR) return;
  const float* R = raw + (size_t)i * 2;
  float s = R[0], c = R[1];
  float inv = rsqrtf(fmaxf(L2_EPS, s * s + c * c));
  float* A = ang + (size_t)i * 2;
  A[0] = s * inv; A[1] = c * inv;
}

// and its backward: the same projection-out-of-the-radial-direction the
// quaternion normalisation has, in two dimensions.
__global__ void normAnglesBwdK(const float* __restrict__ raw,
                               const float* __restrict__ dang,
                               float* __restrict__ draw, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * NTOR) return;
  const float* R = raw + (size_t)i * 2;
  const float* D = dang + (size_t)i * 2;
  float s = R[0], c = R[1];
  float n2 = fmaxf(L2_EPS, s * s + c * c);
  float inv = rsqrtf(n2);
  float hs = s * inv, hc = c * inv;
  float dot = hs * D[0] + hc * D[1];
  float* O = draw + (size_t)i * 2;
  O[0] = inv * (D[0] - hs * dot);
  O[1] = inv * (D[1] - hc * dot);
}

struct Sidechain {
  int N = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;

  float *wi0 = nullptr, *bi0 = nullptr, *wi1 = nullptr, *bi1 = nullptr;
  float *wr1[NRES_BLOCK] = {}, *br1[NRES_BLOCK] = {};
  float *wr2[NRES_BLOCK] = {}, *br2[NRES_BLOCK] = {};
  float *wa = nullptr, *ba = nullptr;

  // forward state, all kept: this module is N-shaped, so the save-set is
  // 128 floats a residue per stage -- nothing worth replaying.  A residual
  // block's INPUT is just the previous stage's output, so it is a pointer and
  // not a copy; only the tensors a backward actually re-reads are buffers.
  float *rin0 = nullptr, *rin1 = nullptr;       // relu of the two inputs
  float *act0 = nullptr;                        // the summed projection
  const float* blkIn[NRES_BLOCK] = {};          // borrowed, see above
  float *blkH[NRES_BLOCK] = {};                 // the block's middle value
  float *blkOut[NRES_BLOCK] = {};
  float *rb1[NRES_BLOCK] = {}, *rb2[NRES_BLOCK] = {};
  const float* actF = nullptr;                  // the last block's output
  float *rF = nullptr;
  float *raw = nullptr, *ang = nullptr;
  float *dtmp = nullptr, *dtmp2 = nullptr, *dtmp3 = nullptr, *draw = nullptr;

  void alloc(int N_, cudaStream_t s = nullptr) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, st));
    CUB(cublasSetMathMode(cub, CUBLAS_PEDANTIC_MATH));
    size_t ni = (size_t)N * CIN, nc = (size_t)N * C;
    CU(cudaMalloc(&rin0, ni * 4)); CU(cudaMalloc(&rin1, ni * 4));
    CU(cudaMalloc(&act0, nc * 4));
    for (int b = 0; b < NRES_BLOCK; ++b) {
      CU(cudaMalloc(&blkH[b], nc * 4));
      CU(cudaMalloc(&blkOut[b], nc * 4));
      CU(cudaMalloc(&rb1[b], nc * 4));
      CU(cudaMalloc(&rb2[b], nc * 4));
    }
    CU(cudaMalloc(&rF, nc * 4));
    CU(cudaMalloc(&raw, (size_t)N * NANG * 4));
    CU(cudaMalloc(&ang, (size_t)N * NANG * 4));
    CU(cudaMalloc(&dtmp, nc * 4)); CU(cudaMalloc(&dtmp2, nc * 4));
    CU(cudaMalloc(&dtmp3, nc * 4));
    CU(cudaMalloc(&draw, (size_t)N * NANG * 4));
  }

  void gemm(int M, int Nn, int K, const float* A, const float* B, float* D,
            float beta = 0.f) {
    const float alpha = 1.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, Nn, M, K, &alpha, B, Nn, A,
                    K, &beta, D, Nn));
  }
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
    auto get = [&](const std::string& nm, float** dst, size_t cnt) {
      // THE PREFIX IS THE DUMP'S.  A --part sc dump builds these Linears at
      // the top level; inside a FoldIteration they are scoped under
      // rigid_sidechain/.  Reading which one is present beats hardcoding.
      const std::string k = o.has(P + "rigid_sidechain/" + nm)
                                ? P + "rigid_sidechain/" + nm : P + nm;
      Tensor& t = o[k];
      if (t.n() != cnt) {
        fprintf(stderr, "sidechain: %s has %zu, expected %zu\n", k.c_str(),
                t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("input_projection/weights", &wi0, (size_t)CIN * C);
    get("input_projection/bias", &bi0, C);
    get("input_projection_1/weights", &wi1, (size_t)CIN * C);
    get("input_projection_1/bias", &bi1, C);
    // haiku's suffixing: the first residual block's Linears are `resblock1`
    // and `resblock2`, the second's are `resblock1_1` and `resblock2_1`.
    for (int b = 0; b < NRES_BLOCK; ++b) {
      std::string sfx = b ? "_1" : "";
      get("resblock1" + sfx + "/weights", &wr1[b], (size_t)C * C);
      get("resblock1" + sfx + "/bias", &br1[b], C);
      get("resblock2" + sfx + "/weights", &wr2[b], (size_t)C * C);
      get("resblock2" + sfx + "/bias", &br2[b], C);
    }
    get("unnormalized_angles/weights", &wa, (size_t)C * NANG);
    get("unnormalized_angles/bias", &ba, NANG);
  }

  // `act` is the current iteration's activation, `initial` the structure
  // module's input single -- the same tensor for all eight iterations.
  void shareWeights(const Sidechain& o) {
    wi0 = o.wi0; bi0 = o.bi0; wi1 = o.wi1; bi1 = o.bi1;
    for (int b = 0; b < NRES_BLOCK; ++b) {
      wr1[b] = o.wr1[b]; br1[b] = o.br1[b];
      wr2[b] = o.wr2[b]; br2[b] = o.br2[b];
    }
    wa = o.wa; ba = o.ba;
  }

  void forward(const float* act, const float* initial) {
    size_t ni = (size_t)N * CIN, nc = (size_t)N * C;
    int gi = (int)((ni + 255) / 256), gc = (int)((nc + 255) / 256);
    // TWO REPRESENTATIONS, EACH WITH ITS OWN PROJECTION, SUMMED.
    reluToK<<<gi, 256, 0, st>>>(act, rin0, ni);
    reluToK<<<gi, 256, 0, st>>>(initial, rin1, ni);
    gemm(N, C, CIN, rin0, wi0, act0);
    addRowBiasK<<<gc, 256, 0, st>>>(act0, bi0, C, nc);
    gemm(N, C, CIN, rin1, wi1, dtmp);
    addRowBiasK<<<gc, 256, 0, st>>>(dtmp, bi1, C, nc);
    accSumK<<<gc, 256, 0, st>>>(act0, dtmp, nc);
    const float* cur = act0;
    for (int b = 0; b < NRES_BLOCK; ++b) {
      blkIn[b] = cur;
      reluToK<<<gc, 256, 0, st>>>(blkIn[b], rb1[b], nc);
      gemm(N, C, C, rb1[b], wr1[b], blkH[b]);
      addRowBiasK<<<gc, 256, 0, st>>>(blkH[b], br1[b], C, nc);
      reluToK<<<gc, 256, 0, st>>>(blkH[b], rb2[b], nc);
      gemm(N, C, C, rb2[b], wr2[b], blkOut[b]);
      addRowBiasK<<<gc, 256, 0, st>>>(blkOut[b], br2[b], C, nc);
      accSumK<<<gc, 256, 0, st>>>(blkOut[b], blkIn[b], nc);
      cur = blkOut[b];
    }
    actF = cur;
    reluToK<<<gc, 256, 0, st>>>(actF, rF, nc);
    gemm(N, NANG, C, rF, wa, raw);
    addRowBiasK<<<(int)(((size_t)N * NANG + 255) / 256), 256, 0, st>>>(
        raw, ba, NANG, (size_t)N * NANG);
    normAnglesK<<<(N * NTOR + 255) / 256, 256, 0, st>>>(raw, ang, N);
  }

  // `dang` is the cotangent on the normalised angles.  dact is written;
  // dinitial is ACCUMULATED, because the same initial_act feeds all eight
  // iterations.
  void backward(const float* dang, const float* act, const float* initial,
                float* dact, float* dinitial, bool accInitial) {
    size_t ni = (size_t)N * CIN, nc = (size_t)N * C;
    int gi = (int)((ni + 255) / 256), gc = (int)((nc + 255) / 256);
    normAnglesBwdK<<<(N * NTOR + 255) / 256, 256, 0, st>>>(raw, dang, draw, N);
    gemmNT(N, C, NANG, draw, wa, dtmp);
    reluToBwdK<<<gc, 256, 0, st>>>(dtmp, actF, dtmp, nc, false);
    for (int b = NRES_BLOCK - 1; b >= 0; --b) {
      gemmNT(N, C, C, dtmp, wr2[b], dtmp2);
      reluToBwdK<<<gc, 256, 0, st>>>(dtmp2, blkH[b], dtmp2, nc, false);
      gemmNT(N, C, C, dtmp2, wr1[b], dtmp3);
      reluToBwdK<<<gc, 256, 0, st>>>(dtmp3, blkIn[b], dtmp3, nc, false);
      accSumK<<<gc, 256, 0, st>>>(dtmp, dtmp3, nc);   // + the block's residual
    }
    // the two input projections.  rin0/rin1 are done with as activations by
    // now -- the relu's sign comes from the INPUTS, which the caller still
    // owns -- so they serve as the gradient scratch.
    gemmNT(N, CIN, C, dtmp, wi0, rin0);
    reluToBwdK<<<gi, 256, 0, st>>>(rin0, act, dact, ni, false);
    gemmNT(N, CIN, C, dtmp, wi1, rin1);
    reluToBwdK<<<gi, 256, 0, st>>>(rin1, initial, dinitial, ni, accInitial);
  }
};

}  // namespace sidechain
