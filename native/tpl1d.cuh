// template_embedding_1d: the MSA ROW THE TEMPLATE ADDS.
//
// This is the piece the one-call step was missing, and nothing upstream
// could see it: every per-component gate hands the trunk an MSA that JAX
// built, so none of them notices that the trunk's MSA is not the
// embedder's.  modules_multimer concatenates a template row onto both the
// msa and its mask before the Evoformer
// (`evoformer_input['msa'] = concatenate([msa, template_features])`), so a
// step that runs the trunk at depth 1 is running a different model.
//
// NO BACKWARD.  Its inputs are the template COORDINATES (an input, not a
// prediction) and template_aatype, which bc2 derives from
// `sequence.argmax(-1)` -- so the whole row is a constant of the step with
// respect to the designed logits.  It still has to be recomputed every
// step, because the argmax moves.
//
// The chi angles are AF2's own: four dihedrals per residue, over the atom
// quadruples chi_angles_atoms names, masked both by which chis the residue
// type has and by whether all four atoms were resolved.
#pragma once
#include "common.cuh"

namespace tpl1d {

constexpr int NAA1H = 22;     // one_hot(template_aatype, 22)
constexpr int NCHI = 4;
constexpr int NFEAT = NAA1H + 3 * NCHI;   // 34
constexpr int CM = 256;       // msa_channel
constexpr int ATOM = 37;

struct Tables {
  const int32_t* chi_atom_indices;   // [22, 4, 4]
  const float*   chi_angles_mask;    // [22, 4]
};

// dihedral_angle(a,b,c,d) as geometry/vector.py computes it -- NOT the
// textbook atan2(|v2| * (v1 x v2) . v3, (v1 x v2) . (v2 x v3)): the signs
// and the operand order differ and the two disagree by pi on half the
// inputs.
__device__ __forceinline__ float dihedral(const float* a, const float* b,
                                          const float* c, const float* d) {
  float v1[3], v2[3], v3[3];
  for (int i = 0; i < 3; ++i) { v1[i] = a[i] - b[i]; v2[i] = b[i] - c[i];
                                v3[i] = d[i] - c[i]; }
  auto cross = [](const float* x, const float* y, float* o) {
    o[0] = x[1] * y[2] - x[2] * y[1];
    o[1] = x[2] * y[0] - x[0] * y[2];
    o[2] = x[0] * y[1] - x[1] * y[0];
  };
  float c1[3], c2[3], c3[3];
  cross(v1, v2, c1);
  cross(v3, v2, c2);
  cross(c2, c1, c3);
  float v2m = sqrtf(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);
  float num = c3[0]*v2[0] + c3[1]*v2[1] + c3[2]*v2[2];
  float den = v2m * (c1[0]*c2[0] + c1[1]*c2[1] + c1[2]*c2[2]);
  return atan2f(num, den);
}

// one residue a thread: the 34 features and the row's mask
__global__ void featK(const int32_t* __restrict__ t_aatype,  // [NMAX]
                      const float* __restrict__ t_pos,       // [NMAX,37,3]
                      const float* __restrict__ t_mask,      // [NMAX,37]
                      const int32_t* __restrict__ np,
                      Tables tab,
                      float* __restrict__ feat,              // [NMAX,34]
                      float* __restrict__ row_mask,          // [NMAX]
                      int NMAX) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NMAX) return;
  float* f = feat + (size_t)i * NFEAT;
  for (int k = 0; k < NFEAT; ++k) f[k] = 0.f;
  if (i >= *np) { row_mask[i] = 0.f; return; }

  // jnp's gather CLAMPS an out-of-range index; aatype 21 with a 21-row
  // table therefore reads row 20, and the tables below carry a 22nd row
  // that repeats row 20 so this reads the same thing without a branch.
  int a = t_aatype[i];
  a = a < 0 ? 0 : (a > NAA1H - 1 ? NAA1H - 1 : a);
  f[a] = 1.f;                                   // one_hot(aatype, 22)

  for (int ch = 0; ch < NCHI; ++ch) {
    const int32_t* idx = tab.chi_atom_indices + ((size_t)a * NCHI + ch) * 4;
    float m = tab.chi_angles_mask[(size_t)a * NCHI + ch];
    for (int k = 0; k < 4; ++k) m *= t_mask[(size_t)i * ATOM + idx[k]];
    const float* p[4];
    for (int k = 0; k < 4; ++k)
      p[k] = t_pos + ((size_t)i * ATOM + idx[k]) * 3;
    const float ang = dihedral(p[0], p[1], p[2], p[3]);
    f[NAA1H + ch]            = sinf(ang) * m;
    f[NAA1H + NCHI + ch]     = cosf(ang) * m;
    f[NAA1H + 2 * NCHI + ch] = m;
    if (ch == 0) row_mask[i] = m;
  }
}

// feat -> Linear(256) -> relu -> Linear(256), the row the trunk reads
struct Tpl1D {
  int N = 0;
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  float *w0 = nullptr, *b0 = nullptr, *w1 = nullptr, *b1 = nullptr;
  float *feat = nullptr, *h0 = nullptr;

  void alloc(int N_, cudaStream_t s) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, s));
    CU(cudaMalloc(&feat, (size_t)N * NFEAT * 4));
    CU(cudaMalloc(&h0, (size_t)N * CM * 4));
  }

  void setStream(cudaStream_t s) {
    st = s; CUB(cublasSetStream(cub, st));
  }

  void setWeights(Oracle& o, const std::string& P = "param/") {
    auto get = [&](const std::string& nm, float** dst, size_t cnt) {
      Tensor& t = o[P + nm];
      if (t.n() != cnt) {
        fprintf(stderr, "tpl1d: %s has %zu, want %zu\n", nm.c_str(), t.n(), cnt);
        abort();
      }
      CU(cudaMalloc(dst, cnt * 4));
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("template_single_embedding/weights", &w0, (size_t)NFEAT * CM);
    get("template_single_embedding/bias", &b0, CM);
    get("template_projection/weights", &w1, (size_t)CM * CM);
    get("template_projection/bias", &b1, CM);
  }

  // out is [N, CM] -- ONE row; bc2 carries exactly one target template
  void forward(const int32_t* t_aatype, const float* t_pos,
               const float* t_mask, const int32_t* n, Tables tab,
               float* out, float* row_mask) {
    const int g1 = (N + 127) / 128;
    featK<<<g1, 128, 0, st>>>(t_aatype, t_pos, t_mask, n, tab, feat,
                              row_mask, N);
    // `template_features.astype(bfloat16)` and both Linears under
    // bfloat16_context -- see roundBf16K
    roundBf16K<<<(int)(((size_t)N * NFEAT + 255) / 256), 256, 0, st>>>(
        feat, (size_t)N * NFEAT);
    const float alpha = 1.f, beta = 0.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, CM, N, NFEAT, &alpha,
                    w0, CM, feat, NFEAT, &beta, h0, CM));
    const size_t nc = (size_t)N * CM;
    addRowBiasK<<<(int)((nc + 255) / 256), 256, 0, st>>>(h0, b0, CM, nc);
    roundBf16K<<<(int)((nc + 255) / 256), 256, 0, st>>>(h0, nc);
    foldit::reluK<<<(int)((nc + 255) / 256), 256, 0, st>>>(h0, nc);
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, CM, N, CM, &alpha,
                    w1, CM, h0, CM, &beta, out, CM));
    addRowBiasK<<<(int)((nc + 255) / 256), 256, 0, st>>>(out, b1, CM, nc);
    roundBf16K<<<(int)((nc + 255) / 256), 256, 0, st>>>(out, nc);
  }
};

}  // namespace tpl1d
