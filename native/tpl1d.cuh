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
// THE MONOMER'S ROW IS A DIFFERENT ROW.  alphafold2_ptm embeds all SEVEN
// torsions (pre_omega, phi, psi, chi1..4), their ALTERNATIVES under the
// naming ambiguities, and a mask for each -- 22 + 14 + 14 + 7 = 57 -- where
// the multimer embeds the four chis as sin, cos and mask, 22 + 12 = 34.  The
// weight shape says so: template_single_embedding is [57, 256] against
// [34, 256].
constexpr int NTORS = 7;
constexpr int NFEAT_PTM = NAA1H + 2 * 2 * NTORS + NTORS;   // 57
constexpr int NFEAT_MAX = NFEAT_PTM;
constexpr int CM = 256;       // msa_channel
constexpr int ATOM = 37;

struct Tables {
  const int32_t* chi_atom_indices;   // [22, 4, 4]
  const float*   chi_angles_mask;    // [22, 4]
  // MONOMER ONLY: which chis are 180-degree ambiguous, which is what the
  // alternative angles mirror.  Null on the multimer path.
  const float*   chi_pi_periodic = nullptr;   // [22, 4]
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

// The MONOMER's 57 features, one residue a thread.
//
//   [one_hot(aatype, 22) | sin,cos x 7 | alt sin,cos x 7 | mask x 7]
//
// The seven torsions are AF2's own order -- pre_omega, phi, psi, chi1..4 --
// and the first two READ THE PREVIOUS RESIDUE, which is the only reason this
// cannot be a per-residue function of its own atoms.  Residue 0's previous
// atoms are ZEROS, exactly as atom37_to_torsion_angles pads them, and its
// pre_omega and phi are computed from those zeros rather than skipped: the
// MASK is zero there but the FEATURE IS NOT MASKED, so the value has to be
// the same garbage AF2 produces.  That is what makes the frame construction
// below worth writing out instead of calling dihedral():
// vecs_robust_normalize divides by sqrt(|v|^2 + 1e-8), so a degenerate frame
// is finite and reproducible, and atan2 of the same inputs is not.
__device__ __forceinline__ void torsionSinCos(const float* p0, const float* p1,
                                              const float* p2, const float* p3,
                                              float* sc) {
  // rigids_from_3_points(point_on_neg_x_axis=p1, origin=p2, point_on_xy_plane=p0)
  float e0[3], e1[3];
  for (int k = 0; k < 3; ++k) { e0[k] = p2[k] - p1[k]; e1[k] = p0[k] - p2[k]; }
  auto rnorm = [](float* v) {
    const float n = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + 1e-8f);
    for (int k = 0; k < 3; ++k) v[k] /= n;
  };
  rnorm(e0);
  const float c = e1[0]*e0[0] + e1[1]*e0[1] + e1[2]*e0[2];
  for (int k = 0; k < 3; ++k) e1[k] -= c * e0[k];
  rnorm(e1);
  const float e2[3] = {e0[1]*e1[2] - e0[2]*e1[1],
                       e0[2]*e1[0] - e0[0]*e1[2],
                       e0[0]*e1[1] - e0[1]*e1[0]};
  // the fourth atom in that frame: R^T (p3 - origin)
  const float d[3] = {p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]};
  const float y = d[0]*e1[0] + d[1]*e1[1] + d[2]*e1[2];
  const float z = d[0]*e2[0] + d[1]*e2[1] + d[2]*e2[2];
  const float n = sqrtf(z * z + y * y + 1e-8f);
  sc[0] = z / n;   // sin
  sc[1] = y / n;   // cos
}

__global__ void featPtmK(const int32_t* __restrict__ t_aatype,  // [NMAX]
                         const float* __restrict__ t_pos,       // [NMAX,37,3]
                         const float* __restrict__ t_mask,      // [NMAX,37]
                         const int32_t* __restrict__ np,
                         Tables tab,
                         float* __restrict__ feat,              // [NMAX,57]
                         float* __restrict__ row_mask,          // [NMAX]
                         int NMAX) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NMAX) return;
  float* f = feat + (size_t)i * NFEAT_PTM;
  for (int k = 0; k < NFEAT_PTM; ++k) f[k] = 0.f;
  if (i >= *np) { row_mask[i] = 0.f; return; }

  // THE ONE-HOT IS NOT CLAMPED TO 20 AND THE TABLES ARE.  The embedder does
  // one_hot(template_aatype, 22) on the raw aatype -- bc2 writes 21 for a
  // residue with no template -- while atom37_to_torsion_angles starts with
  // `aatype = jnp.minimum(aatype, 20)`.  Two different indices from one
  // array, which is exactly the kind of thing that gates green at 21 and
  // fails on a real campaign's untemplated binder.
  int a = t_aatype[i];
  a = a < 0 ? 0 : (a > NAA1H - 1 ? NAA1H - 1 : a);
  f[a] = 1.f;
  const int ac = a > 20 ? 20 : a;

  const float* P = t_pos + (size_t)i * ATOM * 3;
  const float* M = t_mask + (size_t)i * ATOM;
  const float zero3[3] = {0.f, 0.f, 0.f};
  const bool first = (i == 0);
  const float* PV = first ? nullptr : t_pos + (size_t)(i - 1) * ATOM * 3;
  const float* MV = first ? nullptr : t_mask + (size_t)(i - 1) * ATOM;
  auto prevPos = [&](int at) { return first ? zero3 : PV + at * 3; };
  auto prevMask = [&](int at) { return first ? 0.f : MV[at]; };

  // N, CA, C, O in atom37
  const int N_ = 0, CA = 1, C_ = 2, O_ = 4;
  float sc[2];
  for (int t = 0; t < NTORS; ++t) {
    const float *p0, *p1, *p2, *p3;
    float m;
    if (t == 0) {            // pre_omega: prev CA, prev C, this N, this CA
      p0 = prevPos(CA); p1 = prevPos(C_); p2 = P + N_ * 3; p3 = P + CA * 3;
      m = prevMask(CA) * prevMask(C_) * M[N_] * M[CA];
    } else if (t == 1) {     // phi: prev C, this N, CA, C
      p0 = prevPos(C_); p1 = P + N_ * 3; p2 = P + CA * 3; p3 = P + C_ * 3;
      m = prevMask(C_) * M[N_] * M[CA] * M[C_];
    } else if (t == 2) {     // psi: this N, CA, C, O
      p0 = P + N_ * 3; p1 = P + CA * 3; p2 = P + C_ * 3; p3 = P + O_ * 3;
      m = M[N_] * M[CA] * M[C_] * M[O_];
    } else {
      const int ch = t - 3;
      const int32_t* idx = tab.chi_atom_indices + ((size_t)ac * NCHI + ch) * 4;
      p0 = P + idx[0] * 3; p1 = P + idx[1] * 3;
      p2 = P + idx[2] * 3; p3 = P + idx[3] * 3;
      m = tab.chi_angles_mask[(size_t)ac * NCHI + ch];
      for (int k = 0; k < 4; ++k) m *= M[idx[k]];
    }
    torsionSinCos(p0, p1, p2, p3, sc);
    // psi is MIRRORED, because it is computed from the oxygen
    if (t == 2) { sc[0] = -sc[0]; sc[1] = -sc[1]; }
    // and the alternative mirrors the chis that are 180-degree ambiguous
    const float mir = (t < 3) ? 1.f
        : 1.f - 2.f * tab.chi_pi_periodic[(size_t)ac * NCHI + (t - 3)];
    f[NAA1H + 2 * t + 0] = sc[0];
    f[NAA1H + 2 * t + 1] = sc[1];
    f[NAA1H + 2 * NTORS + 2 * t + 0] = sc[0] * mir;
    f[NAA1H + 2 * NTORS + 2 * t + 1] = sc[1] * mir;
    f[NAA1H + 4 * NTORS + t] = m;
    // the MSA row's mask is the PSI mask -- "it only depends on the backbone
    // atoms from a single residue", which pre_omega and phi do not
    if (t == 2) row_mask[i] = m;
  }
}

// feat -> Linear(256) -> relu -> Linear(256), the row the trunk reads
struct Tpl1D {
  int N = 0;
  // set before setWeights: it decides the feature width and so the shape of
  // template_single_embedding
  bool monomer = false;
  int nfeat() const { return monomer ? NFEAT_PTM : NFEAT; }
  cudaStream_t st = nullptr;
  cublasHandle_t cub = nullptr;
  float *w0 = nullptr, *b0 = nullptr, *w1 = nullptr, *b1 = nullptr;
  float *feat = nullptr, *h0 = nullptr;

  void alloc(int N_, cudaStream_t s) {
    N = N_; st = s;
    CUB(cublasCreate(&cub));
    CUB(cublasSetStream(cub, s));
    // NFEAT_MAX, so the same object serves either width -- alloc() runs before
    // the caller has necessarily decided, and it runs AGAIN when the step is
    // re-derived at a shorter length (nbrealloc), where asking for a bigger
    // buffer than the first call took is an error.
    CU(cudaMalloc(&feat, (size_t)N * NFEAT_MAX * 4));
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
      recordWeightPack(*dst);
      CU(cudaMemcpy(*dst, t.host.data(), cnt * 4, cudaMemcpyHostToDevice));
    };
    get("template_single_embedding/weights", &w0, (size_t)nfeat() * CM);
    get("template_single_embedding/bias", &b0, CM);
    get("template_projection/weights", &w1, (size_t)CM * CM);
    get("template_projection/bias", &b1, CM);
  }

  // out is [N, CM] -- ONE row; bc2 carries exactly one target template
  void forward(const int32_t* t_aatype, const float* t_pos,
               const float* t_mask, const int32_t* n, Tables tab,
               float* out, float* row_mask) {
    const int g1 = (N + 127) / 128;
    const int nf = nfeat();
    if (monomer)
      featPtmK<<<g1, 128, 0, st>>>(t_aatype, t_pos, t_mask, n, tab, feat,
                                   row_mask, N);
    else
      featK<<<g1, 128, 0, st>>>(t_aatype, t_pos, t_mask, n, tab, feat,
                                row_mask, N);
    // `template_features.astype(bfloat16)` and both Linears under
    // bfloat16_context -- see roundBf16K
    roundBf16K<<<(int)(((size_t)N * nf + 255) / 256), 256, 0, st>>>(
        feat, (size_t)N * nf);
    const float alpha = 1.f, beta = 0.f;
    CUB(cublasSgemm(cub, CUBLAS_OP_N, CUBLAS_OP_N, CM, N, nf, &alpha,
                    w0, CM, feat, nf, &beta, h0, CM));
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
