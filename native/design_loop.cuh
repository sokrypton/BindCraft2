// ONE DESIGN STEP, END TO END, IN C++: sequence -> model -> loss -> sequence.
//
// design.cu already has this shape against Oracle dumps; this is the same
// chain driven from a campaign file, with the state binding bc2 uses.
#pragma once
#include "design_step.cuh"
#include "design_inputs.cuh"
#include "prod_losses.cuh"
#include "act_const.cuh"

namespace dsloop {

// The residue-constant tables, uploaded once.  They are NOT optional: a
// default-constructed inputfeat::Tables has null pointers and the embedder
// fails at its first matmul with `embed.cuh:184 cublas: 13`, which names the
// matmul and not the missing table.
struct ConstTables {
  float *a14m = nullptr, *a37m = nullptr, *chim = nullptr;
  int32_t *a1437 = nullptr, *a3714 = nullptr, *chii = nullptr;
  void upload() {
    auto F = [&](float** p, const float* h, size_t n) {
      CU(::cudaMalloc(p, n * 4)); CU(cudaMemcpy(*p, h, n * 4, cudaMemcpyHostToDevice)); };
    auto I = [&](int32_t** p, const int* h, size_t n) {
      std::vector<int32_t> t(h, h + n);
      CU(::cudaMalloc(p, n * 4)); CU(cudaMemcpy(*p, t.data(), n * 4, cudaMemcpyHostToDevice)); };
    F(&a14m, RES_ATOM14_MASK, 308); F(&a37m, RES_ATOM37_MASK, 814);
    F(&chim, RES_CHI_MASK, 88);
    I(&a1437, RES_ATOM14_TO_ATOM37, 308); I(&a3714, RES_ATOM37_TO_ATOM14, 814);
    I(&chii, RES_CHI_ATOM_INDICES, 352);
  }
  inputfeat::Tables feat() const { return inputfeat::Tables{a14m, a37m, a1437, a3714}; }
  tpl1d::Tables chi() const { return tpl1d::Tables{chii, chim}; }
};

// One state's device buffers: the inputs the step reads and the outputs it
// writes.  Held for the life of a trajectory, since the shapes do not move.
struct DeviceState {
  int n = 0, nmax = 0;
  float *sequence = nullptr, *atoms = nullptr, *atom_mask = nullptr;
  float *seq_mask = nullptr, *act_const = nullptr, *scalars = nullptr;
  uint8_t* flags = nullptr;
  int32_t *residue_index = nullptr, *asym_id = nullptr, *entity_id = nullptr;
  int32_t *interface_asym = nullptr, *n_dev = nullptr;
  uint32_t* seed = nullptr;
  // outputs
  float *o_atoms = nullptr, *o_plddt = nullptr, *o_pae = nullptr;
  float *o_ptm = nullptr, *o_iptm = nullptr, *o_iptm_res = nullptr;
  float *o_expres = nullptr, *o_dgram = nullptr;
  bool* o_amask = nullptr;
  // cotangents
  float *c_atoms = nullptr, *c_plddt = nullptr, *c_pae = nullptr;
  float *c_ptm = nullptr, *c_iptm = nullptr, *c_iptm_res = nullptr;
  float *c_expres = nullptr, *c_dgram = nullptr;
  float* d_sequence = nullptr;

  void alloc(const dsin::State& h) {
    n = h.n; nmax = h.nmax;
    const size_t n1 = nmax, nn = (size_t)nmax * nmax;
    auto F = [&](float** p, size_t k) { CU(::cudaMalloc(p, k * 4)); CU(cudaMemset(*p, 0, k * 4)); };
    F(&sequence, n1 * dsin::NAA); F(&atoms, n1 * 37 * 3); F(&atom_mask, n1 * 37);
    F(&seq_mask, n1); F(&act_const, nn * 64); F(&scalars, 4);
    CU(::cudaMalloc(&flags, n1)); CU(cudaMemset(flags, 0, n1));
    auto I = [&](int32_t** p, size_t k) { CU(::cudaMalloc(p, k * 4)); CU(cudaMemset(*p, 0, k * 4)); };
    I(&residue_index, n1); I(&asym_id, n1); I(&entity_id, n1);
    I(&interface_asym, n1); I(&n_dev, 1);
    CU(::cudaMalloc(&seed, 2 * 4)); CU(cudaMemset(seed, 0, 2 * 4));
    F(&o_atoms, n1 * 37 * 3); F(&o_plddt, n1); F(&o_pae, nn); F(&o_ptm, 1);
    F(&o_iptm, 1); F(&o_iptm_res, n1); F(&o_expres, n1);
    F(&o_dgram, nn * design_step::DBINS);
    CU(::cudaMalloc(&o_amask, n1 * 37)); CU(cudaMemset(o_amask, 0, n1 * 37));
    F(&c_atoms, n1 * 37 * 3); F(&c_plddt, n1); F(&c_pae, nn); F(&c_ptm, 1);
    F(&c_iptm, 1); F(&c_iptm_res, n1); F(&c_expres, n1);
    F(&c_dgram, nn * design_step::DBINS);
    F(&d_sequence, n1 * dsin::NAA);
    // THE RUNTIME LENGTH LIVES ON THE DEVICE, which is what lets one
    // allocation serve every length -- see design_step.cuh's note on in.n.
    CU(cudaMemcpy(n_dev, &n, 4, cudaMemcpyHostToDevice));
  }

  void upload(const dsin::State& h) {
    const size_t n1 = nmax;
    CU(cudaMemcpy(sequence, h.sequence.data(), n1 * dsin::NAA * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(atoms, h.atoms.data(), n1 * 37 * 3 * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(atom_mask, h.atom_mask.data(), n1 * 37 * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(seq_mask, h.seq_mask.data(), n1 * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(flags, h.flags.data(), n1, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(residue_index, h.residue_index.data(), n1 * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(asym_id, h.asym_id.data(), n1 * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(entity_id, h.entity_id.data(), n1 * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(interface_asym, h.interface_asym.data(), n1 * 4, cudaMemcpyHostToDevice));
  }

  design_step::Inputs inputs() const {
    design_step::Inputs in{};
    in.sequence = sequence; in.atoms = atoms; in.atom_mask = atom_mask;
    in.flags = flags; in.residue_index = residue_index; in.asym_id = asym_id;
    in.entity_id = entity_id; in.interface_asym = interface_asym;
    in.seq_mask = seq_mask; in.act_const = act_const; in.n = n_dev;
    in.seed = seed; in.scalars = scalars;
    return in;
  }
  design_step::Outputs outputs() const {
    design_step::Outputs o{};
    o.atoms = o_atoms; o.atom_mask = o_amask; o.plddt = o_plddt; o.pae = o_pae;
    o.ptm = o_ptm; o.iptm = o_iptm; o.iptm_per_residue = o_iptm_res;
    o.expres_ca = o_expres; o.distogram = o_dgram;
    return o;
  }
  design_step::Cotangents cotangents() const {
    design_step::Cotangents c{};
    c.atoms = c_atoms; c.plddt = c_plddt; c.pae = c_pae; c.ptm = c_ptm;
    c.iptm = c_iptm; c.iptm_per_residue = c_iptm_res; c.expres_ca = c_expres;
    c.distogram = c_dgram;
    return c;
  }
  void zeroCotangents() {
    const size_t n1 = nmax, nn = (size_t)nmax * nmax;
    CU(cudaMemset(c_atoms, 0, n1 * 37 * 3 * 4)); CU(cudaMemset(c_plddt, 0, n1 * 4));
    CU(cudaMemset(c_pae, 0, nn * 4)); CU(cudaMemset(c_ptm, 0, 4));
    CU(cudaMemset(c_iptm, 0, 4)); CU(cudaMemset(c_iptm_res, 0, n1 * 4));
    CU(cudaMemset(c_expres, 0, n1 * 4));
    CU(cudaMemset(c_dgram, 0, nn * (size_t)design_step::DBINS * 4));
  }
};

// THE FOUR DESIGN SCALARS, which bc2 RAMPS across a trajectory:
// softmax_weight, one_hot_weight, temperature, logit_scale.  They ride in a
// buffer rather than an attribute precisely so the ramp does not recompile.
inline void setScalars(DeviceState& d, float softmax_w, float one_hot_w,
                       float temperature, float logit_scale) {
  const float s[4] = {softmax_w, one_hot_w, temperature, logit_scale};
  CU(cudaMemcpy(d.scalars, s, 4 * 4, cudaMemcpyHostToDevice));
}

// Adam, as design.cu's loop has it.
struct Adam {
  std::vector<float> m, v;
  int t = 0;
  float lr = 0.1f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
  void step(float* x, const float* g, size_t n) {
    if (m.empty()) { m.assign(n, 0.f); v.assign(n, 0.f); }
    ++t;
    const float c1 = 1.f - powf(b1, (float)t), c2 = 1.f - powf(b2, (float)t);
    for (size_t i = 0; i < n; ++i) {
      m[i] = b1 * m[i] + (1.f - b1) * g[i];
      v[i] = b2 * v[i] + (1.f - b2) * g[i] * g[i];
      x[i] -= lr * (m[i] / c1) / (sqrtf(v[i] / c2) + eps);
    }
  }
};

}  // namespace dsloop
