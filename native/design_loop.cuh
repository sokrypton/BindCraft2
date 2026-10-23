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
  float *chipp = nullptr;           // chi_pi_periodic, the monomer row only
  int32_t *a1437 = nullptr, *a3714 = nullptr, *chii = nullptr;
  void upload() {
    auto F = [&](float** p, const float* h, size_t n) {
      CU(::cudaMalloc(p, n * 4)); CU(cudaMemcpy(*p, h, n * 4, cudaMemcpyHostToDevice)); };
    auto I = [&](int32_t** p, const int* h, size_t n) {
      std::vector<int32_t> t(h, h + n);
      CU(::cudaMalloc(p, n * 4)); CU(cudaMemcpy(*p, t.data(), n * 4, cudaMemcpyHostToDevice)); };
    F(&a14m, RES_ATOM14_MASK, 308); F(&a37m, RES_ATOM37_MASK, 814);
    F(&chim, RES_CHI_MASK, 88);
    // MONOMER ONLY, but uploaded unconditionally: 88 floats, and a table that
    // is present-but-unread is cheaper than a second upload path that is
    // skipped on the branch nobody runs.
    F(&chipp, RES_CHI_PI_PERIODIC, 88);
    I(&a1437, RES_ATOM14_TO_ATOM37, 308); I(&a3714, RES_ATOM37_TO_ATOM14, 814);
    I(&chii, RES_CHI_ATOM_INDICES, 352);
  }
  inputfeat::Tables feat() const { return inputfeat::Tables{a14m, a37m, a1437, a3714}; }
  tpl1d::Tables chi() const { return tpl1d::Tables{chii, chim, chipp}; }
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
  // the composition bias, [NAA], uploaded once -- see Inputs::aa_bias
  float* aa_bias = nullptr;
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
    F(&aa_bias, dsin::NAA);
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
    in.seed = seed; in.scalars = scalars; in.aa_bias = aa_bias;
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
  void zeroCotangents(cudaStream_t st = nullptr) {
    const size_t n1 = nmax, nn = (size_t)nmax * nmax;
    CU(cudaMemsetAsync(c_atoms, 0, n1 * 37 * 3 * 4, st)); CU(cudaMemsetAsync(c_plddt, 0, n1 * 4, st));
    CU(cudaMemsetAsync(c_pae, 0, nn * 4, st)); CU(cudaMemsetAsync(c_ptm, 0, 4, st));
    CU(cudaMemsetAsync(c_iptm, 0, 4, st)); CU(cudaMemsetAsync(c_iptm_res, 0, n1 * 4, st));
    CU(cudaMemsetAsync(c_expres, 0, n1 * 4, st));
    CU(cudaMemsetAsync(c_dgram, 0, nn * (size_t)design_step::DBINS * 4, st));
  }
};

// ONE ROUND'S PREDICTION, KEPT.  bc2's run_gradient_design_stage keeps the
// round with the lowest design loss (`best_protein_states,
// best_predictions`) and the stage ENDS THERE: its filters judge that
// round's prediction and the next stage starts from that round's sequence.
// The device only ever holds the latest forward, so the outputs of the best
// round are copied aside when it is seen and copied back when the stage
// ends -- the prediction itself, not a re-run of it, because a re-run
// redraws the design model and the dropout.
struct OutputSnapshot {
  size_t n1 = 0, nn = 0;
  float *atoms = nullptr, *plddt = nullptr, *pae = nullptr, *ptm = nullptr;
  float *iptm = nullptr, *iptm_res = nullptr, *expres = nullptr, *dgram = nullptr;
  bool* amask = nullptr;
  void alloc(const DeviceState& d) {
    n1 = (size_t)d.nmax; nn = n1 * n1;
    auto F = [&](float** p, size_t k) { CU(::cudaMalloc(p, k * 4)); };
    F(&atoms, n1 * 37 * 3); F(&plddt, n1); F(&pae, nn); F(&ptm, 1);
    F(&iptm, 1); F(&iptm_res, n1); F(&expres, n1);
    F(&dgram, nn * design_step::DBINS);
    CU(::cudaMalloc(&amask, n1 * 37));
  }
  static void copy(float* dst, const float* src, size_t k) {
    CU(cudaMemcpy(dst, src, k * 4, cudaMemcpyDeviceToDevice));
  }
  void save(const DeviceState& d) {
    copy(atoms, d.o_atoms, n1 * 37 * 3); copy(plddt, d.o_plddt, n1);
    copy(pae, d.o_pae, nn); copy(ptm, d.o_ptm, 1); copy(iptm, d.o_iptm, 1);
    copy(iptm_res, d.o_iptm_res, n1); copy(expres, d.o_expres, n1);
    copy(dgram, d.o_dgram, nn * design_step::DBINS);
    CU(cudaMemcpy(amask, d.o_amask, n1 * 37, cudaMemcpyDeviceToDevice));
  }
  void restore(DeviceState& d) const {
    copy(d.o_atoms, atoms, n1 * 37 * 3); copy(d.o_plddt, plddt, n1);
    copy(d.o_pae, pae, nn); copy(d.o_ptm, ptm, 1); copy(d.o_iptm, iptm, 1);
    copy(d.o_iptm_res, iptm_res, n1); copy(d.o_expres, expres, n1);
    copy(d.o_dgram, dgram, nn * design_step::DBINS);
    CU(cudaMemcpy(d.o_amask, amask, n1 * 37, cudaMemcpyDeviceToDevice));
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

// THE STAGE SCHEDULE, as bindcraft/trajectory.py sets it up.  The four
// scalars ride in a device buffer precisely so these can move without
// recompiling anything -- see the note in ds_gate.py.
//
//   screen  softmax 0.0 -> 0.9 linear,  one_hot 0, temperature 1
//   refine  softmax 0.9 -> 1.0 linear,  one_hot 0, temperature 1
//   anneal  softmax 1.0,                one_hot 0, temperature 1 -> 0.01 QUADRATIC
//   harden  softmax 1.0,                one_hot 1, temperature 0.01
//
// A flat setting is not a neutral choice: the whole point of the ramp is
// that a soft sequence explores and a hard one commits, and running the
// hard end from the start is a different algorithm.
enum Stage { SCREEN = 0, REFINE, ANNEAL, HARDEN, NSTAGES };

// settings.py's DESIGN_STAGE_DEFAULT_ROUNDS.  A flat count per stage is not
// the same schedule: screen is 50 rounds and harden is 5, so an equal split
// spends ten times too long committing and a tenth as long exploring.
inline int stageRounds(int stage) {
  static const int r[] = {50, 25, 45, 5};
  return r[stage];
}
inline const char* stageName(int s) {
  static const char* n[] = {"screen", "refine", "anneal", "harden"};
  return n[s];
}
struct Scalars { float softmax_w, one_hot_w, temperature, logit_scale; };

inline Scalars stageScalars(int stage, int it, int iters) {
  // (it + 1) / iters, NOT it / iters.  Both of bc2's schedules are written
  // that way -- _linear_sequence_schedule and _quadratic_temperature_schedule
  // -- so the LAST iteration lands exactly on the end value.  Using it/iters
  // leaves every ramp short: screen finished at softmax 0.79 instead of 0.9
  // and anneal at temperature 0.025 instead of 0.01, which is visible in the
  // trace and easy to read past.
  const float t = iters > 0 ? (float)(std::min(it, iters - 1) + 1) / (float)iters : 1.f;
  Scalars s{1.f, 0.f, 1.f, 2.f};
  switch (stage) {
    case SCREEN: s.softmax_w = 0.0f + (0.9f - 0.0f) * t; break;
    case REFINE: s.softmax_w = 0.9f + (1.0f - 0.9f) * t; break;
    case ANNEAL: {
      // quadratic, not linear: trajectory.py's _quadratic_temperature_schedule
      const float u = 1.f - t;
      s.temperature = 0.01f + (1.0f - 0.01f) * u * u;
      break;
    }
    case HARDEN: s.one_hot_w = 1.f; s.temperature = 0.01f; break;
  }
  return s;
}

// SGD, WHICH IS WHAT bc2 USES.  sequence_optimization.py:67 --
// `optax.sgd(learning_rate)` with learning_rate=0.1 -- not Adam.
//
// The difference is not cosmetic.  Adam rescales each element to about the
// learning rate, so after norm_seq_grad (which already fixes the gradient's
// SCALE) it discards the one thing left in it, the relative magnitude
// between positions, and takes a step about five times larger.  design.cu's
// harness uses Adam; the product has to use what bc2 uses.
// THE LOGITS LIVE IN FLOAT16 BETWEEN STEPS, which is bc2's and not a
// choice this port gets to make.  `Protein.empty` builds the sequence as
// float16 and `GradientSequenceOptimizer._update` ends with
//
//     .replace(sequence=updated_logits.astype(shared.sequence.dtype))
//
// so the arithmetic is f32 -- read, normalise, Adam, add -- and the RESULT
// is quantised back to f16 every single step.  Keeping f32 across the
// trajectory is a different optimiser: not a more accurate one, a different
// one, and the port diverges from bc2 at step 1 rather than at the end.
//
// BC2_F32_LOGITS=1 keeps f32, for measuring what the quantisation is worth.
inline float toF16(float v) {
  return (float)__float2half(v);
}
inline void quantiseLogits(float* x, size_t n) {
  if (getenv("BC2_F32_LOGITS")) return;
  for (size_t i = 0; i < n; ++i) x[i] = toF16(x[i]);
}

// WHAT THE QUANTISATION IS WORTH, MEASURED WHERE IT HAPPENS.
//
// The obvious experiment -- run the trajectory with f16 and with f32 and
// compare the sequences -- measures nothing: the control (the SAME arm
// twice, same seed) already diverges completely, because the backward is
// not bit-reproducible (repro_gate, ~9.1e-04 relative).  Any difference
// between the arms is buried under that.
//
// So the quantity is measured at the step instead, where it is
// deterministic: of the entries this update moved, how many land back on
// the SAME f16 grid point -- i.e. how many updates the storage throws away.
// BC2_LOGIT_QUANT=1 prints it per step.
// Returns the count MOVED as well as the fraction lost, because 0/0 and
// 0/N print the same otherwise -- and they mean opposite things.  Harden's
// gradient is exactly zero (harden_gradient_gate), so its rows are 0/0:
// "no updates to lose", which read identically to "no updates lost" in the
// first version of this and would have hidden the end of anneal.
inline float quantiseLogitsCounting(float* x, const float* before, size_t n,
                                    size_t* moved_out) {
  if (getenv("BC2_F32_LOGITS")) { *moved_out = 0; return 0.f; }
  size_t moved = 0, lost = 0;
  for (size_t i = 0; i < n; ++i) {
    const float q = toF16(x[i]);
    if (x[i] != before[i]) {
      ++moved;
      if (q == toF16(before[i])) ++lost;
    }
    x[i] = q;
  }
  *moved_out = moved;
  return moved ? (float)lost / (float)moved : 0.f;
}

struct Sgd {
  float lr = 0.1f;
  std::vector<float> before;
  void step(float* x, const float* g, size_t n) {
    const bool count = getenv("BC2_LOGIT_QUANT") != nullptr;
    if (count) before.assign(x, x + n);
    for (size_t i = 0; i < n; ++i) x[i] -= lr * g[i];
    if (count) {
      size_t moved = 0;
      const float lost = quantiseLogitsCounting(x, before.data(), n, &moved);
      if (moved == 0)
        printf("      logit quantisation: NOTHING MOVED (%zu entries)\n", n);
      else
        printf("      logit quantisation: %.1f%% of %zu moved entries round "
               "back\n", 100.f * lost, moved);
    } else {
      quantiseLogits(x, n);
    }
  }
};

// Adam, as design.cu's harness has it, kept for comparison.
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
    // the Adam MOMENTS stay f32 -- bc2's optimizer_state is built from the
    // f32 view and never round-trips.  Only the logits are quantised.
    quantiseLogits(x, n);
  }
};

}  // namespace dsloop
