// THE SEMI-GREEDY MUTATE STAGE, the first of the two gradient-free ones.
//
// bc2's DESIGN_STAGE_NAMES is six and only the four gradient stages were
// ported.  This is the fifth: fifteen rounds of one-residue mutation, each
// scored by a PREDICTION and kept only if the loss improves.  It is the
// answer to soft-decode collapse -- the gradient stages optimise a mixture
// and the argmax of that mixture is worse than the mixture, so something
// has to search in the discrete space the argmax actually lands in.
//
// It is forward-only, which is why the predict call came first: a
// prediction is a third of a design step (176 ms against 589 at complex
// 175), so fifteen rounds of this cost about five gradient steps.
//
// Transcribed from bindcraft/sequence_optimization.py's
// SemigreedySequenceSampler and bindcraft/trajectory.py's
// run_sequence_mutation_stage.  The four things that read one way and mean
// another are marked WHERE THEY ARE, not here.
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace mutate {

// bc2's OMITTED_AMINO_ACID_LOGIT, and the test it is read back with.  A
// logit at or below half of it means "this residue may not be this amino
// acid" -- which is how aa_bias's zero weight becomes a ban (see the
// aa_bias note in bindcraft_cli.cu).
constexpr float OMITTED_LOGIT = -10000.0f;
inline bool omitted(float logit) { return logit <= OMITTED_LOGIT / 2; }

constexpr float INTERFACE_CUTOFF = 8.0f;

// THE PSEUDO-BETA INTERFACE, as pseudo_beta_interface_mask has it: CB where
// the residue has one and CA where it does not (glycine), and a binder
// residue is at the interface if ANY target pseudo-beta is within 8 A.
//
// The coordinates are the PREDICTION's, not the input's.  That is what
// makes this a measurement of the fold rather than of the template, and it
// is why the mask is recomputed every round.
inline std::vector<char> interfaceMask(const std::vector<float>& atoms,
                                       const std::vector<float>& atom_mask,
                                       const std::vector<uint8_t>& flags,
                                       int n, uint8_t design_flag,
                                       float cutoff = INTERFACE_CUTOFF) {
  // atom 3 is CB in the 37-atom layout, atom 1 is CA
  auto pb = [&](int i, float* xyz) -> bool {
    const size_t b = (size_t)i * 37;
    const int a = atom_mask[b + 3] > 0 ? 3 : 1;
    if (atom_mask[b + a] <= 0) return false;
    const float* p = &atoms[(b + a) * 3];
    xyz[0] = p[0]; xyz[1] = p[1]; xyz[2] = p[2];
    return true;
  };
  std::vector<char> at_interface((size_t)n, 0);
  std::vector<float> tgt;
  for (int j = 0; j < n; ++j) {
    if (flags[j] & design_flag) continue;
    float x[3];
    if (pb(j, x)) tgt.insert(tgt.end(), x, x + 3);
  }
  const float c2 = cutoff * cutoff;
  for (int i = 0; i < n; ++i) {
    if (!(flags[i] & design_flag)) continue;
    float x[3];
    if (!pb(i, x)) continue;
    for (size_t k = 0; k + 2 < tgt.size(); k += 3) {
      const float dx = x[0] - tgt[k], dy = x[1] - tgt[k + 1],
                  dz = x[2] - tgt[k + 2];
      if (dx * dx + dy * dy + dz * dz < c2) { at_interface[i] = 1; break; }
    }
  }
  return at_interface;
}

// SemigreedySequenceSampler, with mutation_weighting='interface_iptm' --
// which is what a SINGLE prepared target selects.  ('plddt' is the
// multitarget branch and leaves the weights at 1.)
struct Semigreedy {
  std::mt19937 rng;
  // per-binder-residue, the MEAN over the states this chain appears in.
  // The binder is in both `complex` and `binder_alone`, so this is the
  // average of two pLDDTs and not the complex's alone.
  std::vector<float> plddt;
  std::vector<float> iface_w;        // 1.0 where not computed
  bool scored = false;
  float best = 0.f;
  std::vector<float> best_logits;
  // THE BANNED LETTERS, PER LETTER AND KEPT.  bc2 caches
  // `omitted_amino_acid_mask(sequence).any(0)` once per chain (setdefault),
  // so a ban survives the one-hot a mutation writes over its row.  Reading
  // it off the row each time lost it the moment a row went one-hot, and
  // the driver's raw logits never carried it to begin with.
  std::vector<char> banned;
  bool isBanned(const float* row, int a) const {
    return omitted(row[a]) || (a < (int)banned.size() && banned[(size_t)a]);
  }

  explicit Semigreedy(uint32_t seed, int binder_len)
      : rng(seed), plddt((size_t)binder_len, 0.f),
        iface_w((size_t)binder_len, 1.f) {}

  // interface_confidence_weights: (1 - iptm_per_residue) over the interface,
  // normalised by its own mean, and 1.0 off the interface.  A residue the
  // model is UNSURE about at the interface is mutated more often.
  void setInterfaceWeights(const std::vector<float>& iptm_res,
                           const std::vector<char>& at_interface,
                           const std::vector<uint8_t>& flags, int n,
                           uint8_t design_flag) {
    double sum = 0; int cnt = 0;
    for (int i = 0; i < n; ++i)
      if (at_interface[i]) { sum += 1.0 - iptm_res[i]; ++cnt; }
    const double mean = cnt ? sum / cnt : 0.0;
    // STOP AT THE PROTOMER.  `iface_w` is sized by binder_len -- ONE
    // protomer's logits, which is what the mutation is written into --
    // and the designed rows of a `copies` campaign are binder_len*copies.
    // Without this bound a two-copy campaign wrote 120 weights into a
    // 60-entry vector and corrupted the heap; it surfaced as "corrupted
    // size vs. prev_size" in the middle of the final filter table, two
    // stages later.  The plddt loop three lines up already bounds itself
    // the same way.
    int b = 0;
    for (int i = 0; i < n && b < (int)iface_w.size(); ++i) {
      if (!(flags[i] & design_flag)) continue;
      iface_w[b++] = (at_interface[i] && mean > 0)
                         ? (float)((1.0 - iptm_res[i]) / std::max(mean, 1e-8))
                         : 1.0f;
    }
  }

  // propose_sequence_mutation.  Two draws: WHICH residue, weighted by
  // (1 - plddt) * interface weight, and WHICH amino acid, from the
  // residue's own logits with the current one excluded.
  //
  // `logits` is the binder's [L, 20] working array and the mutation is
  // written into it as a literal ONE-HOT -- which is what bc2 does, and it
  // matters: predict runs with one_hot_weight 1 at temperature 0.01, so the
  // row's magnitude is irrelevant, but the NEXT proposal reads this row as
  // logits and a one-hot row makes the 19 alternatives equiprobable.
  // THE RESIDUE DISTRIBUTION, exposed.  propose() draws from it, and a
  // draw cannot be compared against bc2's (different RNG) -- the
  // DISTRIBUTION can, and it is where the semantics live.
  std::vector<double> residueProbs(int binder_len) const {
    std::vector<double> p((size_t)binder_len);
    double tot = 0;
    for (int i = 0; i < binder_len; ++i) {
      p[i] = std::max(0.0, (1.0 - (double)plddt[i]) * (double)iface_w[i]);
      tot += p[i];
    }
    // bc2 divides by (sum + 1e-8); jax.random.choice then renormalises,
    // so the epsilon only matters when the sum is zero
    for (int i = 0; i < binder_len; ++i) p[i] /= (tot + 1e-8);
    return p;
  }

  // ...and the amino-acid distribution for one residue: softmax over the
  // row with the CURRENT acid pushed down and every omitted one banned
  std::vector<double> acidProbs(const std::vector<float>& logits, int res,
                                int naa) const {
    const float* row = &logits[(size_t)res * naa];
    int cur = 0;
    for (int a = 1; a < naa; ++a) if (row[a] > row[cur]) cur = a;
    std::vector<double> q((size_t)naa);
    double mx = -1e300;
    for (int a = 0; a < naa; ++a) {
      double v = isBanned(row, a) ? (double)OMITTED_LOGIT : (double)row[a];
      if (a == cur) v -= 1e8;
      q[a] = v; mx = std::max(mx, v);
    }
    double qs = 0;
    for (int a = 0; a < naa; ++a) { q[a] = std::exp(q[a] - mx); qs += q[a]; }
    for (int a = 0; a < naa; ++a) q[a] /= qs;
    return q;
  }

  bool propose(std::vector<float>& logits, int binder_len, int naa,
               int* out_res, int* out_aa) {
    std::vector<double> p((size_t)binder_len);
    double tot = 0;
    for (int i = 0; i < binder_len; ++i) {
      p[i] = std::max(0.0, (1.0 - (double)plddt[i]) * (double)iface_w[i]);
      tot += p[i];
    }
    // bc2 divides by (sum + 1e-8) and then jax.random.choice renormalises;
    // an all-zero vector would be uniform there, so it is uniform here.
    if (tot <= 0) { for (int i = 0; i < binder_len; ++i) p[i] = 1.0; tot = binder_len; }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng) * tot, acc = 0; int res = binder_len - 1;
    for (int i = 0; i < binder_len; ++i) {
      acc += p[i];
      if (r <= acc) { res = i; break; }
    }
    float* row = &logits[(size_t)res * naa];
    int cur = 0;
    for (int a = 1; a < naa; ++a) if (row[a] > row[cur]) cur = a;
    // softmax over the row with the CURRENT amino acid pushed to -1e8 and
    // every omitted one to OMITTED_LOGIT -- so a banned residue (aa_bias
    // weight <= 0) can never be proposed.
    std::vector<double> q((size_t)naa);
    double mx = -1e300;
    for (int a = 0; a < naa; ++a) {
      double v = isBanned(row, a) ? (double)OMITTED_LOGIT : (double)row[a];
      if (a == cur) v -= 1e8;
      q[a] = v; mx = std::max(mx, v);
    }
    double qs = 0;
    for (int a = 0; a < naa; ++a) { q[a] = std::exp(q[a] - mx); qs += q[a]; }
    r = U(rng) * qs; acc = 0; int aa = cur;
    for (int a = 0; a < naa; ++a) {
      acc += q[a];
      if (r <= acc) { aa = a; break; }
    }
    if (aa == cur) return false;
    for (int a = 0; a < naa; ++a) row[a] = (a == aa) ? 1.f : 0.f;
    *out_res = res; *out_aa = aa;
    return true;
  }
};

}  // namespace mutate
