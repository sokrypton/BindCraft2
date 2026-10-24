// relax.cuh -- `relax_accepted_designs`, which is a restrained minimiser
// and not a force field.
//
// TODO.md said this "needs an energy minimiser" and left it for last.
// `protein.relax_protein_complex` is about seventy lines: Adam over the
// RESOLVED atoms of the complex with three quadratic terms, all O(N^2),
// all differentiable by hand.  No OpenMM, no AMBER, no hydrogens.
//
//   clash      0.5 * w_clash * sum_ij C_ij * relu(closest_ij - d_ij)^2
//   bond       w_bond * sum_b (d_b - d_b0)^2
//   restraint  sum_i pw_i * |x_i - x0_i|^2
//
// where `closest_ij = max(vdw_i + vdw_j - overlap_tol, min_sep)`, a bond
// is a same-residue pair under 1.9 A plus a C-N amide under 1.9 and an
// S-S under 2.5, and a clashing pair is any pair that is NOT in
// neighbouring residues of the same chain and NOT bonded.
//
// THE BOND LENGTHS ARE THE INPUT'S OWN.  `bond_lengths` is read off the
// structure being relaxed, so the bond term does not pull towards ideal
// geometry -- it holds the geometry the model predicted while the clash
// term pushes overlapping atoms apart.  A relaxation that pulled towards
// ideal bond lengths would be a different operation with the same name.
//
// AND THE TOLERANCE IS NOT A ROUNDING DETAIL.  bc2's own comment: "a
// restrained minimum settles short of the separation it is pushed
// towards, so the tolerance has to leave a relieved contact clear of the
// 2.5 A a clash is counted at".  overlap_tol 0.4 against a 2.5 A clash
// cutoff is what makes `Relaxed_Backbone_Clashes` lower than
// `Backbone_Clashes` rather than equal to it.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace relaxer {

struct Params {
  int steps = 200;
  double learning_rate = 0.02;
  double restraint_backbone = 10.0;
  double restraint_sidechain = 0.5;
  double weight_bond = 100.0;
  double weight_clash = 5.0;
  double overlap_tol = 0.4;
  double min_sep = 2.5;
};

// DERIVED FROM THE ATOM NAMES, not transcribed.  bc2 builds both tables
// the same way -- `RELAX_VDW_RADIUS_BY_ELEMENT.get(name[0], C)` and
// `name in ('N','CA','C','O','OXT')` -- so deriving them here from the
// same 37 names means a table cannot drift from the names it indexes.
inline double vdwRadius(const char* atom_name) {
  switch (atom_name[0]) {
    case 'N': return 1.55;
    case 'O': return 1.52;
    case 'S': return 1.80;
    default:  return 1.70;      // C, and anything unrecognised, as bc2's .get
  }
}

inline bool isBackbone(const char* atom_name) {
  const std::string n(atom_name);
  return n == "N" || n == "CA" || n == "C" || n == "O" || n == "OXT";
}

// `atoms` is [n][37][3] and `atom_mask` [n][37], both in the port's own
// row order; `asym_id` and `residue_index` are per RESIDUE.  Only the
// resolved slots move, and they are written back in place.
//
// `atom_names` is the 37-name table (pdbin::ATOM_TYPES), passed rather
// than included so this header has no dependency on the PDB reader.
inline void relax(std::vector<float>& atoms,
                  const std::vector<float>& atom_mask,
                  const std::vector<int32_t>& asym_id,
                  const std::vector<int>& residue_index,
                  int n, const char* const* atom_names,
                  const Params& p) {
  // ---- gather the resolved slots ---------------------------------------
  std::vector<int> slot_residue, slot_atom;
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < 37; ++a)
      if (atom_mask[(size_t)i * 37 + a] > 0) {
        slot_residue.push_back(i);
        slot_atom.push_back(a);
      }
  const int M = (int)slot_residue.size();
  if (M < 2) return;
  std::vector<double> x((size_t)M * 3), x0((size_t)M * 3), radius((size_t)M),
      pw((size_t)M);
  for (int s = 0; s < M; ++s) {
    const int i = slot_residue[s], a = slot_atom[s];
    for (int c = 0; c < 3; ++c)
      x[(size_t)s * 3 + c] = x0[(size_t)s * 3 + c] =
          atoms[((size_t)i * 37 + a) * 3 + c];
    radius[(size_t)s] = vdwRadius(atom_names[a]);
    pw[(size_t)s] = isBackbone(atom_names[a]) ? p.restraint_backbone
                                              : p.restraint_sidechain;
  }

  // ---- the pair classification, on the INPUT coordinates ---------------
  //
  // Built once: bc2 computes `bonded` and `clashing_pairs` from
  // `relax_geometry`'s distances, which are the input's, and holds them
  // fixed through the optimisation.  A bond list that moved with the
  // coordinates would let the minimiser break a bond by moving out of the
  // 1.9 A window and then stop restraining it.
  auto dist = [&](int s, int t) {
    double q = 0;
    for (int c = 0; c < 3; ++c) {
      const double d = x0[(size_t)s * 3 + c] - x0[(size_t)t * 3 + c];
      q += d * d;
    }
    return std::sqrt(q > 0 ? q : 0.0);
  };
  std::vector<int> bond_a, bond_b;
  std::vector<double> bond_len;
  std::vector<char> clashing((size_t)M * M, 0);
  std::vector<double> closest((size_t)M * M, 0.0);
  for (int s = 0; s < M; ++s) {
    for (int t = 0; t < M; ++t) {
      if (s == t) continue;
      const int ri = slot_residue[s], rj = slot_residue[t];
      const int ai = slot_atom[s], aj = slot_atom[t];
      const double d = dist(s, t);
      const bool same_chain = asym_id[ri] == asym_id[rj];
      const bool same_residue = same_chain && residue_index[ri] == residue_index[rj];
      const bool neighbouring =
          same_chain && std::abs(residue_index[ri] - residue_index[rj]) <= 1;
      const bool sulfur_i = ai == 10 || ai == 18;      // SG, SD
      const bool sulfur_j = aj == 10 || aj == 18;
      const bool disulfide = sulfur_i && sulfur_j && d < 2.5;
      // C index 2, N index 0 in the 37-atom layout
      const bool amide = (((ai == 2 && aj == 0) || (ai == 0 && aj == 2)) && d < 1.9);
      const bool bonded = disulfide || amide || (same_residue && d < 1.9);
      if (bonded && s < t) { bond_a.push_back(s); bond_b.push_back(t); bond_len.push_back(d); }
      clashing[(size_t)s * M + t] = (!neighbouring && !disulfide && !amide) ? 1 : 0;
      closest[(size_t)s * M + t] =
          std::max(radius[(size_t)s] + radius[(size_t)t] - p.overlap_tol, p.min_sep);
    }
  }

  // ---- Adam, as optax.adam runs it -------------------------------------
  //
  // b1 0.9, b2 0.999, eps 1e-8, WITH bias correction -- the first fifty of
  // two hundred steps are the ones the correction changes, and dropping it
  // makes the early steps a twentieth of their size.
  const double b1 = 0.9, b2 = 0.999, eps = 1e-8;
  std::vector<double> m((size_t)M * 3, 0.0), v((size_t)M * 3, 0.0),
      g((size_t)M * 3, 0.0);
  for (int step = 1; step <= p.steps; ++step) {
    std::fill(g.begin(), g.end(), 0.0);
    // clash.  The sum runs over ORDERED pairs and carries a 0.5, so the
    // gradient on one atom collects both orders: the factor below is
    // 0.5 * 2 (the square) * 2 (both orders) = 2.
    for (int s = 0; s < M; ++s) {
      for (int t = 0; t < M; ++t) {
        if (s == t || !clashing[(size_t)s * M + t]) continue;
        double q = 0; double d3[3];
        for (int c = 0; c < 3; ++c) {
          d3[c] = x[(size_t)s * 3 + c] - x[(size_t)t * 3 + c];
          q += d3[c] * d3[c];
        }
        const double d = std::sqrt(q + 1e-6);
        const double r = closest[(size_t)s * M + t] - d;
        if (r <= 0) continue;
        const double k = -2.0 * p.weight_clash * r / d;
        for (int c = 0; c < 3; ++c) g[(size_t)s * 3 + c] += k * d3[c];
      }
    }
    // bond
    for (size_t b = 0; b < bond_a.size(); ++b) {
      const int s = bond_a[b], t = bond_b[b];
      double q = 0; double d3[3];
      for (int c = 0; c < 3; ++c) {
        d3[c] = x[(size_t)s * 3 + c] - x[(size_t)t * 3 + c];
        q += d3[c] * d3[c];
      }
      const double d = std::sqrt(q + 1e-6);
      const double k = 2.0 * p.weight_bond * (d - bond_len[b]) / d;
      for (int c = 0; c < 3; ++c) {
        g[(size_t)s * 3 + c] += k * d3[c];
        g[(size_t)t * 3 + c] -= k * d3[c];
      }
    }
    // restraint
    for (int s = 0; s < M; ++s)
      for (int c = 0; c < 3; ++c)
        g[(size_t)s * 3 + c] +=
            2.0 * pw[(size_t)s] * (x[(size_t)s * 3 + c] - x0[(size_t)s * 3 + c]);
    const double c1 = 1.0 - std::pow(b1, (double)step);
    const double c2 = 1.0 - std::pow(b2, (double)step);
    for (size_t k = 0; k < x.size(); ++k) {
      m[k] = b1 * m[k] + (1.0 - b1) * g[k];
      v[k] = b2 * v[k] + (1.0 - b2) * g[k] * g[k];
      x[k] -= p.learning_rate * (m[k] / c1) / (std::sqrt(v[k] / c2) + eps);
    }
  }

  // ---- back into the 37-atom layout, resolved slots only ---------------
  for (int s = 0; s < M; ++s) {
    const int i = slot_residue[s], a = slot_atom[s];
    for (int c = 0; c < 3; ++c)
      atoms[((size_t)i * 37 + a) * 3 + c] = (float)x[(size_t)s * 3 + c];
  }
}

}  // namespace relaxer
