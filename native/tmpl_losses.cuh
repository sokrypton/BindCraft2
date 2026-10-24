// THE THREE LOSSES THAT COMPARE THE FOLD WITH ITS INPUT.
//
//   target_rmsd      has the target moved from its crystal coordinates?
//   target_rigidity  the same question asked ONLY of pairs that cross a
//                    receptor chain break -- "did the two chains move
//                    relative to each other", not "did either one bend"
//   com_distance     is the binder's centroid within 0.9 binder-radii of
//                    the target's?
//
// The first two are one function, bc2's template_distance_deviation, under
// two pair masks.  That is the whole difference, and writing them as two
// transcriptions would gate one reading of loss.py twice.
//
// All three read CA only and are differentiated here on the host, like the
// termini geometry in geom_losses.cuh; the cotangent is scattered into
// c_atoms at the CA slots.  The TEMPLATE side takes no gradient -- it is an
// input, not an output -- which is also why the gate had to learn the
// difference between protein_states and predictions before any of this
// could be checked.
#pragma once
#include <cmath>
#include <vector>
#include "geom_losses.cuh"

namespace tmplloss {

using geomloss::V3;
using geomloss::ca;
using geomloss::dot;

// bc2 passes eps=1e-8 explicitly here, NOT pairwise_atom_distances' 1e-4
// default.  sqrt(1e-8) is 1e-4 and sqrt(1e-4) is 1e-2, so the diagonal --
// which the pair mask keeps -- differs by two orders of magnitude between
// them and the mean carries it.
inline double pdist(const V3& a, const V3& b) {
  const V3 d = a - b;
  return sqrt(dot(d, d) + 1e-8);
}

// template_distance_deviation.
//
//   valid      = template CA mask * predicted CA mask     (per row)
//   pair_mask  = selected * valid[:,None] * valid[None,:]
//   err        = pdist(pred) - pdist(tmpl)
//   loss       = sqrt(masked_mean(err^2, pair_mask) + 1e-8) * (pairs > 0)
//
// `rows` selects the chain; `selected` is the extra pair predicate
// (nullptr = bc2's scalar 1.0, i.e. every pair including the diagonal --
// where both distances are 1e-4 and the error is exactly zero), laid out
// with row stride `sel_stride`.  THE STRIDE IS AN ARGUMENT because every
// other pair mask in the driver is sized by the PADDED length and this
// function walks the real one; reading an NB-strided mask at stride n
// silently selects a shifted set of pairs, which is a number that looks
// like an answer (0.320 against bc2's 0.340).
struct Deviation {
  double value = 0;
  double qsum = 0, psum = 0;     // the masked sum and the mask's weight
};

inline Deviation deviation(const std::vector<float>& pat,
                           const std::vector<float>& pam,
                           const std::vector<float>& tat,
                           const std::vector<float>& tam,
                           const std::vector<char>& rows, int n,
                           const std::vector<char>* selected, int sel_stride,
                           std::vector<float>* grad, double s) {
  Deviation o;
  std::vector<int> idx;
  idx.reserve(n);
  for (int i = 0; i < n; ++i)
    if (rows[i] && tam[(size_t)i * 37 + 1] > 0 && pam[(size_t)i * 37 + 1] > 0)
      idx.push_back(i);
  const int k = (int)idx.size();
  if (k == 0) return o;
  // one pass for the value; the backward needs the finished value, so the
  // pair distances are recomputed rather than stored (k is the target's
  // length, a few hundred, and k^2 doubles would be the larger cost)
  double q = 0, p = 0;
  for (int a = 0; a < k; ++a)
    for (int b = 0; b < k; ++b) {
      const int i = idx[a], j = idx[b];
      if (selected && !(*selected)[(size_t)i * sel_stride + j]) continue;
      const double e = pdist(ca(pat, i), ca(pat, j)) -
                       pdist(ca(tat, i), ca(tat, j));
      q += e * e;
      p += 1.0;
    }
  if (p == 0) return o;                      // (pair_mask.sum() > 0) is 0
  o.qsum = q; o.psum = p;
  const double mean = q / (p + 1e-8);
  o.value = sqrt(mean + 1e-8);
  if (!grad || s == 0 || o.value <= 0) return o;
  // dL/d(err_ij) = err_ij / ((p + 1e-8) * L)      [the 2's cancel]
  const double c = 1.0 / ((p + 1e-8) * o.value) * s;
  for (int a = 0; a < k; ++a)
    for (int b = 0; b < k; ++b) {
      const int i = idx[a], j = idx[b];
      if (selected && !(*selected)[(size_t)i * sel_stride + j]) continue;
      if (i == j) continue;                  // d/dx of a self-distance is 0
      const V3 xi = ca(pat, i), xj = ca(pat, j);
      const double dp = pdist(xi, xj);
      const double e = dp - pdist(ca(tat, i), ca(tat, j));
      const V3 u = (xi - xj) * (1.0 / dp);
      const V3 g = u * (c * e);
      const size_t oi = ((size_t)i * 37 + 1) * 3;
      const size_t oj = ((size_t)j * 37 + 1) * 3;
      (*grad)[oi + 0] += (float)g.x;
      (*grad)[oi + 1] += (float)g.y;
      (*grad)[oi + 2] += (float)g.z;
      (*grad)[oj + 0] -= (float)g.x;
      (*grad)[oj + 1] -= (float)g.y;
      (*grad)[oj + 2] -= (float)g.z;
    }
  return o;
}

// receptor_chain_index: 0 until the residue numbering jumps by at least
// RECEPTOR_CHAIN_BREAK_GAP, then 1, and so on.  bc2 computes it over the
// TARGET protein alone, so the diff is taken within the selected rows in
// their own order -- not over the whole complex.
inline std::vector<int> receptorChainIndex(const std::vector<int32_t>& ri,
                                           const std::vector<char>& rows,
                                           int n, int gap) {
  std::vector<int> out((size_t)n, -1);
  int c = 0, prev = 0;
  bool first = true;
  for (int i = 0; i < n; ++i) {
    if (!rows[i]) continue;
    if (!first && ri[i] - prev >= gap) ++c;
    out[i] = c;
    prev = ri[i];
    first = false;
  }
  return out;
}

// com_distance:
//
//   bc        masked CA centroid of the binder chain
//   R         sqrt(masked_mean(|x - bc|^2, m) + 1e-8)      the binder radius
//   D         |bc - tc|
//   loss      relu(D - radius_ratio * R)
//
// The gradient is the easy place to lose a term: a binder row enters BOTH
// D (through bc) and R (directly AND through bc), and the two have opposite
// signs.  Pulling the centroid in shrinks D and grows nothing -- unless the
// row is also what sets the radius.
inline double comDistance(const std::vector<float>& pat,
                          const std::vector<float>& pam,
                          const std::vector<char>& brows,
                          const std::vector<char>& trows, int n,
                          double radius_ratio, std::vector<float>* grad,
                          double s) {
  double mb = 0, mt = 0;
  V3 bc, tc;
  for (int i = 0; i < n; ++i) {
    if (pam[(size_t)i * 37 + 1] <= 0) continue;
    if (brows[i]) { bc = bc + ca(pat, i); mb += 1.0; }
    if (trows[i]) { tc = tc + ca(pat, i); mt += 1.0; }
  }
  // masked_coordinate_centroid divides by mask.sum() + 1e-8, so an empty
  // chain gives the origin rather than a NaN -- and the loss is then a
  // distance from the origin, which is what bc2 computes too.
  const double invb = 1.0 / (mb + 1e-8), invt = 1.0 / (mt + 1e-8);
  bc = bc * invb;
  tc = tc * invt;
  double sq = 0;
  for (int i = 0; i < n; ++i) {
    if (!brows[i] || pam[(size_t)i * 37 + 1] <= 0) continue;
    const V3 d = ca(pat, i) - bc;
    sq += dot(d, d);
  }
  const double R = sqrt(sq * invb + 1e-8);
  const V3 dv = bc - tc;
  // jnp.linalg.norm, no epsilon
  const double D = sqrt(dot(dv, dv));
  const double val = D - radius_ratio * R;
  if (val <= 0) return 0.0;                  // relu: value AND gradient
  if (!grad || s == 0) return val;
  const V3 u = D > 0 ? dv * (1.0 / D) : V3{};
  // dL/dR = -radius_ratio ; dR/d(sq*invb) = 1/(2R)
  const double dsq = -radius_ratio / (2.0 * R) * invb;
  // |x-bc|^2 summed also depends on bc, and bc depends on every binder
  // row.  d(sum)/d(bc) = -2 * sum_i (x_i - bc), which is 1e-8*bc rather
  // than exactly zero because the centroid's denominator carries the eps.
  V3 resid;
  for (int i = 0; i < n; ++i) {
    if (!brows[i] || pam[(size_t)i * 37 + 1] <= 0) continue;
    resid = resid + (ca(pat, i) - bc);
  }
  const V3 dbc_from_r = resid * (-2.0 * dsq);
  const V3 per_binder = (u + dbc_from_r) * invb;
  const V3 per_target = u * (-invt);
  auto add = [&](int i, V3 g) {
    const size_t o = ((size_t)i * 37 + 1) * 3;
    (*grad)[o + 0] += (float)(g.x * s);
    (*grad)[o + 1] += (float)(g.y * s);
    (*grad)[o + 2] += (float)(g.z * s);
  };
  for (int i = 0; i < n; ++i) {
    if (pam[(size_t)i * 37 + 1] <= 0) continue;
    if (brows[i]) {
      const V3 d = ca(pat, i) - bc;
      add(i, per_binder + d * (2.0 * dsq));
    }
    if (trows[i]) add(i, per_target);
  }
  return val;
}

}  // namespace tmplloss
