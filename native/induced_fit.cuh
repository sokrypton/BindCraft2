// induced_fit_global and induced_fit_interface: how much does the binder
// CHANGE when it binds?
//
// Both compare the binder's rows in the COMPLEX against the same binder's
// rows in its own BINDER_ALONE fold -- two predictions, not a prediction
// against a template -- and both are "reward a change", so they are relu'd
// the other way round from the template losses:
//
//   induced_fit_global     relu(TM(bound, unbound) - 0.6)^2
//   induced_fit_interface  relu(3.0 - interface RMSD)^2
//
// THE ROTATION IS STOP_GRADIENT'D, which is the whole reason these are
// tractable on the host.  bc2's align_binder_coordinates differentiates
// through `coordinates - center` and not through the Kabsch rotation, so
// no derivative of an SVD is needed; the centroid still carries one.
//
// They differ in which rows do what.  global aligns on and scores the same
// set.  interface aligns on the rows that do NOT face the target and
// scores the ones that do -- "hold the core still and ask how far the
// interface moved", which is a different question from "how far did
// everything move" and gives a different number on the same two folds.
#pragma once
#include <cmath>
#include <vector>
#include "geom_losses.cuh"
#include "kabsch.cuh"

namespace indfit {

using geomloss::V3;

struct Out {
  double value = 0;
  double raw = 0;             // the TM or the RMSD, before relu and square
  std::vector<float> dx;        // d loss / d bound CA      [n*3]
  std::vector<float> dr;        // d loss / d unbound CA    [n*3]
  // d loss / d score_i and d loss / d align_i, when asked (want_wgrad).
  // 1.0.4 weights these masks by CONFIDENCE (`confident_binder_comparison`
  // multiplies the valid mask by relu(min pLDDT - floor) / (1 - floor)),
  // and nothing stop_gradients it -- so the masks are differentiable
  // inputs now, and their cotangent is how the loss reaches the pLDDT
  // head of both predictions.
  std::vector<double> ds, da;
};

// `score` is the mask the deviation is averaged over, `align` the mask the
// superposition is computed from.  They are the same array for global and
// complementary for interface.
// `scored_length`: 1.0.4's `aligned_binder_tm_score(..., scored_length)`.
// TM-score's distance scale is defined on the length of the FOLD, and with
// a confidence-weighted mask the weight sum is not that length -- reading
// the scale off it shrinks it toward its 0.5 A floor and calls sub-Angstrom
// noise a different fold.  Negative keeps the old rule (the mask's sum).
inline Out compare(const std::vector<double>& x,      // bound CA   [n*3]
                   const std::vector<double>& r,      // unbound CA [n*3]
                   const std::vector<double>& score,
                   const std::vector<double>& align,
                   int n, bool tm_mode, double target,
                   bool want_grad, bool plain_rmsd = false,
                   double scored_length = -1.0, bool want_wgrad = false) {
  Out o;
  double ALs = 0, SCs = 0;
  for (int i = 0; i < n; ++i) { ALs += align[(size_t)i]; SCs += score[(size_t)i]; }
  if (ALs < 3 || SCs < 3) return o;             // bc2's (mask.sum() >= 3)
  const double alden = ALs + 1e-8, scden = SCs + 1e-8;
  V3 c, rc;
  for (int i = 0; i < n; ++i) {
    const double w = align[(size_t)i];
    if (w == 0) continue;
    c.x += w * x[(size_t)i * 3]; c.y += w * x[(size_t)i * 3 + 1];
    c.z += w * x[(size_t)i * 3 + 2];
    rc.x += w * r[(size_t)i * 3]; rc.y += w * r[(size_t)i * 3 + 1];
    rc.z += w * r[(size_t)i * 3 + 2];
  }
  c = c * (1.0 / alden); rc = rc * (1.0 / alden);
  float cov[9] = {0,0,0,0,0,0,0,0,0};
  for (int i = 0; i < n; ++i) {
    const double w = align[(size_t)i];
    if (w == 0) continue;
    const double px[3] = {x[(size_t)i*3] - c.x, x[(size_t)i*3+1] - c.y,
                          x[(size_t)i*3+2] - c.z};
    const double qx[3] = {r[(size_t)i*3] - rc.x, r[(size_t)i*3+1] - rc.y,
                          r[(size_t)i*3+2] - rc.z};
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        cov[a * 3 + b] += (float)(w * px[a] * qx[b]);
  }
  float R[9];
  kabsch::rotationFromCov(cov, R);
  // aligned_i = R (x_i - c) + rc, written as v @ R^T like binderRmsd
  std::vector<double> dev((size_t)n * 3, 0.0), d2((size_t)n, 0.0);
  for (int i = 0; i < n; ++i) {
    if (score[(size_t)i] == 0 && align[(size_t)i] == 0) continue;
    double v[3] = {x[(size_t)i*3] - c.x, x[(size_t)i*3+1] - c.y,
                   x[(size_t)i*3+2] - c.z};
    double q = 0;
    for (int d = 0; d < 3; ++d) {
      double acc = 0;
      for (int e = 0; e < 3; ++e) acc += v[e] * (double)R[d * 3 + e];
      const double D = acc + (d == 0 ? rc.x : d == 1 ? rc.y : rc.z)
                     - r[(size_t)i * 3 + d];
      dev[(size_t)i * 3 + d] = D;
      q += D * D;
    }
    d2[(size_t)i] = q;
  }
  // dL/d(dev_i), before the rotation and the centroid are undone
  std::vector<double> a((size_t)n * 3, 0.0);
  if (tm_mode) {
    // distance_scale depends on the MASK SUM alone, so it is a constant
    // here -- differentiating it would be differentiating a residue count
    const double length = scored_length >= 0 ? scored_length : SCs;
    const double s = std::max(1.24 * pow(std::max(length - 15.0, 1e-8), 1.0 / 3.0)
                              - 1.8, 0.5);
    const double s2 = s * s;
    double tm = 0;
    std::vector<double> f((size_t)n, 0.0);
    for (int i = 0; i < n; ++i) {
      if (score[(size_t)i] == 0 && !(want_wgrad && align[(size_t)i] != 0)) continue;
      f[(size_t)i] = 1.0 / (1.0 + (d2[(size_t)i] + 1e-8) / s2);
      tm += score[(size_t)i] * f[(size_t)i];
    }
    tm /= scden;
    o.raw = tm;
    const double rel = std::max(0.0, tm - target);
    o.value = rel * rel;
    if (want_wgrad && rel > 0) {
      // d tm / d score_k = (f_k - tm) / scden
      o.ds.assign((size_t)n, 0.0);
      for (int k = 0; k < n; ++k)
        o.ds[(size_t)k] = 2.0 * rel * (f[(size_t)k] - tm) / scden;
    }
    if (want_grad && rel > 0) {
      const double dLdtm = 2.0 * rel;
      for (int i = 0; i < n; ++i) {
        if (score[(size_t)i] == 0) continue;
        // d f / d(d2) = -f^2 / s^2
        const double k = dLdtm * score[(size_t)i] / scden *
                         (-f[(size_t)i] * f[(size_t)i] / s2) * 2.0;
        for (int d = 0; d < 3; ++d)
          a[(size_t)i * 3 + d] = k * dev[(size_t)i * 3 + d];
      }
    }
  } else {
    double q = 0;
    for (int i = 0; i < n; ++i) q += score[(size_t)i] * d2[(size_t)i];
    // Induced_Fit_RMSD divides by the mask sum with NO epsilon and takes
    // no 1e-8 under the root; core_aligned_interface_rmsd does both.
    const double rmsd = plain_rmsd ? sqrt(q / SCs) : sqrt(q / scden + 1e-8);
    o.raw = rmsd;
    const double rel = std::max(0.0, target - rmsd);
    o.value = rel * rel;
    if (want_wgrad && rel > 0 && rmsd > 0) {
      // d rmsd / d score_k = (d2_k - q / den) / (2 rmsd den)
      const double den = plain_rmsd ? SCs : scden;
      o.ds.assign((size_t)n, 0.0);
      for (int k = 0; k < n; ++k)
        o.ds[(size_t)k] = -2.0 * rel * (d2[(size_t)k] - q / den) / (2.0 * rmsd * den);
    }
    if (want_grad && rel > 0 && rmsd > 0) {
      const double dLdr = -2.0 * rel;
      for (int i = 0; i < n; ++i) {
        if (score[(size_t)i] == 0) continue;
        const double k = dLdr * score[(size_t)i] / scden / (2.0 * rmsd) * 2.0;
        for (int d = 0; d < 3; ++d)
          a[(size_t)i * 3 + d] = k * dev[(size_t)i * 3 + d];
      }
    }
  }
  if (want_wgrad && !o.ds.empty()) {
    double sa[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i)
      for (int d = 0; d < 3; ++d) sa[d] += a[(size_t)i * 3 + d];
    // a is only filled when want_grad; the weight path asks for both
    o.da.assign((size_t)n, 0.0);
    for (int k = 0; k < n; ++k) {
      if (align[(size_t)k] == 0 && score[(size_t)k] == 0) continue;
      double dot = 0;
      for (int d = 0; d < 3; ++d) dot += sa[d] * dev[(size_t)k * 3 + d];
      o.da[(size_t)k] = -dot / alden;
    }
  }
  if (!want_grad) return o;
  // undo the rotation and the two centroids.
  //   d/dx_j = R^T ( a_j - align_j/ALden * sum_i a_i )
  //   d/dr_i = -a_i + align_i/ALden * sum_j a_j
  double sum[3] = {0, 0, 0};
  for (int i = 0; i < n; ++i)
    for (int d = 0; d < 3; ++d) sum[d] += a[(size_t)i * 3 + d];
  o.dx.assign((size_t)n * 3, 0.f);
  o.dr.assign((size_t)n * 3, 0.f);
  for (int i = 0; i < n; ++i) {
    double t[3];
    for (int d = 0; d < 3; ++d)
      t[d] = a[(size_t)i * 3 + d] - align[(size_t)i] / alden * sum[d];
    for (int e = 0; e < 3; ++e) {
      double acc = 0;
      for (int d = 0; d < 3; ++d) acc += t[d] * (double)R[d * 3 + e];
      o.dx[(size_t)i * 3 + e] = (float)acc;
    }
    for (int d = 0; d < 3; ++d)
      o.dr[(size_t)i * 3 + d] =
          (float)(-a[(size_t)i * 3 + d] + align[(size_t)i] / alden * sum[d]);
  }
  return o;
}

// THE SAME SUPERPOSITION, READ AS A METRIC.  Three of bc2's filters --
// Induced_Fit_TM, Induced_Fit_RMSD and Induced_Fit_Interface_RMSD -- are
// the quantities `compare` computes BEFORE the relu and the square, so
// `Out::raw` carries them out rather than a second transcription of the
// superposition doing it again.  The fold_switch modality makes
// Induced_Fit_TM mandatory at 0.6, so this is not a spare metric: it is
// the check that decides whether a fold-switch design is accepted.
//
// Induced_Fit_RMSD is NOT the interface RMSD with a full mask: it divides
// by valid_mask.sum() with no epsilon and takes no 1e-8 under the root,
// where the interface one does both.  Two RMSDs, two formulas, and
// `plain_rmsd` is which one.
struct Scores { double tm = 0, rmsd = 0, iface_rmsd = 0;
                bool has_global = false, has_iface = false; };

// 1.0.4: the three metrics read the CONFIDENCE-weighted mask `w` (see
// `confident_binder_comparison`), and the TM reads its distance scale off
// the length of the whole fold (`scored_length` = the unweighted valid
// count).  The interface masks are built from w the way
// induced_fit_interface_masks builds them: iface = facing * w,
// align = (1 - iface) * w.
inline Scores scores(const std::vector<double>& x,
                     const std::vector<double>& r,
                     const std::vector<double>& valid,
                     const std::vector<double>& w,
                     const std::vector<double>& facing, int n) {
  Scores o;
  double vs = 0, ws = 0, is = 0, as = 0;
  std::vector<double> iface((size_t)n), align((size_t)n);
  for (int i = 0; i < n; ++i) {
    vs += valid[(size_t)i]; ws += w[(size_t)i];
    iface[(size_t)i] = facing[(size_t)i] * w[(size_t)i];
    align[(size_t)i] = (1.0 - iface[(size_t)i]) * w[(size_t)i];
    is += iface[(size_t)i]; as += align[(size_t)i];
  }
  if (ws >= 3) {
    o.tm = compare(x, r, w, w, n, true, 0.0, false, false, vs).raw;
    o.rmsd = compare(x, r, w, w, n, false, 0.0, false,
                     /*plain_rmsd*/ true).raw;
    o.has_global = true;
  }
  if (is >= 3 && as >= 3) {
    o.iface_rmsd = compare(x, r, iface, align, n, false, 0.0, false).raw;
    o.has_iface = true;
  }
  return o;
}

// the confidence weight itself, one place for the loss and the filters:
//   valid * relu(min(pLDDT_bound, pLDDT_free) - floor) / (1 - floor + 1e-8)
inline std::vector<double> confidenceWeights(const std::vector<double>& valid,
                                             const std::vector<double>& pc,
                                             const std::vector<double>& pb,
                                             double floor) {
  std::vector<double> w(valid.size(), 0.0);
  for (size_t i = 0; i < valid.size(); ++i)
    w[i] = valid[i] * std::max(0.0, std::min(pc[i], pb[i]) - floor) /
           (1.0 - floor + 1e-8);
  return w;
}

}  // namespace indfit
