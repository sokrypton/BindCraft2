// EXPOSURE: how solvent-accessible is a residue, and what that is for.
//
// Both losses in this file are built on bc2's binder_residue_exposure --
// a soft neighbour count turned into a soft "is it on the surface":
//
//   neighbours_i = sum_j sigmoid((10 - d_ij) / 2) * weight_j  -  1
//   exposure_i   = sigmoid((16 - neighbours_i) / 4)
//
// The -1 is the residue's own term, which is sigmoid(10/2) ~ 1 because
// pairwise_atom_distances adds eps=1e-4 under the root and a self-distance
// is therefore 1e-2, not 0.  Both constants matter and neither is round:
// using 0.0 for the self-distance changes every count by a thousandth, and
// subtracting nothing changes it by one whole neighbour.
//
//   exposed_termini  the mean exposure of the first and last three rows
//   exposed_loops    exposure x "is this a loop" x a 3-window of itself
//
// `protease_stable` is the property preset that turns both on, at 0.5
// each, with measure="plddt" -- so loopiness is read off the confidence
// head, not off the geometry.  The other two measures ("geometry" and
// "both") need the CA pseudo-dihedral and the strand propensity and are
// NOT implemented; the driver refuses them at startup rather than
// silently running the plddt branch under their name.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "geom_losses.cuh"

namespace exposeloss {

using geomloss::V3;
using geomloss::ca;
using geomloss::dot;

inline double sgm(double x) { return 1.0 / (1.0 + exp(-x)); }

struct Exposure {
  std::vector<double> value;      // per complex row (binder rows filled)
  std::vector<double> neighbours;
};

// `brows` are the rows the exposure is wanted for (the binder chain);
// `w` is complex_residue_weights -- the REAL-residue weight of every row,
// taken from the STATE's flags, not from the prediction.
inline Exposure exposure(const std::vector<float>& pat,
                         const std::vector<float>& w,
                         const std::vector<char>& brows, int n,
                         double radius = 10.0, double contact_t = 2.0,
                         double nbr_thr = 16.0, double temp = 4.0) {
  Exposure e;
  e.value.assign((size_t)n, 0.0);
  e.neighbours.assign((size_t)n, 0.0);
  for (int i = 0; i < n; ++i) {
    if (!brows[i]) continue;
    const V3 xi = ca(pat, i);
    double c = 0;
    for (int j = 0; j < n; ++j) {
      if (w[j] <= 0) continue;
      const V3 d = xi - ca(pat, j);
      // pairwise_atom_distances' DEFAULT eps, 1e-4 -- not the 1e-8 that
      // template_distance_deviation passes explicitly
      const double dij = sqrt(dot(d, d) + 1e-4);
      c += sgm((radius - dij) / contact_t) * w[j];
    }
    c -= 1.0;
    e.neighbours[i] = c;
    e.value[i] = sgm((nbr_thr - c) / temp);
  }
  return e;
}

// d(exposure_i)/d(CA) for every row, scaled by ge[i], added into `grad`.
inline void exposureBackward(const std::vector<float>& pat,
                             const std::vector<float>& w,
                             const std::vector<char>& brows, int n,
                             const Exposure& e,
                             const std::vector<double>& ge,
                             std::vector<float>* grad,
                             double radius = 10.0, double contact_t = 2.0,
                             double temp = 4.0) {
  auto add = [&](int i, V3 g) {
    const size_t o = ((size_t)i * 37 + 1) * 3;
    (*grad)[o + 0] += (float)g.x;
    (*grad)[o + 1] += (float)g.y;
    (*grad)[o + 2] += (float)g.z;
  };
  for (int i = 0; i < n; ++i) {
    if (!brows[i] || ge[i] == 0.0) continue;
    const double s = e.value[i];
    // d sigmoid((T - c)/temp) / dc = -s(1-s)/temp
    const double dnc = ge[i] * (-s * (1.0 - s) / temp);
    const V3 xi = ca(pat, i);
    for (int j = 0; j < n; ++j) {
      if (w[j] <= 0) continue;
      const V3 dv = xi - ca(pat, j);
      const double dij = sqrt(dot(dv, dv) + 1e-4);
      const double t = sgm((radius - dij) / contact_t);
      // d(sigmoid((R - d)/ct) * w_j)/dd = -t(1-t)/ct * w_j
      const double dd = dnc * (-t * (1.0 - t) / contact_t) * (double)w[j];
      if (dd == 0.0) continue;
      const V3 u = dv * (1.0 / dij);
      add(i, u * dd);
      add(j, u * (-dd));
    }
  }
}

// jnp.convolve(x, ones(k)/k, mode='same'), for EVEN k as well as odd.
//
// numpy's 'same' is the middle m of the full convolution (length m+k-1),
// starting at (k-1)//2 -- so for even k the window is NOT centred: it
// reaches further back than forward.  The general rule is
//
//     y[i] = (1/k) * sum_{j=0}^{k-1} x[i + (k-1)/2 - j]
//
// which for k=3 is the centred window this file already had and for k=4
// is x[i-2..i+1].  The geometry measure smooths with k=4, so the odd-only
// version would have been quietly wrong by one position.
inline std::vector<double> convSame(const std::vector<double>& x, int k) {
  const int m = (int)x.size();
  const int off = (k - 1) / 2;
  std::vector<double> y((size_t)m, 0.0);
  for (int i = 0; i < m; ++i) {
    double a = 0;
    for (int j = 0; j < k; ++j) {
      const int q = i + off - j;
      if (q >= 0 && q < m) a += x[(size_t)q];
    }
    y[(size_t)i] = a / (double)k;
  }
  return y;
}

// and its transpose: d(sum_i g_i y_i)/dx
inline std::vector<double> convSameBwd(const std::vector<double>& g, int k) {
  const int m = (int)g.size();
  const int off = (k - 1) / 2;
  std::vector<double> gx((size_t)m, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      const int q = i + off - j;
      if (q >= 0 && q < m) gx[(size_t)q] += g[(size_t)i] / (double)k;
    }
  }
  return gx;
}

// ---- THE GEOMETRY MEASURES ---------------------------------------------
//
// `exposed_loops_measure` is "plddt" in every preset bc2 ships, and the
// other two read the BACKBONE instead of the confidence head:
//
//   geometry   loopiness = 1 - structured, where structured is a smoothed
//              soft "is residue i within 7 A of residue i+3" -- the i,i+3
//              distance is what separates a helix or a turn from an
//              extended run.
//   both       the elementwise MAXIMUM of that and the plddt one.
//
// and `exposed_loops_distinguish_sheets` (TRUE by default) adds the other
// half of "structured": a strand is extended, so its i..i+3 distance is
// large and the distance measure alone calls a beta strand a loop.  The
// strand propensity is read off the CA PSEUDO-DIHEDRAL -- -cos(phi) is
// near 1 for an extended chain -- and maxed in.
inline V3 cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

// bc2's binder_ca_pseudo_dihedral, one window.  Written the way bc2
// writes it (project b1 and b3 off b2_hat, then atan2) rather than as the
// textbook torsion, because the two differ in their eps handling and the
// gradient has to match the one jax takes through THIS expression.
inline double pseudoDihedral(V3 p0, V3 p1, V3 p2, V3 p3, double eps = 1e-8) {
  const V3 b1 = p1 - p0, b2 = p2 - p1, b3 = p3 - p2;
  const double n2 = sqrt(dot(b2, b2));
  const V3 b2h = b2 * (1.0 / (n2 + eps));
  const V3 v = b1 - b2h * dot(b1, b2h);
  const V3 w = b3 - b2h * dot(b3, b2h);
  return atan2(dot(cross(b2h, v), w), dot(v, w));
}

// d(phi)/d(p0..p3) * gphi, accumulated into out[0..3]
inline void pseudoDihedralBwd(V3 p0, V3 p1, V3 p2, V3 p3, double gphi,
                              V3 out[4], double eps = 1e-8) {
  const V3 b1 = p1 - p0, b2 = p2 - p1, b3 = p3 - p2;
  const double n2 = sqrt(dot(b2, b2));
  const double inv = 1.0 / (n2 + eps);
  const V3 b2h = b2 * inv;
  const double d1 = dot(b1, b2h), d3 = dot(b3, b2h);
  const V3 v = b1 - b2h * d1, w = b3 - b2h * d3;
  const double X = dot(v, w), Y = dot(cross(b2h, v), w);
  const double den = X * X + Y * Y + 1e-300;
  const double gY = gphi * X / den, gX = gphi * (-Y) / den;
  // Y = b2h . (v x w)
  V3 gv = cross(w, b2h) * gY + w * gX;      // dY/dv = w x b2h ; dX/dv = w
  V3 gw = cross(b2h, v) * gY + v * gX;      // dY/dw = b2h x v ; dX/dw = v
  V3 gb2h = cross(v, w) * gY;               // dY/db2h = v x w
  // v = b1 - b2h * (b1 . b2h)
  V3 gb1 = gv - b2h * dot(gv, b2h);
  gb2h = gb2h - (gv * d1 + b2h * dot(gv, b1));
  // w = b3 - b2h * (b3 . b2h)
  V3 gb3 = gw - b2h * dot(gw, b2h);
  gb2h = gb2h - (gw * d3 + b2h * dot(gw, b3));
  // b2h = b2 / (|b2| + eps)
  V3 gb2 = gb2h * inv - b2 * (dot(gb2h, b2) * inv * inv / (n2 + 1e-300));
  // b1 = p1 - p0, b2 = p2 - p1, b3 = p3 - p2
  out[0] = out[0] - gb1;
  out[1] = out[1] + gb1 - gb2;
  out[2] = out[2] + gb2 - gb3;
  out[3] = out[3] + gb3;
}

// bc2's binder_strand_propensity, over a chain's CA coordinates
inline std::vector<double> strandPropensity(const std::vector<V3>& ca,
                                            const std::vector<double>& w,
                                            double ext_cut = 0.3,
                                            double temp = 0.2,
                                            int smoothing = 4,
                                            std::vector<double>* raw = nullptr) {
  const int L = (int)ca.size();
  if (L < 4) return std::vector<double>((size_t)L, 0.0);
  std::vector<double> prob((size_t)L, 0.0);     // L-3 entries then 3 zeros
  for (int t = 0; t + 3 < L; ++t) {
    const double ext = -cos(pseudoDihedral(ca[t], ca[t + 1], ca[t + 2], ca[t + 3]));
    prob[(size_t)t] = sgm((ext - ext_cut) / temp) * w[(size_t)t] * w[(size_t)t + 3];
  }
  if (raw) *raw = prob;
  std::vector<double> y = convSame(prob, smoothing);
  for (double& q : y) q = q < 0 ? 0 : (q > 1 ? 1 : q);
  return y;
}

// bc2's binder_loop_geometry_propensity: 1 - structured
inline std::vector<double> loopGeometry(const std::vector<V3>& ca,
                                        const std::vector<double>& w,
                                        bool distinguish_sheets,
                                        double cutoff = 7.0,
                                        double temp = 0.8,
                                        int smoothing = 4,
                                        double strand_cut = 0.3,
                                        double strand_temp = 0.2) {
  const int L = (int)ca.size();
  if (L < 4) return std::vector<double>((size_t)L, 1.0);
  std::vector<double> close((size_t)L, 0.0);
  for (int t = 0; t + 3 < L; ++t) {
    const V3 d = ca[(size_t)t] - ca[(size_t)t + 3];
    const double dist = sqrt(dot(d, d) + 1e-8);
    close[(size_t)t] = sgm((cutoff - dist) / temp) * w[(size_t)t] * w[(size_t)t + 3];
  }
  std::vector<double> structured = convSame(close, smoothing);
  for (double& q : structured) q = q < 0 ? 0 : (q > 1 ? 1 : q);
  if (distinguish_sheets) {
    const std::vector<double> strand =
        strandPropensity(ca, w, strand_cut, strand_temp, smoothing);
    for (int i = 0; i < L; ++i)
      structured[(size_t)i] = std::max(structured[(size_t)i], strand[(size_t)i]);
  }
  std::vector<double> out((size_t)L, 0.0);
  for (int i = 0; i < L; ++i) out[(size_t)i] = 1.0 - structured[(size_t)i];
  return out;
}

// d(sum_i gl_i * loopiness_i)/d(CA), accumulated into `grad` at `rows`
inline void loopGeometryBwd(const std::vector<V3>& ca,
                            const std::vector<double>& w,
                            const std::vector<double>& gl,
                            bool distinguish_sheets,
                            std::vector<V3>* grad,
                            double cutoff = 7.0, double temp = 0.8,
                            int smoothing = 4, double strand_cut = 0.3,
                            double strand_temp = 0.2) {
  const int L = (int)ca.size();
  if (L < 4) return;
  // forward again, keeping what the backward needs
  std::vector<double> close((size_t)L, 0.0);
  for (int t = 0; t + 3 < L; ++t) {
    const V3 d = ca[(size_t)t] - ca[(size_t)t + 3];
    const double dist = sqrt(dot(d, d) + 1e-8);
    close[(size_t)t] = sgm((cutoff - dist) / temp) * w[(size_t)t] * w[(size_t)t + 3];
  }
  std::vector<double> conv = convSame(close, smoothing);
  std::vector<double> clipped = conv;
  for (double& q : clipped) q = q < 0 ? 0 : (q > 1 ? 1 : q);
  std::vector<double> strand_raw, strand;
  if (distinguish_sheets)
    strand = strandPropensity(ca, w, strand_cut, strand_temp, smoothing,
                              &strand_raw);
  // loopiness = 1 - max(clipped, strand)  ->  d/d(the winner) = -gl
  std::vector<double> gclip((size_t)L, 0.0), gstrand((size_t)L, 0.0);
  for (int i = 0; i < L; ++i) {
    const bool sheet_wins = distinguish_sheets && strand[(size_t)i] > clipped[(size_t)i];
    (sheet_wins ? gstrand : gclip)[(size_t)i] = -gl[(size_t)i];
  }
  // through the clip, then the convolution
  for (int i = 0; i < L; ++i)
    if (conv[(size_t)i] < 0.0 || conv[(size_t)i] > 1.0) gclip[(size_t)i] = 0.0;
  const std::vector<double> gclose = convSameBwd(gclip, smoothing);
  for (int t = 0; t + 3 < L; ++t) {
    const double ww = w[(size_t)t] * w[(size_t)t + 3];
    if (ww == 0.0 || gclose[(size_t)t] == 0.0) continue;
    const V3 d = ca[(size_t)t] - ca[(size_t)t + 3];
    const double dist = sqrt(dot(d, d) + 1e-8);
    const double sg = sgm((cutoff - dist) / temp);
    const double gd = gclose[(size_t)t] * ww * sg * (1.0 - sg) * (-1.0 / temp);
    const V3 u = d * (gd / dist);
    (*grad)[(size_t)t] = (*grad)[(size_t)t] + u;
    (*grad)[(size_t)t + 3] = (*grad)[(size_t)t + 3] - u;
  }
  if (!distinguish_sheets) return;
  // and the strand branch, through its own clip, convolution and dihedral
  std::vector<double> sconv = convSame(strand_raw, smoothing);
  for (int i = 0; i < L; ++i)
    if (sconv[(size_t)i] < 0.0 || sconv[(size_t)i] > 1.0) gstrand[(size_t)i] = 0.0;
  const std::vector<double> gprob = convSameBwd(gstrand, smoothing);
  for (int t = 0; t + 3 < L; ++t) {
    const double ww = w[(size_t)t] * w[(size_t)t + 3];
    if (ww == 0.0 || gprob[(size_t)t] == 0.0) continue;
    const double phi = pseudoDihedral(ca[t], ca[t + 1], ca[t + 2], ca[t + 3]);
    const double ext = -cos(phi);
    const double sg = sgm((ext - strand_cut) / strand_temp);
    const double gext = gprob[(size_t)t] * ww * sg * (1.0 - sg) / strand_temp;
    const double gphi = gext * sin(phi);        // d(-cos phi)/d phi = sin phi
    V3 out[4] = {V3(), V3(), V3(), V3()};
    pseudoDihedralBwd(ca[t], ca[t + 1], ca[t + 2], ca[t + 3], gphi, out);
    for (int q = 0; q < 4; ++q)
      (*grad)[(size_t)t + q] = (*grad)[(size_t)t + q] + out[q];
  }
}

}  // namespace exposeloss
