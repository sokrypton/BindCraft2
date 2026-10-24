// The termini / centre-of-mass geometry losses, on the host.
//
// All five are functions of the predicted CA coordinates alone -- no
// distogram, no pae -- so they are computed and differentiated here and the
// cotangent is scattered into c_atoms at the CA slots.  A few hundred
// floats per step against a forward pass; the arithmetic is not the cost.
//
// They share one quantity, bc2's terminus_target_direction_cosine:
//
//   a = binder_centre - target_centre          "away" from the target
//   t = terminus_position - binder_centre      where the terminus points
//   cos = a.t / (|a||t| + eps)
//
// and differ only in what they do with it:
//
//   termini_angle      arccos(clip(cos))  * valid    both termini, midpoint
//   n_terminus_away    relu(-cos)         * valid    the N terminus
//   c_terminus_away    relu(-cos)         * valid    the C terminus
//
// `valid` is a PRODUCT OF MASKS, not a branch: if either centroid is empty
// or the terminus has no CA the loss is zero AND so is its gradient.
#pragma once
#include <cmath>
#include <cstring>
#include <vector>

namespace geomloss {

struct V3 { double x = 0, y = 0, z = 0; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline double len(V3 a) { return sqrt(dot(a, a) + 1e-8); }

// `rows` selects a chain; returns its masked CA centroid and the mask sum
inline V3 centroid(const std::vector<float>& atoms,
                   const std::vector<float>& amask,
                   const std::vector<char>& rows, int n, double* msum) {
  V3 c; double m = 0;
  for (int i = 0; i < n; ++i) {
    if (!rows[i]) continue;
    const double w = amask[(size_t)i * 37 + 1] > 0 ? 1.0 : 0.0;
    if (w == 0) continue;
    c.x += atoms[((size_t)i * 37 + 1) * 3 + 0];
    c.y += atoms[((size_t)i * 37 + 1) * 3 + 1];
    c.z += atoms[((size_t)i * 37 + 1) * 3 + 2];
    m += 1.0;
  }
  *msum = m;
  return m > 0 ? c * (1.0 / m) : V3{};
}

inline V3 ca(const std::vector<float>& atoms, int i) {
  return {atoms[((size_t)i * 37 + 1) * 3 + 0],
          atoms[((size_t)i * 37 + 1) * 3 + 1],
          atoms[((size_t)i * 37 + 1) * 3 + 2]};
}

// which == 0 n-terminus, 1 c-terminus, 2 both (the midpoint)
struct Cosine {
  double value = 0;
  double valid = 0;
  V3 a, t;
  double la = 0, lt = 0;      // len(): the eps is inside the sqrt, as bc2
  double mb = 0, mt = 0;      // the two centroids' mask sums
};

inline Cosine cosineTo(const std::vector<float>& atoms,
                       const std::vector<float>& amask,
                       const std::vector<char>& brows,
                       const std::vector<char>& trows, int n,
                       int first, int last, int which) {
  Cosine o;
  double mb = 0, mt = 0;
  const V3 bc = centroid(atoms, amask, brows, n, &mb);
  const V3 tc = centroid(atoms, amask, trows, n, &mt);
  const bool h0 = amask[(size_t)first * 37 + 1] > 0;
  const bool hc = amask[(size_t)last * 37 + 1] > 0;
  o.mb = mb; o.mt = mt;
  o.valid = (mb > 0 && mt > 0 &&
             (which == 2 ? (h0 && hc) : (which == 0 ? h0 : hc))) ? 1.0 : 0.0;
  if (!o.valid) return o;
  const V3 tp = which == 2 ? (ca(atoms, first) + ca(atoms, last)) * 0.5
                           : ca(atoms, which == 0 ? first : last);
  o.a = bc - tc;
  o.t = tp - bc;
  o.la = len(o.a); o.lt = len(o.t);
  o.value = dot(o.a, o.t) / (o.la * o.lt + 1e-8);
  return o;
}

// d(cos)/d(CA) for every row, added into `grad` (n*37*3) scaled by s.
//
// a = bc - tc and t = tp - bc, so a BINDER row moves both: it enters bc,
// which raises a and lowers t.  Writing only the `a` half is the easy
// error and leaves a gradient that is plausible and wrong.
//
//   binder row i (with CA):  (ga - gt) / Mb
//     and if i is the terminus:  + gt * (1 for that end, 0.5 when "both")
//   target row j (with CA):  -ga / Mt
inline void cosineBackward(const Cosine& c, const std::vector<float>& amask,
                           const std::vector<char>& brows,
                           const std::vector<char>& trows, int n,
                           int first, int last, int which,
                           double mb, double mt, double s,
                           std::vector<float>* grad) {
  if (!c.valid || s == 0 || mb <= 0 || mt <= 0) return;
  const double den = c.la * c.lt + 1e-8;
  // d la / d a = a / la with the eps inside the sqrt, and den carries its
  // own + 1e-8, so value * lt / (den * la) -- not value / la^2
  const V3 ga = (c.t * (1.0 / den)) - (c.a * (c.value * c.lt / (den * c.la)));
  const V3 gt = (c.a * (1.0 / den)) - (c.t * (c.value * c.la / (den * c.lt)));
  auto add = [&](int i, V3 g) {
    const size_t o = ((size_t)i * 37 + 1) * 3;
    (*grad)[o + 0] += (float)(g.x * s);
    (*grad)[o + 1] += (float)(g.y * s);
    (*grad)[o + 2] += (float)(g.z * s);
  };
  const V3 per_binder = (ga - gt) * (1.0 / mb);
  for (int i = 0; i < n; ++i) {
    if (!brows[i] || amask[(size_t)i * 37 + 1] <= 0) continue;
    add(i, per_binder);
  }
  const V3 per_target = ga * (-1.0 / mt);
  for (int j = 0; j < n; ++j) {
    if (!trows[j] || amask[(size_t)j * 37 + 1] <= 0) continue;
    add(j, per_target);
  }
  if (which == 2) {
    add(first, gt * 0.5);
    add(last, gt * 0.5);
  } else {
    add(which == 0 ? first : last, gt);
  }
}

}  // namespace geomloss
