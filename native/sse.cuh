// SECONDARY STRUCTURE: P-SEA, because that is what bc2 uses.
//
// The beta-sheet re-optimisation trigger (`betasheet_reopt_trigger`, 0.15
// by default) fires on the binder's SHEET FRACTION, and bc2 gets that from
// `biotite.structure.annotate_sse` -- the P-SEA algorithm of Labesse et al.
// 1997, which reads CA COORDINATES ONLY: three distances (i to i+2, i+3,
// i+4), one angle and one dihedral.
//
// NOT DSSP, and the difference is not cosmetic.  DSSP (and PyDSSP, its
// simplified re-implementation) assigns from BACKBONE HYDROGEN BONDS --
// the Kabsch-Sander electrostatic energy over N, CA, C, O and an imputed
// H.  The two disagree on real structures, especially at the ends of
// elements and on distorted or low-confidence geometry, which is exactly
// what a half-finished design is.  A DSSP-style assignment here would be a
// better algorithm answering a different question: the trigger would fire
// at a different fraction than bc2's, and the extra refine/anneal rounds
// would be spent on different trajectories.
//
// So this is P-SEA, transcribed from biotite/structure/sse.py, and
// sse_gate.py diffs it against biotite residue by residue.  If a
// DSSP-style assignment is ever wanted it belongs BESIDE this one, under
// its own name, not in place of it.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace sse {

// the P-SEA windows, in radians and angstroms
static const double R_HELIX[2]  = {(89 - 12) * M_PI / 180.0, (89 + 12) * M_PI / 180.0};
static const double A_HELIX[2]  = {(50 - 20) * M_PI / 180.0, (50 + 20) * M_PI / 180.0};
static const double D3_HELIX[2] = {5.3 - 0.5, 5.3 + 0.5};
static const double D4_HELIX[2] = {6.4 - 0.6, 6.4 + 0.6};

static const double R_STRAND[2]  = {(124 - 14) * M_PI / 180.0, (124 + 14) * M_PI / 180.0};
// the dihedral window WRAPS: [-180, -125] and [145, 180]
static const double A_STRAND[4]  = {-M_PI, -125 * M_PI / 180.0,
                                    145 * M_PI / 180.0, M_PI};
static const double D2_STRAND[2] = {6.7 - 0.6, 6.7 + 0.6};
static const double D3_STRAND[2] = {9.9 - 0.9, 9.9 + 0.9};
static const double D4_STRAND[2] = {12.4 - 1.1, 12.4 + 1.1};

inline double dist(const float* a, const float* b) {
  const double x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
  return std::sqrt(x * x + y * y + z * z);
}

inline double angleAt(const float* a, const float* b, const float* c) {
  double u[3], v[3];
  for (int i = 0; i < 3; ++i) { u[i] = a[i] - b[i]; v[i] = c[i] - b[i]; }
  const double du = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
  const double dv = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (du == 0 || dv == 0) return NAN;
  double d = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) / (du * dv);
  d = d > 1.0 ? 1.0 : (d < -1.0 ? -1.0 : d);
  return std::acos(d);
}

// the signed torsion a-b-c-d, in (-pi, pi]
inline double dihedral(const float* a, const float* b, const float* c,
                       const float* d) {
  double b1[3], b2[3], b3[3];
  for (int i = 0; i < 3; ++i) {
    b1[i] = b[i] - a[i]; b2[i] = c[i] - b[i]; b3[i] = d[i] - c[i];
  }
  auto cross = [](const double* p, const double* q, double* o) {
    o[0] = p[1]*q[2] - p[2]*q[1];
    o[1] = p[2]*q[0] - p[0]*q[2];
    o[2] = p[0]*q[1] - p[1]*q[0];
  };
  double n1[3], n2[3], m[3];
  cross(b1, b2, n1);
  cross(b2, b3, n2);
  const double lb2 = std::sqrt(b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2]);
  if (lb2 == 0) return NAN;
  double u[3];
  for (int i = 0; i < 3; ++i) u[i] = b2[i] / lb2;
  cross(u, n1, m);
  const double x = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2];
  const double y = m[0]*n2[0] + m[1]*n2[1] + m[2]*n2[2];
  // THE SIGN IS BIOTITE'S, and it is not a convention one may pick.  The
  // helix window is +50 +/- 20 degrees and the strand window wraps through
  // +/-180; a flipped torsion puts a right-handed helix at -50 and the
  // strict helix test then fails wherever d4 is undefined -- which is only
  // the last residues of the chain.  Six of eight gate cases still matched
  // and the two that did not differed by ONE residue at position n-2.
  // Checked against struc.dihedral, not reasoned about.
  return std::atan2(y, x);
}

inline bool between(double v, const double lo, const double hi) {
  return !std::isnan(v) && v >= lo && v <= hi;
}

// biotite's _mask_consecutive: true wherever `number` consecutive trues
// cover the position
inline std::vector<char> maskConsecutive(const std::vector<char>& m, int number) {
  const int n = (int)m.size();
  std::vector<char> out((size_t)n, 0);
  if (n < number) return out;
  for (int s = 0; s + number <= n; ++s) {
    bool all = true;
    for (int k = 0; k < number && all; ++k) all = m[s + k] != 0;
    if (all) for (int k = 0; k < number; ++k) out[s + k] = 1;
  }
  return out;
}

// _extend_region: one element at each END of a true run, if it meets the
// relaxed condition.  The ends are found from the run boundaries, so an
// isolated relaxed element far from any run is NOT promoted.
inline std::vector<char> extendRegion(const std::vector<char>& base,
                                      const std::vector<char>& relaxed) {
  const int n = (int)base.size();
  std::vector<char> out = base;
  for (int i = 0; i < n; ++i) {
    if (base[i]) continue;
    const bool left_of_run = (i + 1 < n) && base[i + 1];
    const bool right_of_run = (i > 0) && base[i - 1];
    if ((left_of_run || right_of_run) && relaxed[i]) out[i] = 1;
  }
  return out;
}

// _mask_regions_with_contacts: a candidate RUN survives when its residues
// together have at least `min_contacts` CA neighbours in
// (min_distance, max_distance].
inline std::vector<char> maskRegionsWithContacts(
    const std::vector<float>& ca, const std::vector<char>& valid,
    const std::vector<char>& candidate, int min_contacts,
    double min_distance, double max_distance) {
  const int n = (int)candidate.size();
  std::vector<int> contacts((size_t)n, 0);
  for (int i = 0; i < n; ++i) {
    if (!candidate[i] || !valid[i]) continue;
    int c = 0;
    for (int j = 0; j < n; ++j) {
      if (!valid[j]) continue;
      const double d = dist(&ca[(size_t)i * 3], &ca[(size_t)j * 3]);
      if (d > min_distance && d <= max_distance) ++c;
    }
    contacts[i] = c;
  }
  std::vector<char> out((size_t)n, 0);
  int i = 0;
  while (i < n) {
    if (!candidate[i]) { ++i; continue; }
    int j = i, total = 0;
    while (j < n && candidate[j]) { total += contacts[j]; ++j; }
    if (total >= min_contacts) for (int k = i; k < j; ++k) out[k] = 1;
    i = j;
  }
  return out;
}

// `ca` is [n, 3]; `valid` marks residues that have a CA at all.  Returns
// 'a' (helix), 'b' (strand), 'c' (coil), 0 (no CA).
inline std::vector<char> annotate(const std::vector<float>& ca,
                                  const std::vector<char>& valid) {
  const int n = (int)valid.size();
  std::vector<char> out((size_t)n, 'c');
  for (int i = 0; i < n; ++i) if (!valid[i]) out[i] = 0;
  if (n <= 5) return out;      // too short to measure anything

  std::vector<double> d2(n, NAN), d3(n, NAN), d4(n, NAN), ri(n, NAN), ai(n, NAN);
  auto V = [&](int i) { return i >= 0 && i < n && valid[i]; };
  for (int i = 1; i < n - 1; ++i)
    if (V(i - 1) && V(i + 1)) d2[i] = dist(&ca[(size_t)(i-1)*3], &ca[(size_t)(i+1)*3]);
  for (int i = 1; i < n - 2; ++i)
    if (V(i - 1) && V(i + 2)) d3[i] = dist(&ca[(size_t)(i-1)*3], &ca[(size_t)(i+2)*3]);
  for (int i = 1; i < n - 3; ++i)
    if (V(i - 1) && V(i + 3)) d4[i] = dist(&ca[(size_t)(i-1)*3], &ca[(size_t)(i+3)*3]);
  for (int i = 1; i < n - 1; ++i)
    if (V(i - 1) && V(i) && V(i + 1))
      ri[i] = angleAt(&ca[(size_t)(i-1)*3], &ca[(size_t)i*3], &ca[(size_t)(i+1)*3]);
  for (int i = 1; i < n - 2; ++i)
    if (V(i - 1) && V(i) && V(i + 1) && V(i + 2))
      ai[i] = dihedral(&ca[(size_t)(i-1)*3], &ca[(size_t)i*3],
                       &ca[(size_t)(i+1)*3], &ca[(size_t)(i+2)*3]);

  std::vector<char> relaxed_h(n, 0), strict_h(n, 0), relaxed_s(n, 0), strict_s(n, 0);
  for (int i = 0; i < n; ++i) {
    relaxed_h[i] = between(d3[i], D3_HELIX[0], D3_HELIX[1]) ||
                   between(ri[i], R_HELIX[0], R_HELIX[1]);
    strict_h[i] = (between(d3[i], D3_HELIX[0], D3_HELIX[1]) &&
                   between(d4[i], D4_HELIX[0], D4_HELIX[1])) ||
                  (between(ri[i], R_HELIX[0], R_HELIX[1]) &&
                   between(ai[i], A_HELIX[0], A_HELIX[1]));
    relaxed_s[i] = between(d3[i], D3_STRAND[0], D3_STRAND[1]);
    strict_s[i] = (between(d2[i], D2_STRAND[0], D2_STRAND[1]) &&
                   between(d3[i], D3_STRAND[0], D3_STRAND[1]) &&
                   between(d4[i], D4_STRAND[0], D4_STRAND[1])) ||
                  (between(ri[i], R_STRAND[0], R_STRAND[1]) &&
                   (between(ai[i], A_STRAND[0], A_STRAND[1]) ||
                    between(ai[i], A_STRAND[2], A_STRAND[3])));
  }

  std::vector<char> helix = extendRegion(maskConsecutive(strict_h, 5), relaxed_h);
  std::vector<char> strand4 = maskConsecutive(strict_s, 4);
  // a THREE-residue strand counts only if the run is packed against
  // something: five CA neighbours between 4.2 and 5.2 A, which is a sheet
  // partner and not the chain itself
  std::vector<char> strand3 = maskRegionsWithContacts(
      ca, valid, maskConsecutive(strict_s, 3), 5, 4.2, 5.2);
  std::vector<char> both(n, 0);
  for (int i = 0; i < n; ++i) both[i] = strand4[i] || strand3[i];
  std::vector<char> strand = extendRegion(both, relaxed_s);

  for (int i = 0; i < n; ++i) {
    if (helix[i]) out[i] = 'a';
    if (strand[i]) out[i] = 'b';     // strand wins, as biotite assigns it last
    if (!valid[i]) out[i] = 0;
  }
  return out;
}

inline float fraction(const std::vector<char>& sse, char code) {
  int hit = 0, tot = 0;
  for (char c : sse) {
    if (c == 0) continue;
    ++tot;
    hit += (code == 'c') ? (c != 'a' && c != 'b') : (c == code);
  }
  return tot ? (float)hit / (float)tot : 0.f;
}

}  // namespace sse
