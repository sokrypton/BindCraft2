// forced_targeting -- bc2's epitope_targeting.py, transcribed.
//
// Before screen, every surface-exposed target residue outside `epitope_cutoff`
// (forced_targeting_shell, default 10.0 A) of the declared hotspots (and not
// already lysine) has its sidechain rebuilt as the least-clashing of 81
// canonical lysine rotamers, and its sequence row replaced with one-hot K.
// Before harden, the wild-type target is restored (`restore_wild_type_targets`).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "design_inputs.cuh"

namespace epitar {

constexpr double SURFACE_NEIGHBOUR_RADIUS = 10.0;
constexpr double MAX_SURFACE_NEIGHBOURS = 22.0;
constexpr double EPITOPE_CUTOFF = 10.0;
constexpr double CLASH_ALLOWANCE = 0.6;

// ATOM_NAMES in bindcraft/protein.py:
//  0:N   1:CA  2:C   3:CB  4:O   5:CG  6:CG1 7:CG2 8:OG  9:OG1 10:SG
// 11:CD 12:CD1 13:CD2 14:ND1 15:ND2 16:OD1 17:OD2 18:SD 19:CE 20:CE1
// 21:CE2 22:CE3 23:NE 24:NE1 25:NE2 26:OE1 27:OE2 28:CH2 29:NH1 30:NH2
// 31:OH 32:CZ 33:CZ2 34:CZ3 35:NZ 36:OXT
static constexpr int IDX_N = 0, IDX_CA = 1, IDX_C = 2, IDX_CB = 3, IDX_O = 4;
static constexpr int IDX_CG = 5, IDX_CD = 11, IDX_CE = 19, IDX_NZ = 35, IDX_OXT = 36;
static constexpr int K_INDEX = 11;  // 'K' in ARNDCQEGHILKMFPSTWYV

static const double ATOM_VDW_RADII[37] = {
    1.55, 1.70, 1.70, 1.70, 1.52, 1.70, 1.70, 1.70, 1.52, 1.52,
    1.80, 1.70, 1.70, 1.70, 1.55, 1.55, 1.52, 1.52, 1.80, 1.70,
    1.70, 1.70, 1.70, 1.55, 1.55, 1.55, 1.52, 1.52, 1.70, 1.55,
    1.55, 1.52, 1.70, 1.70, 1.70, 1.55, 1.52};

inline bool isBackboneAtom(int a) {
  return a == IDX_N || a == IDX_CA || a == IDX_C || a == IDX_O || a == IDX_OXT;
}

inline bool isLysineSidechainAtom(int a) {
  return a == IDX_CB || a == IDX_CG || a == IDX_CD || a == IDX_CE || a == IDX_NZ;
}

inline bool isRebuiltSidechainAtom(int a) {
  return !isBackboneAtom(a) && a != IDX_CB;
}

struct V3 {
  double x = 0, y = 0, z = 0;
};

inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 mul(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(V3 a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
inline V3 unit(V3 a) {
  const double n = norm(a);
  return {a.x / n, a.y / n, a.z / n};
}

inline V3 virtualBetaCarbon(V3 n, V3 ca, V3 c) {
  const V3 amide = sub(ca, n);
  const V3 carbonyl = sub(c, ca);
  const V3 cr = cross(amide, carbonyl);
  // bc2's term order, so the sum rounds the same way before the f16 cast
  return {-0.58273431 * cr.x + 0.56802827 * amide.x - 0.54067466 * carbonyl.x + ca.x,
          -0.58273431 * cr.y + 0.56802827 * amide.y - 0.54067466 * carbonyl.y + ca.y,
          -0.58273431 * cr.z + 0.56802827 * amide.z - 0.54067466 * carbonyl.z + ca.z};
}

inline V3 placeSidechainAtom(V3 first, V3 second, V3 third, double bond_length,
                             double bond_angle_deg, double torsion_angle_deg) {
  constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
  const double ba = bond_angle_deg * DEG2RAD;
  const double ta = torsion_angle_deg * DEG2RAD;
  const V3 bd = unit(sub(third, second));
  const V3 pn = unit(cross(sub(second, first), bd));
  const V3 pnbd = cross(pn, bd);
  const double cba = std::cos(ba), sba = std::sin(ba);
  const double cta = std::cos(ta), sta = std::sin(ta);
  return {third.x + bond_length * (-bd.x * cba + pnbd.x * (sba * cta) + pn.x * (sba * sta)),
          third.y + bond_length * (-bd.y * cba + pnbd.y * (sba * cta) + pn.y * (sba * sta)),
          third.z + bond_length * (-bd.z * cba + pnbd.z * (sba * cta) + pn.z * (sba * sta))};
}

// Modifies target rows [t0, cx.n) of `cx` in-place, matching `lysinate_target`.
// Returns false with `err` populated if hotspots or target backbone are missing.
inline bool lysinateTarget(dsin::State& cx, int t0, double epitope_cutoff = EPITOPE_CUTOFF,
                           bool verbose = true, std::string* err = nullptr) {
  const int T = cx.n - t0;
  if (T <= 0) return true;

  std::vector<V3> atoms((size_t)T * 37);
  std::vector<uint8_t> atom_mask((size_t)T * 37, 0);
  std::vector<uint8_t> hotspot((size_t)T, 0);
  bool any_hotspot = false;
  bool backbone_ok = true;

  for (int r = 0; r < T; ++r) {
    const int gr = t0 + r;
    hotspot[r] = (cx.flags[gr] & dsin::F_HOTSPOT) ? 1 : 0;
    if (hotspot[r]) any_hotspot = true;
    for (int a = 0; a < 37; ++a) {
      const size_t o = ((size_t)gr * 37 + a) * 3;
      atoms[(size_t)r * 37 + a] = {
          (double)cx.atoms[o + 0], (double)cx.atoms[o + 1], (double)cx.atoms[o + 2]};
      atom_mask[(size_t)r * 37 + a] = cx.atom_mask[(size_t)gr * 37 + a] > 0.f ? 1 : 0;
    }
    if (!atom_mask[(size_t)r * 37 + IDX_N] ||
        !atom_mask[(size_t)r * 37 + IDX_CA] ||
        !atom_mask[(size_t)r * 37 + IDX_C]) {
      backbone_ok = false;
    }
  }
  if (!any_hotspot) {
    if (err)
      *err = "Epitope-focused design needs target hotspots to define the region kept at wild type";
    return false;
  }
  if (!backbone_ok) {
    if (err)
      *err = "Epitope-focused design needs a target with a complete backbone to identify surface residues";
    return false;
  }

  // hotspot_coordinates: resolved atoms on hotspot residues
  std::vector<V3> hs_coords;
  for (int r = 0; r < T; ++r) {
    if (!hotspot[r]) continue;
    for (int a = 0; a < 37; ++a) {
      if (atom_mask[(size_t)r * 37 + a]) hs_coords.push_back(atoms[(size_t)r * 37 + a]);
    }
  }

  // epitope_residues: min resolved atom distance to any resolved hotspot atom <= epitope_cutoff
  std::vector<uint8_t> epitope_mask((size_t)T, 0);
  int epitope_count = 0;
  for (int r = 0; r < T; ++r) {
    double min_d = INFINITY;
    for (int a = 0; a < 37; ++a) {
      if (!atom_mask[(size_t)r * 37 + a]) continue;
      const V3 p = atoms[(size_t)r * 37 + a];
      for (const V3& h : hs_coords) {
        const double d = norm(sub(p, h));
        if (d < min_d) min_d = d;
      }
    }
    if (min_d <= epitope_cutoff) {
      epitope_mask[r] = 1;
      ++epitope_count;
    }
  }

  // surface_exposed_residues & ~epitope & (argmax(sequence) != K)
  std::vector<uint8_t> lysinated((size_t)T, 0);
  int lysinated_count = 0;
  for (int r = 0; r < T; ++r) {
    const V3 ca_r = atoms[(size_t)r * 37 + IDX_CA];
    int nbrs = 0;
    for (int j = 0; j < T; ++j) {
      if (norm(sub(ca_r, atoms[(size_t)j * 37 + IDX_CA])) < SURFACE_NEIGHBOUR_RADIUS)
        ++nbrs;
    }
    const bool surface = (nbrs - 1) < MAX_SURFACE_NEIGHBOURS;
    const int gr = t0 + r;
    int argmax_aa = 0;
    float best_s = cx.sequence[(size_t)gr * dsin::NAA];
    for (int aa = 1; aa < dsin::NAA; ++aa) {
      const float v = cx.sequence[(size_t)gr * dsin::NAA + aa];
      if (v > best_s) { best_s = v; argmax_aa = aa; }
    }
    if (surface && !epitope_mask[r] && argmax_aa != K_INDEX) {
      lysinated[r] = 1;
      ++lysinated_count;
    }
  }

  // Virtual CB placement for lysinated residues without resolved CB
  for (int r = 0; r < T; ++r) {
    if (!lysinated[r]) continue;
    if (!atom_mask[(size_t)r * 37 + IDX_CB]) {
      atoms[(size_t)r * 37 + IDX_CB] = virtualBetaCarbon(
          atoms[(size_t)r * 37 + IDX_N],
          atoms[(size_t)r * 37 + IDX_CA],
          atoms[(size_t)r * 37 + IDX_C]);
    }
    atom_mask[(size_t)r * 37 + IDX_CB] = 1;
  }

  // environment_atom_mask = atom_mask & ~(lysinated[:, None] & REBUILT_SIDECHAIN_MASK)
  std::vector<uint8_t> env_mask((size_t)T * 37, 0);
  for (int r = 0; r < T; ++r) {
    for (int a = 0; a < 37; ++a) {
      env_mask[(size_t)r * 37 + a] =
          (atom_mask[(size_t)r * 37 + a] && !(lysinated[r] && isRebuiltSidechainAtom(a))) ? 1 : 0;
    }
  }

  static const double CHI1[3] = {-67.0, -177.0, 62.0};
  static const double CHI234[3] = {180.0, -65.0, 65.0};
  static const double SC_RADII[4] = {1.70, 1.70, 1.70, 1.55};  // CG, CD, CE, NZ
  static const int SC_IDX[4] = {IDX_CG, IDX_CD, IDX_CE, IDX_NZ};

  std::vector<V3> env_coords;
  std::vector<double> env_radii;
  env_coords.reserve((size_t)T * 10);
  env_radii.reserve((size_t)T * 10);

  for (int r = 0; r < T; ++r) {
    if (!lysinated[r]) continue;
    env_coords.clear();
    env_radii.clear();
    for (int j = 0; j < T; ++j) {
      if (j == r) continue;
      for (int a = 0; a < 37; ++a) {
        if (env_mask[(size_t)j * 37 + a]) {
          env_coords.push_back(atoms[(size_t)j * 37 + a]);
          env_radii.push_back(ATOM_VDW_RADII[a]);
        }
      }
    }

    const V3 pN = atoms[(size_t)r * 37 + IDX_N];
    const V3 pCA = atoms[(size_t)r * 37 + IDX_CA];
    const V3 pCB = atoms[(size_t)r * 37 + IDX_CB];

    double best_clash = INFINITY;
    V3 best_sc[4];

    for (int i1 = 0; i1 < 3; ++i1) {
      const V3 pCG = placeSidechainAtom(pN, pCA, pCB, 1.52, 114.1, CHI1[i1]);
      for (int i2 = 0; i2 < 3; ++i2) {
        const V3 pCD = placeSidechainAtom(pCA, pCB, pCG, 1.52, 111.3, CHI234[i2]);
        for (int i3 = 0; i3 < 3; ++i3) {
          const V3 pCE = placeSidechainAtom(pCB, pCG, pCD, 1.52, 111.3, CHI234[i3]);
          for (int i4 = 0; i4 < 3; ++i4) {
            const V3 pNZ = placeSidechainAtom(pCG, pCD, pCE, 1.489, 111.9, CHI234[i4]);
            const V3 rot[4] = {pCG, pCD, pCE, pNZ};
            double clash = 0.0;
            for (int s = 0; s < 4; ++s) {
              const double r_sc = SC_RADII[s];
              const V3 ps = rot[s];
              for (size_t e = 0; e < env_coords.size(); ++e) {
                const double d = norm(sub(ps, env_coords[e]));
                const double ov = r_sc + env_radii[e] - CLASH_ALLOWANCE - d;
                if (ov > 0.0) clash += ov;
              }
            }
            if (clash < best_clash) {
              best_clash = clash;
              for (int s = 0; s < 4; ++s) best_sc[s] = rot[s];
            }
          }
        }
      }
    }

    for (int s = 0; s < 4; ++s)
      atoms[(size_t)r * 37 + SC_IDX[s]] = best_sc[s];
    for (int a = 0; a < 37; ++a) {
      atom_mask[(size_t)r * 37 + a] =
          ((atom_mask[(size_t)r * 37 + a] && isBackboneAtom(a)) || isLysineSidechainAtom(a)) ? 1 : 0;
      env_mask[(size_t)r * 37 + a] = atom_mask[(size_t)r * 37 + a];
    }
  }

  // Write modified sequence, atoms, and atom_mask back to cx
  for (int r = 0; r < T; ++r) {
    if (!lysinated[r]) continue;
    const int gr = t0 + r;
    for (int aa = 0; aa < dsin::NAA; ++aa)
      cx.sequence[(size_t)gr * dsin::NAA + aa] = (aa == K_INDEX) ? 1.f : 0.f;
    for (int a = 0; a < 37; ++a) {
      const size_t o = ((size_t)gr * 37 + a) * 3;
      // Protein.atoms in bc2 is float16, and lysinate_target ends with
      // `jnp.asarray(atoms, dtype=target.atoms.dtype)`.
      cx.atoms[o + 0] = pdbin::half2f(atoms[(size_t)r * 37 + a].x);
      cx.atoms[o + 1] = pdbin::half2f(atoms[(size_t)r * 37 + a].y);
      cx.atoms[o + 2] = pdbin::half2f(atoms[(size_t)r * 37 + a].z);
      cx.atom_mask[(size_t)gr * 37 + a] = atom_mask[(size_t)r * 37 + a] ? 1.f : 0.f;
    }
  }

  if (verbose) {
    printf("forced targeting epitope=%d lysinated=%d residues=%d\n",
           epitope_count, lysinated_count, T);
    fflush(stdout);
  }
  return true;
}

}  // namespace epitar
