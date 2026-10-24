// SOLVENT-ACCESSIBLE SURFACE AREA, as biotite computes it.
//
// Three of bc2's filters read it, and two of them REJECT on a shipped
// preset: settings/property/protease_stable.json sets
// max_exposed_loop_fraction_final 0.7 and max_terminus_exposure_final 0.6.
// So this is not an optional nicety -- a protease_stable campaign was
// being accepted on two checks fewer than bc2 applies.
//
// bc2 calls `struc.sasa(probe_radius=1.4, point_number=100)`, which is
// Shrake-Rupley: put 100 points on a sphere of radius (vdw + probe) around
// each atom, and count the ones no other atom's sphere swallows.  Three
// things have to match or the number does, quietly, not:
//
//   the POINT SET      biotite's golden-section spiral, not any other
//                      equidistant-ish set.  Reproduced exactly below.
//   the RADII          ProtOr, keyed by (residue, atom) -- generated into
//                      sasa_tables.cuh, not transcribed.
//   the FALLBACK       bc2 catches biotite's KeyError on a (residue, atom)
//                      pair the table does not carry and re-runs the WHOLE
//                      array with ELEMENT radii.  An atom-by-atom fallback
//                      would be a different answer on exactly the inputs
//                      the comment in bc2 was written for.
//
// The cost is ~3600 atoms x 100 points x the occluders in range, once per
// design at the final filters.  Not worth a kernel.
#pragma once
#include <cmath>
#include <vector>
#include "sasa_tables.cuh"

namespace sasa {

// biotite's _create_fibonacci_points, element for element.
inline void fibonacci(int n, std::vector<float>* out) {
  out->assign((size_t)n * 3, 0.f);
  const double golden = (3.0 - sqrt(5.0)) * M_PI;
  for (int i = 0; i < n; ++i) {
    const double phi = golden * (double)i;
    // np.linspace(1 - 1/n, 1/n - 1, n)
    const double z = (n == 1) ? (1.0 - 1.0 / n)
                             : (1.0 - 1.0 / n) +
                                   ((1.0 / n - 1.0) - (1.0 - 1.0 / n)) *
                                       (double)i / (double)(n - 1);
    const double r = sqrt(1.0 - z * z);
    (*out)[(size_t)i * 3 + 0] = (float)(r * cos(phi));
    (*out)[(size_t)i * 3 + 1] = (float)(r * sin(phi));
    (*out)[(size_t)i * 3 + 2] = (float)z;
  }
}

// Per-RESIDUE area over `rows`, in the caller's row order.  `aa` is the
// per-row amino-acid index (20 = unknown, which has no ProtOr entry and
// therefore forces the element fallback, as it does in bc2).
inline std::vector<double> residueArea(const std::vector<float>& atoms,
                                       const std::vector<float>& atom_mask,
                                       const std::vector<int>& aa, int n,
                                       int points = 100,
                                       float probe = 1.4f) {
  // THE RADIUS SET IS CHOSEN ONCE, FOR EVERY ATOM.  bc2's try/except wraps
  // the whole sasa() call, so one unknown pair moves every radius.
  bool protor_ok = true;
  for (int i = 0; i < n && protor_ok; ++i)
    for (int a = 0; a < sasatab::NATOM; ++a) {
      if (atom_mask[(size_t)i * sasatab::NATOM + a] <= 0) continue;
      const int k = aa[(size_t)i];
      if (k < 0 || k >= sasatab::NAA ||
          sasatab::PROTOR[(size_t)k * sasatab::NATOM + a] < 0) {
        protor_ok = false;
        break;
      }
    }
  // flatten the present atoms
  std::vector<float> xyz;
  std::vector<float> rad, rad2;
  std::vector<int> row;
  for (int i = 0; i < n; ++i)
    for (int a = 0; a < sasatab::NATOM; ++a) {
      if (atom_mask[(size_t)i * sasatab::NATOM + a] <= 0) continue;
      const int k = aa[(size_t)i];
      const float base =
          protor_ok ? sasatab::PROTOR[(size_t)k * sasatab::NATOM + a]
                    : sasatab::elementRadius(sasatab::ELEMENT[a]);
      const float r = base + probe;
      rad.push_back(r);
      rad2.push_back(r * r);
      row.push_back(i);
      for (int c = 0; c < 3; ++c)
        xyz.push_back(atoms[((size_t)i * sasatab::NATOM + a) * 3 + c]);
    }
  const int m = (int)row.size();
  std::vector<double> out((size_t)n, 0.0);
  if (m == 0) return out;
  std::vector<float> sph;
  fibonacci(points, &sph);
  const double per_point = 4.0 * M_PI / (double)points;
  std::vector<int> occ;
  occ.reserve((size_t)m);
  for (int i = 0; i < m; ++i) {
    const float xi = xyz[(size_t)i * 3], yi = xyz[(size_t)i * 3 + 1],
                zi = xyz[(size_t)i * 3 + 2];
    const float ri = rad[(size_t)i];
    occ.clear();
    for (int j = 0; j < m; ++j) {
      const float dx = xyz[(size_t)j * 3] - xi,
                  dy = xyz[(size_t)j * 3 + 1] - yi,
                  dz = xyz[(size_t)j * 3 + 2] - zi;
      const float d2 = dx * dx + dy * dy + dz * dz;
      // biotite drops d2 == 0 -- the atom itself, AND any atom sitting on
      // top of it -- and keeps strictly-closer-than the radius sum
      if (d2 != 0.f && d2 < (rad[(size_t)j] + ri) * (rad[(size_t)j] + ri))
        occ.push_back(j);
    }
    int acc = points;
    for (int p = 0; p < points; ++p) {
      const float px = sph[(size_t)p * 3] * ri + xi;
      const float py = sph[(size_t)p * 3 + 1] * ri + yi;
      const float pz = sph[(size_t)p * 3 + 2] * ri + zi;
      for (int k : occ) {
        const float dx = px - xyz[(size_t)k * 3],
                    dy = py - xyz[(size_t)k * 3 + 1],
                    dz = pz - xyz[(size_t)k * 3 + 2];
        if (dx * dx + dy * dy + dz * dz < rad2[(size_t)k]) { --acc; break; }
      }
    }
    out[(size_t)row[(size_t)i]] +=
        per_point * (double)acc * (double)(ri * ri);
  }
  return out;
}

// MAX_RESIDUE_SASA, bc2's own table, by amino-acid index in AMINO_ACIDS
// (A R N D C Q E G H I L K M F P S T W Y V).
inline double maxResidueArea(int aa) {
  static const double T[20] = {121.0, 265.0, 187.0, 187.0, 148.0, 214.0,
                               214.0, 97.0,  216.0, 195.0, 191.0, 230.0,
                               203.0, 228.0, 154.0, 143.0, 163.0, 264.0,
                               255.0, 165.0};
  return (aa >= 0 && aa < 20) ? T[aa] : 121.0;
}

}  // namespace sasa
