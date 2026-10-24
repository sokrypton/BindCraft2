// multidomain: make the binder TWO domains, not one blob.
//
// The longest loss in bc2 and the only one that pulls on all three
// cotangents at once -- the pAE, the CA coordinates and the sequence.  It
// asks three things of a split that `binder_domain_ids` fixed in advance
// (see domains.cuh; the split is a DRAW and is reproduced exactly):
//
//   within-domain pAE should be low
//   between-domain pAE should exceed it by a margin
//   each domain should be globular for its own size
//
// and it discounts the LINKER while doing so: residues whose pAE says they
// belong to no domain in particular are down-weighted, by an amount scaled
// by how hard the sequence is.  That discount is not stop_gradient'd, so
// the pAE reaches the loss twice -- once as the quantity being minimised
// and once through the weights that decide what is being averaged.
//
// Both optional terms ARE implemented, and that is not optional: the
// shipped `multidomain` modality sets domain_sep_weight 1.0 and
// domain_linker_helix_weight 0.5, so refusing them would have rejected the
// one preset the loss exists for.  (The first version of this file did
// refuse them, on the strength of their being 0.0 "in every shipped
// preset" -- which was read off the signature rather than off the preset.)
//
//   domain_sep_weight          a soft contact count BETWEEN the domains,
//                              pushed down: the two should not touch
//   domain_linker_helix_weight the i,i+3 helix propensity ALONG THE
//                              LINKER, pushed up, scaled by how hard the
//                              sequence is
#pragma once
#include <cmath>
#include <vector>

namespace multidom {

inline double sgm(double x) { return 1.0 / (1.0 + exp(-x)); }
inline double globularRadius(double n) { return 2.38 * pow(n, 0.365); }

struct Out {
  double value = 0;
  std::vector<float> dpae;    // [L*L]  d loss / d (raw pae, NOT /31)
  std::vector<float> dx;      // [L*3]  d loss / d CA
  std::vector<float> dprob;   // [L*20] d loss / d amino-acid probabilities
  std::vector<float> ddgram;  // [L*L*B] d loss / d distogram logits
};

struct Cfg {
  int nd = 2;                 // the domain count the membership spans
  double rg_weight = 0.3;
  double pae_margin = 0.15;
  double linker_gap = 0.1;
  double linker_sharpness = 0.03;
  double sep_weight = 0.0;
  double contact_cutoff = 8.0;
  double linker_helix_weight = 0.0;
};

// ---- domain_linker_membership, on its own --------------------------------
//
// "is this residue's pAE no closer to another domain than to its own?",
// as a soft membership in [0, 1].  It is the middle of `multidomain`'s
// forward and it is ALSO what `mpnn_fix_linker` thresholds to decide which
// residues ProteinMPNN may not touch (`multidomain_linker_residues`), so
// it is lifted out rather than written twice -- two copies of a sigmoid
// over a ratio of masked means would drift, and only one of them is gated.
//
// `pae` is the RAW binder-block pae (0..31); bc2 divides by 31 inside.
inline std::vector<double> linkerMembership(const std::vector<float>& pae,
                                            const std::vector<int>& dom,
                                            const std::vector<double>& w,
                                            int L, int ND, double gap,
                                            double sharpness) {
  const double eps = 1e-8;
  auto Mof = [&](int i, int k) { return (dom[(size_t)i] == k) ? w[(size_t)i] : 0.0; };
  std::vector<double> domsum((size_t)ND, 0.0);
  for (int k = 0; k < ND; ++k)
    for (int i = 0; i < L; ++i) domsum[(size_t)k] += Mof(i, k);
  std::vector<double> out((size_t)L, 0.0);
  std::vector<double> ptd((size_t)ND, 0.0);
  for (int i = 0; i < L; ++i) {
    for (int k = 0; k < ND; ++k) {
      double c = 0, acc = 0;
      for (int j = 0; j < L; ++j) {
        if (std::abs(i - j) <= 2) continue;        // separated_pairs
        const double mk = Mof(j, k);
        c += mk;
        acc += ((double)pae[(size_t)i * L + j] / 31.0) * mk;
      }
      ptd[(size_t)k] = acc / (c + eps);
    }
    double own = 0;
    for (int k = 0; k < ND; ++k) own += ptd[(size_t)k] * Mof(i, k);
    double best = 1e30;
    for (int k = 0; k < ND; ++k) {
      const double v = ptd[(size_t)k] + Mof(i, k) * 1e9 +
                       (domsum[(size_t)k] < 0.5 ? 1e9 : 0.0);
      best = std::min(best, v);
    }
    out[(size_t)i] = sgm((gap - (best - own)) / sharpness);
  }
  return out;
}

// `pae` is the RAW binder-block pae (0..31); bc2 divides by 31 inside.
// `dom` is the per-row domain id, `w` the real-residue weight.
inline Out multidomain(const std::vector<float>& pae,
                       const std::vector<double>& x,
                       const std::vector<float>& prob,
                       const std::vector<int>& dom,
                       const std::vector<double>& w, int L, int NAA,
                       const Cfg& cfg, bool want_grad,
                       const std::vector<float>& dgram = {}, int NB_ = 0) {
  Out o;
  const int ND = cfg.nd;
  const double eps = 1e-8;
  auto Mof = [&](int i, int k) { return (dom[(size_t)i] == k) ? w[(size_t)i] : 0.0; };
  // S, Dd and the pair masks are constant: they come from the split
  std::vector<double> P((size_t)L * L, 0.0), S((size_t)L * L, 0.0),
      Dd((size_t)L * L, 0.0);
  double ddsum = 0;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      P[(size_t)i * L + j] = (double)pae[(size_t)i * L + j] / 31.0;
      double s = 0;
      for (int k = 0; k < ND; ++k) s += Mof(i, k) * Mof(j, k);
      S[(size_t)i * L + j] = s;
      Dd[(size_t)i * L + j] = (1.0 - s) * w[(size_t)i] * w[(size_t)j];
      ddsum += Dd[(size_t)i * L + j];
    }
  if (ddsum <= 0) return o;              // bc2's (different_domain.sum() > 0)

  // ---- sequence hardness -------------------------------------------
  double M = 0, hs = 0;
  std::vector<double> rowmax((size_t)L, 0.0);
  for (int i = 0; i < L; ++i) {
    double b = -1;
    for (int a = 0; a < NAA; ++a)
      b = std::max(b, (double)prob[(size_t)i * NAA + a]);
    rowmax[(size_t)i] = b;
    if (w[(size_t)i] > 0) { M += w[(size_t)i]; hs += b * w[(size_t)i]; }
  }
  const double h = hs / (M + eps);

  // ---- linker membership --------------------------------------------
  // "is this residue's pAE no closer to its own domain than to another?"
  std::vector<double> counts((size_t)L * ND, 0.0), ptd((size_t)L * ND, 0.0);
  std::vector<double> own((size_t)L, 0.0), nearv((size_t)L, 0.0);
  std::vector<int> nearest((size_t)L, -1);
  std::vector<double> domsum((size_t)ND, 0.0);
  for (int k = 0; k < ND; ++k)
    for (int i = 0; i < L; ++i) domsum[(size_t)k] += Mof(i, k);
  for (int i = 0; i < L; ++i) {
    for (int k = 0; k < ND; ++k) {
      double c = 0, acc = 0;
      for (int j = 0; j < L; ++j) {
        if (std::abs(i - j) <= 2) continue;        // separated_pairs
        const double mk = Mof(j, k);
        c += mk;
        acc += P[(size_t)i * L + j] * mk;
      }
      counts[(size_t)i * ND + k] = c;
      ptd[(size_t)i * ND + k] = acc / (c + eps);
    }
    double ow = 0;
    for (int k = 0; k < ND; ++k) ow += ptd[(size_t)i * ND + k] * Mof(i, k);
    own[(size_t)i] = ow;
    double best = 1e30; int bk = -1;
    for (int k = 0; k < ND; ++k) {
      const double v = ptd[(size_t)i * ND + k] + Mof(i, k) * 1e9 +
                       (domsum[(size_t)k] < 0.5 ? 1e9 : 0.0);
      if (v < best) { best = v; bk = k; }
    }
    nearv[(size_t)i] = best;
    nearest[(size_t)i] = bk;
  }
  std::vector<double> lmr((size_t)L, 0.0), u((size_t)L, 0.0);
  for (int i = 0; i < L; ++i) {
    lmr[(size_t)i] = sgm((cfg.linker_gap -
                          (nearv[(size_t)i] - own[(size_t)i])) /
                         cfg.linker_sharpness);
    u[(size_t)i] = (1.0 - h * lmr[(size_t)i] * w[(size_t)i]) * w[(size_t)i];
  }

  // ---- the two pAE means ---------------------------------------------
  double An = 0, Ad = 0, Bn = 0, Bd = 0;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      const double U = u[(size_t)i] * u[(size_t)j];
      const double a = S[(size_t)i * L + j] * U, b = Dd[(size_t)i * L + j] * U;
      Ad += a; Bd += b;
      if (a > 0) An += P[(size_t)i * L + j] * a;
      if (b > 0) Bn += P[(size_t)i * L + j] * b;
    }
  const double within = An / (Ad + eps), between = Bn / (Bd + eps);
  const double rel = std::max(0.0, cfg.pae_margin - (between - within));

  // ---- the per-domain radii ------------------------------------------
  std::vector<double> Wd((size_t)ND, 0.0);
  for (int k = 0; k < ND; ++k)
    for (int i = 0; i < L; ++i) Wd[(size_t)k] += Mof(i, k) * u[(size_t)i];
  std::vector<double> C((size_t)ND * 3, 0.0);
  for (int k = 0; k < ND; ++k) {
    for (int i = 0; i < L; ++i) {
      const double mu = Mof(i, k) * u[(size_t)i];
      for (int d = 0; d < 3; ++d) C[(size_t)k * 3 + d] += mu * x[(size_t)i * 3 + d];
    }
    for (int d = 0; d < 3; ++d) C[(size_t)k * 3 + d] /= (Wd[(size_t)k] + eps);
  }
  std::vector<double> sq((size_t)L, 0.0);
  for (int i = 0; i < L; ++i) {
    const int k = dom[(size_t)i];
    double q = 0;
    for (int d = 0; d < 3; ++d) {
      const double dd = x[(size_t)i * 3 + d] -
                        (w[(size_t)i] > 0 ? C[(size_t)k * 3 + d] : 0.0);
      q += dd * dd;
    }
    sq[(size_t)i] = q;
  }
  std::vector<double> T((size_t)ND, 0.0), r((size_t)ND, 0.0);
  double rl_num = 0, rl_den = 0;
  std::vector<double> elu_d((size_t)ND, 0.0);
  for (int k = 0; k < ND; ++k) {
    for (int i = 0; i < L; ++i)
      T[(size_t)k] += Mof(i, k) * sq[(size_t)i] * u[(size_t)i];
    r[(size_t)k] = sqrt(T[(size_t)k] / (Wd[(size_t)k] + eps) + eps);
    if (domsum[(size_t)k] > 0.5) {
      const double z = r[(size_t)k] - globularRadius(domsum[(size_t)k] + eps);
      rl_num += z > 0 ? z : (exp(z) - 1.0);
      elu_d[(size_t)k] = z > 0 ? 1.0 : exp(z);
      rl_den += 1.0;
    }
  }
  const double radius_loss = rl_num / (rl_den + eps);
  o.value = within + rel + cfg.rg_weight * radius_loss;

  // ---- domain_sep_weight: the two domains should not touch -----------
  // A soft CA contact count over the between-domain pairs, weighted by the
  // same linker-discounted weights.  pairwise_atom_distances here is
  // called with eps=1e-8 explicitly, not the 1e-4 default.
  double sepn = 0, sepd = 0;
  std::vector<double> sepv;
  if (cfg.sep_weight != 0.0) {
    sepv.assign((size_t)L * L, 0.0);
    for (int i = 0; i < L; ++i)
      for (int j = 0; j < L; ++j) {
        double q = 0;
        for (int d = 0; d < 3; ++d) {
          const double dd = x[(size_t)i * 3 + d] - x[(size_t)j * 3 + d];
          q += dd * dd;
        }
        const double dij = sqrt(q + 1e-8);
        const double v = sgm((cfg.contact_cutoff - dij) / 2.0);
        sepv[(size_t)i * L + j] = v;
        const double m = Dd[(size_t)i * L + j] * u[(size_t)i] * u[(size_t)j];
        sepd += m;
        if (m > 0) sepn += v * m;
      }
    o.value += cfg.sep_weight * (sepn / (sepd + eps));
  }

  // ---- domain_linker_helix_weight: make the LINKER a helix -----------
  // The i,i+3 diagonal of the binary distogram contact loss, averaged
  // under the linker membership of both ends, SUBTRACTED (so a helical
  // linker lowers the loss) and scaled by the sequence hardness.
  const int B = (L > 0 && !dgram.empty())
                    ? (int)(dgram.size() / ((size_t)NB_ * NB_)) : 0;
  double hln = 0, hld = 0;
  std::vector<double> hv, hq;
  if (cfg.linker_helix_weight != 0.0 && B > 0 && L > 3) {
    hv.assign((size_t)L - 3, 0.0);
    hq.assign((size_t)L - 3, 0.0);
    // distogram_bin_distances(B) < 6.0
    std::vector<char> sel((size_t)B, 0);
    sel[0] = (0.0f < 6.0f);
    for (int k = 1; k < B; ++k) {
      const double e = 2.3125 + (21.6875 - 2.3125) * (double)(k - 1) /
                                    (double)(B - 2);
      sel[(size_t)k] = e < 6.0 ? 1 : 0;
    }
    for (int i = 0; i + 3 < L; ++i) {
      const float* row = &dgram[(((size_t)i * NB_) + (i + 3)) * B];
      double m = row[0];
      for (int k = 1; k < B; ++k) m = std::max(m, (double)row[k]);
      double z = 0, q = 0;
      for (int k = 0; k < B; ++k) {
        const double e = exp((double)row[k] - m);
        z += e;
        if (sel[(size_t)k]) q += e;
      }
      q /= z;
      hq[(size_t)i] = q;
      hv[(size_t)i] = -log(q + 1e-8);
      const double lw = lmr[(size_t)i] * w[(size_t)i] *
                        lmr[(size_t)i + 3] * w[(size_t)i + 3];
      hld += lw;
      if (lw > 0) hln += hv[(size_t)i] * lw;
    }
    o.value -= cfg.linker_helix_weight * h * (hln / (hld + eps));
  }
  if (!want_grad) return o;

  o.dpae.assign((size_t)L * L, 0.f);
  o.dx.assign((size_t)L * 3, 0.f);
  o.dprob.assign((size_t)L * NAA, 0.f);
  std::vector<double> du((size_t)L, 0.0), dP((size_t)L * L, 0.0);
  double dh = 0;
  std::vector<double> dlmr0((size_t)L, 0.0);

  // ---- the two optional terms, backward ------------------------------
  if (cfg.sep_weight != 0.0) {
    const double num = sepn, den = sepd + eps;
    const double dn = cfg.sep_weight / den;
    const double dd = -cfg.sep_weight * num / (den * den);
    for (int i = 0; i < L; ++i)
      for (int j = 0; j < L; ++j) {
        const double c = Dd[(size_t)i * L + j];
        const double m = c * u[(size_t)i] * u[(size_t)j];
        double dm = dd;
        if (m > 0) {
          dm += dn * sepv[(size_t)i * L + j];
          // d sigmoid((C - d)/2) / dd = -s(1-s)/2
          const double v = sepv[(size_t)i * L + j];
          double q = 0;
          for (int d = 0; d < 3; ++d) {
            const double de = x[(size_t)i * 3 + d] - x[(size_t)j * 3 + d];
            q += de * de;
          }
          const double dij = sqrt(q + 1e-8);
          const double gd = dn * m * (-v * (1.0 - v) / 2.0) / dij;
          for (int d = 0; d < 3; ++d) {
            const double de = x[(size_t)i * 3 + d] - x[(size_t)j * 3 + d];
            o.dx[(size_t)i * 3 + d] += (float)(gd * de);
            o.dx[(size_t)j * 3 + d] -= (float)(gd * de);
          }
        }
        du[(size_t)i] += dm * c * u[(size_t)j];
        du[(size_t)j] += dm * c * u[(size_t)i];
      }
  }
  if (cfg.linker_helix_weight != 0.0 && B > 0 && L > 3) {
    o.ddgram.assign((size_t)L * L * B, 0.f);
    const double mean = hln / (hld + eps);
    dh += -cfg.linker_helix_weight * mean;
    const double g = -cfg.linker_helix_weight * h;
    const double dn = g / (hld + eps);
    const double dd = -g * hln / ((hld + eps) * (hld + eps));
    std::vector<char> sel((size_t)B, 0);
    sel[0] = 1;
    for (int k = 1; k < B; ++k) {
      const double e = 2.3125 + (21.6875 - 2.3125) * (double)(k - 1) /
                                    (double)(B - 2);
      sel[(size_t)k] = e < 6.0 ? 1 : 0;
    }
    for (int i = 0; i + 3 < L; ++i) {
      const double lw = lmr[(size_t)i] * w[(size_t)i] *
                        lmr[(size_t)i + 3] * w[(size_t)i + 3];
      double dlw = dd;
      if (lw > 0) {
        dlw += dn * hv[(size_t)i];
        // d(-log(q + 1e-8))/d(dgram_k) through the softmax
        const double q = hq[(size_t)i];
        const double gq = -dn * lw / (q + 1e-8);
        const float* row = &dgram[(((size_t)i * NB_) + (i + 3)) * B];
        double m = row[0];
        for (int k = 1; k < B; ++k) m = std::max(m, (double)row[k]);
        double z = 0;
        for (int k = 0; k < B; ++k) z += exp((double)row[k] - m);
        for (int k = 0; k < B; ++k) {
          const double pk = exp((double)row[k] - m) / z;
          o.ddgram[(((size_t)i * L) + (i + 3)) * B + k] +=
              (float)(gq * pk * ((sel[(size_t)k] ? 1.0 : 0.0) - q));
        }
      }
      if (w[(size_t)i] > 0 && w[(size_t)i + 3] > 0) {
        dlmr0[(size_t)i] += dlw * lmr[(size_t)i + 3] * w[(size_t)i] *
                            w[(size_t)i + 3];
        dlmr0[(size_t)i + 3] += dlw * lmr[(size_t)i] * w[(size_t)i] *
                                w[(size_t)i + 3];
      }
    }
  }

  // within and between
  const double g_within = 1.0 + (rel > 0 ? 1.0 : 0.0);
  const double g_between = rel > 0 ? -1.0 : 0.0;
  auto meanBwd = [&](const std::vector<double>& maskc, double num, double den,
                     double g) {
    if (g == 0) return;
    const double dn = g / (den + eps), dd = -g * num / ((den + eps) * (den + eps));
    for (int i = 0; i < L; ++i)
      for (int j = 0; j < L; ++j) {
        const double c = maskc[(size_t)i * L + j];
        const double U = u[(size_t)i] * u[(size_t)j];
        const double m = c * U;
        // d/dm: the numerator term only where m > 0, the denominator
        // everywhere -- mask.sum() counts the zeros too
        double dm = dd;
        if (m > 0) { dm += dn * P[(size_t)i * L + j];
                     dP[(size_t)i * L + j] += dn * m; }
        du[(size_t)i] += dm * c * u[(size_t)j];
        du[(size_t)j] += dm * c * u[(size_t)i];
      }
  };
  meanBwd(S, An, Ad, g_within);
  meanBwd(Dd, Bn, Bd, g_between);

  // the radii
  std::vector<double> dWd((size_t)ND, 0.0), dT((size_t)ND, 0.0);
  for (int k = 0; k < ND; ++k) {
    if (domsum[(size_t)k] <= 0.5) continue;
    const double gr = cfg.rg_weight * elu_d[(size_t)k] / (rl_den + eps);
    const double inv = 1.0 / (Wd[(size_t)k] + eps);
    dT[(size_t)k] += gr / (2.0 * r[(size_t)k]) * inv;
    dWd[(size_t)k] += -gr / (2.0 * r[(size_t)k]) * T[(size_t)k] * inv * inv;
  }
  std::vector<double> dsq((size_t)L, 0.0);
  for (int k = 0; k < ND; ++k) {
    if (dT[(size_t)k] == 0.0) continue;
    for (int i = 0; i < L; ++i) {
      const double m = Mof(i, k);
      if (m == 0) continue;
      dsq[(size_t)i] += dT[(size_t)k] * m * u[(size_t)i];
      du[(size_t)i] += dT[(size_t)k] * m * sq[(size_t)i];
    }
  }
  for (int k = 0; k < ND; ++k)
    if (dWd[(size_t)k] != 0.0)
      for (int i = 0; i < L; ++i) du[(size_t)i] += dWd[(size_t)k] * Mof(i, k);
  // sq -> x and the centroid; the centroid carries both x and u
  std::vector<double> dC((size_t)ND * 3, 0.0);
  for (int i = 0; i < L; ++i) {
    if (dsq[(size_t)i] == 0.0) continue;
    const int k = dom[(size_t)i];
    for (int d = 0; d < 3; ++d) {
      const double dd = x[(size_t)i * 3 + d] -
                        (w[(size_t)i] > 0 ? C[(size_t)k * 3 + d] : 0.0);
      o.dx[(size_t)i * 3 + d] += (float)(dsq[(size_t)i] * 2.0 * dd);
      if (w[(size_t)i] > 0) dC[(size_t)k * 3 + d] -= dsq[(size_t)i] * 2.0 * dd;
    }
  }
  for (int k = 0; k < ND; ++k) {
    const double inv = 1.0 / (Wd[(size_t)k] + eps);
    for (int i = 0; i < L; ++i) {
      const double m = Mof(i, k);
      if (m == 0) continue;
      for (int d = 0; d < 3; ++d) {
        o.dx[(size_t)i * 3 + d] += (float)(dC[(size_t)k * 3 + d] * m *
                                           u[(size_t)i] * inv);
        du[(size_t)i] += dC[(size_t)k * 3 + d] * m *
                         (x[(size_t)i * 3 + d] - C[(size_t)k * 3 + d]) * inv;
      }
    }
  }

  // u -> h and the linker membership
  std::vector<double> dlmr = dlmr0;
  for (int i = 0; i < L; ++i) {
    if (du[(size_t)i] == 0.0 || w[(size_t)i] == 0) continue;
    dh += du[(size_t)i] * (-lmr[(size_t)i] * w[(size_t)i] * w[(size_t)i]);
    dlmr[(size_t)i] += du[(size_t)i] * (-h * w[(size_t)i] * w[(size_t)i]);
  }
  // h -> the row maxima, split across ties exactly as jnp.max does
  for (int i = 0; i < L; ++i) {
    if (w[(size_t)i] <= 0 || dh == 0.0) continue;
    int ties = 0;
    for (int a = 0; a < NAA; ++a)
      ties += (prob[(size_t)i * NAA + a] == (float)rowmax[(size_t)i]) ? 1 : 0;
    if (!ties) continue;
    const double share = dh * w[(size_t)i] / (M + eps) / (double)ties;
    for (int a = 0; a < NAA; ++a)
      if (prob[(size_t)i * NAA + a] == (float)rowmax[(size_t)i])
        o.dprob[(size_t)i * NAA + a] += (float)share;
  }
  // the linker membership -> the pAE, through own and the nearest other
  std::vector<double> dptd((size_t)L * ND, 0.0);
  for (int i = 0; i < L; ++i) {
    if (dlmr[(size_t)i] == 0.0) continue;
    const double dz = dlmr[(size_t)i] * lmr[(size_t)i] * (1.0 - lmr[(size_t)i]) /
                      cfg.linker_sharpness;
    for (int k = 0; k < ND; ++k) dptd[(size_t)i * ND + k] += dz * Mof(i, k);
    if (nearest[(size_t)i] >= 0)
      dptd[(size_t)i * ND + nearest[(size_t)i]] -= dz;
  }
  for (int i = 0; i < L; ++i)
    for (int k = 0; k < ND; ++k) {
      const double g = dptd[(size_t)i * ND + k];
      if (g == 0.0) continue;
      const double inv = 1.0 / (counts[(size_t)i * ND + k] + eps);
      for (int j = 0; j < L; ++j) {
        if (std::abs(i - j) <= 2) continue;
        const double mk = Mof(j, k);
        if (mk != 0.0) dP[(size_t)i * L + j] += g * mk * inv;
      }
    }
  // and out through the /31
  for (size_t q = 0; q < dP.size(); ++q)
    o.dpae[q] = (float)(dP[q] / 31.0);
  return o;
}

}  // namespace multidom
