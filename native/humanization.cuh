// humanization: how strongly does this sequence look like an MHC epitope?
//
// bc2 scores every 9-residue window of the binder against a panel of MHC
// alleles, takes a soft maximum over alleles, a soft maximum over windows,
// and a soft maximum over binder chains -- then adds class I and class II.
// The panels are in mhc_tables.cuh, GENERATED from bc2's own module rather
// than typed out here.
//
// THE COUPLING TERM COLLAPSES.  bc2 builds, per allele and per coupled
// anchor pair (f, s, w), the outer product w * A[f] (x) A[s] and contracts
// it with core[f] (x) core[s].  That is exactly
//
//     w * (core[f] . A[f]) * (core[s] . A[s])
//
// so the AxPx20x20 tensor never has to exist.  Worth saying out loud
// because the two forms are only equal for a RANK-ONE coupling matrix,
// which is what an outer product is -- if bc2 ever couples with something
// else this shortcut is silently wrong, and the gate is what would say so.
//
// soft_maximum(v, T) = sum_i v_i * softmax(v/T)_i, whose derivative is
//     d/dv_j = s_j * (1 + (v_j - m) / T),   m = soft_maximum(v, T)
// which is 1 at the argmax and ~0 elsewhere as T -> 0, as it should be.
#pragma once
#include <cmath>
#include <vector>
#include "mhc_tables.cuh"

namespace human {

// softmax weights and the soft maximum, in one pass
inline double softMax(const std::vector<double>& v, double T,
                      std::vector<double>* s) {
  const size_t n = v.size();
  s->assign(n, 0.0);
  if (!n) return 0.0;
  double m = v[0];
  for (double x : v) m = std::max(m, x);
  double z = 0;
  for (size_t i = 0; i < n; ++i) { (*s)[i] = exp((v[i] - m) / T); z += (*s)[i]; }
  double acc = 0;
  for (size_t i = 0; i < n; ++i) { (*s)[i] /= z; acc += v[i] * (*s)[i]; }
  return acc;
}

// d(softMax)/dv_j = s_j (1 + (v_j - m)/T), accumulated into `g` scaled by u
inline void softMaxBwd(const std::vector<double>& v,
                       const std::vector<double>& s, double m, double T,
                       double u, std::vector<double>* g) {
  g->assign(v.size(), 0.0);
  for (size_t j = 0; j < v.size(); ++j)
    (*g)[j] = u * s[j] * (1.0 + (v[j] - m) / T);
}

// One chain's epitope score against one panel, and its gradient wrt the
// per-row amino acid probabilities.  `prob` is L x 20, `redes` the
// redesignable mask; `dprob` is accumulated into (L x 20) when non-null.
inline double epitopeScore(const std::vector<float>& prob,
                           const std::vector<char>& redes, int L,
                           const mhc::Panel& P, double coupling_w,
                           double hydro_w, double T,
                           double upstream, std::vector<double>* dprob) {
  if (P.n == 0 || L < mhc::CORE) return 0.0;
  const int W = L - mhc::CORE + 1;
  // per window: the allele scores, their soft max, and the pieces the
  // backward needs (the two anchor dot products per coupled pair)
  std::vector<double> core_scores((size_t)W, 0.0);
  std::vector<std::vector<double>> allele_v((size_t)W), allele_s((size_t)W);
  std::vector<std::vector<double>> pair_f((size_t)W), pair_s((size_t)W);
  for (int c = 0; c < W; ++c) {
    bool real = true;
    for (int k = 0; k < mhc::CORE; ++k) real &= redes[(size_t)c + k] != 0;
    if (!real) { core_scores[(size_t)c] = -1e9; continue; }
    double burial = 0;
    for (int p = 0; p < mhc::CORE; ++p) {
      double h = 0;
      for (int x = 0; x < mhc::NAA; ++x)
        h += (double)prob[((size_t)c + p) * mhc::NAA + x] * mhc::HYDRO[x];
      burial += h * (double)P.burial[p];
    }
    std::vector<double> v((size_t)P.n, 0.0);
    std::vector<double> pf((size_t)P.n * P.npair, 0.0),
                        ps((size_t)P.n * P.npair, 0.0);
    for (int a = 0; a < P.n; ++a) {
      const float* A = P.anchor + (size_t)a * mhc::CORE * mhc::NAA;
      double match = 0;
      for (int p = 0; p < mhc::CORE; ++p)
        for (int x = 0; x < mhc::NAA; ++x)
          match += (double)A[p * mhc::NAA + x] *
                   (double)prob[((size_t)c + p) * mhc::NAA + x];
      double coup = 0;
      for (int k = 0; k < P.npair; ++k) {
        const int f = P.pair[k * 2], sp = P.pair[k * 2 + 1];
        double df = 0, ds = 0;
        for (int x = 0; x < mhc::NAA; ++x) {
          df += (double)A[f * mhc::NAA + x] *
                (double)prob[((size_t)c + f) * mhc::NAA + x];
          ds += (double)A[sp * mhc::NAA + x] *
                (double)prob[((size_t)c + sp) * mhc::NAA + x];
        }
        pf[(size_t)a * P.npair + k] = df;
        ps[(size_t)a * P.npair + k] = ds;
        coup += (double)P.pairw[k] * df * ds;
      }
      v[(size_t)a] = match + coupling_w * coup + hydro_w * burial;
    }
    std::vector<double> sa;
    core_scores[(size_t)c] = softMax(v, T, &sa);
    allele_v[(size_t)c] = v; allele_s[(size_t)c] = sa;
    pair_f[(size_t)c] = pf; pair_s[(size_t)c] = ps;
  }
  std::vector<double> sw;
  const double m = softMax(core_scores, T, &sw);
  const double out = m / (double)mhc::CORE;
  if (!dprob || upstream == 0.0) return out;
  std::vector<double> gw;
  softMaxBwd(core_scores, sw, m, T, upstream / (double)mhc::CORE, &gw);
  for (int c = 0; c < W; ++c) {
    if (core_scores[(size_t)c] <= -1e8) continue;
    std::vector<double> gv;
    softMaxBwd(allele_v[(size_t)c], allele_s[(size_t)c],
               core_scores[(size_t)c], T, gw[(size_t)c], &gv);
    for (int a = 0; a < P.n; ++a) {
      const double ga = gv[(size_t)a];
      if (ga == 0.0) continue;
      const float* A = P.anchor + (size_t)a * mhc::CORE * mhc::NAA;
      for (int p = 0; p < mhc::CORE; ++p)
        for (int x = 0; x < mhc::NAA; ++x)
          (*dprob)[((size_t)c + p) * mhc::NAA + x] +=
              ga * (double)A[p * mhc::NAA + x];
      for (int k = 0; k < P.npair; ++k) {
        const int f = P.pair[k * 2], sp = P.pair[k * 2 + 1];
        const double w = (double)P.pairw[k] * coupling_w * ga;
        const double df = pair_f[(size_t)c][(size_t)a * P.npair + k];
        const double ds = pair_s[(size_t)c][(size_t)a * P.npair + k];
        for (int x = 0; x < mhc::NAA; ++x) {
          (*dprob)[((size_t)c + f) * mhc::NAA + x] +=
              w * ds * (double)A[f * mhc::NAA + x];
          (*dprob)[((size_t)c + sp) * mhc::NAA + x] +=
              w * df * (double)A[sp * mhc::NAA + x];
        }
      }
      // the burial term is shared by every allele, so it is weighted by
      // the SUM of the allele gradients -- adding it once per allele is
      // right and adding it once per window is not
      if (hydro_w != 0.0)
        for (int p = 0; p < mhc::CORE; ++p)
          for (int x = 0; x < mhc::NAA; ++x)
            (*dprob)[((size_t)c + p) * mhc::NAA + x] +=
                ga * hydro_w * (double)P.burial[p] * (double)mhc::HYDRO[x];
    }
  }
  return out;
}

}  // namespace human
