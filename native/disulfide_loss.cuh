// disulfide -- bc2's cysteine-pairing loss.
//
//   sequence      = softmax(logits)
//   cys_i         = sequence[i][C]
//   hardness      = masked_mean(max_a sequence[i][a], real)
//   bq[i][j]      = exp(-((|CB_i - CB_j| - 3.8) / 1.5)^2)
//   pairable      = (|i - j| >= 3) & real_i & real_j        <- ARRAY POSITION
//   pb[i][j]      = bq[i][j] * stop_grad(cys_j) * pairable[i][j]
//   chosen        = softmax(pb / 0.1, axis=-1)
//   mutual[i][j]  = chosen[i][j] * chosen[j][i]
//   sat_i         = sum_j pb[i][j] * mutual[i][j]
//   loss          = hardness * sum_i cys_i * relu(1 - sat_i) * real_i
//
// THREE THINGS THAT PRODUCE A WORKING DESIGN IF WRONG:
//
//   the cysteine probability appears TWICE and only one occurrence is
//   differentiable -- the partner's is under stop_gradient.  Differentiating
//   both double-counts and gives a gradient right in sign, wrong in scale.
//
//   the coordinates are CB, not CA.  Every other geometry loss here takes
//   CA, so the convenient default is the wrong one.
//
//   `|i - j| < 3` is on ARRAY POSITION, not residue_index.  Everywhere else
//   in this port the separation is in residue numbering; on a scaffold with
//   shifted numbering those differ.
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

namespace disul {

constexpr int NAA = 20, CYS = 4;          // ARNDC...: C is index 4

// bc2's amino_acid_probabilities, which is a PER-ROW decision:
//
//   jnp.where((p.min(-1, keepdims=True) >= 0) &
//             isclose(p.sum(-1, keepdims=True), 1.0),  p,  softmax(p))
//
// `keepdims=True` inside the where makes the test per residue, so a row
// that is ALREADY a distribution passes through untouched and only the
// others are softmaxed.  After the mutate stage the logit array holds BOTH
// -- one-hot rows where a residue was fixed, raw logits elsewhere -- so
// softmaxing the lot turns every settled residue back into a near-uniform
// smear.  disulfide is the only loss that goes through this; protease_sites
// and sequence_entropy take a plain softmax and are unaffected.
inline void aminoAcidProbabilities(const float* logits, int n,
                                   std::vector<float>* out) {
  out->assign((size_t)n * NAA, 0.f);
  for (int i = 0; i < n; ++i) {
    const float* l = logits + (size_t)i * NAA;
    float mn = 1e30f, sum = 0.f;
    for (int a = 0; a < NAA; ++a) { mn = std::min(mn, l[a]); sum += l[a]; }
    if (mn >= 0.f && std::fabs(sum - 1.f) <= 1e-5f + 1e-8f * std::fabs(sum)) {
      for (int a = 0; a < NAA; ++a) (*out)[(size_t)i * NAA + a] = l[a];
      continue;
    }
    float mx = -1e30f;
    for (int a = 0; a < NAA; ++a) mx = std::max(mx, l[a]);
    double z = 0;
    for (int a = 0; a < NAA; ++a) z += exp((double)l[a] - mx);
    for (int a = 0; a < NAA; ++a)
      (*out)[(size_t)i * NAA + a] = (float)(exp((double)l[a] - mx) / z);
  }
}

struct Out {
  float loss = 0.f;
  std::vector<float> dprob;               // d loss / d sequence prob  [n*20]
  std::vector<float> dcb;                 // d loss / d CB             [n*3]
};

inline Out disulfide(const std::vector<float>& prob,     // [n][20]
                     const std::vector<float>& cb,       // [n][3]
                     const std::vector<char>& cbok,      // CB resolved
                     const std::vector<char>& real, int n,
                     float distance, float sigma, int sep, float T,
                     bool want_grad) {
  Out o;
  std::vector<double> cys(n, 0.0), mx(n, 0.0);
  std::vector<int> amax(n, 0);
  double M = 0, hsum = 0;
  for (int i = 0; i < n; ++i) {
    cys[i] = prob[(size_t)i * NAA + CYS];
    double b = -1; int ba = 0;
    for (int a = 0; a < NAA; ++a) {
      const double v = prob[(size_t)i * NAA + a];
      if (v > b) { b = v; ba = a; }
    }
    mx[i] = b; amax[i] = ba;
    if (real[i]) { M += 1.0; hsum += b; }
  }
  const double H = M > 0 ? hsum / (M + 1e-8) : 0.0;

  std::vector<double> bq((size_t)n * n, 0.0), pb((size_t)n * n, 0.0);
  std::vector<char> pairable((size_t)n * n, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const bool ok = real[i] && real[j] && std::abs(i - j) >= sep;
      pairable[(size_t)i * n + j] = ok ? 1 : 0;
      if (!ok || !cbok[i] || !cbok[j]) continue;
      double d2 = 1e-8;
      for (int c = 0; c < 3; ++c) {
        const double dd = (double)cb[(size_t)i * 3 + c] - cb[(size_t)j * 3 + c];
        d2 += dd * dd;
      }
      const double d = sqrt(d2);
      const double q = exp(-((d - distance) / sigma) * ((d - distance) / sigma));
      bq[(size_t)i * n + j] = q;
      pb[(size_t)i * n + j] = q * cys[j];      // cys_j is stop_gradient'd
    }

  // chosen = softmax(pb / T, -1); mutual = chosen * chosen^T
  std::vector<double> ch((size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i) {
    double m = -1e300;
    for (int j = 0; j < n; ++j) m = std::max(m, pb[(size_t)i * n + j] / T);
    double z = 0;
    for (int j = 0; j < n; ++j) z += exp(pb[(size_t)i * n + j] / T - m);
    for (int j = 0; j < n; ++j)
      ch[(size_t)i * n + j] = exp(pb[(size_t)i * n + j] / T - m) / z;
  }
  std::vector<double> sat(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double s = 0;
    for (int j = 0; j < n; ++j)
      s += pb[(size_t)i * n + j] *
           (ch[(size_t)i * n + j] * ch[(size_t)j * n + i]);
    sat[i] = s;
  }
  double S = 0;
  for (int i = 0; i < n; ++i)
    if (real[i]) S += cys[i] * std::max(0.0, 1.0 - sat[i]);
  o.loss = (float)(H * S);
  if (!want_grad) return o;

  o.dprob.assign((size_t)n * NAA, 0.f);
  o.dcb.assign((size_t)n * 3, 0.f);
  // dL/dcys_i (the DIFFERENTIABLE occurrence only) and dL/dH
  std::vector<double> gsat(n, 0.0);
  for (int i = 0; i < n; ++i) {
    if (!real[i]) continue;
    o.dprob[(size_t)i * NAA + CYS] += (float)(H * std::max(0.0, 1.0 - sat[i]));
    gsat[i] = (sat[i] < 1.0) ? -H * cys[i] : 0.0;
  }
  // THE MAX'S GRADIENT IS SPLIT ACROSS TIES.  jnp.max sends the cotangent
  // to EVERY column that attains the maximum, divided equally -- and a tie
  // is not a measure-zero curiosity here: the logits are float16, so two
  // amino acids landing on the same probability is routine.  One row in
  // sixty on the gate campaign.  Giving it all to the first argmax made
  // that row's gradient twice the right size in one column and zero in
  // another, which only a gradient gate can see.
  for (int i = 0; i < n; ++i) {
    if (!real[i]) continue;
    int ties = 0;
    for (int a = 0; a < NAA; ++a)
      ties += (prob[(size_t)i * NAA + a] == (float)mx[i]) ? 1 : 0;
    if (ties == 0) { o.dprob[(size_t)i * NAA + amax[i]] +=
                         (float)(S / (M + 1e-8)); continue; }
    const double share = S / (M + 1e-8) / (double)ties;
    for (int a = 0; a < NAA; ++a)
      if (prob[(size_t)i * NAA + a] == (float)mx[i])
        o.dprob[(size_t)i * NAA + a] += (float)share;
  }

  // dL/dpb, through sat = sum_j pb * chosen_ij * chosen_ji
  std::vector<double> V(n, 0.0);
  for (int a = 0; a < n; ++a) {
    double v = 0;
    for (int i = 0; i < n; ++i)
      v += gsat[i] * pb[(size_t)i * n + a] *
           (ch[(size_t)i * n + a] * ch[(size_t)a * n + i]);
    V[a] = v;
  }
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      if (!pairable[(size_t)a * n + b]) continue;
      const double Mab = ch[(size_t)a * n + b] * ch[(size_t)b * n + a];
      const double Mba = ch[(size_t)b * n + a] * ch[(size_t)a * n + b];
      double g = gsat[a] * Mab
               + gsat[a] / T * (pb[(size_t)a * n + b] * Mab -
                                ch[(size_t)a * n + b] * sat[a])
               + gsat[b] * pb[(size_t)b * n + a] * Mba / T
               - ch[(size_t)a * n + b] * V[a] / T;
      if (g == 0) continue;
      // pb = bq * cys_b(stop_grad) * pairable  ->  only bq carries gradient
      const double gbq = g * cys[b];
      if (gbq == 0 || !cbok[a] || !cbok[b]) continue;
      double d2 = 1e-8, dv[3];
      for (int c = 0; c < 3; ++c) {
        dv[c] = (double)cb[(size_t)a * 3 + c] - cb[(size_t)b * 3 + c];
        d2 += dv[c] * dv[c];
      }
      const double d = sqrt(d2);
      const double dq = bq[(size_t)a * n + b] *
                        (-2.0 * (d - distance) / (sigma * sigma));
      for (int c = 0; c < 3; ++c) {
        const double t = gbq * dq * dv[c] / d;
        o.dcb[(size_t)a * 3 + c] += (float)t;
        o.dcb[(size_t)b * 3 + c] -= (float)t;
      }
    }
  return o;
}

}  // namespace disul
