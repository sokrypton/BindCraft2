// protease_sites -- bc2's developability panel, on the host.
//
// A function of the SEQUENCE alone, like sequence_entropy: its gradient
// goes straight onto the logits and no cotangent reaches the model.
//
// Four proteases, each with a P1 preference and a P1' block:
//
//   trypsin       1.0   K,R          blocked by P at P1'
//   chymotrypsin  1.0   F,Y,W, L/M .5   blocked by P
//   elastase      0.4   A,V, G/S .6, L/I .4   blocked by P
//   pepsin        0.5   F,L,W,Y      no block
//
// score_i = sum_k w_k * p1_k(i) * (1 - block_k(i+1))
// loss    = masked_mean(score, redesignable)
//
// THE P1' TERM IS 1.0 WHERE THE NEXT RESIDUE IS NOT REAL, not 0 -- so the
// last residue of a chain contributes NOTHING (1 - 1 = 0).  A cleavage site
// needs a real residue on both sides, and defaulting that to 0 would invent
// a site at every chain end.
#pragma once
#include <cmath>
#include <vector>

namespace devel {

constexpr int NPROT = 4, NAA = 20;
// AF2's alphabet, which is bc2's AMINO_ACIDS: checked, not assumed
// A R N D C Q E G H I L K M F P S T W Y V
//  0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
inline const float* p1Table() {
  static float t[NPROT][NAA] = {};
  static bool init = false;
  if (!init) {
    auto set = [&](int k, int a, float v) { t[k][a] = v; };
    set(0, 11, 1.0f); set(0, 1, 1.0f);                       // trypsin K,R
    set(1, 13, 1.0f); set(1, 18, 1.0f); set(1, 17, 1.0f);    // chymo F,Y,W
    set(1, 10, 0.5f); set(1, 12, 0.5f);                      //        L,M
    set(2, 0, 1.0f);  set(2, 19, 1.0f);                      // elastase A,V
    set(2, 7, 0.6f);  set(2, 15, 0.6f);                      //          G,S
    set(2, 10, 0.4f); set(2, 9, 0.4f);                       //          L,I
    set(3, 13, 1.0f); set(3, 10, 1.0f);                      // pepsin F,L
    set(3, 17, 1.0f); set(3, 18, 1.0f);                      //        W,Y
    init = true;
  }
  return &t[0][0];
}
inline const float* blockTable() {
  static float t[NPROT][NAA] = {};
  static bool init = false;
  if (!init) { t[0][14] = 1.f; t[1][14] = 1.f; t[2][14] = 1.f; init = true; }
  return &t[0][0];                                  // pepsin is not blocked
}
inline const float* weightTable() {
  static const float w[NPROT] = {1.0f, 1.0f, 0.4f, 0.5f};
  return w;
}

// `prob` is [n][20] softmax of the logits; `mask` the redesignable rows.
// Returns the loss and, when `dprob` is non-null, d(loss)/d(prob).
inline float proteaseScore(const std::vector<float>& prob,
                           const std::vector<char>& mask, int n,
                           std::vector<float>* dprob) {
  const float* P1 = p1Table();
  const float* BL = blockTable();
  const float* W = weightTable();
  std::vector<float> p1p((size_t)n * NPROT, 0.f), blp((size_t)n * NPROT, 0.f);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < NPROT; ++k) {
      float a = 0, b = 0;
      for (int x = 0; x < NAA; ++x) {
        a += prob[(size_t)i * NAA + x] * P1[k * NAA + x];
        b += prob[(size_t)i * NAA + x] * BL[k * NAA + x];
      }
      p1p[(size_t)i * NPROT + k] = a;
      blp[(size_t)i * NPROT + k] = b;
    }
  double msum = 0, acc = 0;
  std::vector<float> per(n, 0.f);
  for (int i = 0; i < n; ++i) {
    if (!mask[i]) continue;
    msum += 1.0;
    float s = 0;
    for (int k = 0; k < NPROT; ++k) {
      // 1.0 when the next residue is not real -- the site is blocked
      const float blk = (i + 1 < n && mask[i + 1])
                            ? blp[(size_t)(i + 1) * NPROT + k] : 1.0f;
      s += W[k] * p1p[(size_t)i * NPROT + k] * (1.0f - blk);
    }
    per[i] = s;
    acc += s;
  }
  const float loss = msum > 0 ? (float)(acc / (msum + 1e-8)) : 0.f;
  if (dprob) {
    dprob->assign((size_t)n * NAA, 0.f);
    const double inv = msum > 0 ? 1.0 / (msum + 1e-8) : 0.0;
    for (int i = 0; i < n; ++i) {
      if (!mask[i]) continue;
      for (int k = 0; k < NPROT; ++k) {
        const bool nxt = (i + 1 < n && mask[i + 1]);
        const float blk = nxt ? blp[(size_t)(i + 1) * NPROT + k] : 1.0f;
        // d/d p1p(i,k)
        const double g1 = inv * W[k] * (1.0 - blk);
        for (int x = 0; x < NAA; ++x)
          (*dprob)[(size_t)i * NAA + x] += (float)(g1 * P1[k * NAA + x]);
        // d/d blp(i+1,k), only when the next residue is real
        if (nxt) {
          const double g2 = -inv * W[k] * p1p[(size_t)i * NPROT + k];
          for (int x = 0; x < NAA; ++x)
            (*dprob)[(size_t)(i + 1) * NAA + x] += (float)(g2 * BL[k * NAA + x]);
        }
      }
    }
  }
  return loss;
}

}  // namespace devel
