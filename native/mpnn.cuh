// ProteinMPNN, IN C++, ON THE CPU.
//
// The last of bc2's six stages needs a SECOND NETWORK: the MPNN stage
// redesigns the binder's sequence on the backbone the trajectory produced
// and refolds it.  This is that network -- bindcraft/mpnn/modules.py,
// sample.py and score.py, transcribed.
//
// WHY THE CPU, in a directory that is otherwise all CUDA.  It is 1.66M
// parameters against AlphaFold's 93M, the hidden width is 128, and the
// decoder is AUTOREGRESSIVE: L sequential steps, each a handful of
// [48 x 384] x [384 x 128] matmuls.  That is launch-latency bound on a
// GPU and about two GFLOP in total -- well under a second here, with no
// kernel to get wrong and no device memory to find in the middle of a
// design step that is already holding 11 GB.  If it ever becomes the
// bottleneck the shapes are all here to move.
//
// Everything is deterministic except the gumbel draw in sample(): bc2
// configures augment_eps 0.0 and dropout 0.0 (proteinmpnn.py), so score()
// is a pure function of the backbone and mpnn_gate.py diffs it against
// bc2's own.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <random>
#include <vector>

#include "ckpt.cuh"

namespace mpnn {

constexpr int H = 128;           // hidden
constexpr int VOCAB = 21;
constexpr int NRBF = 16;
constexpr int NPOS = 16;         // positional embedding width
constexpr int MAXREL = 32;
constexpr int EDGE_IN = NPOS + NRBF * 25;   // 416
constexpr int NENC = 3, NDEC = 3;

// hk.Linear holds w as [in, out] and b as [out]
struct Linear {
  int in = 0, out = 0;
  const float* w = nullptr;
  const float* b = nullptr;
  void apply(const float* x, float* y) const {
    for (int o = 0; o < out; ++o) y[o] = b ? b[o] : 0.f;
    for (int i = 0; i < in; ++i) {
      const float xi = x[i];
      if (xi == 0.f) continue;
      const float* wr = w + (size_t)i * out;
      for (int o = 0; o < out; ++o) y[o] += xi * wr[o];
    }
  }
};

struct Norm {                    // hk.LayerNorm over the last axis
  const float* scale = nullptr;
  const float* offset = nullptr;
  int n = 0;
  void apply(float* x) const {
    double m = 0;
    for (int i = 0; i < n; ++i) m += x[i];
    m /= n;
    double v = 0;
    for (int i = 0; i < n; ++i) v += (x[i] - m) * (x[i] - m);
    v /= n;
    const float inv = (float)(1.0 / std::sqrt(v + 1e-5));
    for (int i = 0; i < n; ++i)
      x[i] = (float)((x[i] - m) * inv) * scale[i] + offset[i];
  }
};

// EXACT gelu, not the tanh approximation: modules.py pins
// `functools.partial(jax.nn.gelu, approximate=False)`, and the two differ
// by ~1e-3 at the knee -- enough to walk a 175-step autoregressive decode
// somewhere else.
inline float gelu(float x) {
  return 0.5f * x * (1.f + std::erf(x * 0.70710678118654752f));
}

struct Enc {
  Linear W1, W2, W3, W11, W12, W13, Win, Wout;
  Norm n1, n2, n3;
};
struct Dec {
  Linear W1, W2, W3, Win, Wout;
  Norm n1, n2;
};

struct Weights {
  Linear edge_embedding, pos_embedding, W_e, W_out;
  Norm norm_edges;
  const float* W_s = nullptr;        // [21, 128]
  Enc enc[NENC];
  Dec dec[NDEC];
  int k_neighbors = 48;
  std::map<std::string, ckpt::Array> store;
  bool ok = false;
};

inline const float* get(Weights& w, const std::string& key, size_t want) {
  auto it = w.store.find(key);
  if (it == w.store.end()) {
    fprintf(stderr, "mpnn: no %s in the checkpoint\n", key.c_str());
    return nullptr;
  }
  if (want && it->second.v.size() != want) {
    fprintf(stderr, "mpnn: %s is %zu floats, expected %zu\n", key.c_str(),
            it->second.v.size(), want);
    return nullptr;
  }
  return it->second.v.data();
}

inline bool load(Weights& w, const std::string& npz) {
  // the MPNN checkpoint separates module from parameter with '|', not the
  // '//' AlphaFold's uses
  std::map<std::string, std::map<std::string, ckpt::Array>> t;
  if (!ckpt::readNpz(npz, &t, "|")) return false;
  for (auto& mod : t)
    for (auto& pr : mod.second)
      w.store[mod.first + "|" + pr.first] = std::move(pr.second);
  // num_edges travels as its own entry with no separator; readNpz skips
  // those, so k comes from the file name's convention or the default 48.
  const std::string P = "protein_mpnn/~/";
  auto lin = [&](Linear& L, const std::string& k, int in, int out, bool bias) {
    L.in = in; L.out = out;
    L.w = get(w, k + "|w", (size_t)in * out);
    L.b = bias ? get(w, k + "|b", (size_t)out) : nullptr;
    return L.w && (!bias || L.b);
  };
  auto nrm = [&](Norm& N, const std::string& k, int n) {
    N.n = n;
    N.scale = get(w, k + "|scale", (size_t)n);
    N.offset = get(w, k + "|offset", (size_t)n);
    return N.scale && N.offset;
  };
  bool ok = true;
  ok &= lin(w.pos_embedding, P + "protein_features/~/positional_encodings/~/embedding_linear",
            2 * MAXREL + 2, NPOS, true);
  ok &= lin(w.edge_embedding, P + "protein_features/~/edge_embedding", EDGE_IN, H, false);
  ok &= nrm(w.norm_edges, P + "protein_features/~/norm_edges", H);
  ok &= lin(w.W_e, P + "W_e", H, H, true);
  ok &= lin(w.W_out, P + "W_out", H, VOCAB, true);
  w.W_s = get(w, P + "embed_token|W_s", (size_t)VOCAB * H);
  ok &= w.W_s != nullptr;
  for (int i = 0; i < NENC; ++i) {
    const std::string m = P + "enc_layer" + (i ? "_" + std::to_string(i) : "") + "/~/";
    const std::string e = "enc" + std::to_string(i);
    ok &= lin(w.enc[i].W1, m + e + "_W1", 3 * H, H, true);
    ok &= lin(w.enc[i].W2, m + e + "_W2", H, H, true);
    ok &= lin(w.enc[i].W3, m + e + "_W3", H, H, true);
    ok &= lin(w.enc[i].W11, m + e + "_W11", 3 * H, H, true);
    ok &= lin(w.enc[i].W12, m + e + "_W12", H, H, true);
    ok &= lin(w.enc[i].W13, m + e + "_W13", H, H, true);
    ok &= nrm(w.enc[i].n1, m + e + "_norm1", H);
    ok &= nrm(w.enc[i].n2, m + e + "_norm2", H);
    ok &= nrm(w.enc[i].n3, m + e + "_norm3", H);
    const std::string f = m + "position_wise_feed_forward/~/" + e + "_dense_";
    ok &= lin(w.enc[i].Win, f + "W_in", H, 4 * H, true);
    ok &= lin(w.enc[i].Wout, f + "W_out", 4 * H, H, true);
  }
  for (int i = 0; i < NDEC; ++i) {
    const std::string m = P + "dec_layer" + (i ? "_" + std::to_string(i) : "") + "/~/";
    const std::string e = "dec" + std::to_string(i);
    ok &= lin(w.dec[i].W1, m + e + "_W1", 4 * H, H, true);
    ok &= lin(w.dec[i].W2, m + e + "_W2", H, H, true);
    ok &= lin(w.dec[i].W3, m + e + "_W3", H, H, true);
    ok &= nrm(w.dec[i].n1, m + e + "_norm1", H);
    ok &= nrm(w.dec[i].n2, m + e + "_norm2", H);
    const std::string f = m + "position_wise_feed_forward/~/" + e + "_dense_";
    ok &= lin(w.dec[i].Win, f + "W_in", H, 4 * H, true);
    ok &= lin(w.dec[i].Wout, f + "W_out", 4 * H, H, true);
  }
  w.ok = ok;
  return ok;
}

// ---- the featuriser --------------------------------------------------------
struct Features {
  int L = 0, K = 0;
  std::vector<int> E_idx;            // [L, K]
  std::vector<float> h_E;            // [L, K, H]
};

struct Input {
  int L = 0;
  std::vector<float> X;              // [L, 4, 3]  N, CA, C, O
  std::vector<float> mask;           // [L]
  std::vector<int> residue_idx;      // [L]
  std::vector<int> chain_idx;        // [L]
};

inline void features(const Weights& w, const Input& in, Features* out) {
  const int L = in.L, K = std::min(w.k_neighbors, L);
  out->L = L; out->K = K;
  // Y = [N, CA, C, O, Cb]; Cb from the first three, exactly as modules.py
  std::vector<float> Y((size_t)5 * L * 3);
  auto Yp = [&](int a, int i) { return &Y[((size_t)a * L + i) * 3]; };
  for (int i = 0; i < L; ++i)
    for (int a = 0; a < 4; ++a)
      for (int c = 0; c < 3; ++c)
        Yp(a, i)[c] = in.X[((size_t)i * 4 + a) * 3 + c];
  for (int i = 0; i < L; ++i) {
    float b[3], c[3], cr[3];
    for (int k = 0; k < 3; ++k) { b[k] = Yp(1, i)[k] - Yp(0, i)[k];
                                  c[k] = Yp(2, i)[k] - Yp(1, i)[k]; }
    cr[0] = b[1]*c[2] - b[2]*c[1];
    cr[1] = b[2]*c[0] - b[0]*c[2];
    cr[2] = b[0]*c[1] - b[1]*c[0];
    for (int k = 0; k < 3; ++k)
      Yp(4, i)[k] = -0.58273431f * cr[k] + 0.56802827f * b[k]
                  - 0.54067466f * c[k] + Yp(1, i)[k];
  }
  // kNN on CA-CA.  A masked-out partner is pushed to the row's MAXIMUM
  // rather than to infinity, which is what _get_edge_idx does -- and it
  // matters, because a row with fewer than K real partners then keeps
  // whatever the max is instead of a sentinel.
  std::vector<float> D((size_t)L * L);
  for (int i = 0; i < L; ++i) {
    float mx = 0;
    for (int j = 0; j < L; ++j) {
      const float* a = Yp(1, i); const float* bb = Yp(1, j);
      const float dx = a[0]-bb[0], dy = a[1]-bb[1], dz = a[2]-bb[2];
      const float d = std::sqrt(dx*dx + dy*dy + dz*dz + 1e-6f);
      D[(size_t)i * L + j] = d;
      mx = std::max(mx, d);
    }
    for (int j = 0; j < L; ++j)
      if (!(in.mask[i] > 0 && in.mask[j] > 0)) D[(size_t)i * L + j] = mx;
  }
  out->E_idx.assign((size_t)L * K, 0);
  std::vector<int> ord(L);
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) ord[j] = j;
    const float* row = &D[(size_t)i * L];
    std::partial_sort(ord.begin(), ord.begin() + K, ord.end(),
                      [&](int a, int b) {
                        return row[a] != row[b] ? row[a] < row[b] : a < b; });
    for (int k = 0; k < K; ++k) out->E_idx[(size_t)i * K + k] = ord[k];
  }
  // 25 ordered atom pairs, in modules.py's order
  static const int PAIR[25][2] = {
      {1,1},{0,0},{2,2},{3,3},{4,4},{1,0},{1,2},{1,3},{1,4},{0,2},
      {0,3},{0,4},{4,2},{4,3},{3,2},{0,1},{2,1},{3,1},{4,1},{2,0},
      {3,0},{4,0},{2,4},{3,4},{2,3}};
  const float D_min = 2.f, D_max = 22.f;
  const float sigma = (D_max - D_min) / NRBF;
  std::vector<float> E((size_t)L * K * EDGE_IN, 0.f);
  for (int i = 0; i < L; ++i)
    for (int k = 0; k < K; ++k) {
      const int j = out->E_idx[(size_t)i * K + k];
      float* dst = &E[((size_t)i * K + k) * EDGE_IN + NPOS];
      for (int p = 0; p < 25; ++p) {
        const float* a = Yp(PAIR[p][0], i);
        const float* b = Yp(PAIR[p][1], j);
        const float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
        const float d = std::sqrt(dx*dx + dy*dy + dz*dz + 1e-6f);
        for (int r = 0; r < NRBF; ++r) {
          // jnp.linspace(2, 22, 16) -- endpoint INCLUDED, so the step is
          // 20/15 and not 20/16.  D_sigma is 20/16 regardless; the two
          // denominators are different numbers and swapping them is a
          // silent change of basis.
          const float mu = D_min + (D_max - D_min) * (float)r / (float)(NRBF - 1);
          const float z = (d - mu) / sigma;
          dst[p * NRBF + r] = std::exp(-z * z);
        }
      }
    }
  // positional encoding: clip(offset + 32, 0, 64) where the chains match,
  // 65 where they do not; one-hot over 66, then Linear(16)
  std::vector<float> oh(2 * MAXREL + 2);
  for (int i = 0; i < L; ++i)
    for (int k = 0; k < K; ++k) {
      const int j = out->E_idx[(size_t)i * K + k];
      const int same = in.chain_idx[i] == in.chain_idx[j] ? 1 : 0;
      const long off = (long)in.residue_idx[i] - in.residue_idx[j];
      long d = std::min<long>(std::max<long>(off + MAXREL, 0), 2 * MAXREL);
      d = d * same + (1 - same) * (2 * MAXREL + 1);
      std::fill(oh.begin(), oh.end(), 0.f);
      oh[(size_t)d] = 1.f;
      w.pos_embedding.apply(oh.data(), &E[((size_t)i * K + k) * EDGE_IN]);
    }
  // edge_embedding (no bias) then norm_edges, then W_e
  out->h_E.assign((size_t)L * K * H, 0.f);
  std::vector<float> tmp(H);
  for (int i = 0; i < L; ++i)
    for (int k = 0; k < K; ++k) {
      w.edge_embedding.apply(&E[((size_t)i * K + k) * EDGE_IN], tmp.data());
      w.norm_edges.apply(tmp.data());
      w.W_e.apply(tmp.data(), &out->h_E[((size_t)i * K + k) * H]);
    }
}

// ---- the encoder -----------------------------------------------------------
inline void encode(const Weights& w, const Input& in, Features& f,
                   std::vector<float>* h_V_out) {
  const int L = f.L, K = f.K;
  std::vector<float> h_V((size_t)L * H, 0.f);
  std::vector<float> ev(3 * H), a(H), b(H), msg(H), dh(H), ff(4 * H);
  for (int l = 0; l < NENC; ++l) {
    const Enc& E = w.enc[l];
    // pass one: the node update
    std::vector<float> acc((size_t)L * H, 0.f);
    for (int i = 0; i < L; ++i)
      for (int k = 0; k < K; ++k) {
        const int j = f.E_idx[(size_t)i * K + k];
        // h_EV = [ h_V[i] | h_E[i,k] | h_V[j] ]
        std::copy(&h_V[(size_t)i * H], &h_V[(size_t)i * H] + H, ev.begin());
        std::copy(&f.h_E[((size_t)i * K + k) * H],
                  &f.h_E[((size_t)i * K + k) * H] + H, ev.begin() + H);
        std::copy(&h_V[(size_t)j * H], &h_V[(size_t)j * H] + H, ev.begin() + 2 * H);
        E.W1.apply(ev.data(), a.data());
        for (int c = 0; c < H; ++c) a[c] = gelu(a[c]);
        E.W2.apply(a.data(), b.data());
        for (int c = 0; c < H; ++c) b[c] = gelu(b[c]);
        E.W3.apply(b.data(), msg.data());
        const float ma = in.mask[i] * in.mask[j];
        for (int c = 0; c < H; ++c) acc[(size_t)i * H + c] += ma * msg[c];
      }
    for (int i = 0; i < L; ++i) {
      for (int c = 0; c < H; ++c) h_V[(size_t)i * H + c] += acc[(size_t)i * H + c] / 30.f;
      E.n1.apply(&h_V[(size_t)i * H]);
      E.Win.apply(&h_V[(size_t)i * H], ff.data());
      for (int c = 0; c < 4 * H; ++c) ff[c] = gelu(ff[c]);
      E.Wout.apply(ff.data(), dh.data());
      for (int c = 0; c < H; ++c) h_V[(size_t)i * H + c] += dh[c];
      E.n2.apply(&h_V[(size_t)i * H]);
      for (int c = 0; c < H; ++c) h_V[(size_t)i * H + c] *= in.mask[i];
    }
    // pass two: the edge update, on the NEW h_V
    for (int i = 0; i < L; ++i)
      for (int k = 0; k < K; ++k) {
        const int j = f.E_idx[(size_t)i * K + k];
        std::copy(&h_V[(size_t)i * H], &h_V[(size_t)i * H] + H, ev.begin());
        std::copy(&f.h_E[((size_t)i * K + k) * H],
                  &f.h_E[((size_t)i * K + k) * H] + H, ev.begin() + H);
        std::copy(&h_V[(size_t)j * H], &h_V[(size_t)j * H] + H, ev.begin() + 2 * H);
        E.W11.apply(ev.data(), a.data());
        for (int c = 0; c < H; ++c) a[c] = gelu(a[c]);
        E.W12.apply(a.data(), b.data());
        for (int c = 0; c < H; ++c) b[c] = gelu(b[c]);
        E.W13.apply(b.data(), msg.data());
        float* e = &f.h_E[((size_t)i * K + k) * H];
        for (int c = 0; c < H; ++c) e[c] += msg[c];
        E.n3.apply(e);
      }
  }
  *h_V_out = std::move(h_V);
}

// one decoder layer over one position: h_node [H], h_E [K, 4H], mask scalar
inline void decLayer(const Dec& D, const float* h_node, const float* h_Ein,
                     int K, float mask_V, const float* mask_attend,
                     float* out) {
  std::vector<float> ev(4 * H), a(H), b(H), msg(H), acc(H, 0.f), ff(4 * H), dh(H);
  for (int k = 0; k < K; ++k) {
    std::copy(h_node, h_node + H, ev.begin());
    std::copy(h_Ein + (size_t)k * 3 * H, h_Ein + (size_t)(k + 1) * 3 * H,
              ev.begin() + H);
    D.W1.apply(ev.data(), a.data());
    for (int c = 0; c < H; ++c) a[c] = gelu(a[c]);
    D.W2.apply(a.data(), b.data());
    for (int c = 0; c < H; ++c) b[c] = gelu(b[c]);
    D.W3.apply(b.data(), msg.data());
    const float m = mask_attend ? mask_attend[k] : 1.f;
    for (int c = 0; c < H; ++c) acc[c] += m * msg[c];
  }
  for (int c = 0; c < H; ++c) out[c] = h_node[c] + acc[c] / 30.f;
  D.n1.apply(out);
  D.Win.apply(out, ff.data());
  for (int c = 0; c < 4 * H; ++c) ff[c] = gelu(ff[c]);
  D.Wout.apply(ff.data(), dh.data());
  for (int c = 0; c < H; ++c) out[c] += dh[c];
  D.n2.apply(out);
  for (int c = 0; c < H; ++c) out[c] *= mask_V;
}


// ---- score(): the DETERMINISTIC path, which is what the gate can use -----
//
// bc2 configures augment_eps 0.0 and dropout 0.0, so with a given sequence
// and a given autoregressive mask this is a pure function of the backbone.
// sample() differs from it only by drawing each residue instead of being
// handed it, so a score() that matches is a sample() that matches up to the
// gumbel.
//
// `S` is [L, 21] in the MPNN alphabet; `ar_mask` is [L, L] with 1 where i
// may see j.
inline void score(const Weights& w, const Input& in,
                  const std::vector<float>& S,
                  const std::vector<float>& ar_mask,
                  std::vector<float>* logits) {
  Features f;
  features(w, in, &f);
  const int L = f.L, K = f.K;
  std::vector<float> h_V;
  encode(w, in, f, &h_V);

  // h_S = W_s(S), a [21,128] embedding table applied to a soft one-hot
  std::vector<float> h_S((size_t)L * H, 0.f);
  for (int i = 0; i < L; ++i)
    for (int v = 0; v < VOCAB; ++v) {
      const float s = S[(size_t)i * VOCAB + v];
      if (s == 0.f) continue;
      for (int c = 0; c < H; ++c)
        h_S[(size_t)i * H + c] += s * w.W_s[(size_t)v * H + c];
    }

  // h_EXV_encoder = [ h_E | 0 | h_V[nb] ]   (cat_neighbors_nodes twice)
  // h_ES          = [ h_E | h_S[nb] ]
  std::vector<float> hEXV((size_t)L * K * 3 * H, 0.f);
  std::vector<float> hES((size_t)L * K * 2 * H, 0.f);
  for (int i = 0; i < L; ++i)
    for (int k = 0; k < K; ++k) {
      const int j = f.E_idx[(size_t)i * K + k];
      const float* e = &f.h_E[((size_t)i * K + k) * H];
      float* x = &hEXV[((size_t)i * K + k) * 3 * H];
      std::copy(e, e + H, x);                              // h_E
      // the middle H stays ZERO: cat_neighbors_nodes(zeros_like(h_V), ...)
      std::copy(&h_V[(size_t)j * H], &h_V[(size_t)j * H] + H, x + 2 * H);
      float* y = &hES[((size_t)i * K + k) * 2 * H];
      std::copy(e, e + H, y);
      std::copy(&h_S[(size_t)j * H], &h_S[(size_t)j * H] + H, y + H);
    }

  std::vector<float> esv((size_t)K * 3 * H), hv_new((size_t)L * H);
  for (int l = 0; l < NDEC; ++l) {
    for (int i = 0; i < L; ++i) {
      for (int k = 0; k < K; ++k) {
        const int j = f.E_idx[(size_t)i * K + k];
        const float att = ar_mask[(size_t)i * L + j];
        const float bw = in.mask[i] * att, fw = in.mask[i] * (1.f - att);
        float* d = &esv[(size_t)k * 3 * H];
        const float* es = &hES[((size_t)i * K + k) * 2 * H];
        const float* ex = &hEXV[((size_t)i * K + k) * 3 * H];
        // h_ESV = mask_bw * [ h_E | h_S[nb] | h_V[nb] ] + mask_fw * h_EXV
        for (int c = 0; c < 2 * H; ++c) d[c] = bw * es[c] + fw * ex[c];
        for (int c = 0; c < H; ++c)
          d[2 * H + c] = bw * h_V[(size_t)j * H + c] + fw * ex[2 * H + c];
      }
      decLayer(w.dec[l], &h_V[(size_t)i * H], esv.data(), K, in.mask[i],
               nullptr, &hv_new[(size_t)i * H]);
    }
    h_V.swap(hv_new);
  }
  logits->assign((size_t)L * VOCAB, 0.f);
  for (int i = 0; i < L; ++i)
    w.W_out.apply(&h_V[(size_t)i * H], &(*logits)[(size_t)i * VOCAB]);
}


// ---- the alphabets -------------------------------------------------------
// AF2's is "ARNDCQEGHILKMFPSTWYVX" and ProteinMPNN's is
// "ACDEFGHIKLMNPQRSTVWYX".  They are the same 21 letters in a DIFFERENT
// ORDER, and nothing raises if the mapping is dropped -- the sequence just
// comes back scrambled in a way that still folds to something.
static const char AF2_AA[22]  = "ARNDCQEGHILKMFPSTWYVX";
static const char MPNN_AA[22] = "ACDEFGHIKLMNPQRSTVWYX";
inline int af2ToMpnn(int a) {
  for (int i = 0; i < 21; ++i) if (MPNN_AA[i] == AF2_AA[a]) return i;
  return 20;
}
inline int mpnnToAf2(int a) {
  for (int i = 0; i < 21; ++i) if (AF2_AA[i] == MPNN_AA[a]) return i;
  return 20;
}

// ---- sample(): the autoregressive decode ---------------------------------
//
// One residue at a time in `decoding_order`, each conditioned on the ones
// already placed.  `fixed` holds the positions whose sequence is given --
// the target's -- and they are decoded FIRST (priority -1 in
// proteinmpnn_input_features) with a 1e7 bias on their own letter, so the
// redesigned positions see a real neighbourhood rather than an empty one.
//
// Returns the sampled sequence in the MPNN alphabet, [L] indices.
// `force` (optional) pins the residue chosen at every position and
// `order_in` (optional) pins the decoding order.  With both supplied the
// sampler becomes DETERMINISTIC and its per-position logits are exactly
// what score() computes for that sequence and that autoregressive mask --
// which is how the SAMPLING LOOP gets gated at all.  Its masks (mask_bw,
// mask_fw, h_EXV) exist nowhere else, and a wrong one gives sequences
// that still look like proteins.
// TIED GROUPS: one letter for several positions.
//
// `predict_tied_candidates` puts every target state in ONE ProteinMPNN
// input, 1000 A apart so no state sees another's backbone, and ties the
// binder's rows across them: the group is decoded in one step, its
// members' logits are AVERAGED, and a single gumbel draw picks the letter
// for all of them.  bc2's own note on why the mean and one draw: averaging
// a draw per state shrinks it by sqrt(N) and quietly lowers the sampling
// temperature.
//
// `groups` is the full partition -- every position appears in exactly one
// group, untied ones in a group of their own -- which is what
// `tied_chain_residue_groups` builds (it pads the singletons out to the
// group width, and a repeated index is the same position written twice).
inline void sample(const Weights& w, const Input& in,
                   const std::vector<char>& fixed,
                   const std::vector<int>& fixed_aa_mpnn,
                   float temperature, uint32_t seed,
                   std::vector<int>* out_seq,
                   const std::vector<int>* force = nullptr,
                   const std::vector<int>* order_in = nullptr,
                   std::vector<float>* out_logits = nullptr,
                   const std::vector<std::vector<int>>* groups = nullptr) {
  Features f;
  features(w, in, &f);
  const int L = f.L, K = f.K;
  std::vector<float> h_V_enc;
  encode(w, in, f, &h_V_enc);

  // the decoding order: a uniform priority per position, minus one where
  // the sequence is fixed and plus one where the residue is unresolved
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  std::vector<std::pair<float, int>> pri((size_t)L);
  for (int i = 0; i < L; ++i) {
    float p = U(rng);
    if (!(in.mask[i] > 0)) p += 1.f;
    if (fixed[i]) p -= 1.f;
    pri[i] = {p, i};
  }
  std::sort(pri.begin(), pri.end());
  std::vector<int> order((size_t)L), step((size_t)L);
  for (int t = 0; t < L; ++t) { order[t] = pri[t].second; step[order[t]] = t; }
  if (order_in) {
    order = *order_in;
    for (int t = 0; t < L; ++t) step[order[t]] = t;
  }
  // the groups, in the order of their FIRST member's priority -- bc2's
  // `tied_residue_groups[decoding_priorities[tied_residue_groups[:, 0]]
  // .argsort()]` -- and every member of a group shares its step, which is
  // what makes ar_mask treat them as one decoding position.
  std::vector<std::vector<int>> plan;
  if (groups) {
    std::vector<float> key((size_t)L, 0.f);
    if (order_in) {
      // a caller that names the order gets it: the groups follow the
      // position of their FIRST member in it.  (The gate drives the
      // decode this way; a campaign leaves it to the priorities.)
      for (int t = 0; t < L; ++t) key[(size_t)(*order_in)[(size_t)t]] = (float)t;
    } else {
      for (int i = 0; i < L; ++i)
        key[(size_t)pri[(size_t)i].second] = pri[(size_t)i].first;
    }
    std::vector<std::pair<float, int>> gp;
    for (size_t g = 0; g < groups->size(); ++g)
      gp.push_back({key[(size_t)(*groups)[g].front()], (int)g});
    std::stable_sort(gp.begin(), gp.end());
    for (const auto& q : gp) plan.push_back((*groups)[(size_t)q.second]);
    for (size_t t = 0; t < plan.size(); ++t)
      for (int i : plan[t]) step[(size_t)i] = (int)t;
  } else {
    for (int t = 0; t < L; ++t) plan.push_back({order[t]});
  }
  if (out_logits) out_logits->assign((size_t)L * VOCAB, 0.f);

  // h_EXV_encoder = [ h_E | 0 | h_V[nb] ] * mask_fw
  std::vector<float> hEXV((size_t)L * K * 3 * H, 0.f);
  for (int i = 0; i < L; ++i)
    for (int k = 0; k < K; ++k) {
      const int j = f.E_idx[(size_t)i * K + k];
      const float att = step[i] > step[j] ? 1.f : 0.f;   // ar_mask[i][j]
      const float fw = in.mask[i] * (1.f - att);
      float* x = &hEXV[((size_t)i * K + k) * 3 * H];
      const float* e = &f.h_E[((size_t)i * K + k) * H];
      for (int c = 0; c < H; ++c) x[c] = fw * e[c];
      for (int c = 0; c < H; ++c)
        x[2 * H + c] = fw * h_V_enc[(size_t)j * H + c];
    }

  // h_V[l] for l = 0..NDEC, h_V[0] being the encoder's
  std::vector<std::vector<float>> hV(NDEC + 1,
                                     std::vector<float>((size_t)L * H, 0.f));
  hV[0] = h_V_enc;
  std::vector<float> h_S((size_t)L * H, 0.f);
  out_seq->assign((size_t)L, 20);

  std::vector<float> esv((size_t)K * 3 * H), node(H), logits(VOCAB);
  std::vector<float> glog(VOCAB);
  for (size_t gt = 0; gt < plan.size(); ++gt) {
   // the group's members are decoded against the state as it stands at
   // the START of the step: each writes hV[l+1] and reads hV[l], and h_S
   // is written only once the whole group has a letter.
   std::fill(glog.begin(), glog.end(), 0.f);
   int members = 0;
   for (int i : plan[gt]) {
    ++members;
    for (int l = 0; l < NDEC; ++l) {
      for (int k = 0; k < K; ++k) {
        const int j = f.E_idx[(size_t)i * K + k];
        const float att = step[i] > step[j] ? 1.f : 0.f;
        const float bw = in.mask[i] * att;
        float* dst = &esv[(size_t)k * 3 * H];
        const float* e = &f.h_E[((size_t)i * K + k) * H];
        const float* ex = &hEXV[((size_t)i * K + k) * 3 * H];
        // mask_bw * [ h_E | h_S[nb] | h_V_l[nb] ] + h_EXV_encoder
        for (int c = 0; c < H; ++c) dst[c] = bw * e[c] + ex[c];
        for (int c = 0; c < H; ++c)
          dst[H + c] = bw * h_S[(size_t)j * H + c] + ex[H + c];
        for (int c = 0; c < H; ++c)
          dst[2 * H + c] = bw * hV[l][(size_t)j * H + c] + ex[2 * H + c];
      }
      decLayer(w.dec[l], &hV[l][(size_t)i * H], esv.data(), K, in.mask[i],
               nullptr, node.data());
      std::copy(node.begin(), node.end(), &hV[l + 1][(size_t)i * H]);
    }
    w.W_out.apply(node.data(), logits.data());
    if (out_logits)
      std::copy(logits.begin(), logits.end(),
                out_logits->begin() + (size_t)i * VOCAB);
    // the bias: a fixed position is pinned to its own letter.  Added
    // BEFORE the mean, as bc2 adds it before `logits_t.mean(0)`.
    if (fixed[i]) logits[(size_t)fixed_aa_mpnn[i]] += 1e7f;
    for (int v = 0; v < VOCAB; ++v) glog[(size_t)v] += logits[(size_t)v];
   }
   // the MEAN over the group, then ONE gumbel draw, and the argmax over
   // the FIRST TWENTY -- X is never sampled
   int best = 0; float bv = -1e30f;
   for (int v = 0; v < 20; ++v) {
     const float u = std::max(U(rng), 1e-20f);
     const float g = -std::log(-std::log(u));
     const float z = glog[(size_t)v] / (float)members / temperature + g;
     if (z > bv) { bv = z; best = v; }
   }
   if (force) best = (*force)[(size_t)plan[gt].front()];   // teacher forcing
   for (int i : plan[gt]) {
     (*out_seq)[(size_t)i] = best;
     for (int c = 0; c < H; ++c)
       h_S[(size_t)i * H + c] = w.W_s[(size_t)best * H + c];
   }
  }
}

}  // namespace mpnn