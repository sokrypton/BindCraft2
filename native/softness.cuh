// collective_softness -- the elastic-network loss, on the host.
//
// bc2 builds a 3N x 3N anisotropic-network Hessian from the binder's CA
// coordinates, projects out the six rigid-body modes, inverts it, and reads
// two numbers off the resulting covariance: how MANY residues the softest
// motions involve (`collectivity`, the exponential of the fluctuation-share
// entropy) and how much of the total fluctuation the softest mode carries
// (`concentration`, a Rayleigh quotient after twelve power iterations).
// The loss is 1 - collectivity * concentration: a binder that breathes as
// one piece scores low.
//
//   loss.py:775  elastic_network_covariance
//   loss.py:794  collective_softness_loss
//
// A function of the predicted CA alone, so it lives here with the other
// host losses and its cotangent is scattered into c_atoms at the CA slots.
// Unlike those, it is not a few hundred floats: the forward is one dense
// LU of a 3N x 3N matrix and the backward is two dense products of the
// same size.  Everything ELSE is kept to O(6 * (3N)^2) by never forming
// the projector -- P = I - Q Q^T with Q only 3N x 6, so P X is
// X - Q (Q^T X).  MEASURED, forward and backward together: 0.02 s at
// N=60, 0.15 s at N=120, 0.49 s at N=180 -- so on a 180-mer this term
// roughly doubles the design step, and on a 60-mer it is noise.  Only
// paid when the campaign weights it, which no shipped preset does.
//
// THE BACKWARD GOES THROUGH A QR AND AN INVERSE.  Both have closed forms:
//
//   inv:  Z = A^-1          =>  gA = -Z^T gZ Z^T
//   qr:   A = Q R (thin)    =>  gA = (gQ + Q W) R^-T,
//                               W = copyltu(-gQ^T Q)
//
// and the QR rule is the transpose of JAX's own qr_jvp_rule, derived
// rather than copied: with A = Q^T gQ and K = tril(A - A^T, -1), JAX's
// JVP transposes to gA = (gQ + Q (K - A)) R^-T, which is the same matrix
// -- (K - A)_ij is -A_ji above the diagonal, -A_ij below it, -A_ii on it.
// Q's SIGN does not matter: Q -> -Q sends R -> -R and gQ -> -gQ, and the
// formula is invariant, so this does not depend on LAPACK's choice.
//
// K IS IDENTICALLY ZERO FOR THIS LOSS AND THE GATE CANNOT SEE IT.  K is
// the part of the gradient that rotates the basis within its own span,
// and the loss reads Q only through P = I - Q Q^T, which is invariant
// under Q -> Q O.  Perturbing K by 2% moves softness_gate by nothing,
// where perturbing the inverse rule, the normalisation or the distance
// derivative each move it by 1e-2 or more.  It is kept because it is the
// correct general rule and the next loss to take a QR may not be
// gauge-invariant -- but it is UNCHECKED, and saying so is the point.
//
// The masks are NOT differentiated (they are the CA atom mask, a
// selection).  A masked row is anchored with PADDING_ANCHOR_STIFFNESS on
// its own diagonal block, which is what keeps the Hessian invertible when
// the state is padded.
#pragma once
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

namespace softloss {

constexpr double PADDING_ANCHOR_STIFFNESS = 1000.0;
constexpr int RIGID = 6;

struct Out {
  double value = 0.0;
  double collectivity = 0.0;   // reported, not scored
  double concentration = 0.0;
  std::vector<double> dx;      // N*3 cotangent on the CA coordinates
};

// ---- dense helpers, all row-major --------------------------------------

// C = A B, A is n x k, B is k x m.  ikj order: one pass over B per row of
// A, which is the only thing that keeps the 3N^3 terms tolerable.
inline void gemm(const double* A, const double* B, double* C,
                 int n, int k, int m) {
  for (int i = 0; i < n; ++i) {
    double* c = C + (size_t)i * m;
    for (int j = 0; j < m; ++j) c[j] = 0.0;
    for (int p = 0; p < k; ++p) {
      const double a = A[(size_t)i * k + p];
      if (a == 0.0) continue;
      const double* b = B + (size_t)p * m;
      for (int j = 0; j < m; ++j) c[j] += a * b[j];
    }
  }
}

// C = A^T B, A is n x k (so A^T is k x n), B is n x m
inline void gemmTN(const double* A, const double* B, double* C,
                   int n, int k, int m) {
  for (int p = 0; p < k; ++p)
    for (int j = 0; j < m; ++j) C[(size_t)p * m + j] = 0.0;
  for (int i = 0; i < n; ++i)
    for (int p = 0; p < k; ++p) {
      const double a = A[(size_t)i * k + p];
      if (a == 0.0) continue;
      const double* b = B + (size_t)i * m;
      double* c = C + (size_t)p * m;
      for (int j = 0; j < m; ++j) c[j] += a * b[j];
    }
}

// in-place inverse by Gauss-Jordan with partial pivoting; returns false on
// a singular matrix rather than writing nan into the loss
inline bool inverse(std::vector<double>& A, int n) {
  std::vector<double> I((size_t)n * n, 0.0);
  for (int i = 0; i < n; ++i) I[(size_t)i * n + i] = 1.0;
  for (int c = 0; c < n; ++c) {
    int piv = c; double best = std::fabs(A[(size_t)c * n + c]);
    for (int r = c + 1; r < n; ++r) {
      const double v = std::fabs(A[(size_t)r * n + c]);
      if (v > best) { best = v; piv = r; }
    }
    if (best == 0.0) return false;
    if (piv != c)
      for (int j = 0; j < n; ++j) {
        std::swap(A[(size_t)c * n + j], A[(size_t)piv * n + j]);
        std::swap(I[(size_t)c * n + j], I[(size_t)piv * n + j]);
      }
    const double d = 1.0 / A[(size_t)c * n + c];
    for (int j = 0; j < n; ++j) {
      A[(size_t)c * n + j] *= d; I[(size_t)c * n + j] *= d;
    }
    for (int r = 0; r < n; ++r) {
      if (r == c) continue;
      const double f = A[(size_t)r * n + c];
      if (f == 0.0) continue;
      for (int j = 0; j < n; ++j) {
        A[(size_t)r * n + j] -= f * A[(size_t)c * n + j];
        I[(size_t)r * n + j] -= f * I[(size_t)c * n + j];
      }
    }
  }
  A.swap(I);
  return true;
}

// thin QR of A (m x RIGID) by Householder, the same factorisation LAPACK's
// geqrf+orgqr produce (including the sign of R's diagonal, which does not
// matter here -- see the note at the top).
inline void thinQR(const std::vector<double>& Ain, int m,
                   std::vector<double>& Q, std::vector<double>& R) {
  std::vector<double> A = Ain;
  std::vector<std::vector<double>> vs;
  R.assign(RIGID * RIGID, 0.0);
  for (int k = 0; k < RIGID; ++k) {
    double nrm = 0.0;
    for (int i = k; i < m; ++i) nrm += A[(size_t)i * RIGID + k] * A[(size_t)i * RIGID + k];
    nrm = sqrt(nrm);
    std::vector<double> v((size_t)m, 0.0);
    if (nrm > 0.0) {
      const double a0 = A[(size_t)k * RIGID + k];
      const double alpha = a0 >= 0 ? -nrm : nrm;
      for (int i = k; i < m; ++i) v[i] = A[(size_t)i * RIGID + k];
      v[k] -= alpha;
      double vn = 0.0;
      for (int i = k; i < m; ++i) vn += v[i] * v[i];
      if (vn > 0.0) {
        const double s = 1.0 / sqrt(vn);
        for (int i = k; i < m; ++i) v[i] *= s;
        // A <- (I - 2 v v^T) A
        for (int j = k; j < RIGID; ++j) {
          double d = 0.0;
          for (int i = k; i < m; ++i) d += v[i] * A[(size_t)i * RIGID + j];
          d *= 2.0;
          for (int i = k; i < m; ++i) A[(size_t)i * RIGID + j] -= d * v[i];
        }
      } else {
        for (int i = 0; i < m; ++i) v[i] = 0.0;
      }
    }
    vs.push_back(v);
  }
  for (int i = 0; i < RIGID; ++i)
    for (int j = i; j < RIGID; ++j) R[(size_t)i * RIGID + j] = A[(size_t)i * RIGID + j];
  // Q = H_0 H_1 ... H_5 applied to the first RIGID columns of I
  Q.assign((size_t)m * RIGID, 0.0);
  for (int j = 0; j < RIGID; ++j) Q[(size_t)j * RIGID + j] = 1.0;
  for (int k = RIGID - 1; k >= 0; --k) {
    const std::vector<double>& v = vs[(size_t)k];
    for (int j = 0; j < RIGID; ++j) {
      double d = 0.0;
      for (int i = k; i < m; ++i) d += v[i] * Q[(size_t)i * RIGID + j];
      d *= 2.0;
      if (d == 0.0) continue;
      for (int i = k; i < m; ++i) Q[(size_t)i * RIGID + j] -= d * v[i];
    }
  }
}

// solve X R^T = Y for X, R upper triangular (RIGID x RIGID), row by row
inline void solveRT(const std::vector<double>& R, std::vector<double>& Y, int m) {
  for (int i = 0; i < m; ++i) {
    double* y = &Y[(size_t)i * RIGID];
    for (int k = RIGID - 1; k >= 0; --k) {
      double s = y[k];
      for (int j = k + 1; j < RIGID; ++j) s -= R[(size_t)k * RIGID + j] * y[j];
      const double d = R[(size_t)k * RIGID + k];
      y[k] = d != 0.0 ? s / d : 0.0;
    }
  }
}

// X <- P X, with P = I - Q Q^T; X is m x c
inline void projectLeft(const std::vector<double>& Q, std::vector<double>& X,
                        int m, int c) {
  std::vector<double> t((size_t)RIGID * c, 0.0);
  gemmTN(Q.data(), X.data(), t.data(), m, RIGID, c);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < c; ++j) {
      double s = 0.0;
      for (int p = 0; p < RIGID; ++p) s += Q[(size_t)i * RIGID + p] * t[(size_t)p * c + j];
      X[(size_t)i * c + j] -= s;
    }
}

// X <- X P, X is r x m
inline void projectRight(const std::vector<double>& Q, std::vector<double>& X,
                         int r, int m) {
  std::vector<double> t((size_t)r * RIGID, 0.0);
  gemm(X.data(), Q.data(), t.data(), r, m, RIGID);   // t = X Q
  for (int i = 0; i < r; ++i)
    for (int j = 0; j < m; ++j) {
      double s = 0.0;
      for (int p = 0; p < RIGID; ++p) s += t[(size_t)i * RIGID + p] * Q[(size_t)j * RIGID + p];
      X[(size_t)i * m + j] -= s;
    }
}

// ---- the loss -----------------------------------------------------------
//
// X is N*3 CA coordinates, mask is N (1.0 for a real residue).  `cd` is
// contact_decay, `damping` the Tikhonov factor, `iters` the power
// iterations.  want_grad fills Out::dx.
inline Out softness(const std::vector<double>& X, const std::vector<double>& mask,
                    int N, double cd, double damping, int iters, double eps,
                    bool want_grad) {
  Out out;
  if (N <= 0) return out;
  const int M = 3 * N;
  const double cd2 = cd * cd;

  // --- the Hessian ---------------------------------------------------
  std::vector<double> H((size_t)M * M, 0.0);
  {
    std::vector<double> anch((size_t)N * 9, 0.0);
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        double o[3], d2 = 0.0;
        for (int a = 0; a < 3; ++a) {
          o[a] = X[(size_t)j * 3 + a] - X[(size_t)i * 3 + a];
          d2 += o[a] * o[a];
        }
        const double r = 1.0 / sqrt(d2 + eps);
        const double k = exp(-d2 / cd2) * mask[i] * mask[j];
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            const double blk = -k * (o[a] * r) * (o[b] * r);
            H[(size_t)(3 * i + a) * M + (3 * j + b)] = blk;
            anch[(size_t)i * 9 + a * 3 + b] -= blk;
          }
      }
    for (int i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          H[(size_t)(3 * i + a) * M + (3 * i + b)] =
              anch[(size_t)i * 9 + a * 3 + b] +
              (a == b ? (1.0 - mask[i]) * PADDING_ANCHOR_STIFFNESS : 0.0);
  }

  // --- the rigid-body basis ------------------------------------------
  double msum = 0.0, cen[3] = {0, 0, 0};
  for (int i = 0; i < N; ++i) {
    msum += mask[i];
    for (int a = 0; a < 3; ++a) cen[a] += X[(size_t)i * 3 + a] * mask[i];
  }
  for (int a = 0; a < 3; ++a) cen[a] /= (msum + 1e-8);
  std::vector<double> cc((size_t)N * 3);
  for (int i = 0; i < N; ++i)
    for (int a = 0; a < 3; ++a)
      cc[(size_t)i * 3 + a] = (X[(size_t)i * 3 + a] - cen[a]) * mask[i];

  std::vector<double> A((size_t)M * RIGID, 0.0);
  for (int i = 0; i < N; ++i) {
    const double cx = cc[(size_t)i * 3 + 0], cy = cc[(size_t)i * 3 + 1],
                 cz = cc[(size_t)i * 3 + 2];
    A[(size_t)(3 * i + 0) * RIGID + 0] = mask[i];
    A[(size_t)(3 * i + 1) * RIGID + 1] = mask[i];
    A[(size_t)(3 * i + 2) * RIGID + 2] = mask[i];
    A[(size_t)(3 * i + 1) * RIGID + 3] = -cz;
    A[(size_t)(3 * i + 2) * RIGID + 3] =  cy;
    A[(size_t)(3 * i + 0) * RIGID + 4] =  cz;
    A[(size_t)(3 * i + 2) * RIGID + 4] = -cx;
    A[(size_t)(3 * i + 0) * RIGID + 5] = -cy;
    A[(size_t)(3 * i + 1) * RIGID + 5] =  cx;
  }
  std::vector<double> Q, R;
  thinQR(A, M, Q, R);

  // --- Hint = P H P, then the damped inverse -------------------------
  std::vector<double> Hint = H;
  projectLeft(Q, Hint, M, M);
  projectRight(Q, Hint, M, M);

  std::vector<double> mrep((size_t)M);
  for (int i = 0; i < N; ++i)
    for (int a = 0; a < 3; ++a) mrep[(size_t)3 * i + a] = mask[i];

  double dsum = 0.0;
  for (int k = 0; k < M; ++k) dsum += Hint[(size_t)k * M + k] * mrep[k];
  const double denom = std::max(1.0, 3.0 * msum - 6.0);
  const double mc_raw = dsum / denom;
  const double mc = std::max(eps, mc_raw);
  const double lam = damping * mc;

  std::vector<double> Z = Hint;
  for (int k = 0; k < M; ++k) Z[(size_t)k * M + k] += lam;
  if (!inverse(Z, M)) {
    fprintf(stderr, "collective_softness: the damped Hessian is singular "
                    "(N=%d) -- the loss is 0 and so is its gradient\n", N);
    if (want_grad) out.dx.assign((size_t)N * 3, 0.0);
    return out;
  }

  // C = P Z P
  std::vector<double> C = Z;
  projectLeft(Q, C, M, M);
  projectRight(Q, C, M, M);

  // --- the two scores -------------------------------------------------
  std::vector<double> f((size_t)N, 0.0);
  std::vector<char> fpos((size_t)N, 0);
  double F = 0.0;
  for (int i = 0; i < N; ++i) {
    double raw = 0.0;
    for (int a = 0; a < 3; ++a) raw += C[(size_t)(3 * i + a) * M + (3 * i + a)];
    fpos[i] = raw > 0.0;
    f[i] = (raw > 0.0 ? raw : 0.0) * mask[i];
    F += f[i];
  }
  std::vector<double> s((size_t)N);
  double S = 0.0;
  for (int i = 0; i < N; ++i) {
    s[i] = f[i] / (F + eps);
    if (s[i] > 0.0) S += s[i] * log(s[i] + eps);
  }
  const double coll_raw = exp(-S) / (msum + eps);
  const double coll = std::min(1.0, std::max(0.0, coll_raw));

  // twelve power iterations from the centred coordinates
  std::vector<std::vector<double>> win;        // w going IN to each iteration
  std::vector<double> nrm;                     // |C w| at each iteration
  std::vector<double> w = cc;                  // cc is already N*3 == M
  for (int it = 0; it < iters; ++it) {
    win.push_back(w);
    std::vector<double> y((size_t)M, 0.0);
    gemm(C.data(), w.data(), y.data(), M, M, 1);
    double n2 = 0.0;
    for (int k = 0; k < M; ++k) n2 += y[k] * y[k];
    const double n = sqrt(n2);
    nrm.push_back(n);
    for (int k = 0; k < M; ++k) y[k] /= (n + eps);
    w.swap(y);
  }
  std::vector<double> Cw((size_t)M, 0.0);
  gemm(C.data(), w.data(), Cw.data(), M, M, 1);
  double quad = 0.0;
  for (int k = 0; k < M; ++k) quad += w[k] * Cw[k];
  double T = 0.0;
  for (int k = 0; k < M; ++k) T += C[(size_t)k * M + k] * mrep[k];
  const double conc_raw = quad / (T + eps);
  const double conc = std::min(1.0, std::max(0.0, conc_raw));

  out.value = 1.0 - coll * conc;
  out.collectivity = coll;
  out.concentration = conc;
  if (!want_grad) return out;

  // ---- backward ------------------------------------------------------
  out.dx.assign((size_t)N * 3, 0.0);
  std::vector<double> gC((size_t)M * M, 0.0);
  std::vector<double> gcc((size_t)N * 3, 0.0);

  const double g_coll = (coll_raw > 0.0 && coll_raw < 1.0) ? -conc : 0.0;
  const double g_conc = (conc_raw > 0.0 && conc_raw < 1.0) ? -coll : 0.0;

  // concentration = quad / (T + eps)
  {
    const double g_quad = g_conc / (T + eps);
    const double g_T = -g_conc * quad / ((T + eps) * (T + eps));
    for (int k = 0; k < M; ++k) gC[(size_t)k * M + k] += g_T * mrep[k];
    // quad = w . (C w):  gC += g_quad * w w^T,  gw += g_quad (C + C^T) w
    std::vector<double> gw((size_t)M, 0.0);
    for (int i = 0; i < M; ++i) {
      double ctw = 0.0;
      for (int k = 0; k < M; ++k) ctw += C[(size_t)k * M + i] * w[k];
      gw[i] = g_quad * (Cw[i] + ctw);
      const double wi = g_quad * w[i];
      if (wi != 0.0) {
        double* row = &gC[(size_t)i * M];
        for (int j = 0; j < M; ++j) row[j] += wi * w[j];
      }
    }
    // the unrolled power iteration, in reverse
    for (int it = iters - 1; it >= 0; --it) {
      const std::vector<double>& wi = win[(size_t)it];
      const double n = nrm[(size_t)it];
      std::vector<double> y((size_t)M, 0.0);
      gemm(C.data(), wi.data(), y.data(), M, M, 1);
      // w_out = y / (n + eps), n = |y|
      double gy_dot = 0.0;
      for (int k = 0; k < M; ++k) gy_dot += gw[k] * y[k];
      std::vector<double> gy((size_t)M);
      const double inv = 1.0 / (n + eps);
      const double corr = (n > 0.0) ? gy_dot / (n * (n + eps) * (n + eps)) : 0.0;
      for (int k = 0; k < M; ++k) gy[k] = gw[k] * inv - y[k] * corr;
      // y = C w_in
      for (int i = 0; i < M; ++i) {
        if (gy[i] == 0.0) continue;
        double* row = &gC[(size_t)i * M];
        for (int j = 0; j < M; ++j) row[j] += gy[i] * wi[j];
      }
      for (int j = 0; j < M; ++j) {
        double s2 = 0.0;
        for (int i = 0; i < M; ++i) s2 += C[(size_t)i * M + j] * gy[i];
        gw[j] = s2;
      }
    }
    // the seed is the centred coordinates
    for (int k = 0; k < M; ++k) gcc[k] += gw[k];
  }

  // collectivity = exp(-S) / (msum + eps)
  {
    const double gS = -g_coll * coll_raw;
    std::vector<double> gf((size_t)N, 0.0);
    double gF = 0.0;
    for (int i = 0; i < N; ++i) {
      if (s[i] <= 0.0) continue;
      const double gs = gS * (log(s[i] + eps) + s[i] / (s[i] + eps));
      gf[i] += gs / (F + eps);
      gF += -gs * f[i] / ((F + eps) * (F + eps));
    }
    for (int i = 0; i < N; ++i) {
      const double g = (gf[i] + gF) * mask[i] * (fpos[i] ? 1.0 : 0.0);
      if (g == 0.0) continue;
      for (int a = 0; a < 3; ++a)
        gC[(size_t)(3 * i + a) * M + (3 * i + a)] += g;
    }
  }

  // ---- C = P Z P -----------------------------------------------------
  // gZ = P gC P (P symmetric); and gP reaches Q only through gP Q and
  // gP^T Q, each of which is a handful of 3N x 6 products.
  std::vector<double> gQ((size_t)M * RIGID, 0.0);
  std::vector<double> gZ = gC;
  projectLeft(Q, gZ, M, M);
  projectRight(Q, gZ, M, M);

  // gP from Y = P Z P:  gP = gC (Z P)^T + (P Z)^T gC
  //   gP Q    = gC (P Z^T Q) + Z^T (P (gC Q))
  //   gP^T Q  = Z (P (gC^T Q)) + gC^T (P (Z Q))
  auto applyLeftT = [&](const std::vector<double>& Mat,
                        const std::vector<double>& Bm,
                        std::vector<double>& Rm) {            // Rm = Mat^T Bm
    Rm.assign((size_t)M * RIGID, 0.0);
    gemmTN(Mat.data(), Bm.data(), Rm.data(), M, M, RIGID);
  };
  auto applyLeft = [&](const std::vector<double>& Mat,
                       const std::vector<double>& Bm,
                       std::vector<double>& Rm) {             // Rm = Mat Bm
    Rm.assign((size_t)M * RIGID, 0.0);
    gemm(Mat.data(), Bm.data(), Rm.data(), M, M, RIGID);
  };
  auto accumGQ = [&](const std::vector<double>& Mat,
                     const std::vector<double>& Cot) {
    // Mat is the matrix the projector sandwiches (Z or H); Cot the cotangent
    // on P Mat P.  Adds both gP Q and gP^T Q, negated for P = I - Q Q^T.
    std::vector<double> t1, t2;
    applyLeftT(Mat, Q, t1); projectLeft(Q, t1, M, RIGID);  // t1 = P Mat^T Q
    applyLeft(Cot, t1, t2);                                 // t2 = Cot t1
    std::vector<double> u1, u2;
    applyLeft(Cot, Q, u1); projectLeft(Q, u1, M, RIGID);    // u1 = P Cot Q
    applyLeftT(Mat, u1, u2);                                // u2 = Mat^T u1
    std::vector<double> p1, p2;
    applyLeftT(Cot, Q, p1); projectLeft(Q, p1, M, RIGID);   // p1 = P Cot^T Q
    applyLeft(Mat, p1, p2);                                 // p2 = Mat p1
    std::vector<double> q1, q2;
    applyLeft(Mat, Q, q1); projectLeft(Q, q1, M, RIGID);    // q1 = P Mat Q
    applyLeftT(Cot, q1, q2);                                // q2 = Cot^T q1
    for (size_t k = 0; k < gQ.size(); ++k)
      gQ[k] -= t2[k] + u2[k] + p2[k] + q2[k];
  };
  accumGQ(Z, gC);

  // ---- Z = inv(Hint + lam I) -----------------------------------------
  std::vector<double> gG((size_t)M * M, 0.0);
  {
    std::vector<double> t((size_t)M * M, 0.0);
    // gG = -Z^T gZ Z^T
    gemmTN(Z.data(), gZ.data(), t.data(), M, M, M);     // t = Z^T gZ
    std::vector<double> u((size_t)M * M, 0.0);
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < M; ++j) {
        double sm = 0.0;
        const double* tr = &t[(size_t)i * M];
        for (int p = 0; p < M; ++p) sm += tr[p] * Z[(size_t)j * M + p];  // Z^T
        u[(size_t)i * M + j] = -sm;
      }
    gG.swap(u);
  }
  std::vector<double> gHint = gG;
  // lam = damping * mc, mc = clip(dsum/denom, eps, inf)
  {
    double gl = 0.0;
    for (int k = 0; k < M; ++k) gl += gG[(size_t)k * M + k];
    const double gmc = (mc_raw > eps) ? gl * damping : 0.0;
    const double gdsum = gmc / denom;
    if (gdsum != 0.0)
      for (int k = 0; k < M; ++k) gHint[(size_t)k * M + k] += gdsum * mrep[k];
  }

  // ---- Hint = P H P ---------------------------------------------------
  accumGQ(H, gHint);
  std::vector<double> gH = gHint;
  projectLeft(Q, gH, M, M);
  projectRight(Q, gH, M, M);

  // ---- P = I - Q Q^T  ->  A = Q R  ------------------------------------
  {
    std::vector<double> Aq((size_t)RIGID * RIGID, 0.0);
    gemmTN(Q.data(), gQ.data(), Aq.data(), M, RIGID, RIGID);   // Aq = Q^T gQ
    std::vector<double> W((size_t)RIGID * RIGID, 0.0);
    for (int i = 0; i < RIGID; ++i)
      for (int j = 0; j < RIGID; ++j) {
        const double k = (i > j) ? (Aq[(size_t)i * RIGID + j] -
                                   Aq[(size_t)j * RIGID + i]) : 0.0;
        W[(size_t)i * RIGID + j] = k - Aq[(size_t)i * RIGID + j];
      }
    std::vector<double> gA((size_t)M * RIGID, 0.0);
    gemm(Q.data(), W.data(), gA.data(), M, RIGID, RIGID);
    for (size_t k = 0; k < gA.size(); ++k) gA[k] += gQ[k];
    solveRT(R, gA, M);
    // the three rotation columns are the only ones that carry coordinates
    for (int i = 0; i < N; ++i) {
      const double* g0 = &gA[(size_t)(3 * i + 0) * RIGID];
      const double* g1 = &gA[(size_t)(3 * i + 1) * RIGID];
      const double* g2 = &gA[(size_t)(3 * i + 2) * RIGID];
      gcc[(size_t)i * 3 + 0] += -g2[4] + g1[5];
      gcc[(size_t)i * 3 + 1] +=  g2[3] - g0[5];
      gcc[(size_t)i * 3 + 2] += -g1[3] + g0[4];
    }
  }

  // ---- the centred coordinates ----------------------------------------
  {
    double gcen[3] = {0, 0, 0};
    for (int i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a) {
        const double g = gcc[(size_t)i * 3 + a] * mask[i];
        out.dx[(size_t)i * 3 + a] += g;
        gcen[a] -= g;
      }
    for (int i = 0; i < N; ++i)
      for (int a = 0; a < 3; ++a)
        out.dx[(size_t)i * 3 + a] += gcen[a] * mask[i] / (msum + 1e-8);
  }

  // ---- the Hessian build ----------------------------------------------
  // The diagonal block is a SUM over the row, so each off-diagonal block
  // gets its own cotangent plus the row's anchored one, with a sign.
  for (int i = 0; i < N; ++i) {
    double ganch[9];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        ganch[a * 3 + b] = gH[(size_t)(3 * i + a) * M + (3 * i + b)];
    for (int j = 0; j < N; ++j) {
      if (i == j) continue;
      double o[3], d2 = 0.0;
      for (int a = 0; a < 3; ++a) {
        o[a] = X[(size_t)j * 3 + a] - X[(size_t)i * 3 + a];
        d2 += o[a] * o[a];
      }
      const double r = 1.0 / sqrt(d2 + eps);
      const double k = exp(-d2 / cd2) * mask[i] * mask[j];
      double u[3];
      for (int a = 0; a < 3; ++a) u[a] = o[a] * r;
      double gb[9];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          gb[a * 3 + b] = gH[(size_t)(3 * i + a) * M + (3 * j + b)]
                        - ganch[a * 3 + b];
      // blk = -k u_a u_b
      double gk = 0.0;
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) gk -= gb[a * 3 + b] * u[a] * u[b];
      double gu[3] = {0, 0, 0};
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          gu[a] -= k * (gb[a * 3 + b] + gb[b * 3 + a]) * u[b];
      double gd2 = gk * k * (-1.0 / cd2);
      double go[3];
      double gr = 0.0;
      for (int a = 0; a < 3; ++a) { go[a] = gu[a] * r; gr += gu[a] * o[a]; }
      gd2 += gr * (-0.5 * r * r * r);
      for (int a = 0; a < 3; ++a) go[a] += gd2 * 2.0 * o[a];
      for (int a = 0; a < 3; ++a) {
        out.dx[(size_t)j * 3 + a] += go[a];
        out.dx[(size_t)i * 3 + a] -= go[a];
      }
    }
  }
  return out;
}

}  // namespace softloss
