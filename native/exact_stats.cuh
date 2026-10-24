// statistics.pstdev, BIT FOR BIT.
//
// `summarize_campaign` writes one row per metric with mean, std, min and
// max, and every cell goes through `recorded_number` -- two decimals.  So
// the standard deviation of two accepted designs at i_pDAE 0.04 and 0.01
// is 0.015, and 0.015 is a TIE: "%.2f" reads the double either side of it
// as 0.01 or 0.02 depending on which way the arithmetic landed.  The
// naive float formula lands on 0.015000000000000001 and prints 0.02;
// CPython's `statistics.pstdev` is exact -- it sums the squares as
// RATIONALS and takes a correctly-rounded square root -- lands on exactly
// 0.015, and prints 0.01.
//
// The port had the right FORMULA (population, not sample) and the wrong
// ARITHMETIC, and it passed two suite runs before a campaign produced a
// tie.  Same lesson as `fsum`: where bc2's number is a CPython primitive,
// the primitive is the specification.
//
// CPython, statistics._ss / _float_sqrt_of_frac / _integer_sqrt_of_frac_rto:
//
//   ssd = (count * sxx - sx * sx) / count     exactly, in Fractions
//   mss = ssd / count
//   pstdev = correctly-rounded sqrt(mss)
//
// Every double is exactly n * 2^-e, so with E = max(e_i) and
// a_i = n_i << (E - e_i):  sx = A * 2^-E and sxx = B * 2^-2E, with
// A = sum a_i and B = sum a_i^2 -- all integers, no rounding anywhere.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace exstat {

// ---- the little unsigned bignum these four steps need -------------------
struct Big {
  std::vector<uint32_t> w;             // little endian, no leading zeros
  Big() {}
  explicit Big(uint64_t v) { w.push_back((uint32_t)v); w.push_back((uint32_t)(v >> 32)); trim(); }
  void trim() { while (!w.empty() && w.back() == 0) w.pop_back(); }
  bool zero() const { return w.empty(); }
  size_t bits() const {
    if (w.empty()) return 0;
    size_t b = (w.size() - 1) * 32;
    uint32_t t = w.back();
    while (t) { ++b; t >>= 1; }
    return b;
  }
  bool bit(size_t i) const {
    return i / 32 < w.size() && ((w[i / 32] >> (i % 32)) & 1u);
  }
};

inline int cmp(const Big& a, const Big& b) {
  if (a.w.size() != b.w.size()) return a.w.size() < b.w.size() ? -1 : 1;
  for (size_t i = a.w.size(); i-- > 0;)
    if (a.w[i] != b.w[i]) return a.w[i] < b.w[i] ? -1 : 1;
  return 0;
}

inline Big add(const Big& a, const Big& b) {
  Big r; uint64_t carry = 0;
  for (size_t i = 0; i < a.w.size() || i < b.w.size() || carry; ++i) {
    uint64_t s = carry;
    if (i < a.w.size()) s += a.w[i];
    if (i < b.w.size()) s += b.w[i];
    r.w.push_back((uint32_t)s);
    carry = s >> 32;
  }
  r.trim(); return r;
}

// a - b, and a >= b is the caller's promise
inline Big sub(const Big& a, const Big& b) {
  Big r; int64_t borrow = 0;
  for (size_t i = 0; i < a.w.size(); ++i) {
    int64_t s = (int64_t)a.w[i] - borrow - (i < b.w.size() ? (int64_t)b.w[i] : 0);
    borrow = s < 0;
    if (s < 0) s += ((int64_t)1 << 32);
    r.w.push_back((uint32_t)s);
  }
  r.trim(); return r;
}

inline Big mul(const Big& a, const Big& b) {
  if (a.zero() || b.zero()) return Big();
  std::vector<uint64_t> acc(a.w.size() + b.w.size(), 0);
  for (size_t i = 0; i < a.w.size(); ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < b.w.size(); ++j) {
      const uint64_t cur = acc[i + j] + (uint64_t)a.w[i] * b.w[j] + carry;
      acc[i + j] = cur & 0xffffffffu;
      carry = cur >> 32;
    }
    size_t k = i + b.w.size();
    while (carry) { const uint64_t cur = acc[k] + carry; acc[k] = cur & 0xffffffffu; carry = cur >> 32; ++k; }
  }
  Big r; r.w.assign(acc.begin(), acc.end()); r.trim(); return r;
}

inline Big shl(const Big& a, size_t s) {
  if (a.zero()) return a;
  Big r; r.w.assign(s / 32, 0);
  const size_t sh = s % 32;
  uint32_t carry = 0;
  for (size_t i = 0; i < a.w.size(); ++i) {
    r.w.push_back((uint32_t)((a.w[i] << sh) | carry));
    carry = sh ? (uint32_t)(a.w[i] >> (32 - sh)) : 0;
  }
  if (carry) r.w.push_back(carry);
  r.trim(); return r;
}

inline Big shr(const Big& a, size_t s) {
  Big r;
  const size_t word = s / 32, sh = s % 32;
  for (size_t i = word; i < a.w.size(); ++i) {
    uint64_t v = a.w[i] >> sh;
    if (sh && i + 1 < a.w.size()) v |= (uint64_t)a.w[i + 1] << (32 - sh);
    r.w.push_back((uint32_t)v);
  }
  r.trim(); return r;
}

// a / d for a 64-bit d, floor
inline Big divSmall(const Big& a, uint64_t d) {
  Big r; r.w.assign(a.w.size(), 0);
  // long division in 32-bit limbs with a 64-bit remainder is only safe
  // while d < 2^32; above that, fall back to a shift-and-subtract over
  // the bits, which is what the count^2 denominators need.
  if (d < (1ull << 32)) {
    uint64_t rem = 0;
    for (size_t i = a.w.size(); i-- > 0;) {
      const uint64_t cur = (rem << 32) | a.w[i];
      r.w[i] = (uint32_t)(cur / d);
      rem = cur % d;
    }
    r.trim(); return r;
  }
  Big q, rem;
  const Big bd(d);
  for (size_t i = a.bits(); i-- > 0;) {
    rem = shl(rem, 1);
    if (a.bit(i)) rem = add(rem, Big(1));
    if (cmp(rem, bd) >= 0) { rem = sub(rem, bd); q = add(q, shl(Big(1), i)); }
  }
  return q;
}

// floor(sqrt(n)), by the binary digit method -- shifts, compares and
// subtracts only, which is all this needs
inline Big isqrt(const Big& n) {
  if (n.zero()) return Big();
  Big rem = n, root, bit = shl(Big(1), ((n.bits() - 1) / 2) * 2);
  while (!bit.zero()) {
    const Big t = add(root, bit);
    if (cmp(rem, t) >= 0) { rem = sub(rem, t); root = add(shr(root, 1), bit); }
    else root = shr(root, 1);
    bit = shr(bit, 2);
  }
  return root;
}

// CPython's _integer_sqrt_of_frac_rto: sqrt(n/m) to the nearest integer,
// rounded to odd -- the trick that makes the final conversion correctly
// rounded whatever the discarded tail was
inline Big isqrtFracRto(const Big& n, const Big& m, uint64_t m_small, bool m_is_small) {
  const Big q = m_is_small ? divSmall(n, m_small) : Big();
  Big a = isqrt(q);
  const Big aam = mul(mul(a, a), m);
  if (cmp(aam, n) != 0 && !a.bit(0)) a = add(a, Big(1));   // set the low bit
  return a;
}

// the big integer as a double, round-half-even -- what Python's int/int
// does for a power-of-two denominator
inline double toDouble(const Big& a) {
  const size_t b = a.bits();
  if (b == 0) return 0.0;
  if (b <= 53) {
    double v = 0;
    for (size_t i = b; i-- > 0;) v = v * 2.0 + (a.bit(i) ? 1.0 : 0.0);
    return v;
  }
  const size_t drop = b - 54;              // keep 54 bits, then round
  Big top = shr(a, drop);
  bool sticky = false;
  for (size_t i = 0; i < drop && !sticky; ++i) sticky = a.bit(i);
  uint64_t t = 0;
  for (size_t i = top.bits(); i-- > 0;) t = (t << 1) | (top.bit(i) ? 1u : 0u);
  const uint64_t guard = t & 1u;
  t >>= 1;
  if (guard && (sticky || (t & 1u))) ++t;
  return std::ldexp((double)t, (int)(drop + 1));
}

// ---- pstdev --------------------------------------------------------------
inline double pstdev(const std::vector<double>& values) {
  const size_t count = values.size();
  if (count < 2) return 0.0;
  // every double is n * 2^-e exactly
  std::vector<int64_t> num(count);
  std::vector<int> ex(count);
  int E = -100000;
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) return std::nan("");
    int exp2 = 0;
    const double m = std::frexp(values[i], &exp2);      // |m| in [0.5, 1)
    num[i] = (int64_t)std::ldexp(m, 53);
    ex[i] = 53 - exp2;                                  // x = num * 2^-ex
    if (ex[i] > E) E = ex[i];
  }
  // A = sum a_i and B = sum a_i^2, with a_i = num_i << (E - ex_i)
  Big pos, neg, B;
  for (size_t i = 0; i < count; ++i) {
    const uint64_t mag = (uint64_t)(num[i] < 0 ? -num[i] : num[i]);
    const Big a = shl(Big(mag), (size_t)(E - ex[i]));
    (num[i] < 0 ? neg : pos) = add(num[i] < 0 ? neg : pos, a);
    B = add(B, mul(a, a));
  }
  const bool a_neg = cmp(pos, neg) < 0;
  const Big A = a_neg ? sub(neg, pos) : sub(pos, neg);  // |A|; A^2 needs no sign
  const Big cB = mul(Big((uint64_t)count), B);
  const Big AA = mul(A, A);
  if (cmp(cB, AA) <= 0) return 0.0;                     // every value equal
  // mss = (count*B - A^2) / (count^2 * 2^(2E))
  Big N = sub(cB, AA);
  uint64_t Msmall = (uint64_t)count * (uint64_t)count;
  size_t Mshift = 0;
  if (E >= 0) Mshift = (size_t)(2 * E);
  else N = shl(N, (size_t)(-2 * E));
  Big M = shl(Big(Msmall), Mshift);
  // _float_sqrt_of_frac
  const int sqrt_bit_width = 109;                       // 2 * 53 + 3
  const long nb = (long)N.bits(), mb = (long)M.bits();
  long q = (nb - mb - sqrt_bit_width);
  q = (long)std::floor((double)q / 2.0);
  Big numerator;
  long den_shift = 0;
  if (q >= 0) {
    const Big m2 = shl(M, (size_t)(2 * q));
    // M is count^2 << s: a division by a small factor and a shift
    const Big qq = shr(divSmall(N, Msmall), (size_t)(Mshift + 2 * q));
    Big a = isqrt(qq);
    const Big aam = mul(mul(a, a), m2);
    if (cmp(aam, N) != 0 && !a.bit(0)) a = add(a, Big(1));
    numerator = shl(a, (size_t)q);
  } else {
    const Big n2 = shl(N, (size_t)(-2 * q));
    const Big qq = shr(divSmall(n2, Msmall), Mshift);
    Big a = isqrt(qq);
    const Big aam = mul(mul(a, a), M);
    if (cmp(aam, n2) != 0 && !a.bit(0)) a = add(a, Big(1));
    numerator = a;
    den_shift = -q;
  }
  return std::ldexp(toDouble(numerator), (int)-den_shift);
}

}  // namespace exstat
