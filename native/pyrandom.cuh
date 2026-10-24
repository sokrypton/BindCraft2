// `random.Random(<a string>)`, the seeding half.
//
// bc2 draws the autotune's weight alternation from
// `random.Random(f"{campaign_seed}:{len(trajectory_rows)}")`, and a STRING
// seed is not an integer seed with extra steps.  CPython:
//
//     a = int.from_bytes(a.encode() + sha512(a.encode()).digest())
//
// -- big-endian, so the text is the HIGH bytes -- and then seeds
// init_by_array with that integer's 32-bit limbs, little-endian.  A port
// that hashed the string any other way draws a different multiplier and
// the campaign tunes itself differently; there is no "close" here.
//
// The generator itself is domains.cuh's MT, which the domain split
// needed first; `seedString` drives it through the same init_by_array
// with the string's limbs instead of an integer's.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace pyrandom {

// ---- SHA-512 (FIPS 180-4) ----------------------------------------------
struct Sha512 {
  uint64_t h[8] = {0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull,
                   0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
                   0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
                   0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull};
  unsigned char buf[128];
  size_t len = 0;          // bytes buffered
  uint64_t total = 0;      // bytes seen

  static uint64_t ror(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }

  void block(const unsigned char* p) {
    static const uint64_t K[80] = {
      0x428a2f98d728ae22ull,0x7137449123ef65cdull,0xb5c0fbcfec4d3b2full,0xe9b5dba58189dbbcull,
      0x3956c25bf348b538ull,0x59f111f1b605d019ull,0x923f82a4af194f9bull,0xab1c5ed5da6d8118ull,
      0xd807aa98a3030242ull,0x12835b0145706fbeull,0x243185be4ee4b28cull,0x550c7dc3d5ffb4e2ull,
      0x72be5d74f27b896full,0x80deb1fe3b1696b1ull,0x9bdc06a725c71235ull,0xc19bf174cf692694ull,
      0xe49b69c19ef14ad2ull,0xefbe4786384f25e3ull,0x0fc19dc68b8cd5b5ull,0x240ca1cc77ac9c65ull,
      0x2de92c6f592b0275ull,0x4a7484aa6ea6e483ull,0x5cb0a9dcbd41fbd4ull,0x76f988da831153b5ull,
      0x983e5152ee66dfabull,0xa831c66d2db43210ull,0xb00327c898fb213full,0xbf597fc7beef0ee4ull,
      0xc6e00bf33da88fc2ull,0xd5a79147930aa725ull,0x06ca6351e003826full,0x142929670a0e6e70ull,
      0x27b70a8546d22ffcull,0x2e1b21385c26c926ull,0x4d2c6dfc5ac42aedull,0x53380d139d95b3dfull,
      0x650a73548baf63deull,0x766a0abb3c77b2a8ull,0x81c2c92e47edaee6ull,0x92722c851482353bull,
      0xa2bfe8a14cf10364ull,0xa81a664bbc423001ull,0xc24b8b70d0f89791ull,0xc76c51a30654be30ull,
      0xd192e819d6ef5218ull,0xd69906245565a910ull,0xf40e35855771202aull,0x106aa07032bbd1b8ull,
      0x19a4c116b8d2d0c8ull,0x1e376c085141ab53ull,0x2748774cdf8eeb99ull,0x34b0bcb5e19b48a8ull,
      0x391c0cb3c5c95a63ull,0x4ed8aa4ae3418acbull,0x5b9cca4f7763e373ull,0x682e6ff3d6b2b8a3ull,
      0x748f82ee5defb2fcull,0x78a5636f43172f60ull,0x84c87814a1f0ab72ull,0x8cc702081a6439ecull,
      0x90befffa23631e28ull,0xa4506cebde82bde9ull,0xbef9a3f7b2c67915ull,0xc67178f2e372532bull,
      0xca273eceea26619cull,0xd186b8c721c0c207ull,0xeada7dd6cde0eb1eull,0xf57d4f7fee6ed178ull,
      0x06f067aa72176fbaull,0x0a637dc5a2c898a6ull,0x113f9804bef90daeull,0x1b710b35131c471bull,
      0x28db77f523047d84ull,0x32caab7b40c72493ull,0x3c9ebe0a15c9bebcull,0x431d67c49c100d4cull,
      0x4cc5d4becb3e42b6ull,0x597f299cfc657e2aull,0x5fcb6fab3ad6faecull,0x6c44198c4a475817ull};
    uint64_t w[80];
    for (int i = 0; i < 16; ++i) {
      w[i] = 0;
      for (int b = 0; b < 8; ++b) w[i] = (w[i] << 8) | p[i * 8 + b];
    }
    for (int i = 16; i < 80; ++i) {
      const uint64_t s0 = ror(w[i - 15], 1) ^ ror(w[i - 15], 8) ^ (w[i - 15] >> 7);
      const uint64_t s1 = ror(w[i - 2], 19) ^ ror(w[i - 2], 61) ^ (w[i - 2] >> 6);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 80; ++i) {
      const uint64_t S1 = ror(e, 14) ^ ror(e, 18) ^ ror(e, 41);
      const uint64_t ch = (e & f) ^ (~e & g);
      const uint64_t t1 = hh + S1 + ch + K[i] + w[i];
      const uint64_t S0 = ror(a, 28) ^ ror(a, 34) ^ ror(a, 39);
      const uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint64_t t2 = S0 + maj;
      hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  void update(const unsigned char* p, size_t n) {
    total += n;
    while (n) {
      const size_t take = (128 - len) < n ? (128 - len) : n;
      memcpy(buf + len, p, take);
      len += take; p += take; n -= take;
      if (len == 128) { block(buf); len = 0; }
    }
  }

  std::vector<unsigned char> digest() {
    Sha512 c = *this;
    const uint64_t bits = c.total * 8;
    unsigned char pad = 0x80;
    c.update(&pad, 1);
    const unsigned char zero = 0;
    while (c.len != 112) c.update(&zero, 1);
    unsigned char tail[16] = {0};
    for (int i = 0; i < 8; ++i) tail[15 - i] = (unsigned char)(bits >> (8 * i));
    c.update(tail, 16);
    std::vector<unsigned char> out(64);
    for (int i = 0; i < 8; ++i)
      for (int b = 0; b < 8; ++b)
        out[(size_t)i * 8 + b] = (unsigned char)(c.h[i] >> (56 - 8 * b));
    return out;
  }
};

inline std::vector<unsigned char> sha512(const std::string& s) {
  Sha512 d;
  d.update((const unsigned char*)s.data(), s.size());
  return d.digest();
}

// the 32-bit limbs, LITTLE-endian, of `int.from_bytes(a + sha512(a))`.
// init_by_array wants them in that order; the byte string is big-endian,
// so the last byte is the least significant.
inline std::vector<uint32_t> seedLimbs(const std::string& text) {
  std::vector<unsigned char> bytes(text.begin(), text.end());
  const std::vector<unsigned char> d = sha512(text);
  bytes.insert(bytes.end(), d.begin(), d.end());
  std::vector<uint32_t> limbs;
  size_t i = bytes.size();
  while (i > 0) {
    uint32_t limb = 0;
    for (int b = 0; b < 4 && i > 0; ++b) limb |= (uint32_t)bytes[--i] << (8 * b);
    limbs.push_back(limb);
  }
  while (limbs.size() > 1 && limbs.back() == 0) limbs.pop_back();
  return limbs;
}

// init_by_array over an arbitrary key -- what `random_seed` does once it
// has turned the seed into an integer, whether that integer came from a
// number or from a hashed string.  (domains.cuh's MT::seed is this with
// the limbs of a uint64.)
template <class MT>
inline void seedKey(MT& g, const std::vector<uint32_t>& key) {
  g.initGenrand(19650218u);
  size_t i = 1, j = 0;
  size_t k = key.size() > 624 ? key.size() : 624;
  for (; k; --k) {
    g.mt[i] = (uint32_t)((g.mt[i] ^ ((g.mt[i - 1] ^ (g.mt[i - 1] >> 30)) * 1664525u))
                         + key[j] + (uint32_t)j);
    ++i; ++j;
    if (i >= 624) { g.mt[0] = g.mt[623]; i = 1; }
    if (j >= key.size()) j = 0;
  }
  for (k = 623; k; --k) {
    g.mt[i] = (uint32_t)((g.mt[i] ^ ((g.mt[i - 1] ^ (g.mt[i - 1] >> 30)) * 1566083941u))
                         - (uint32_t)i);
    ++i;
    if (i >= 624) { g.mt[0] = g.mt[623]; i = 1; }
  }
  g.mt[0] = 0x80000000u;
  g.idx = 624;
}

template <class MT>
inline void seedString(MT& g, const std::string& text) {
  seedKey(g, seedLimbs(text));
}

}  // namespace pyrandom
