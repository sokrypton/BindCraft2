// binder_domain_ids: WHERE DOES THE BINDER SPLIT INTO DOMAINS?
//
// bc2 decides with `random.Random(seed)` and `random.choice`, which is
// CPython's Mersenne Twister.  Nothing about the split is derivable from
// the sequence or the structure -- it is a draw, and the multidomain loss
// is a different objective under a different draw.  So this reproduces
// MT19937 and CPython's `_randbelow_with_getrandbits` exactly rather than
// substituting another generator and matching "the distribution".
//
// That is the opposite of the choice made for the design-model pool and
// for dropout, where JAX's threefry stream was NOT reproduced -- and the
// difference is worth stating.  There the draw changes which model or
// which mask, and every draw is an equally valid campaign.  Here the draw
// changes the LOSS: domain 0 ending at residue 57 or at residue 63 is two
// different targets, and a port that picks its own is optimising
// something bc2 never asked for.  It is also cheap: one integer stream,
// seeded 0, a few dozen draws per trajectory.
#pragma once
#include "pyrandom.cuh"
#include <cstdint>
#include <vector>

namespace domains {

// CPython's `random.Random`, enough of it: init_by_array seeding from an
// integer seed, genrand_uint32, getrandbits and _randbelow.
struct MT {
  uint32_t mt[624];
  int idx = 625;

  void initGenrand(uint32_t s) {
    mt[0] = s;
    for (int i = 1; i < 624; ++i)
      mt[i] = (uint32_t)(1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i);
    idx = 624;
  }
  // random_seed() for a non-negative int key: init_by_array over the key's
  // 32-bit limbs.  A Python int seed is converted with abs() and then
  // split little-endian; seed 0 gives a single zero limb.
  // ONE init_by_array, in pyrandom.cuh.  A STRING seed hashes to a much
  // bigger integer than a uint64 and seeds the same way, so the loop
  // lives there and this is its uint64 case -- two copies of
  // init_by_array is exactly the drift `three code copies` is about.
  void seed(uint64_t s) {
    std::vector<uint32_t> key;
    if (s == 0) key.push_back(0);
    while (s) { key.push_back((uint32_t)(s & 0xffffffffu)); s >>= 32; }
    pyrandom::seedKey(*this, key);
  }
  uint32_t next() {
    if (idx >= 624) {
      static const uint32_t mag01[2] = {0u, 0x9908b0dfu};
      for (int kk = 0; kk < 624 - 397; ++kk) {
        const uint32_t y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
        mt[kk] = mt[kk + 397] ^ (y >> 1) ^ mag01[y & 1u];
      }
      for (int kk = 624 - 397; kk < 623; ++kk) {
        const uint32_t y = (mt[kk] & 0x80000000u) | (mt[kk + 1] & 0x7fffffffu);
        mt[kk] = mt[kk + (397 - 624)] ^ (y >> 1) ^ mag01[y & 1u];
      }
      const uint32_t y = (mt[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
      mt[623] = mt[396] ^ (y >> 1) ^ mag01[y & 1u];
      idx = 0;
    }
    uint32_t y = mt[idx++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
  }
  // getrandbits(k) for k <= 32: CPython takes the TOP k bits
  uint32_t getrandbits(int k) {
    if (k == 0) return 0;
    return next() >> (32 - k);
  }
  // _randbelow_with_getrandbits: rejection sampling on k = bit_length(n)
  int randbelow(int n) {
    // ONLY n == 0 SHORT-CIRCUITS.  CPython returns early for a falsy n
    // and otherwise always draws -- so _randbelow(1) draws getrandbits(1)
    // and REJECTS until it gets a zero, consuming a random number of
    // draws.  Returning 0 without drawing is the obvious optimisation and
    // it desynchronises the stream from the first single-element choice
    // onward: both remaining gate cases were ones where the size cap left
    // exactly one extendable domain.
    if (n == 0) return 0;
    // k = n.bit_length(), NOT (n-1).bit_length().  CPython says so in a
    // comment ("don't use (n-1) here because n can be 1") and the
    // difference is one extra bit, so the rejection loop runs with twice
    // the range and consumes draws at a different rate.  With (n-1) the
    // stream stays valid and the SPLIT MOVES BY ONE RESIDUE -- four of
    // nine gate cases, each off by exactly one, which is what a wrong
    // reproduction looks like when it is nearly right.
    int k = 0;
    for (int v = n; v; v >>= 1) ++k;
    uint32_t r = getrandbits(k);
    while ((int)r >= n) r = getrandbits(k);
    return (int)r;
  }
};

// bc2's binder_domain_ids, line for line.
inline std::vector<int> binderDomainIds(int binder_length, int n_domains,
                                        int min_domain_size,
                                        int max_domain_size, uint64_t seed) {
  const int minimum_size = min_domain_size > 1 ? min_domain_size : 1;
  const int domain_count = std::min(n_domains, binder_length / minimum_size);
  if (domain_count < 2) return std::vector<int>((size_t)binder_length, 0);
  MT rng;
  rng.seed(seed);
  std::vector<int> sizes((size_t)domain_count, minimum_size);
  const int cap = max_domain_size > minimum_size ? max_domain_size : minimum_size;
  for (int t = 0; t < binder_length - minimum_size * domain_count; ++t) {
    std::vector<int> extendable;
    for (int i = 0; i < domain_count; ++i)
      if (sizes[(size_t)i] < cap) extendable.push_back(i);
    // random.choice(seq) is seq[_randbelow(len(seq))]; randrange(n) on the
    // empty-extendable path is _randbelow(n) too, so the two branches draw
    // the same way from different ranges
    if (!extendable.empty())
      sizes[(size_t)extendable[(size_t)rng.randbelow((int)extendable.size())]] += 1;
    else
      sizes[(size_t)rng.randbelow(domain_count)] += 1;
  }
  std::vector<int> out;
  out.reserve((size_t)binder_length);
  for (int d = 0; d < domain_count; ++d)
    for (int k = 0; k < sizes[(size_t)d]; ++k) out.push_back(d);
  return out;
}

}  // namespace domains
