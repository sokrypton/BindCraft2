// ---------------------------------------------------------------------
// jax.random's threefry stream, enough of it to make the same draws
// ---------------------------------------------------------------------
//
// bc2 draws from `jax.random` wherever a decision is a COIN the campaign
// must be able to repeat: which conformation a round predicts, which
// trajectory key a trajectory gets.  Reproducing the decision means
// reproducing the stream, and the stream is Threefry-2x32 with JAX's own
// key derivation on top -- `PRNGKey`, `fold_in`, `split` and the
// `randint` that reads bits out of it.
//
// WHAT IS HERE IS THE DEFAULT IMPLEMENTATION, not an equivalent one.
// JAX's `randint` is biased on purpose (it draws twice the bits and
// folds them modulo the span to reduce that bias) and any "cleaner"
// uniform integer would be a different draw -- the same distribution and
// a different answer, which is the one kind of difference a campaign
// cannot see and cannot repeat.  jaxrandom_gate.py compares every
// function here against jax itself rather than against a description of
// it.
//
// Four entry points, and the chain bc2 builds out of them:
//
//   key        = prngKey(campaign_seed)
//   trajectory = foldIn(key, trajectory_number)      (1-based, as bc2's
//                                                     claim_trajectory
//                                                     hands it out)
//   init, mut  = split(trajectory)
//   conformation = split(init)[1]
//
// which is `run_trajectory`'s first two lines, and what
// InducedFitSchedule is handed.
#pragma once
#include <cstdint>
#include <vector>

namespace jrandom {

struct Key { uint32_t a = 0, b = 0; };

inline uint32_t rotl32(uint32_t v, int d) {
  return (v << d) | (v >> (32 - d));
}

// threefry2x32, 5 groups of 4 rounds.  The rotation table and the
// 0x1BD11BDA parity constant are Threefry's own.
inline void threefry2x32(uint32_t k0, uint32_t k1, uint32_t& x0,
                         uint32_t& x1) {
  static const int ROT[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
  const uint32_t ks[3] = {k0, k1, k0 ^ k1 ^ 0x1BD11BDAu};
  x0 += ks[0];
  x1 += ks[1];
  for (int i = 0; i < 5; ++i) {
    for (int r = 0; r < 4; ++r) {
      x0 += x1;
      x1 = rotl32(x1, ROT[i % 2][r]);
      x1 ^= x0;
    }
    x0 += ks[(i + 1) % 3];
    x1 += ks[(i + 2) % 3] + (uint32_t)(i + 1);
  }
}

// THE PARTITIONABLE STREAM, which is what JAX runs by default now
// (`jax_threefry_partitionable`, true since 0.4.36 and true in the 0.10
// this port is gated against).  The older implementation fed the block
// cipher a flat `iota(2n)` counter; the partitionable one feeds it a
// 64-bit iota SPLIT INTO ITS TWO HALVES -- element i is the counter
// (0, i) -- which is what makes a draw independent of how the array is
// sharded.  The two give completely different numbers, so the gate asks
// jax for the flag as well as for the values.
inline void counterPair(uint64_t index, uint32_t* hi, uint32_t* lo) {
  *hi = (uint32_t)(index >> 32);
  *lo = (uint32_t)(index & 0xFFFFFFFFull);
}

// `threefry_seed`: the high and low words of the seed, in that order
inline Key prngKey(uint64_t seed) {
  Key k;
  k.a = (uint32_t)(seed >> 32);
  k.b = (uint32_t)(seed & 0xFFFFFFFFull);
  return k;
}

// `_threefry_split_foldlike`: key i is the cipher applied to the
// counter (0, i), and the two output words ARE the new key.
inline std::vector<Key> split(const Key& key, int n) {
  std::vector<Key> out((size_t)n);
  for (int i = 0; i < n; ++i) {
    uint32_t x0, x1;
    counterPair((uint64_t)i, &x0, &x1);
    threefry2x32(key.a, key.b, x0, x1);
    out[i].a = x0;
    out[i].b = x1;
  }
  return out;
}

// `threefry_fold_in(key, data)` = threefry_2x32(key, threefry_seed(data)),
// which is NOT partitionable-dependent: the counter is the seed's own
// two words.
inline Key foldIn(const Key& key, uint64_t data) {
  const Key d = prngKey(data);
  uint32_t x0 = d.a, x1 = d.b;
  threefry2x32(key.a, key.b, x0, x1);
  Key out;
  out.a = x0;
  out.b = x1;
  return out;
}

// `_threefry_random_bits_partitionable(key, 32, ())`: the cipher on the
// counter (0, 0), and the two output words XORed -- not the first word,
// which is what the old implementation returned.
inline uint32_t randomBits(const Key& key) {
  uint32_t x0 = 0, x1 = 0;
  threefry2x32(key.a, key.b, x0, x1);
  return x0 ^ x1;
}

// `jax.random.randint(key, (), minval, maxval)` for int32.
//
// IT IS NOT `bits % span`.  JAX draws TWO words and combines them with a
// multiplier equal to 2**32 mod span, computed as (2**16 mod span)^2 mod
// span to stay inside the type -- the bias of a plain modulo is halved
// that way, and the ANSWER differs from the plain modulo at almost every
// key.
inline int32_t randint(const Key& key, int32_t minval, int32_t maxval) {
  const std::vector<Key> ks = split(key, 2);
  const uint32_t higher = randomBits(ks[0]);
  const uint32_t lower = randomBits(ks[1]);
  uint32_t span = (uint32_t)((int64_t)maxval - (int64_t)minval);
  if (maxval <= minval) span = 1;
  // THE MULTIPLIER WRAPS TOO, and that is not a detail: for any span
  // above 65536 the remainder is 65536 itself, 65536 * 65536 is 2**32,
  // and in uint32 that is ZERO -- so the high word drops out of the
  // draw entirely and the result is just `lower % span`.  Computed in
  // 64 bits it is 954414 instead of 0 and every large-span draw
  // differs.  jax's own comment says the identity avoids overflow; the
  // code overflows anyway, and the code is what runs.
  uint32_t multiplier = 65536u % span;
  multiplier = (uint32_t)(multiplier * multiplier) % span;
  // IN UINT32, AND IT WRAPS.  jax builds the offset with `lax.mul` and
  // `lax.add` on uint32 arrays, so the product overflows and is kept
  // modulo 2**32 before the final remainder.  Computing it in 64 bits
  // gives a different answer for every span big enough to overflow --
  // the small spans agree either way, which is exactly how a gate with
  // only small spans would have missed it.
  uint32_t offset = (uint32_t)((higher % span) * multiplier) + (lower % span);
  offset = offset % span;
  return (int32_t)((int64_t)minval + (int64_t)offset);
}

// bc2's own chain, named once so the driver and the gate cannot drift:
// the conformation key a trajectory's InducedFitSchedule starts from.
inline Key conformationKey(uint64_t campaign_seed, int trajectory_number) {
  const Key traj = foldIn(prngKey(campaign_seed), (uint64_t)trajectory_number);
  const Key init = split(traj, 2)[0];     // binder_initialization_key
  return split(init, 2)[1];               // conformation_random_key
}

}  // namespace jrandom
