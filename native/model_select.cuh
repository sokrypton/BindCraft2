// WHICH MODELS A CAMPAIGN DESIGNS ON, AND WHICH IT HOLDS OUT.
//
// bc2's `select_design_and_validation_models` takes either a LIST of model
// names (or indices into the pool) or a COUNT, and a count is a SEEDED
// SAMPLE of the pool -- `sorted(random.Random(campaign_seed).sample(pool,
// n))`, CPython's Mersenne Twister.  This port read neither:
//
//   design_models: 3        the reader looked for "model_" in "3", found
//                           none, and `continue`d -- so an integer count
//                           silently fell back to ONE model, which is a
//                           different objective (bc2 resamples the model
//                           inside every gradient step)
//   validation_models: 2    took the FIRST two siblings that were not the
//                           designer, where bc2 samples from the held-out
//                           pool with the campaign seed
//
// Both decide which networks a campaign actually uses, and both were
// silent.  The sample is reproducible because CPython's `sample` on a pool
// this small takes the list path -- n (5) <= setsize (21) -- which is
//
//     for i in range(k): j = _randbelow(n - i)
//                        result[i] = pool[j]
//                        pool[j] = pool[n - i - 1]
//
// and `_randbelow` is already here, exact, for the multidomain split.
#pragma once
#include <algorithm>
#include <string>
#include <vector>

#include "domains.cuh"

namespace modelsel {

// the five multimer checkpoints, in bc2's MULTIMER_POOL order
inline std::vector<std::string> multimerPool() {
  std::vector<std::string> p;
  for (int i = 1; i <= 5; ++i)
    p.push_back("model_" + std::to_string(i) + "_multimer_v3");
  return p;
}

// `sorted(random.Random(seed).sample(pool, k))`
inline std::vector<std::string> sample(const std::vector<std::string>& pool,
                                       int k, int seed) {
  domains::MT rng;
  rng.seed((uint64_t)(seed < 0 ? -seed : seed));
  std::vector<std::string> work = pool, out;
  const int n = (int)pool.size();
  for (int i = 0; i < k; ++i) {
    const int j = rng.randbelow(n - i);
    out.push_back(work[(size_t)j]);
    work[(size_t)j] = work[(size_t)(n - i - 1)];
  }
  std::sort(out.begin(), out.end());
  return out;
}

// a requested list: names, or decimal INDICES into the pool
// (resolve_model_pool_selection accepts both).  Returns false and names the
// offender rather than dropping it.
inline bool resolveNames(const std::vector<std::string>& requested,
                         const std::vector<std::string>& pool,
                         std::vector<std::string>* out, std::string* bad) {
  for (const std::string& r : requested) {
    if (!r.empty() && r.find_first_not_of("0123456789") == std::string::npos) {
      const int index = atoi(r.c_str());
      if (index < 0 || index >= (int)pool.size()) {
        *bad = "model index " + r + " is outside the pool";
        return false;
      }
      out->push_back(pool[(size_t)index]);
      continue;
    }
    if (std::find(pool.begin(), pool.end(), r) == pool.end()) {
      *bad = "model " + r + " is outside the pool";
      return false;
    }
    out->push_back(r);
  }
  return true;
}

}  // namespace modelsel
