// prepare_design_sequence_features, the first piece of the glue.
//
//   scaled  = logits * logit_scale / temperature
//   probs   = softmax(scaled + bias)              bias is per amino acid
//   blended = logits, zeroed where the acid is omitted
//   feat    = sw * probs + (1 - sw) * blended
//   out     = ohw * one_hot(argmax probs) + (1 - ohw) * feat
//
// and that twice: once with the caller's one_hot_weight for the features
// AlphaFold reads, once with one_hot_weight = 1 for the profile, each kept
// only where the residue is flagged DESIGN and the input logits used
// everywhere else.
//
// THE ONE-HOT IS STRAIGHT-THROUGH.  Forward it is a hard one-hot; backward
// it is the identity on probs, because the JAX writes it as
// stop_gradient(one_hot - probs) + probs.  A kernel that returns the one-hot
// and differentiates it as a one-hot would silently kill the gradient that
// drives the whole design.
// RELATED, NOT THE SAME: seqxform.cuh does ColabDesign2's own version of
// this -- soft/hard with the same straight-through trick -- and its header
// derives the backward.  BindCraft's formula differs in three ways: the
// bias is added AFTER the logit_scale/temperature scaling rather than
// before, the non-softmax path zeroes omitted acids, and a second output
// (the profile) is produced at one_hot_weight = 1.  Follow seqxform's
// derivation for the backward; do not re-derive it.
#pragma once
#include "common.cuh"

namespace seqfeat {

constexpr int A = 20;                    // amino acids
// bindcraft.sequence_optimization: omitted iff logit <= OMITTED/2
constexpr float OMITTED_HALF = -1e4f;    // set from the caller, see setOmitted

struct Scalars {
  float softmax_weight, one_hot_weight, temperature, logit_scale;
};

// THE FOUR SCALARS HAVE TO BE ABLE TO COME FROM DEVICE MEMORY.
//
// bc2 passes softmax_weight, one_hot_weight, temperature and logit_scale as
// ARGUMENTS to its jitted step, and a trajectory RAMPS them -- so making
// them XLA custom-call attributes puts them in the compiled signature and
// every ramp step recompiles the whole graph.  That is the same mistake as
// passing the length as a shape, and it undoes the same saving.
//
// `sc4` wins when it is non-null; the per-piece FFI still passes attributes
// and leaves it null.
__device__ __forceinline__ Scalars pickScalars(const Scalars& s,
                                               const float* sc4) {
  if (!sc4) return s;
  Scalars o;
  o.softmax_weight = sc4[0]; o.one_hot_weight = sc4[1];
  o.temperature = sc4[2];    o.logit_scale = sc4[3];
  return o;
}

// one residue per thread: A is 20, so a serial pass over it beats a warp
// reduction and keeps the straight-through term exact.
__global__ void seqFeatK(const float* __restrict__ logits,   // [N, A]
                         const uint8_t* __restrict__ flags,  // [N]
                         const float* __restrict__ bias,     // [A] or null
                         Scalars s_, float omitted_half,
                         float* __restrict__ feat,           // [N, A]
                         float* __restrict__ prof,           // [N, A]
                         int N, const float* __restrict__ sc4 = nullptr) {
  const Scalars s = pickScalars(s_, sc4);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float* L = logits + (size_t)i * A;
  float* F = feat + (size_t)i * A;
  float* P = prof + (size_t)i * A;

  // not a designed residue: the logits pass through untouched, both ways
  if (!(flags[i] & 1u)) {                       // ResidueFlags.DESIGN == 1<<0
    for (int a = 0; a < A; ++a) { F[a] = L[a]; P[a] = L[a]; }
    return;
  }

  const float k = s.logit_scale / s.temperature;
  float m = -INFINITY;
  for (int a = 0; a < A; ++a) {
    float v = L[a] * k + (bias ? bias[a] : 0.f);
    if (v > m) m = v;
  }
  float z = 0.f;
  for (int a = 0; a < A; ++a) z += __expf(L[a] * k + (bias ? bias[a] : 0.f) - m);
  const float inv = 1.f / z;

  int amax = 0; float pmax = -INFINITY;
  for (int a = 0; a < A; ++a) {
    float p = __expf(L[a] * k + (bias ? bias[a] : 0.f) - m) * inv;
    if (p > pmax) { pmax = p; amax = a; }
    F[a] = p;                                   // stash probs
  }
  for (int a = 0; a < A; ++a) {
    const float p = F[a];
    const float blended = (L[a] <= omitted_half) ? 0.f : L[a];
    const float f = s.softmax_weight * p + (1.f - s.softmax_weight) * blended;
    const float oh = (a == amax) ? 1.f : 0.f;
    F[a] = s.one_hot_weight * oh + (1.f - s.one_hot_weight) * f;
    P[a] = oh;                                  // profile: one_hot_weight == 1
  }
}

// ---- the backward -----------------------------------------------------
// Both outputs reach the logits through the SAME softmax, because the
// one-hot is straight-through: forward it is discrete, backward it is the
// identity on p.  So the coefficients collapse, exactly as seqxform.cuh
// says for ColabDesign2's variant.
//
//   features: c_f = ohw + (1 - ohw) * sw      through p
//             plus (1 - ohw)(1 - sw) straight through the blended path,
//             which is zero where the acid is omitted
//   profile:  c_p = 1                         through p (one_hot_weight = 1)
//
// and J_softmax^T g = k * p * (g - <p, g>), with k = logit_scale/temperature
// because the softmax sees logits * k.
//
// A NON-DESIGNED RESIDUE PASSES ITS LOGITS THROUGH BOTH OUTPUTS, so its
// cotangent is simply the sum of the two.
__global__ void seqFeatBwdK(const float* __restrict__ logits,  // [N, A]
                            const uint8_t* __restrict__ flags,
                            const float* __restrict__ bias,
                            Scalars s_, float omitted_half,
                            const float* __restrict__ d_feat,  // [N, A]
                            const float* __restrict__ d_prof,  // [N, A]
                            float* __restrict__ d_logits,      // [N, A]
                            int N,
                            const float* __restrict__ sc4 = nullptr) {
  const Scalars s = pickScalars(s_, sc4);
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  const float* L = logits + (size_t)i * A;
  const float* gf = d_feat + (size_t)i * A;
  const float* gp = d_prof + (size_t)i * A;
  float* out = d_logits + (size_t)i * A;

  if (!(flags[i] & 1u)) {
    for (int a = 0; a < A; ++a) out[a] = gf[a] + gp[a];
    return;
  }

  const float k = s.logit_scale / s.temperature;
  float m = -INFINITY;
  for (int a = 0; a < A; ++a)
    m = fmaxf(m, L[a] * k + (bias ? bias[a] : 0.f));
  float z = 0.f;
  for (int a = 0; a < A; ++a) z += __expf(L[a] * k + (bias ? bias[a] : 0.f) - m);
  const float inv = 1.f / z;

  // the combined cotangent arriving at p
  const float cf = s.one_hot_weight + (1.f - s.one_hot_weight) * s.softmax_weight;
  float dot = 0.f;
  for (int a = 0; a < A; ++a) {
    const float p = __expf(L[a] * k + (bias ? bias[a] : 0.f) - m) * inv;
    dot += p * (cf * gf[a] + gp[a]);
  }
  const float straight = (1.f - s.one_hot_weight) * (1.f - s.softmax_weight);
  for (int a = 0; a < A; ++a) {
    const float p = __expf(L[a] * k + (bias ? bias[a] : 0.f) - m) * inv;
    float g = k * p * ((cf * gf[a] + gp[a]) - dot);
    if (L[a] > omitted_half) g += straight * gf[a];
    out[a] = g;
  }
}

}  // namespace seqfeat
