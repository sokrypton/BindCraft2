// THE PRODUCTION DESIGN LOSSES, as bindcraft/loss.py defines them.
//
// Deliberately NOT losses.cuh.  That file's i_con, plddt and pae were
// written to gate kernels and carry the right NAMES with different
// definitions -- interface_contacts in bc2 has hotspots, chain slices,
// contacts_per_residue and contact_residue_count, none of which that i_con
// knows about.  Mixing the two would be the "same name, different thing"
// trap this port is most exposed to, so production definitions live here and
// each one is gated against the registered python function it claims to be
// (loss_gate.py).
//
// EVERY LOSS HERE HAS A BACKWARD.  The design step consumes cotangents on
// its outputs; a forward-only loss cannot drive anything, so a loss without
// one is not "partially done", it is absent.
#pragma once
#include "common.cuh"

namespace prodloss {

// bindcraft/loss.py::_masked_mean's epsilon, which is load-bearing at the
// denominator and NOT a tolerance: a state with no design positions divides
// by it rather than by zero.
static constexpr float MASK_EPS = 1e-8f;

// plddt_loss(protein_states, predictions, prediction_state='binder_alone',
//            chain='binder')
//   = _masked_mean(1 - plddt[binder rows], DESIGN flags)
//   = sum_i m_i (1 - p_i) / (sum_i m_i + eps)
//
// `prediction_state` DEFAULTS TO binder_alone, not the complex -- three of
// the eight default losses do, and the gate caught binder_pae on exactly
// that.  So `p` is the binder-alone state's plddt over its own rows and `m`
// is the DESIGN flag per binder residue; the caller does the row selection.
__global__ void plddtLossK(const float* __restrict__ p,
                           const float* __restrict__ m,
                           int n, float* __restrict__ out) {
  __shared__ float sn[256], sd[256];
  float num = 0.f, den = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float mi = m[i];
    den += mi;                       // mask.sum(), zeros included
    if (mi > 0.f) num += (1.f - p[i]) * mi;   // jnp.where(mask > 0, ...)
  }
  sn[threadIdx.x] = num; sd[threadIdx.x] = den;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { sn[threadIdx.x] += sn[threadIdx.x + s];
                           sd[threadIdx.x] += sd[threadIdx.x + s]; }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = sn[0] / (sd[0] + MASK_EPS);   // the loss
    out[1] = sd[0];                        // the denominator, for the backward
  }
}

// d/d plddt_i = -dL * m_i / (sum(m) + eps) where m_i > 0, else 0.
// `den` is the forward's out[1] rather than a second reduction: the two must
// be the SAME number or the gradient is scaled by a slightly different
// denominator than the value was.
__global__ void plddtLossBwdK(const float* __restrict__ m, float den,
                              float dL, int n, float* __restrict__ dp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float mi = m[i];
  dp[i] = (mi > 0.f) ? -dL * mi / (den + MASK_EPS) : 0.f;
}

}  // namespace prodloss
