// alphafold_input_features, native.
//
// RUNTIME LENGTH IS THE WHOLE POINT.  Every grid here is sized to NMAX and
// every kernel reads `n` from DEVICE memory and masks past it.  Nothing is
// templated on the length and nothing reads it from an attribute, because
// the moment one kernel specialises on it the graph goes back to being
// per-length and the port has bought nothing (design_step.cuh).
//
// Verified as a formula first -- seqfeat_formula.py's sibling,
// inputfeat_formula.py, transcribes all of this in numpy and matches the
// JAX on 21 of 21 arrays.
#pragma once
#include "common.cuh"

namespace inputfeat {

constexpr int A = 20;        // amino acids
constexpr int ATOM = 37;
constexpr int ATOM14 = 14;
constexpr int MSA = 49;      // msa_feat channels
constexpr int CA = 1, CB = 3;        // residue_constants.atom_order
constexpr int GLY = 7;               // restype_order['G']
constexpr int UNK_TEMPLATE = 21;     // aatype for a non-template residue

// bindcraft.protein.ResidueFlags
constexpr uint8_t F_DESIGN = 1u << 0, F_TEMPLATE = 1u << 1, F_CYCLIC = 1u << 6;

struct Tables {                      // residue_constants, uploaded once
  const float*   atom14_mask;        // [22, 14]
  const float*   atom37_mask;        // [22, 37]
  const int32_t* atom14_to_atom37;   // [22, 14]
  const int32_t* atom37_to_atom14;   // [22, 37]
};

// ---- per residue: aatype, the four table gathers, the template arrays --
__global__ void perResidueK(const float* __restrict__ seq,      // [NMAX, A]
                            const float* __restrict__ atoms,    // [NMAX,37,3]
                            const float* __restrict__ amask,    // [NMAX, 37]
                            const uint8_t* __restrict__ flags,  // [NMAX]
                            const float* __restrict__ smask,    // [NMAX]
                            Tables t, const int32_t* __restrict__ np,
                            int32_t* __restrict__ aatype,       // [NMAX]
                            float* __restrict__ a14_exists,     // [NMAX, 14]
                            float* __restrict__ a37_exists,     // [NMAX, 37]
                            int32_t* __restrict__ a14_to_37,    // [NMAX, 14]
                            int32_t* __restrict__ a37_to_14,    // [NMAX, 37]
                            int32_t* __restrict__ t_aatype,     // [NMAX]
                            float* __restrict__ t_pos,          // [NMAX,37,3]
                            float* __restrict__ t_mask,         // [NMAX, 37]
                            float* __restrict__ pbeta,          // [NMAX, 3]
                            float* __restrict__ pbeta_mask,     // [NMAX]
                            int NMAX,
                            // target_flexibility, with the residue index it
                            // selects by; 0 leaves every template residue whole
                            const int32_t* __restrict__ ridx = nullptr,
                            float flex = 0.f) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NMAX) return;
  const int n = *np;                       // runtime, from device memory
  const bool real = i < n;

  // argmax over the 20 acids; a padded row takes aatype 0 and is masked out
  int am = 0; float best = -INFINITY;
  if (real) {
    for (int a = 0; a < A; ++a) {
      float v = seq[(size_t)i * A + a];
      if (v > best) { best = v; am = a; }
    }
  }
  aatype[i] = real ? am : 0;

  const float keep = real ? smask[i] : 0.f;      // seq_mask gates the gathers
  const int row = real ? am : 0;
  for (int k = 0; k < ATOM14; ++k) {
    a14_exists[(size_t)i * ATOM14 + k] = keep ? t.atom14_mask[row * ATOM14 + k] : 0.f;
    a14_to_37 [(size_t)i * ATOM14 + k] = keep ? t.atom14_to_atom37[row * ATOM14 + k] : 0;
  }
  for (int k = 0; k < ATOM; ++k) {
    a37_exists[(size_t)i * ATOM + k] = keep ? t.atom37_mask[row * ATOM + k] : 0.f;
    a37_to_14 [(size_t)i * ATOM + k] = keep ? t.atom37_to_atom14[row * ATOM + k] : 0;
  }

  // the template side: a residue not flagged TEMPLATE contributes nothing
  const bool is_t = real && (flags[i] & F_TEMPLATE);
  // TARGET FLEXIBILITY, as bc2's target_template_features builds it: a
  // template residue that is not designed and that
  // floor((idx+1)*f) > floor(idx*f) selects -- every 1/f-th by RESIDUE INDEX,
  // float32 arithmetic as JAX does it -- reads as a GAP (template aatype 21)
  // and loses its side chain (everything past N, CA, C, CB, O) in the
  // template mask, while its positions stay.  It used to be applied only on
  // the C++ driver's host, by PDB number and to the mask alone, and bc2's
  // fused native call did not apply it here at all: the pair template
  // constant saw it, the torsion row did not.
  const bool flexible =
      is_t && flex > 0.f && ridx && !(flags[i] & F_DESIGN) &&
      floorf((float)(ridx[i] + 1) * flex) > floorf((float)ridx[i] * flex);
  const int ta = (is_t && !flexible) ? am : UNK_TEMPLATE;
  t_aatype[i] = ta;
  for (int k = 0; k < ATOM; ++k) {
    const float mk = (is_t && !(flexible && k > 4)) ? amask[(size_t)i * ATOM + k]
                                                     : 0.f;
    t_mask[(size_t)i * ATOM + k] = mk;
    for (int c = 0; c < 3; ++c) {
      const size_t o = ((size_t)i * ATOM + k) * 3 + c;
      t_pos[o] = is_t ? atoms[o] : 0.f;
    }
  }
  // pseudo-beta: CB, or CA for glycine
  const int slot = (ta == GLY) ? CA : CB;
  for (int c = 0; c < 3; ++c)
    pbeta[(size_t)i * 3 + c] = t_pos[((size_t)i * ATOM + slot) * 3 + c];
  pbeta_mask[i] = t_mask[(size_t)i * ATOM + slot];
}

// ---- chain lengths, one block per row ---------------------------------
__global__ void chainLenK(const int32_t* __restrict__ asym,
                          const float* __restrict__ smask,
                          const int32_t* __restrict__ np,
                          float* __restrict__ out, int NMAX) {
  const int i = blockIdx.x;
  if (i >= NMAX) return;
  const int n = *np;
  // one WARP per row and the xor reduction this codebase uses elsewhere
  // (flash_body.cuh, foldit.cuh), rather than a shared-memory block reduce
  float acc = 0.f;
  if (i < n)
    for (int j = threadIdx.x; j < n; j += 32)
      acc += (asym[i] == asym[j]) ? smask[j] : 0.f;
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (threadIdx.x == 0) out[i] = (i < n) ? acc : 0.f;
}

// ---- the [NMAX, NMAX] relative offsets, cyclic where both ends are -----
__global__ void offsetK(const int32_t* __restrict__ ridx,
                        const int32_t* __restrict__ asym,
                        const uint8_t* __restrict__ flags,
                        const float* __restrict__ smask,
                        const float* __restrict__ chain_len,
                        const int32_t* __restrict__ np,
                        int32_t* __restrict__ out, int NMAX,
                        int mode = 1) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * NMAX) return;
  const int i = (int)(p / NMAX), j = (int)(p % NMAX);
  const int n = *np;
  if (i >= n || j >= n) { out[p] = 0; return; }

  const int off = ridx[i] - ridx[j];
  const bool same = asym[i] == asym[j];
  const float ci = ((flags[i] & F_CYCLIC) != 0) * smask[i];
  const float cj = ((flags[j] & F_CYCLIC) != 0) * smask[j];
  if (!(same && ci > 0.f && cj > 0.f)) { out[p] = off; return; }

  // bc2's cyclic_sequence_offsets, all three CYCLIC_OFFSET_MODES:
  //   0 'distance'    the ring distance, signed by the offset
  //   1 'direction'   ...negated where the short way round is the other way
  //   2 'neighbours'  direction, and anything past 2 saturates at +-32
  const int ao = off < 0 ? -off : off;
  float cd = fminf((float)ao, chain_len[i] - (float)ao);
  if (mode >= 1 && cd < (float)ao) cd = -cd;
  if (mode == 2 && fabsf(cd) > 2.f) cd = 32.f * (float)((cd > 0.f) - (cd < 0.f));
  const int sgn = (off > 0) - (off < 0);
  out[p] = (int32_t)lrintf(cd * sgn);
}

// ---- msa_feat: the sequence in 0:20, the profile in 25:45 --------------
__global__ void msaFeatK(const float* __restrict__ seq,
                         const float* __restrict__ prof,
                         const int32_t* __restrict__ np,
                         float* __restrict__ out, int NMAX) {
  const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= (size_t)NMAX * MSA) return;
  const int i = (int)(p / MSA), c = (int)(p % MSA);
  const int n = *np;
  float v = 0.f;
  if (i < n) {
    if (c < A) v = seq[(size_t)i * A + c];
    else if (c >= 25 && c < 45) v = prof[(size_t)i * A + (c - 25)];
  }
  out[p] = v;
}

// ---- the backward, which is almost nothing ----------------------------
// Only two paths out of this whole block carry a gradient to the sequence:
// target_feat IS the sequence, and msa_feat copies the sequence into
// channels 0:20 and the profile into 25:45.  Everything else -- aatype
// (an argmax), the four residue_constants gathers, the template arrays,
// the cyclic offsets -- is an integer lookup or a mask, and contributes
// none.  Saying so explicitly is worth more than a kernel: a reader who
// assumes otherwise goes looking for a bug that is not there.
__global__ void msaFeatBwdK(const float* __restrict__ d_msa,  // [NMAX, 49]
                            const float* __restrict__ d_target,// [NMAX, A]
                            const int32_t* __restrict__ np,
                            float* __restrict__ d_seq,        // [NMAX, A]
                            float* __restrict__ d_prof,       // [NMAX, A]
                            int NMAX) {
  const size_t q = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= (size_t)NMAX * A) return;
  const int i = (int)(q / A), a = (int)(q % A);
  if (i >= *np) { d_seq[q] = 0.f; d_prof[q] = 0.f; return; }
  d_seq[q] = d_msa[(size_t)i * MSA + a] + (d_target ? d_target[q] : 0.f);
  d_prof[q] = d_msa[(size_t)i * MSA + 25 + a];
}

}  // namespace inputfeat
