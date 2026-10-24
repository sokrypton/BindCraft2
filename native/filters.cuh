// THE SIX DEFAULT FILTERS, and the `final` stage that applies them.
//
// A trajectory that runs is not a design.  bc2's settings/core/default.json
// carries six filters, and without them the port produces a sequence and
// has no opinion about it -- which is the difference between a demo and a
// campaign.  All six are transcribed from bindcraft/filters.py and all six
// read what the step already produces:
//
//   Unbound_Binder_pLDDT  >= 0.80   mean pLDDT of the BINDER_ALONE fold
//   pTM                   >= 0.55   the complex's pTM
//   i_pTM                 >= 0.70   the complex's ipTM
//   i_pAE                 <= 0.35   chain_pair_pae_loss(binder, target)
//   Backbone_Clashes      <= 0      CA pairs across chains within 2.5 A
//   Interface_Residues    >= 7      binder residues within 4 A of target
//
// Two of them are the SAME NUMBERS the loss already computes -- i_pAE is
// interface_pae and nothing else, and Unbound_Binder_pLDDT is plddt_loss's
// input before the 1 - x.  They are recomputed here from the same buffers
// rather than lifted out of the loss, because the loss is weighted and a
// filter is not, and reusing a weighted term is exactly the kind of saving
// that reads fine and silently applies the weight twice.
//
// THE COORDINATES ARE THE PREDICTION'S, not the template's: every
// geometric filter reads predictions[state].protein_complex, which is what
// the model built.
//
// AND "THE BINDER" IS THE CHAIN, NOT THE DESIGNED ROWS.  Every one of these
// took `flags` and `F_DESIGN` and called that the binder, which is correct
// for a de novo binder and wrong for every scaffold: a VHH designs 27 of
// its 115 rows, so the other 88 were being treated as TARGET -- counted as
// contact partners for the designed rows, never counted as interface
// residues themselves, and folded into the binder-vs-target pAE block on
// the wrong side.  filter_gate had never run on a scaffold campaign, so
// three metrics were wrong and green: i_pAE 0.568 against bc2's 0.795,
// Interface_Residues 19 against 13, i_pDAE 0.590 against 0.002.
//
// That is the SAME de novo coincidence recorded in the TODO's traps, found
// for the seventh time, now in the filters.  The masks are therefore an
// explicit chain mask (`is_binder`, the row's POSITION) and the flags are
// passed only where a flag is genuinely what bc2 reads.
#pragma once
#include "design_inputs.cuh"
#include "sse.cuh"
#include "kabsch.cuh"
#include "sasa.cuh"
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <array>

namespace filt {

struct Result {
  const char* name;
  float value, threshold;
  bool higher, passed;
};

// pairwise over the CA atoms of every chain, unique pairs, different
// chains, both atoms resolved.  atom 1 is CA in the 37-atom layout.
inline float backboneClashes(const std::vector<float>& atoms,
                             const std::vector<float>& atom_mask,
                             const std::vector<int32_t>& asym_id, int n,
                             float cutoff = 2.5f) {
  const float c2 = cutoff * cutoff;
  long count = 0;
  for (int i = 0; i < n; ++i) {
    if (atom_mask[(size_t)i * 37 + 1] <= 0) continue;
    const float* a = &atoms[((size_t)i * 37 + 1) * 3];
    for (int j = i + 1; j < n; ++j) {
      if (asym_id[i] == asym_id[j]) continue;
      if (atom_mask[(size_t)j * 37 + 1] <= 0) continue;
      const float* b = &atoms[((size_t)j * 37 + 1) * 3];
      const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
      if (dx * dx + dy * dy + dz * dz <= c2) ++count;
    }
  }
  return (float)count;
}

// a binder residue counts when ANY of its resolved atoms is within 4 A of
// ANY resolved target atom -- all 37, not just the backbone
inline float interfaceResidues(const std::vector<float>& atoms,
                               const std::vector<float>& atom_mask,
                               const std::vector<char>& is_binder, int n,
                               float cutoff = 4.0f) {
  const float c2 = cutoff * cutoff;
  long count = 0;
  for (int i = 0; i < n; ++i) {
    if (!is_binder[i]) continue;
    bool touches = false;
    for (int ai = 0; ai < 37 && !touches; ++ai) {
      if (atom_mask[(size_t)i * 37 + ai] <= 0) continue;
      const float* a = &atoms[((size_t)i * 37 + ai) * 3];
      for (int j = 0; j < n && !touches; ++j) {
        if (is_binder[j]) continue;
        for (int aj = 0; aj < 37; ++aj) {
          if (atom_mask[(size_t)j * 37 + aj] <= 0) continue;
          const float* b = &atoms[((size_t)j * 37 + aj) * 3];
          const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
          if (dx * dx + dy * dy + dz * dz <= c2) { touches = true; break; }
        }
      }
    }
    count += touches;
  }
  return (float)count;
}

// Coldspot_Contact_Fraction: of the TARGET residues the campaign named as
// coldspots, what fraction does the binder touch?  The same 4 A all-atom
// contact as interfaceResidues, read from the other side -- a target row
// counts when any of its atoms is within the cutoff of any binder atom.
//
// bc2 returns None when nothing is flagged, and a filter on None is not
// run; that is the `< 0` return here, and the caller must not record it.
// Returning 0.0 instead would be a PASS on every campaign that names no
// coldspots, which is the wrong answer in the safe-looking direction.
inline float coldspotContactFraction(const std::vector<float>& atoms,
                                     const std::vector<float>& atom_mask,
                                     const std::vector<uint8_t>& flags,
                                     const std::vector<char>& is_binder, int n,
                                     uint8_t cold_flag,
                                     float cutoff = 4.0f) {
  const float c2 = cutoff * cutoff;
  long cold = 0, touched = 0;
  for (int i = 0; i < n; ++i) {
    if (is_binder[i]) continue;                  // target rows only
    if (!(flags[i] & cold_flag)) continue;
    ++cold;
    bool touches = false;
    for (int ai = 0; ai < 37 && !touches; ++ai) {
      if (atom_mask[(size_t)i * 37 + ai] <= 0) continue;
      const float* a = &atoms[((size_t)i * 37 + ai) * 3];
      for (int j = 0; j < n && !touches; ++j) {
        if (!is_binder[j]) continue;
        for (int aj = 0; aj < 37; ++aj) {
          if (atom_mask[(size_t)j * 37 + aj] <= 0) continue;
          const float* b = &atoms[((size_t)j * 37 + aj) * 3];
          const float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
          if (dx * dx + dy * dy + dz * dz <= c2) { touches = true; break; }
        }
      }
    }
    touched += touches;
  }
  return cold ? (float)touched / (float)cold : -1.f;
}

// ---- THE FOUR FILTERS EVERY SHIPPED SCAFFOLD MODALITY MAKES MANDATORY --
//
// Off_Epitope_Contact_Fraction is installed by `modality: binder` and by
// VHH, Fab, scFv and ARP; Off_Paratope_Contact_Fraction and the two
// Scaffold_* checks by the scaffold modalities.  None of them appears in
// any settings JSON -- bc2 derives them in load_settings through
// install_modality_check -- which is why the port's startup guard did not
// name them either.  A VHH campaign here accepted designs on four checks
// fewer than bc2 would have applied.

// Which binder-chain rows touch the target?  interfaceResidues' mask
// rather than its count, because two of the four want the mask.
inline std::vector<char> interfaceRowMask(const std::vector<float>& atoms,
                                          const std::vector<float>& atom_mask,
                                          const std::vector<char>& is_binder,
                                          int n, float cutoff = 4.0f) {
  const float c2 = cutoff * cutoff;
  std::vector<char> out((size_t)n, 0);
  for (int i = 0; i < n; ++i) {
    if (!is_binder[i]) continue;
    bool touches = false;
    for (int ai = 0; ai < 37 && !touches; ++ai) {
      if (atom_mask[(size_t)i * 37 + ai] <= 0) continue;
      const float* a = &atoms[((size_t)i * 37 + ai) * 3];
      for (int j = 0; j < n && !touches; ++j) {
        if (is_binder[j]) continue;
        for (int aj = 0; aj < 37; ++aj) {
          if (atom_mask[(size_t)j * 37 + aj] <= 0) continue;
          const float* b = &atoms[((size_t)j * 37 + aj) * 3];
          const float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
          if (dx*dx + dy*dy + dz*dz <= c2) { touches = true; break; }
        }
      }
    }
    out[(size_t)i] = touches ? 1 : 0;
  }
  return out;
}

// ...and the mirror: which TARGET rows does the binder touch?
inline std::vector<char> contactedTargetMask(const std::vector<float>& atoms,
                                             const std::vector<float>& atom_mask,
                                             const std::vector<char>& is_binder,
                                             int n, float cutoff = 4.0f) {
  const float c2 = cutoff * cutoff;
  std::vector<char> out((size_t)n, 0);
  for (int j = 0; j < n; ++j) {
    if (is_binder[j]) continue;
    bool touches = false;
    for (int aj = 0; aj < 37 && !touches; ++aj) {
      if (atom_mask[(size_t)j * 37 + aj] <= 0) continue;
      const float* b = &atoms[((size_t)j * 37 + aj) * 3];
      for (int i = 0; i < n && !touches; ++i) {
        if (!is_binder[i]) continue;
        for (int ai = 0; ai < 37; ++ai) {
          if (atom_mask[(size_t)i * 37 + ai] <= 0) continue;
          const float* a = &atoms[((size_t)i * 37 + ai) * 3];
          const float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
          if (dx*dx + dy*dy + dz*dz <= c2) { touches = true; break; }
        }
      }
    }
    out[(size_t)j] = touches ? 1 : 0;
  }
  return out;
}

// Off_Paratope_Contact_Fraction: of the binder rows that touch the target,
// what fraction are NOT rows the scaffold declared as paratope (CONTACT)?
//
// bc2 returns 0.0 -- not None -- when nothing touches, so a design that
// misses the target entirely PASSES this check.  That reads wrong and is
// right: the filter asks "is the contact in the wrong place", and no
// contact is not contact in the wrong place.  The other filters catch it.
inline float offParatopeContactFraction(const std::vector<float>& atoms,
                                        const std::vector<float>& atom_mask,
                                        const std::vector<uint8_t>& flags,
                                        const std::vector<char>& is_binder,
                                        int n, uint8_t contact_flag,
                                        float cutoff = 4.0f) {
  const std::vector<char> iface =
      interfaceRowMask(atoms, atom_mask, is_binder, n, cutoff);
  long total = 0, off = 0;
  for (int i = 0; i < n; ++i) {
    if (!iface[(size_t)i]) continue;
    ++total;
    off += (flags[(size_t)i] & contact_flag) ? 0 : 1;
  }
  return total ? (float)off / (float)total : 0.f;
}

// Off_Epitope_Contact_Fraction: of the TARGET rows the binder touches,
// what fraction lie outside the epitope?
//
// The epitope is every target residue with a resolved atom within 10 A of
// a resolved HOTSPOT atom -- and it is measured on the TEMPLATE, the
// crystal coordinates, while the contacts are measured on the PREDICTION.
// Using one array for both would make the epitope move with the fold,
// which is the opposite of what an epitope is.
//
// Returns < 0 for bc2's None: no hotspots, or no resolved hotspot atom.
inline float offEpitopeContactFraction(const std::vector<float>& atoms,
                                       const std::vector<float>& atom_mask,
                                       const std::vector<float>& tmpl_atoms,
                                       const std::vector<float>& tmpl_mask,
                                       const std::vector<uint8_t>& flags,
                                       const std::vector<char>& is_binder,
                                       int n, uint8_t hotspot_flag,
                                       float cutoff = 4.0f,
                                       float epitope_cutoff = 10.0f) {
  bool any_hot = false, any_hot_atom = false;
  for (int i = 0; i < n; ++i) {
    if (is_binder[i] || !(flags[(size_t)i] & hotspot_flag)) continue;
    any_hot = true;
    for (int a = 0; a < 37; ++a)
      if (tmpl_mask[(size_t)i * 37 + a] > 0) { any_hot_atom = true; break; }
  }
  if (!any_hot || !any_hot_atom) return -1.f;          // bc2's None
  const float e2 = epitope_cutoff * epitope_cutoff;
  std::vector<char> epitope((size_t)n, 0);
  for (int i = 0; i < n; ++i) {
    if (is_binder[i]) continue;
    bool near = false;
    for (int a = 0; a < 37 && !near; ++a) {
      if (tmpl_mask[(size_t)i * 37 + a] <= 0) continue;
      const float* p = &tmpl_atoms[((size_t)i * 37 + a) * 3];
      for (int j = 0; j < n && !near; ++j) {
        if (is_binder[j] || !(flags[(size_t)j] & hotspot_flag)) continue;
        for (int b = 0; b < 37; ++b) {
          if (tmpl_mask[(size_t)j * 37 + b] <= 0) continue;
          const float* q = &tmpl_atoms[((size_t)j * 37 + b) * 3];
          const float dx = p[0]-q[0], dy = p[1]-q[1], dz = p[2]-q[2];
          if (dx*dx + dy*dy + dz*dz <= e2) { near = true; break; }
        }
      }
    }
    epitope[(size_t)i] = near ? 1 : 0;
  }
  const std::vector<char> touched =
      contactedTargetMask(atoms, atom_mask, is_binder, n, cutoff);
  long total = 0, off = 0;
  for (int i = 0; i < n; ++i) {
    if (!touched[(size_t)i]) continue;
    ++total;
    off += epitope[(size_t)i] ? 0 : 1;
  }
  return total ? (float)off / (float)total : 0.f;
}

// Scaffold_Sequence_Retained_Fraction: at the FRAMEWORK rows -- the ones
// the scaffold held, TEMPLATE in the design's flags -- does the sequence
// still read as the scaffold's?  VHH ships min_..._final: 1.0, so one
// changed framework residue rejects the design.
//
// `seq` is the design's per-row amino acid index; `scaf_aa` the scaffold's,
// indexed by binder-chain row.  Returns < 0 when there is no scaffold.
inline float scaffoldSequenceRetained(const std::vector<int>& seq,
                                      const std::vector<int>& scaf_aa,
                                      const std::vector<uint8_t>& flags,
                                      int binder_len, int copies,
                                      uint8_t template_flag) {
  long held = 0, total = 0;
  if (scaf_aa.empty()) return -1.f;
  for (int c = 0; c < copies; ++c)
    for (int k = 0; k < binder_len; ++k) {
      const int i = c * binder_len + k;
      if (!(flags[(size_t)i] & template_flag)) continue;
      ++total;
      held += (seq[(size_t)i] == scaf_aa[(size_t)k]) ? 1 : 0;
    }
  return total ? (float)held / (float)total : -1.f;
}

// Scaffold_Framework_RMSD: the predicted framework CA against the
// scaffold's own, Kabsch-aligned over that subset alone.  Not the whole
// binder: a design is allowed to move its CDRs, and aligning on them would
// hide the framework drift this is for.
// ONE ALIGNMENT PER CHAIN, deviations pooled.
//
// bc2's scaffold_framework_chains zips the binder's chains with the
// scaffold's and aligns each pair on its own, then concatenates the SQUARED
// DEVIATIONS and takes one root-mean-square over the lot.  A Fab's two
// chains are placed independently by the model, so one global alignment of
// both frameworks is a different quantity: it read 16.79 A where bc2 reads
// 1.19 on a two-chain scaffold, and the two agree exactly when there is one
// chain -- the ninth convention to be checked only where it coincides.
//
// `chain_lens` is the binder's chains in row order; `scaf_ca` is indexed the
// same way, because both come from the same concatenation of scaffold rows.
inline float scaffoldFrameworkRmsd(const std::vector<float>& atoms,
                                   const std::vector<float>& atom_mask,
                                   const std::vector<float>& scaf_ca,
                                   const std::vector<char>& scaf_ca_mask,
                                   const std::vector<uint8_t>& flags,
                                   const std::vector<int>& chain_lens,
                                   uint8_t template_flag) {
  if (scaf_ca_mask.empty()) return -1.f;
  double ss_all = 0;
  long m_all = 0;
  int base = 0;
  for (size_t ci = 0; ci < chain_lens.size(); ++ci) {
    const int len = chain_lens[ci];
    std::vector<float> p, q;
    for (int k = 0; k < len; ++k) {
      const int i = base + k;
      if (!(flags[(size_t)i] & template_flag)) continue;
      if (atom_mask[(size_t)i * 37 + 1] <= 0 || !scaf_ca_mask[(size_t)i])
        continue;
      for (int d = 0; d < 3; ++d) {
        p.push_back(atoms[((size_t)i * 37 + 1) * 3 + d]);
        q.push_back(scaf_ca[(size_t)i * 3 + d]);
      }
    }
    base += len;
    const int m = (int)(p.size() / 3);
    if (m < 1) continue;
    double pc[3] = {0,0,0}, qc[3] = {0,0,0};
    for (int i = 0; i < m; ++i)
      for (int d = 0; d < 3; ++d) { pc[d] += p[i*3+d]; qc[d] += q[i*3+d]; }
    for (int d = 0; d < 3; ++d) { pc[d] /= m; qc[d] /= m; }
    // THE COVARIANCE ACCUMULATES IN DOUBLE.  It is a 3x3 sum over m points
    // and it was a float[9]: with the loop restructured for the per-chain
    // pass the compiler reassociated it, and the metric moved by 1.45e-05
    // relative -- just over the gate's 1e-5 -- on a VHH whose geometry had
    // not changed at all.  A reduction whose VALUE depends on how it was
    // vectorised is a reduction that will move again.
    double covd[9] = {0,0,0,0,0,0,0,0,0};
    for (int i = 0; i < m; ++i)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          covd[a*3+b] += (p[i*3+a]-pc[a]) * (q[i*3+b]-qc[b]);
    float cov[9];
    for (int k2 = 0; k2 < 9; ++k2) cov[k2] = (float)covd[k2];
    float R[9];
    kabsch::rotationFromCov(cov, R);
    for (int i = 0; i < m; ++i) {
      float v[3];
      for (int d = 0; d < 3; ++d) v[d] = (float)(p[i*3+d] - pc[d]);
      for (int d = 0; d < 3; ++d) {
        float acc = 0;
        for (int e = 0; e < 3; ++e) acc += v[e] * R[d*3+e];
        const double dd = (double)(acc + qc[d]) - q[i*3+d];
        ss_all += dd * dd;
      }
    }
    m_all += m;
  }
  // bc2's guard is on the POOLED count, not the per-chain one
  if (m_all < 3) return -1.f;
  return (float)std::sqrt(ss_all / (double)m_all);
}

// ---- FIVE MORE DERIVED METRICS, each a SECOND REDUCTION ---------------
//
// Of bc2's remaining derived filters, five ask a question the port already
// computes the hard part of.  Hotspot_Contact_Fraction and
// Epitope_Residues_Contacted are the two masks Off_Epitope already builds,
// counted differently; MHC_Anchor_Score and Protease_Site_Score are the
// humanization and protease_sites panels scored on the ARGMAX sequence
// instead of the soft one; Binder_Disulfides is the disulfide loss's
// geometry as a hard count.
//
// The ones NOT here -- Terminus_Exposure, Exposed_Loop_Fraction,
// Framework_Packing_Fraction -- all need solvent-accessible surface area,
// which is a new piece of geometry (biotite's Shrake-Rupley), not a second
// reading of an existing one.

// of the target residues the campaign named as hotspots, what fraction
// does the binder touch?  < 0 for bc2's None (no hotspots at all).
inline float hotspotContactFraction(const std::vector<float>& atoms,
                                    const std::vector<float>& atom_mask,
                                    const std::vector<uint8_t>& flags,
                                    const std::vector<char>& is_binder,
                                    int n, uint8_t hotspot_flag,
                                    float cutoff = 4.0f) {
  long hot = 0, touched = 0;
  for (int i = 0; i < n; ++i)
    if (!is_binder[i] && (flags[(size_t)i] & hotspot_flag)) ++hot;
  if (!hot) return -1.f;
  const std::vector<char> t =
      contactedTargetMask(atoms, atom_mask, is_binder, n, cutoff);
  for (int i = 0; i < n; ++i)
    if (!is_binder[i] && (flags[(size_t)i] & hotspot_flag) && t[(size_t)i])
      ++touched;
  return (float)touched / (float)hot;
}

// how many EPITOPE residues does the binder touch?  A count, not a
// fraction -- bc2's threshold for it is a minimum number.
inline float epitopeResiduesContacted(const std::vector<float>& atoms,
                                      const std::vector<float>& atom_mask,
                                      const std::vector<float>& tmpl_atoms,
                                      const std::vector<float>& tmpl_mask,
                                      const std::vector<uint8_t>& flags,
                                      const std::vector<char>& is_binder,
                                      int n, uint8_t hotspot_flag,
                                      float cutoff = 4.0f,
                                      float epitope_cutoff = 10.0f) {
  bool any_hot = false, any_atom = false;
  for (int i = 0; i < n; ++i) {
    if (is_binder[i] || !(flags[(size_t)i] & hotspot_flag)) continue;
    any_hot = true;
    for (int a = 0; a < 37; ++a)
      if (tmpl_mask[(size_t)i * 37 + a] > 0) { any_atom = true; break; }
  }
  if (!any_hot || !any_atom) return -1.f;
  const float e2 = epitope_cutoff * epitope_cutoff;
  const std::vector<char> touched =
      contactedTargetMask(atoms, atom_mask, is_binder, n, cutoff);
  long count = 0;
  for (int i = 0; i < n; ++i) {
    if (is_binder[i] || !touched[(size_t)i]) continue;
    bool near = false;
    for (int a = 0; a < 37 && !near; ++a) {
      if (tmpl_mask[(size_t)i * 37 + a] <= 0) continue;
      const float* p = &tmpl_atoms[((size_t)i * 37 + a) * 3];
      for (int j = 0; j < n && !near; ++j) {
        if (is_binder[j] || !(flags[(size_t)j] & hotspot_flag)) continue;
        for (int b = 0; b < 37; ++b) {
          if (tmpl_mask[(size_t)j * 37 + b] <= 0) continue;
          const float* q = &tmpl_atoms[((size_t)j * 37 + b) * 3];
          const float dx = p[0]-q[0], dy = p[1]-q[1], dz = p[2]-q[2];
          if (dx*dx + dy*dy + dz*dz <= e2) { near = true; break; }
        }
      }
    }
    count += near;
  }
  return (float)count;
}

// Binder_Disulfides: how many CB pairs are within tolerance of 3.8 A, at
// least three apart, GREEDILY matched shortest-deviation first so no
// cysteine is counted in two bonds.  The greedy order is the metric: a
// plain count of pairable pairs is a larger number on the same fold.
// `paired`, when given, comes back flagging the rows the count paired --
// 1.0.4's `paired_cysteine_mask`, which a disulfide-staple campaign HOLDS
// through ProteinMPNN.  One function for both, so the hold set and the
// metric cannot disagree about which cysteines are bonded.
inline float binderDisulfides(const std::vector<float>& atoms,
                              const std::vector<int>& seq_aa,
                              const std::vector<char>& is_binder, int n,
                              int cys_index, float distance = 3.8f,
                              float tolerance = 1.0f, int separation = 3,
                              std::vector<char>* paired = nullptr) {
  if (paired) paired->assign((size_t)n, 0);
  std::vector<int> cys;
  std::vector<int> row;                 // position WITHIN the binder chain
  int k = 0;
  for (int i = 0; i < n; ++i) {
    if (!is_binder[i]) continue;
    if (seq_aa[(size_t)i] == cys_index) { cys.push_back(i); row.push_back(k); }
    ++k;
  }
  if (cys.size() < 2) return 0.f;
  struct Pair { double dev; int a, b; };
  std::vector<Pair> pairs;
  for (size_t a = 0; a < cys.size(); ++a)
    for (size_t b = a + 1; b < cys.size(); ++b) {
      if (std::abs(row[a] - row[b]) < separation) continue;
      const float* p = &atoms[((size_t)cys[a] * 37 + 3) * 3];   // CB
      const float* q = &atoms[((size_t)cys[b] * 37 + 3) * 3];
      const double dx = p[0]-q[0], dy = p[1]-q[1], dz = p[2]-q[2];
      const double dev = std::fabs(std::sqrt(dx*dx + dy*dy + dz*dz)
                                   - (double)distance);
      if (dev <= (double)tolerance) pairs.push_back({dev, (int)a, (int)b});
    }
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const Pair& x, const Pair& y) { return x.dev < y.dev; });
  std::vector<char> used(cys.size(), 0);
  long bonds = 0;
  for (const Pair& pr : pairs)
    if (!used[(size_t)pr.a] && !used[(size_t)pr.b]) {
      used[(size_t)pr.a] = used[(size_t)pr.b] = 1;
      ++bonds;
      if (paired) {
        (*paired)[(size_t)cys[(size_t)pr.a]] = 1;
        (*paired)[(size_t)cys[(size_t)pr.b]] = 1;
      }
    }
  return (float)bonds;
}

// ---- FOUR MORE: the chain itself, and the domain geometry -------------
//
// Binder_Chain_Breaks and Oligomer_Symmetry_RMSD are two lines of geometry
// each.  The two domain metrics need bc2's `contiguous_domain_split`,
// which is NOT binder_domain_ids: the loss splits the chain by a DRAW
// before anything is folded, and the filter splits the FOLD by maximising
// within-domain contacts.  Two different splits of the same binder, and
// using one for the other would be a plausible number.

// how many consecutive CA pairs are not a plausible bond apart?
// PER CHAIN.  bc2 walks binder_copy_chains and measures CA-CA along each
// one; the junction between a Fab's light and heavy chains is not a bond
// and must not be counted as a broken one.
inline float binderChainBreaks(const std::vector<float>& atoms,
                               const std::vector<float>& atom_mask,
                               const std::vector<int>& chain_lens,
                               float lo = 3.3f, float hi = 4.3f) {
  long breaks = 0; bool any = false;
  int base = 0;
  for (size_t ci = 0; ci < chain_lens.size(); ++ci) {
    const int len = chain_lens[ci];
    for (int k = 0; k + 1 < len; ++k) {
      const int i = base + k, j = i + 1;
      if (atom_mask[(size_t)i * 37 + 1] <= 0 ||
          atom_mask[(size_t)j * 37 + 1] <= 0) continue;
      any = true;
      const float* a = &atoms[((size_t)i * 37 + 1) * 3];
      const float* b = &atoms[((size_t)j * 37 + 1) * 3];
      const double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
      const double d = std::sqrt(dx*dx + dy*dy + dz*dz);
      // 1.0.4: only a STRETCHED bond is a break.  The 3.3 A minimum is
      // gone from binder_chain_breaks_metric (and its parameter with it);
      // `lo` is kept for the signature and ignored.
      (void)lo;
      breaks += (d > (double)hi) ? 1 : 0;
    }
    base += len;
  }
  return any ? (float)breaks : -1.f;        // bc2 returns None when nothing
}

// bc2's contiguous_domain_split: the contiguous cut points that MAXIMISE
// the total within-domain contact count, by dynamic programming over a
// 2-D prefix sum.  Returns the cut ends (the last is always `L`).
inline std::vector<int> contiguousDomainSplit(const std::vector<char>& contact,
                                              int L, int domain_count,
                                              int min_size) {
  const int ms = std::max(1, min_size);
  domain_count = std::max(1, std::min(domain_count, L / ms));
  // integral[(L+1)x(L+1)] prefix sums
  std::vector<long> I((size_t)(L + 1) * (L + 1), 0);
  for (int i = 1; i <= L; ++i)
    for (int j = 1; j <= L; ++j)
      I[(size_t)i * (L + 1) + j] = contact[(size_t)(i - 1) * L + (j - 1)] +
                                   I[(size_t)(i - 1) * (L + 1) + j] +
                                   I[(size_t)i * (L + 1) + (j - 1)] -
                                   I[(size_t)(i - 1) * (L + 1) + (j - 1)];
  auto box = [&](int a, int b) {
    return I[(size_t)b * (L + 1) + b] - I[(size_t)a * (L + 1) + b] -
           I[(size_t)b * (L + 1) + a] + I[(size_t)a * (L + 1) + a];
  };
  // best[end] = (score, cuts); one layer per domain
  std::vector<long> score((size_t)L + 1, -1);
  std::vector<std::vector<int>> cuts((size_t)L + 1);
  std::vector<char> live((size_t)L + 1, 0);
  score[0] = 0; live[0] = 1;
  for (int d = 0; d < domain_count; ++d) {
    const int remaining = (domain_count - d - 1) * ms;
    std::vector<long> ns((size_t)L + 1, -1);
    std::vector<std::vector<int>> nc((size_t)L + 1);
    std::vector<char> nl((size_t)L + 1, 0);
    for (int start = 0; start <= L; ++start) {
      if (!live[(size_t)start]) continue;
      for (int end = start + ms; end <= L - remaining; ++end) {
        const long v = score[(size_t)start] + box(start, end);
        if (!nl[(size_t)end] || v > ns[(size_t)end]) {
          nl[(size_t)end] = 1; ns[(size_t)end] = v;
          nc[(size_t)end] = cuts[(size_t)start];
          nc[(size_t)end].push_back(end);
        }
      }
    }
    score.swap(ns); cuts.swap(nc); live.swap(nl);
  }
  return live[(size_t)L] ? cuts[(size_t)L] : std::vector<int>();
}

// the two metrics that read that split.  `out_inter` is the fraction of
// CA contacts that cross a domain boundary; `out_sep` the smallest
// centroid separation in units of the two domains' radii.  Both < 0 for
// bc2's None.
inline void binderDomainMetrics(const std::vector<float>& atoms,
                                const std::vector<float>& atom_mask,
                                int binder_len, int n_domains,
                                int min_domain_size, float cutoff,
                                float* out_inter, float* out_sep) {
  *out_inter = -1.f; *out_sep = -1.f;
  const int L = binder_len;
  std::vector<char> ok((size_t)L, 0);
  int nvalid = 0;
  for (int i = 0; i < L; ++i) {
    ok[(size_t)i] = atom_mask[(size_t)i * 37 + 1] > 0 ? 1 : 0;
    nvalid += ok[(size_t)i];
  }
  if (L < 2 * min_domain_size || nvalid < 2 * min_domain_size) return;
  std::vector<char> contact((size_t)L * L, 0);
  const double c2 = (double)cutoff * cutoff;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) {
      if (i == j || !ok[(size_t)i] || !ok[(size_t)j]) continue;
      const float* a = &atoms[((size_t)i * 37 + 1) * 3];
      const float* b = &atoms[((size_t)j * 37 + 1) * 3];
      const double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
      contact[(size_t)i * L + j] = (dx*dx + dy*dy + dz*dz) < c2 ? 1 : 0;
    }
  const std::vector<int> ends =
      contiguousDomainSplit(contact, L, n_domains, min_domain_size);
  if (ends.size() < 2) return;
  std::vector<int> starts;
  starts.push_back(0);
  for (size_t k = 0; k + 1 < ends.size(); ++k) starts.push_back(ends[k]);
  long total = 0, within = 0;
  for (int i = 0; i < L; ++i)
    for (int j = 0; j < L; ++j) total += contact[(size_t)i * L + j];
  for (size_t d = 0; d < ends.size(); ++d)
    for (int i = starts[d]; i < ends[d]; ++i)
      for (int j = starts[d]; j < ends[d]; ++j)
        within += contact[(size_t)i * L + j];
  *out_inter = total ? (float)(total - within) / (float)total : 0.f;
  // the separation ratio: the MINIMUM over adjacent domain pairs
  std::vector<std::array<double, 3>> cen(ends.size(), {0, 0, 0});
  std::vector<double> rad(ends.size(), 0.0);
  for (size_t d = 0; d < ends.size(); ++d) {
    double n = 0;
    for (int i = starts[d]; i < ends[d]; ++i) {
      if (!ok[(size_t)i]) continue;
      for (int c = 0; c < 3; ++c)
        cen[d][(size_t)c] += atoms[((size_t)i * 37 + 1) * 3 + c];
      n += 1.0;
    }
    if (n == 0) return;
    for (int c = 0; c < 3; ++c) cen[d][(size_t)c] /= n;
    double q = 0;
    for (int i = starts[d]; i < ends[d]; ++i) {
      if (!ok[(size_t)i]) continue;
      for (int c = 0; c < 3; ++c) {
        const double dd = atoms[((size_t)i * 37 + 1) * 3 + c] - cen[d][(size_t)c];
        q += dd * dd;
      }
    }
    rad[d] = std::sqrt(q / n);
  }
  double best = 1e30;
  for (size_t d = 0; d + 1 < ends.size(); ++d) {
    double q = 0;
    for (int c = 0; c < 3; ++c) {
      const double dd = cen[d][(size_t)c] - cen[d + 1][(size_t)c];
      q += dd * dd;
    }
    best = std::min(best, std::sqrt(q) / (rad[d] + rad[d + 1] + 1e-8));
  }
  *out_sep = (float)best;
}

// Framework_Packing_Fraction: of the FRAMEWORK rows, what fraction touches
// a PARATOPE row at least 20 residues away along the chain?
//
// It wants no solvent-accessible surface area, despite sitting beside the
// three metrics that do -- it is an all-atom contact at 4.5 A between two
// masks the port already builds, and the sequence separation is in ARRAY
// POSITION within the binder chain (bc2 compares the two flatnonzero
// index arrays, not residue_index).  Returns < 0 for bc2's None, which is
// what a binder with no declared paratope gives -- i.e. anything but a
// scaffold.
inline float frameworkPackingFraction(const std::vector<float>& atoms,
                                      const std::vector<float>& atom_mask,
                                      const std::vector<char>& framework,
                                      const std::vector<char>& paratope,
                                      int binder_len, float cutoff = 4.5f,
                                      int separation = 20) {
  std::vector<int> fw, pt;
  for (int i = 0; i < binder_len; ++i) {
    if (framework[(size_t)i]) fw.push_back(i);
    if (paratope[(size_t)i]) pt.push_back(i);
  }
  if (fw.empty() || pt.empty()) return -1.f;
  const double c2 = (double)cutoff * cutoff;
  long packed = 0;
  for (int i : fw) {
    bool any = false;
    for (int j : pt) {
      if (std::abs(i - j) < separation) continue;
      for (int ai = 0; ai < 37 && !any; ++ai) {
        if (atom_mask[(size_t)i * 37 + ai] <= 0) continue;
        const float* a = &atoms[((size_t)i * 37 + ai) * 3];
        for (int aj = 0; aj < 37; ++aj) {
          if (atom_mask[(size_t)j * 37 + aj] <= 0) continue;
          const float* b = &atoms[((size_t)j * 37 + aj) * 3];
          const double dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
          if (dx*dx + dy*dy + dz*dz <= c2) { any = true; break; }
        }
      }
      if (any) break;
    }
    packed += any;
  }
  return (float)packed / (float)fw.size();
}

// ---- THE TWO METRICS THAT READ SOLVENT-ACCESSIBLE AREA ----------------
//
// Both REJECT on settings/property/protease_stable.json
// (max_exposed_loop_fraction_final 0.7, max_terminus_exposure_final 0.6),
// so a protease_stable campaign was being accepted on two checks fewer
// than bc2 applies.  The area itself is sasa.cuh, gated against biotite.
//
// THE AREA IS COMPUTED OVER THE WHOLE COMPLEX and read off the binder's
// rows: bc2's residue_solvent_accessible_area builds the atom array from
// the complete protein_complex and then selects the assembly's residues,
// so a binder face buried against the TARGET is not exposed.  Computing
// it over the binder alone is a different question with the same name.

// Terminus_Exposure: the mean relative area of the first and last
// `terminus_length` rows of each binder protomer.
// `chain_lens` is the binder's chains in row order and `real_counts` the
// REAL residues of each -- bc2 takes the C terminus from
// real_residue_count(flags) and the row offset from len(protein), which are
// the same only when the chain is unpadded.
inline float terminusExposure(const std::vector<double>& area,
                              const std::vector<int>& aa,
                              const std::vector<int>& chain_lens,
                              const std::vector<int>& real_counts,
                              int terminus_length = 3) {
  double sum = 0; long cnt = 0;
  int base = 0;
  for (size_t ci = 0; ci < chain_lens.size(); ++ci) {
    const int len = chain_lens[ci];
    const int real_count = ci < real_counts.size() ? real_counts[ci] : len;
    if (real_count < 1) return -1.f;
    const int cterm = real_count - 1;
    for (int t = 0; t < terminus_length; ++t) {
      for (int which = 0; which < 2; ++which) {
        const int k = which == 0 ? std::min(t, cterm)
                                 : std::max(cterm - t, 0);
        const int i = base + k;
        sum += area[(size_t)i] / sasa::maxResidueArea(aa[(size_t)i]);
        ++cnt;
      }
    }
    base += len;
  }
  return cnt ? (float)(sum / (double)cnt) : -1.f;
}

// Exposed_Loop_Fraction: of the binder rows P-SEA calls neither helix nor
// strand, what fraction has relative area >= 0.2?
inline float exposedLoopFraction(const std::vector<double>& area,
                                 const std::vector<int>& aa,
                                 const std::vector<char>& sse,
                                 int binder_len, int copies,
                                 double surface = 0.2) {
  const int nb = binder_len * copies;
  if ((int)sse.size() != nb) return -1.f;        // bc2's length guard
  long loops = 0, both = 0;
  for (int i = 0; i < nb; ++i) {
    if (sse[(size_t)i] == 'a' || sse[(size_t)i] == 'b') continue;
    ++loops;
    both += (area[(size_t)i] / sasa::maxResidueArea(aa[(size_t)i])
             >= surface) ? 1 : 0;
  }
  if (!loops) return 0.f;                        // bc2 returns 0.0, not None
  return (float)both / (float)loops;
}

// ---- the binder's secondary structure, and whether it keeps its fold ---
//
// Binder_Helix_Fraction / Binder_BetaSheet_Fraction run biotite's
// annotate_sse (P-SEA) over the binder chain of the COMPLEX, PER COPY, and
// count.  sse::annotate is already gated residue-by-residue against biotite
// on eight synthetic shapes and six real chains, so these are a count over
// an answer that is already checked -- which is the only reason they are
// three lines each.
//
// mixed_topology makes both MANDATORY (Binder_BetaSheet_Fraction >= 0.2 and
// Binder_Helix_Fraction <= 0.5 via max_helix_fraction_final), so a port
// without them accepts designs on fewer checks than the campaign asked for.
inline std::vector<char> binderSse(const std::vector<float>& atoms,
                                   const std::vector<float>& atom_mask,
                                   const std::vector<char>& is_binder, int n,
                                   const std::vector<int>& asym) {
  // PER COPY, as binder_copy_chains iterates: a protomer boundary is not a
  // chain break to P-SEA unless the chains are annotated separately, and
  // running it across one invents helices that span the gap.
  std::vector<char> out;
  int lo = 0;
  while (lo < n) {
    while (lo < n && !is_binder[lo]) ++lo;
    if (lo >= n) break;
    int hi = lo;
    while (hi < n && is_binder[hi] && asym[hi] == asym[lo]) ++hi;
    std::vector<float> ca; std::vector<char> valid;
    for (int i = lo; i < hi; ++i) {
      const bool ok = atom_mask[(size_t)i * 37 + 1] > 0;
      valid.push_back(ok ? 1 : 0);
      for (int c = 0; c < 3; ++c)
        ca.push_back(ok ? atoms[((size_t)i * 37 + 1) * 3 + c] : 0.f);
    }
    const std::vector<char> a = sse::annotate(ca, valid);
    out.insert(out.end(), a.begin(), a.end());
    lo = hi;
  }
  return out;
}

// Binder_RMSD: the binder's CA in the COMPLEX against its own BINDER_ALONE
// fold, Kabsch-aligned over the residues resolved in both.  bc2 returns
// None under three valid residues, and None is a filter that does not run.
//
// The rotation comes from kabsch::rotationFromCov -- the same solver the
// template alignment uses -- rather than a second transcription of it.
inline float binderRmsd(const std::vector<float>& atoms,        // complex
                        const std::vector<float>& atom_mask,
                        const std::vector<char>& is_binder, int n,
                        const std::vector<float>& ref_atoms,    // binder_alone
                        const std::vector<float>& ref_mask, int rn) {
  std::vector<float> p, q;
  int k = 0;
  for (int i = 0; i < n && k < rn; ++i) {
    if (!is_binder[i]) continue;
    const bool a = atom_mask[(size_t)i * 37 + 1] > 0;
    const bool b = ref_mask[(size_t)k * 37 + 1] > 0;
    if (a && b)
      for (int c = 0; c < 3; ++c) {
        p.push_back(atoms[((size_t)i * 37 + 1) * 3 + c]);
        q.push_back(ref_atoms[((size_t)k * 37 + 1) * 3 + c]);
      }
    ++k;
  }
  const int m = (int)(p.size() / 3);
  if (m < 3) return -1.f;                 // bc2's None
  double pc[3] = {0, 0, 0}, qc[3] = {0, 0, 0};
  for (int i = 0; i < m; ++i)
    for (int c = 0; c < 3; ++c) { pc[c] += p[i * 3 + c]; qc[c] += q[i * 3 + c]; }
  for (int c = 0; c < 3; ++c) { pc[c] /= m; qc[c] /= m; }
  float cov[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < m; ++i)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        cov[a * 3 + b] += (float)((p[i * 3 + a] - pc[a]) * (q[i * 3 + b] - qc[b]));
  float R[9];
  kabsch::rotationFromCov(cov, R);
  double ss = 0;
  for (int i = 0; i < m; ++i) {
    float v[3];
    for (int c = 0; c < 3; ++c) v[c] = (float)(p[i * 3 + c] - pc[c]);
    for (int c = 0; c < 3; ++c) {
      // v @ R^T, which is what bc2's align_binder_coordinates does
      // (`@ rotation.T`) and what kabsch::applyK does.  Written the other
      // way round it is still a rotation and still gives a plausible
      // number, which is the whole problem with a transposed matrix.
      float acc = 0;
      for (int d = 0; d < 3; ++d) acc += v[d] * R[c * 3 + d];
      const double dd = (double)(acc + qc[c]) - q[i * 3 + c];
      ss += dd * dd;
    }
  }
  return (float)std::sqrt(ss / m);
}

// ---- THE METRICS bc2 RECORDS ON EVERY CAMPAIGN -------------------------
//
// `load_settings` calls `add_design_property_records` for every campaign,
// which installs a dozen checks with a NON-REJECTING threshold (inf, or
// 0.0 on a minimum).  Before the port wrote bc2's tables, a non-rejecting
// filter that did not run changed nothing -- the verdict is the same
// either way, and the startup guard reports it as "inert", which was
// true.  It stopped being true when the output became a RECORD: these are
// columns `bindcraft rank` offers, `bindcraft filter` reapplies and
// `summarize_campaign` summarises, and eleven of them were missing.
//
// A metric that cannot be computed returns -1, this port's None, because
// bc2 returns None for each of them on a binder it cannot read.

// `chain_atom_clash_count` with every atom instead of CA alone.  Unique
// pairs, different chains, both atoms resolved, 2.5 A -- the same count
// `backboneClashes` makes, over 37 atoms a residue rather than one.
inline float allAtomClashes(const std::vector<float>& atoms,
                            const std::vector<float>& atom_mask,
                            const std::vector<int32_t>& asym_id, int n,
                            float cutoff = 2.5f) {
  const float c2 = cutoff * cutoff;
  long count = 0;
  for (int i = 0; i < n; ++i)
    for (int ai = 0; ai < 37; ++ai) {
      if (atom_mask[(size_t)i * 37 + ai] <= 0) continue;
      const float* a = &atoms[((size_t)i * 37 + ai) * 3];
      for (int j = i; j < n; ++j) {
        if (asym_id[i] == asym_id[j]) continue;
        for (int aj = 0; aj < 37; ++aj) {
          if (atom_mask[(size_t)j * 37 + aj] <= 0) continue;
          // UNIQUE PAIRS over the FLATTENED atom index, which is what
          // `jnp.triu(k=1)` takes: i*37+ai < j*37+aj.  Counting residue
          // pairs instead double-counts every cross-chain atom pair.
          if ((size_t)i * 37 + ai >= (size_t)j * 37 + aj) continue;
          const float* b = &atoms[((size_t)j * 37 + aj) * 3];
          const float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
          if (dx*dx + dy*dy + dz*dz <= c2) ++count;
        }
      }
    }
  return (float)count;
}

// `secondary_structure_fraction` with code 'c', which is NOT a P-SEA
// label: bc2 reads it as "neither a nor b", so a residue P-SEA left
// unassigned counts as loop.
inline float sseLoopFraction(const std::vector<char>& ss) {
  if (ss.empty()) return -1.f;
  int k = 0;
  for (char c : ss) k += (c != 'a' && c != 'b') ? 1 : 0;
  return (float)k / (float)ss.size();
}

// `interface_secondary_structure_fraction`: of the binder residues that
// TOUCH the target, what fraction carry this secondary structure?  None
// when nothing touches -- a denominator of zero, which bc2 returns None
// for rather than 0.0.
inline float interfaceSseFraction(const std::vector<char>& ss,
                                  const std::vector<char>& interface_rows,
                                  const std::vector<char>& is_binder, int n,
                                  char code) {
  if (ss.empty()) return -1.f;
  int touched = 0, matching = 0, b = 0;
  for (int i = 0; i < n && b < (int)ss.size(); ++i) {
    if (!is_binder[i]) continue;
    if (interface_rows[i]) {
      ++touched;
      const bool hit = code == 'c' ? (ss[(size_t)b] != 'a' && ss[(size_t)b] != 'b')
                                   : ss[(size_t)b] == code;
      matching += hit ? 1 : 0;
    }
    ++b;
  }
  return touched ? (float)matching / (float)touched : -1.f;
}

// `structured_residue_plddt_metric`: the binder's pLDDT over the residues
// P-SEA called helix or sheet.  ZERO, not None, when nothing is
// structured -- bc2 returns 0.0 there and None only when it could not
// annotate at all, and the two mean different things in a table.
inline float structuredPlddt(const std::vector<float>& plddt,
                             const std::vector<char>& ss,
                             const std::vector<char>& is_binder, int n) {
  if (ss.empty()) return -1.f;
  double acc = 0; int k = 0, b = 0;
  for (int i = 0; i < n && b < (int)ss.size(); ++i) {
    if (!is_binder[i]) continue;
    if (ss[(size_t)b] == 'a' || ss[(size_t)b] == 'b') { acc += plddt[(size_t)i]; ++k; }
    ++b;
  }
  if (!k) return 0.f;
  return (float)(acc / k);
}

// ---- the sequence readings ---------------------------------------------
//
// bc2's own tables, transcribed, because there is nothing to derive them
// from: RESIDUE_MASS is a dictionary of twenty literals.  Everything that
// CAN be derived here is (the vdW radii from the atom names, the MHC
// panels from a generator); this cannot.
// IN pdbin::AA1's ORDER, not alphabetical.  bc2 holds RESIDUE_MASS as a
// dictionary and the lookup is by letter; here the lookup is by index into
// `AA1` = "ARNDCQEGHILKMFPSTWYVX", and a table written alphabetically
// beside it is twenty silently wrong masses.
//                                   A      R       N       D       C
static const double RESIDUE_MASS[20] = {71.08, 156.19, 114.10, 115.09, 103.14,
//                                   Q       E       G      H       I
                                     128.13, 129.12, 57.05, 137.14, 113.16,
//                                   L       K       M       F       P
                                     113.16, 128.17, 131.19, 147.18, 97.12,
//                                   S      T       W       Y       V
                                     87.08, 101.10, 186.21, 163.18, 99.13};

inline int aa1Index(char c) {
  for (int i = 0; i < 20; ++i) if (pdbin::AA1[i] == c) return i;
  return -1;
}
static const double WATER_MASS = 18.02;
static const double REPORTED_PH = 7.4;

// `sequence` is one letter per binder residue, every protomer
// concatenated (`binder_assembly_sequence`).
inline double sequenceMass(const std::string& seq) {
  double m = WATER_MASS;
  for (char c : seq) {
    const int a = aa1Index(c);
    if (a >= 0) m += RESIDUE_MASS[a];
  }
  return m / 1000.0;
}

inline double sequenceNetCharge(const std::string& seq, double ph) {
  // the two termini, then every titratable side chain
  double charge = 1.0 / (1.0 + pow(10.0, ph - 9.69))
                - 1.0 / (1.0 + pow(10.0, 2.34 - ph));
  struct Pk { char aa; double pka; bool basic; };
  static const Pk SIDE[7] = {{'D', 3.86, false}, {'E', 4.25, false},
                             {'C', 8.33, false}, {'Y', 10.07, false},
                             {'H', 6.00, true},  {'K', 10.53, true},
                             {'R', 12.48, true}};
  for (const Pk& p : SIDE) {
    int k = 0;
    for (char c : seq) k += (c == p.aa) ? 1 : 0;
    charge += k * (p.basic ? 1.0 / (1.0 + pow(10.0, ph - p.pka))
                           : -1.0 / (1.0 + pow(10.0, p.pka - ph)));
  }
  return charge;
}

// FIFTY bisections of [0, 14], not a convergence test -- bc2 runs a fixed
// count and the answer is whatever that lands on, which at 50 halvings of
// 14 is below any precision this prints at anyway.
inline double sequenceIsoelectricPoint(const std::string& seq) {
  double lo = 0.0, hi = 14.0;
  for (int i = 0; i < 50; ++i) {
    const double mid = (lo + hi) / 2;
    if (sequenceNetCharge(seq, mid) > 0) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
}

inline double sequenceCysteines(const std::string& seq) {
  int k = 0;
  for (char c : seq) k += (c == 'C') ? 1 : 0;
  return (double)k;
}

// W 5500, Y 1490, and 125 per CYSTINE -- the disulfide count, not the
// cysteine count, so this needs the geometry and not only the sequence.
inline double sequenceExtinction(const std::string& seq, double disulfides) {
  int w = 0, y = 0;
  for (char c : seq) { w += (c == 'W'); y += (c == 'Y'); }
  return 5500.0 * w + 1490.0 * y + 125.0 * disulfides;
}

// `max(0, cysteines - 2 * disulfides)`: the unpaired ones
inline double sequenceFreeCysteines(const std::string& seq, double disulfides) {
  return std::max(0.0, sequenceCysteines(seq) - 2.0 * disulfides);
}

// ---- the SASA readings bc2 records on every campaign --------------------
//
// `residue_solvent_accessible_area` twice: once over the binder chains
// ALONE (unbound) and once over the whole complex (bound).  What the
// interface buries is the difference, and what is "surface" is a residue
// exposing at least a fifth of the area its type can expose.
//
// MAX_RESIDUE_SASA is in pdbin::AA1's order, as RESIDUE_MASS is, and for
// the same reason.
//                                      A      R      N      D      C
static const double MAX_RESIDUE_SASA[20] = {121.0, 265.0, 187.0, 187.0, 148.0,
//                                      Q      E      G     H      I
                                        214.0, 214.0, 97.0, 216.0, 195.0,
//                                      L      K      M      F      P
                                        191.0, 230.0, 203.0, 228.0, 154.0,
//                                      S      T      W      Y      V
                                        143.0, 163.0, 264.0, 255.0, 165.0};
static const double SURFACE_RELATIVE_SASA = 0.2;
// `HYDROPHOBIC_AMINO_ACIDS = frozenset('ACVILMPFWY')`
inline bool isHydrophobic(char c) {
  return c == 'A' || c == 'C' || c == 'V' || c == 'I' || c == 'L' ||
         c == 'M' || c == 'P' || c == 'F' || c == 'W' || c == 'Y';
}

// `Interface_BuriedArea` and its fraction.  `unbound` and `bound` are the
// per-residue areas over the BINDER's rows only, in the same order.
inline float interfaceBuriedArea(const std::vector<double>& unbound,
                                 const std::vector<double>& bound) {
  if (unbound.empty() || unbound.size() != bound.size()) return -1.f;
  double u = 0, b = 0;
  for (size_t i = 0; i < unbound.size(); ++i) { u += unbound[i]; b += bound[i]; }
  return (float)std::max(0.0, u - b);
}

inline float interfaceBuriedAreaFraction(const std::vector<double>& unbound,
                                         const std::vector<double>& bound) {
  if (unbound.empty() || unbound.size() != bound.size()) return -1.f;
  double u = 0, b = 0;
  for (size_t i = 0; i < unbound.size(); ++i) { u += unbound[i]; b += bound[i]; }
  if (u <= 0.0) return -1.f;                       // bc2's None
  return (float)std::max(0.0, (u - b) / u);
}

// `Surface_Hydrophobicity`: of the binder residues exposing at least a
// fifth of their type's maximum area IN THE UNBOUND FOLD, what fraction
// are hydrophobic?  ZERO when nothing is exposed, not None.
inline float surfaceHydrophobicity(const std::vector<double>& unbound,
                                   const std::vector<int>& aatype) {
  if (unbound.empty() || unbound.size() != aatype.size()) return -1.f;
  int surface = 0, phobic = 0;
  for (size_t i = 0; i < unbound.size(); ++i) {
    const int a = aatype[i];
    if (a < 0 || a >= 20) continue;
    if (unbound[i] / MAX_RESIDUE_SASA[a] < SURFACE_RELATIVE_SASA) continue;
    ++surface;
    phobic += isHydrophobic(pdbin::AA1[a]) ? 1 : 0;
  }
  if (!surface) return 0.f;
  return (float)phobic / (float)surface;
}

// `Interface_Hydrophobicity`: of the binder residues that TOUCH the
// target, what fraction are hydrophobic?  None when nothing touches.
inline float interfaceHydrophobicity(const std::vector<char>& interface_rows,
                                     const std::vector<char>& is_binder,
                                     const std::vector<int>& aatype, int n) {
  int touched = 0, phobic = 0;
  for (int i = 0; i < n; ++i) {
    if (!is_binder[i] || !interface_rows[i]) continue;
    ++touched;
    const int a = aatype[(size_t)i];
    phobic += (a >= 0 && a < 20 && isHydrophobic(pdbin::AA1[a])) ? 1 : 0;
  }
  return touched ? (float)phobic / (float)touched : -1.f;
}

// `Receptor_Chains_Contacted`: how many of the target's fused chains the
// binder reaches.  The chain index comes out of the residue NUMBERING --
// a jump of at least RECEPTOR_CHAIN_BREAK_GAP is where
// protein_preparation fused two chains -- so a single-chain target has no
// such jump and the metric is None rather than 1.
inline float receptorChainsContacted(const std::vector<char>& contacted,
                                     const std::vector<char>& is_binder,
                                     const std::vector<int>& residue_index,
                                     int n, int gap) {
  std::vector<int> idx;
  int chain = 0, previous = 0; bool first = true;
  std::vector<int> chain_of;
  for (int i = 0; i < n; ++i) {
    if (is_binder[i]) continue;
    if (!first && residue_index[(size_t)i] - previous >= gap) ++chain;
    previous = residue_index[(size_t)i];
    first = false;
    chain_of.push_back(chain);
    idx.push_back(i);
  }
  if (chain < 1) return -1.f;                      // one chain: bc2's None
  std::set<int> hit;
  for (size_t k = 0; k < idx.size(); ++k)
    if (contacted[(size_t)idx[k]]) hit.insert(chain_of[k]);
  return (float)hit.size();
}

// `Target_RMSD`: how far the predicted target backbone moves from the
// structure it was given, over CA after a Kabsch alignment.  The target
// is a TEMPLATE and should not move; this is the check that says whether
// it did.
inline float targetRmsd(const std::vector<float>& predicted,
                        const std::vector<float>& predicted_mask,
                        const std::vector<float>& reference,
                        const std::vector<float>& reference_mask,
                        const std::vector<char>& is_binder, int n) {
  std::vector<float> p, q;
  for (int i = 0; i < n; ++i) {
    if (is_binder[i]) continue;
    if (predicted_mask[(size_t)i * 37 + 1] <= 0 ||
        reference_mask[(size_t)i * 37 + 1] <= 0) continue;
    for (int c = 0; c < 3; ++c) {
      p.push_back(predicted[((size_t)i * 37 + 1) * 3 + c]);
      q.push_back(reference[((size_t)i * 37 + 1) * 3 + c]);
    }
  }
  const int m = (int)(p.size() / 3);
  if (m < 3) return -1.f;
  double pc[3] = {0, 0, 0}, qc[3] = {0, 0, 0};
  for (int i = 0; i < m; ++i)
    for (int c = 0; c < 3; ++c) { pc[c] += p[i*3+c]; qc[c] += q[i*3+c]; }
  for (int c = 0; c < 3; ++c) { pc[c] /= m; qc[c] /= m; }
  float cov[9] = {0,0,0,0,0,0,0,0,0};
  for (int i = 0; i < m; ++i)
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        cov[a*3+b] += (float)((p[i*3+a] - pc[a]) * (q[i*3+b] - qc[b]));
  float R[9];
  kabsch::rotationFromCov(cov, R);
  double ss = 0;
  for (int i = 0; i < m; ++i) {
    float v[3];
    for (int c = 0; c < 3; ++c) v[c] = (float)(p[i*3+c] - pc[c]);
    for (int c = 0; c < 3; ++c) {
      float acc = 0;
      for (int d = 0; d < 3; ++d) acc += v[d] * R[c*3+d];
      const double dd = (double)(acc + qc[c]) - q[i*3+c];
      ss += dd * dd;
    }
  }
  return (float)std::sqrt(ss / m);
}

// ---- Cyclic_Closure_Distance -------------------------------------------
//
// The gap a cyclic peptide has left between the N of its first residue
// and the C of its last.  bc2 returns None when the binder declares no
// CYCLIC row and when either atom is unresolved -- None, not a number, so
// a linear binder does not appear in the table with a closure distance
// beside it.  -1 is this port's None, as everywhere else here.
//
// THE C TERMINUS IS THE LAST REAL RESIDUE, not the last row: the state is
// padded to nmax and `real_residue_count - 1` is what bc2 indexes.
inline float cyclicClosureDistance(const std::vector<float>& atoms,
                                   const std::vector<float>& atom_mask,
                                   const std::vector<char>& is_binder,
                                   const std::vector<uint8_t>& flags,
                                   const std::vector<float>& seq_mask, int n) {
  int first = -1, last = -1, real = 0;
  bool any_cyclic = false;
  for (int i = 0; i < n; ++i) {
    if (!is_binder[i]) continue;
    any_cyclic |= (flags[(size_t)i] & dsin::F_CYCLIC) != 0;
    if (seq_mask[i] <= 0) continue;
    if (first < 0) first = i;
    last = i;
    ++real;
  }
  if (!any_cyclic || first < 0 || real < 2) return -1.f;
  // N is atom 0 and C is atom 2 in the 37-atom layout
  if (atom_mask[(size_t)first * 37 + 0] <= 0 ||
      atom_mask[(size_t)last * 37 + 2] <= 0) return -1.f;
  const float* a = &atoms[((size_t)first * 37 + 0) * 3];
  const float* b = &atoms[((size_t)last * 37 + 2) * 3];
  double q = 0;
  for (int c = 0; c < 3; ++c) { const double d = (double)a[c] - b[c]; q += d * d; }
  return (float)std::sqrt(q);
}

// ---- induced_fit_mobile_residues, for the MPNN hold set ----------------
//
// Which binder residues MOVE between the bound and the unbound fold.
// `MPNN_stage.induced_fit_mobile_residues`: CA deviation after a Kabsch
// alignment of the unbound fold onto the bound one, thresholded at 2.0 A,
// DILATED by `shell` residues each side, and then capped so that at least
// `designed_share` of the resolved chain is still redesignable -- a
// binder whose whole chain moved would otherwise be held entirely and
// ProteinMPNN would be asked to redesign nothing.
//
// This is not an induced-fit feature.  `prepare_binder_redesign` computes
// it whenever a binder_alone prediction exists, which is every campaign
// that weights a loss on the free fold -- so on bc2's default campaign
// ProteinMPNN is handed a partly frozen binder and the port was handing
// it a free one.
inline std::vector<char> inducedFitMobile(const std::vector<float>& atoms,
                                          const std::vector<float>& atom_mask,
                                          const std::vector<char>& is_binder,
                                          int n,
                                          const std::vector<float>& ref_atoms,
                                          const std::vector<float>& ref_mask,
                                          int rn, float threshold = 2.0f,
                                          int shell = 1,
                                          float designed_share = 0.25f) {
  std::vector<int> rows;              // the complex rows, binder order
  std::vector<float> p, q;            // bound CA, unbound CA
  std::vector<char> valid;
  int k = 0;
  for (int i = 0; i < n && k < rn; ++i) {
    if (!is_binder[i]) continue;
    const bool a = atom_mask[(size_t)i * 37 + 1] > 0;
    const bool b = ref_mask[(size_t)k * 37 + 1] > 0;
    rows.push_back(i);
    valid.push_back(a && b);
    for (int c = 0; c < 3; ++c) {
      p.push_back(atoms[((size_t)i * 37 + 1) * 3 + c]);
      q.push_back(ref_atoms[((size_t)k * 37 + 1) * 3 + c]);
    }
    ++k;
  }
  const int L = (int)rows.size();
  std::vector<char> mobile((size_t)L, 0);
  int m = 0; for (char v : valid) m += v ? 1 : 0;
  if (m < 3) return mobile;
  // the alignment uses the VALID pairs only, as bc2's `valid_mask` does
  double pc[3] = {0, 0, 0}, qc[3] = {0, 0, 0};
  for (int i = 0; i < L; ++i)
    if (valid[i])
      for (int c = 0; c < 3; ++c) { pc[c] += p[i * 3 + c]; qc[c] += q[i * 3 + c]; }
  for (int c = 0; c < 3; ++c) { pc[c] /= m; qc[c] /= m; }
  // q (unbound) onto p (bound), which is bc2's
  // `align_binder_coordinates(reference_coordinates, coordinates, ...)`
  float cov[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < L; ++i) {
    if (!valid[i]) continue;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        cov[a * 3 + b] += (float)((q[i * 3 + a] - qc[a]) * (p[i * 3 + b] - pc[b]));
  }
  float R[9];
  kabsch::rotationFromCov(cov, R);
  std::vector<float> dev((size_t)L, 0.f);
  for (int i = 0; i < L; ++i) {
    if (!valid[i]) continue;
    float v[3];
    for (int c = 0; c < 3; ++c) v[c] = (float)(q[i * 3 + c] - qc[c]);
    double ss = 0;
    for (int c = 0; c < 3; ++c) {
      float acc = 0;
      for (int d = 0; d < 3; ++d) acc += v[d] * R[c * 3 + d];
      const double dd = (double)(acc + pc[c]) - p[i * 3 + c];
      ss += dd * dd;
    }
    dev[i] = (float)std::sqrt(ss + 1e-08);
    mobile[i] = dev[i] > threshold ? 1 : 0;
  }
  // DILATE, then re-mask on resolved.  `jnp.roll(...).at[0].set(False)`
  // is a shift and not a wrap: the first residue never inherits the last
  // one's movement.
  for (int s = 0; s < shell; ++s) {
    std::vector<char> grown = mobile;
    for (int i = 0; i < L; ++i) {
      if (i > 0 && mobile[i - 1]) grown[i] = 1;
      if (i + 1 < L && mobile[i + 1]) grown[i] = 1;
    }
    mobile = grown;
  }
  for (int i = 0; i < L; ++i) if (!valid[i]) mobile[i] = 0;
  // the ceiling: release the LEAST deviant until enough of the chain is
  // redesignable again
  const int ceiling = m - (int)(designed_share * (float)m);
  int held = 0; for (char v : mobile) held += v ? 1 : 0;
  if (held > ceiling) {
    std::vector<int> order;
    for (int i = 0; i < L; ++i) if (mobile[i]) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return dev[a] < dev[b]; });
    for (int i = 0; i < held - ceiling; ++i) mobile[order[i]] = 0;
  }
  return mobile;
}

// chain_pair_pae_loss over (binder rows, target columns), scaled by 1/31 --
// the same primitive interface_pae uses, unweighted.
// THE TWO SIDES ARE TWO MASKS, not a mask and its complement.
//
// bc2 reads `pae[aligned_rows, reference_rows]` for the chain GROUPS it
// is given: i_pAE's aligned group is the ONE chain
// resolve_designed_binder_chain names (binder_0 on an oligomer) and its
// reference group is the target.  A single mask and its complement makes
// the SECOND PROTOMER part of the target -- 0.7029 against bc2's 0.7045
// on a two-copy design, after the first-protomer mask had already fixed
// the larger error.  With copies == 1 the complement is the target and
// the two conventions coincide, which is why this read right for months.
//
// Direction does not matter here and the loop keeps both: the port's pae
// is symmetric to 0.0 (checked on the dump), where bc2 takes the
// binder-row half only.
inline float interfacePae(const std::vector<float>& pae,
                          const std::vector<char>& is_binder,
                          const std::vector<char>& is_target,
                          int n, int nmax) {
  double sum = 0, w = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const bool pair = (is_binder[i] && is_target[j]) ||
                        (is_target[i] && is_binder[j]);
      if (!pair) continue;              // binder x target, both directions
      sum += (double)pae[(size_t)i * nmax + j] / 31.0;
      w += 1.0;
    }
  return w > 0 ? (float)(sum / w) : 0.f;
}

inline float meanOver(const std::vector<float>& v, int n) {
  double s = 0;
  for (int i = 0; i < n; ++i) s += v[i];
  return n ? (float)(s / n) : 0.f;
}

// the same verdict without the table.  The ensemble runs this per model
// per MPNN candidate, and ten candidates x two models is twenty filter
// tables -- which is not a log anybody reads, and the decision is the same
// either way.
inline bool passed(const std::vector<Result>& rs) {
  for (const Result& r : rs) if (!r.passed) return false;
  return true;
}

inline bool report(const std::vector<Result>& rs) {
  int failed = 0;
  printf("    final filters:\n");
  for (const Result& r : rs) {
    printf("      %-22s %10.4f  %s %-6.2f  %s\n", r.name, r.value,
           r.higher ? ">=" : "<=", r.threshold, r.passed ? "pass" : "FAIL");
    failed += !r.passed;
  }
  if (failed)
    printf("    REJECTED: %d of %zu filters failed\n", failed, rs.size());
  else
    printf("    ACCEPTED: all %zu filters passed\n", rs.size());
  return failed == 0;
}


// ---- THE PER-STAGE FILTERS, which ABORT a trajectory mid-run -------------
//
// bc2 checks after EVERY stage, not only at the end (trajectory.py's
// judge_stage), and the defaults are live: min_plddt 0.6 / 0.6 / 0.65 /
// 0.65 / 0.6 / 0.7 and min_iptm -- / -- / 0.5 / 0.5 / 0.5 / 0.7 across
// screen, refine, anneal, harden, mutate, final.  A port that runs all six
// stages regardless spends 125 gradient steps on trajectories bc2 would
// have dropped after 50, and its campaign statistics mean something else.
//
// THE pLDDT IS THE BINDER'S ROWS OF THE COMPLEX, not binder_alone's.
// design_stage_filters binds plddt_metric with prediction_state='complex'
// AND chain='binder', so it is a masked mean over the designed residues of
// the bound fold.  Unbound_Binder_pLDDT, the >= 0.80 final filter, is the
// other one -- same metric, different state, and reading one for the other
// is a check that passes when it should not.
inline float binderPlddtInComplex(const std::vector<float>& plddt,
                                  const std::vector<char>& is_binder,
                                  const std::vector<float>& seq_mask, int n) {
  double s = 0, w = 0;
  for (int i = 0; i < n; ++i) {
    if (!is_binder[i]) continue;
    const float m = seq_mask.empty() ? 1.f : seq_mask[i];
    s += (double)plddt[i] * m;
    w += m;
  }
  return w > 0 ? (float)(s / w) : 0.f;
}

struct StageCheck {
  bool ran = false, passed = true;
  float plddt = 0, iptm = 0, plddt_thr = 0, iptm_thr = 0;
  bool has_plddt = false, has_iptm = false;
};

inline bool reportStage(const char* stage, const StageCheck& c) {
  if (!c.ran) return true;
  char line[256];
  int o = snprintf(line, sizeof line, "      %-6s filters:", stage);
  if (c.has_plddt)
    o += snprintf(line + o, sizeof line - o, "  pLDDT %.3f >= %.2f %s",
                  c.plddt, c.plddt_thr, c.plddt >= c.plddt_thr ? "pass" : "FAIL");
  if (c.has_iptm)
    o += snprintf(line + o, sizeof line - o, "   i_pTM %.3f >= %.2f %s",
                  c.iptm, c.iptm_thr, c.iptm >= c.iptm_thr ? "pass" : "FAIL");
  printf("%s\n", line);
  return c.passed;
}


// ---- i_pDAE, THE RANKING METRIC ------------------------------------------
//
// bc2 ranks accepted designs by i_pDAE (campaign_output.RANKING_METRIC),
// not by i_pTM or by the loss: "how well the two sides agree on the pose".
// Each interface residue gets a TM-like score over its contacts,
//
//   partner_count = number of partners within 8 A (CA-CA, both resolved)
//   scale         = max(1.24 * (max(pc, 19) - 15)^(1/3) - 1.8, 1)
//   score_i       = mean over contacts of 1 / (1 + (pae_ij / scale)^2)
//
// and the metric is the MAXIMUM over residues, taken in BOTH directions
// (binder rows against target columns, and the transpose).  The maximum,
// not the mean: one residue pair the two sides agree on anchors the pose.
//
// The pae here is in ANGSTROMS, not the 0-1 the i_pAE filter divides down
// to -- the same array read with two conventions, which is exactly the
// kind of thing to say out loud.
inline float anchoredMax(const std::vector<float>& pae, int nmax,
                         const std::vector<int>& rows,
                         const std::vector<int>& cols,
                         const std::vector<char>& contact, bool transpose) {
  const int R = transpose ? (int)cols.size() : (int)rows.size();
  const int C = transpose ? (int)rows.size() : (int)cols.size();
  float best = 0.f;
  bool any = false;
  for (int i = 0; i < R; ++i) {
    int pc = 0;
    for (int j = 0; j < C; ++j)
      pc += transpose ? contact[(size_t)j * cols.size() + i]
                      : contact[(size_t)i * cols.size() + j];
    if (!pc) continue;
    any = true;
    const double base = std::max(pc, 19) - 15.0;
    const double scale = std::max(1.24 * std::cbrt(base) - 1.8, 1.0);
    double sum = 0;
    for (int j = 0; j < C; ++j) {
      const bool c = transpose ? contact[(size_t)j * cols.size() + i]
                               : contact[(size_t)i * cols.size() + j];
      if (!c) continue;
      const int r = transpose ? cols[i] : rows[i];
      const int q = transpose ? rows[j] : cols[j];
      const double p = pae[(size_t)r * nmax + q];
      sum += 1.0 / (1.0 + (p / scale) * (p / scale));
    }
    best = std::max(best, (float)(sum / std::max(pc, 1)));
  }
  return any ? best : 0.f;
}

inline float interfacePdae(const std::vector<float>& pae,
                           const std::vector<float>& atoms,
                           const std::vector<float>& atom_mask,
                           const std::vector<char>& is_binder, int n, int nmax,
                           float cutoff = 8.0f) {
  std::vector<int> rows, cols;
  for (int i = 0; i < n; ++i)
    (is_binder[i] ? rows : cols).push_back(i);
  // bc2 returns NONE, not 0.0, when nothing touches: interface_pae_
  // directions bails on `not contact.any()`.  0.0 is a value -- the worst
  // one, since i_pDAE is maximised -- and a design with no interface at
  // all would appear in the table with a number beside it.
  if (rows.empty() || cols.empty()) return -1.f;
  std::vector<char> contact(rows.size() * cols.size(), 0);
  const float c2 = cutoff * cutoff;
  for (size_t a = 0; a < rows.size(); ++a)
    for (size_t b = 0; b < cols.size(); ++b) {
      const int i = rows[a], j = cols[b];
      if (atom_mask[(size_t)i * 37 + 1] <= 0 || atom_mask[(size_t)j * 37 + 1] <= 0)
        continue;
      const float* p = &atoms[((size_t)i * 37 + 1) * 3];
      const float* q = &atoms[((size_t)j * 37 + 1) * 3];
      const float dx = p[0]-q[0], dy = p[1]-q[1], dz = p[2]-q[2];
      if (dx*dx + dy*dy + dz*dz <= c2) contact[a * cols.size() + b] = 1;
    }
  bool any_contact = false;
  for (char c : contact) any_contact |= (c != 0);
  if (!any_contact) return -1.f;                 // bc2's None
  return std::max(anchoredMax(pae, nmax, rows, cols, contact, false),
                  anchoredMax(pae, nmax, rows, cols, contact, true));
}

}  // namespace filt