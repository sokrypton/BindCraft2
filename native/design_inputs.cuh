// BUILDING A DESIGN STEP'S INPUTS FROM A TARGET AND A BINDER LENGTH.
//
// bc2 assembles these in alphafold_input_features(); this is the same
// assembly for the native driver, for the two states a trajectory predicts.
//
// THE TWO STATES ARE DIFFERENT FOLDS, not one with a mask.  `complex` is
// binder + target; `binder_alone` is the binder chain ONLY and is therefore
// SHORTER.  Four of the eight default losses read binder_alone (see
// CPP_PORT.md) and they must see a fold with no target in it at all.
#pragma once
#include <algorithm>
#include <cctype>
#include <cmath>
#include "pdb_in.cuh"
#include "res_tables.cuh"

namespace dsin {

// protein_preparation.RECEPTOR_CHAIN_BREAK_GAP: how far apart the receptor's
// chains are held in the residue numbering once they are fused into one.
constexpr int RECEPTOR_CHAIN_BREAK_GAP = 49;

// bindcraft/protein.py::ResidueFlags
enum : uint8_t {
  F_DESIGN = 1 << 0, F_TEMPLATE = 1 << 1, F_SEQUENCE = 1 << 2,
  F_CONTACT = 1 << 3, F_HOTSPOT = 1 << 4, F_COLDSPOT = 1 << 5,
  F_CYCLIC  = 1 << 6, F_PADDING = 1 << 7,
};

static constexpr int NAA = 20, ATOM = 37;

// ---- binder_sequence_offsets, for a chain that may be CYCLIC ------------
//
// `loss.binder_sequence_offsets`: the signed residue offset, wrapped the
// short way round when BOTH rows are cyclic and real.  Every loss that
// reads "how far apart along the chain" goes through it --
// binder_helicity's i,i+3 band, binder_contacts' separation 9, the
// binder_intra_* separation 20 -- so on a cyclic peptide residue 1 and
// residue 14 are two apart, not thirteen, and a helix that closes round
// the ring is a helix.
//
// The same wrap the EMBEDDER already does (`inputfeat::offsetK`, which is
// bc2's `cyclic_sequence_offsets`).  Two places compute it because they
// run on different sides: that one on the device for the pair features,
// this one on the host for the loss masks.  They have to agree, and the
// cyclic gate checks that they do.
inline int sequenceOffset(int ri, int rj, double real_count, bool both_cyclic) {
  const int off = ri - rj;
  if (!both_cyclic) return off;
  const int ao = off < 0 ? -off : off;
  double cd = std::min((double)ao, real_count - (double)ao);
  if (cd < (double)ao) cd = -cd;                    // 'direction'
  const int sgn = (off > 0) - (off < 0);
  return (int)llrint(cd * sgn);
}

// One state's host-side buffers, padded to nmax.  The device copies are the
// driver's business; this keeps the assembly testable without a GPU.
struct State {
  int n = 0, nmax = 0;
  std::vector<float> sequence, atoms, atom_mask, seq_mask;
  std::vector<uint8_t> flags;
  std::vector<int32_t> residue_index, asym_id, entity_id, interface_asym;

  void resize(int n_, int nmax_) {
    n = n_; nmax = nmax_;
    sequence.assign((size_t)nmax * NAA, 0.f);
    atoms.assign((size_t)nmax * ATOM * 3, 0.f);
    atom_mask.assign((size_t)nmax * ATOM, 0.f);
    seq_mask.assign(nmax, 0.f);
    flags.assign(nmax, 0);
    residue_index.assign(nmax, 0);
    asym_id.assign(nmax, 0);
    entity_id.assign(nmax, 0);
    interface_asym.assign(nmax, 0);
  }
};

inline int padTo(int n, int tile = 32) { return (n + tile - 1) / tile * tile; }

// THE COMPLEX: binder first, then the target's chains.
//
// The order is the binder's because bc2's chain naming puts the binder
// chain first and the losses slice on that assumption -- binder_binding_mask
// and chain_residue_slices both read the chain order, so reversing it here
// would move every interface loss onto the wrong residues with no error.
// `nmax_floor` lets a SHORT state be built in a LONG state's buffers.  A
// campaign samples binder_lengths per trajectory, and every device buffer
// here is sized by the state it was built from -- so a 60-mer and a
// 100-mer would need two allocations unless the short one is laid out at
// the long one's nmax.  The runtime length `n` is what the step reads, so
// the padding costs memory it has already spent and nothing else.
// A hotspot span: a chain and an inclusive residue range.  Declared here
// because buildComplex marks the flags and parseHotspots builds them.
struct Hotspot { char chain; int first, last; };

// `copies` is bc2's oligomer: N binder chains that SHARE ONE SEQUENCE.
// They are separate chains (own asym_id, so the model sees an interface
// between them) but ONE entity -- entity_id 0 for all of them -- which is
// what tells AF2's relative encoding they are copies rather than
// neighbours.  The tie is in the gradient too: every copy's rows
// contribute to the same logits (see the accumulation in the driver).
// COLDSPOTS are the other half of target_binding_site_flags, and they were
// the half nothing wrote.  F_COLDSPOT has been in the enum since the flags
// were ported, at bc2's own bit, and no line set it -- so a target naming
// coldspots (settings/target/hIL7RA.json ships one) lost an entire loss and
// a final filter in silence.  Same shape as Outputs::atom_mask.
// A SCAFFOLD BINDER ROW, as scaf::buildScaffoldChain produces it.  Declared
// here rather than including scaffold.cuh so design_inputs stays free of
// the edit-spec machinery; the caller converts.
struct BinderRow {
  int aatype = 20;                 // 20 = designed, no identity given
  float xyz[37][3] = {};
  float mask[37] = {};
  uint8_t flags = F_DESIGN;
  int residue_index = 0;
  // WHICH BINDER CHAIN THIS ROW BELONGS TO, 0-based.  A Fab or an scFv is
  // ONE binder spanning two chains (L and H), built from a two-chain
  // scaffold -- where `copies` is N chains that SHARE one sequence.  bc2
  // forbids the two together (`resolve_binder_chains` raises on a
  // multi-chain scaffold with copies > 1), so a row's chain is either its
  // copy index or its scaffold chain, never a product of both.
  int chain = 0;
};

// `scaf_chains` is how many CHAINS the scaffold binder spans -- 1 for every
// de novo binder and for a single-chain scaffold, 2 for a Fab or an scFv.
// It is separate from `copies` because the two mean opposite things about
// the SEQUENCE: copies share one logit array and one entity, chains do not.
inline State buildComplex(const pdbin::Structure& tgt, int binder_len,
                          const std::vector<Hotspot>& hotspots,
                          int nmax_floor = 0, float flexibility = 0.f,
                          int copies = 1,
                          const std::vector<Hotspot>& coldspots = {},
                          const std::vector<BinderRow>& scaffold = {},
                          int scaf_chains = 1) {
  State s;
  const int nb = binder_len * copies;
  const int n = nb + tgt.size();
  s.resize(n, std::max(padTo(n), nmax_floor));
  for (int c = 0; c < copies; ++c)
    for (int k = 0; k < binder_len; ++k) {
      const int i = c * binder_len + k;
      if (!scaffold.empty()) {
        // A SCAFFOLD BINDER IS NOT ALL-DESIGNED.  Its fixed rows carry the
        // scaffold's coordinates and identity as TEMPLATE|SEQUENCE and the
        // designed spans are holes -- no coordinates, no mask -- exactly
        // as Protein.with_scaffold builds them.  The residue numbering is
        // the scaffold's own, shifted by the spans' net length change, and
        // NOT 1..n: only differences reach the relative encoding, so a
        // renumbering moves every pair across a resized loop.
        const BinderRow& b = scaffold[k];
        s.flags[i] = b.flags;
        s.residue_index[i] = b.residue_index;
        for (int a2 = 0; a2 < ATOM; ++a2) {
          s.atom_mask[(size_t)i * ATOM + a2] = b.mask[a2];
          for (int c2 = 0; c2 < 3; ++c2)
            s.atoms[((size_t)i * ATOM + a2) * 3 + c2] = b.xyz[a2][c2];
        }
        if (!(b.flags & F_DESIGN) && b.aatype < NAA)
          s.sequence[(size_t)i * NAA + b.aatype] = 1.f;
      } else {
        s.flags[i] = F_DESIGN;                // designed, no template
        // the residue index RESTARTS per copy -- they are the same chain
        // -- and starts at ONE, as Protein.empty numbers it.  Only
        // differences within a chain reach the relative encoding, so the
        // offset is a no-op there; it is matched so that nobody has to
        // re-derive that.
        s.residue_index[i] = k + 1;
      }
      // ASYM SPLITS THE CHAINS, ENTITY SAYS WHICH ARE THE SAME MOLECULE.
      // Copies are one entity (bc2's residue_entity_ids maps every member
      // of a multi_chain_binders group onto the group's first name); a
      // Fab's L and H are two, because nothing ties them.  The product
      // form is general and collapses to `c` or to the chain index,
      // because bc2 refuses both at once.
      const int ch = scaffold.empty() ? 0 : scaffold[k].chain;
      s.asym_id[i] = c * scaf_chains + ch;
      s.entity_id[i] = scaf_chains > 1 ? ch : 0;
      s.interface_asym[i] = 0;
      s.seq_mask[i] = 1.f;
    }
  // THE TARGET IS ONE CHAIN, however many the PDB has.
  //
  // bc2 fuses a multi-chain receptor with merge_receptor_chains: ONE
  // Protein, one asym_id, one entity_id, and the chains held apart by a
  // RESIDUE-INDEX GAP of 49 rather than by a chain id.  Giving each PDB
  // chain its own asym_id instead makes the model see a three-chain
  // complex where bc2 sees two -- a different relative encoding at every
  // pair that crosses the junction -- and leaves the chains' PDB numbering
  // OVERLAPPING, which with one asym_id would place B's residue 1 next to
  // A's.
  //
  // The renumbering is bc2's: each chain after the first starts
  // (its own first index) + (the previous chain's last index) + 49.
  // one chain id for the whole target, after every binder chain
  const int tgt_asym = copies * scaf_chains;
  char last = 0;
  int prev_last = 0, chain_first = 0;
  bool first_chain = true;
  for (int r = 0; r < tgt.size(); ++r) {
    const pdbin::Residue& res = tgt.res[r];
    if (res.chain != last) {
      if (!first_chain) { prev_last = s.residue_index[nb + r - 1];
                          chain_first = res.resseq; }
      else { chain_first = res.resseq; first_chain = false; }
      last = res.chain;
    }
    const int i = nb + r;
    // TEMPLATE | SEQUENCE: the target's coordinates AND its identity are
    // given.  A target marked DESIGN would be redesigned, which is the kind
    // of mistake that shows up as a campaign that quietly optimises the
    // wrong chain.
    s.flags[i] = F_TEMPLATE | F_SEQUENCE;
    for (const Hotspot& h : hotspots)
      if ((h.chain == 0 || h.chain == res.chain) &&
          res.resseq >= h.first && res.resseq <= h.last)
        s.flags[i] |= F_HOTSPOT;
    for (const Hotspot& h : coldspots)
      if ((h.chain == 0 || h.chain == res.chain) &&
          res.resseq >= h.first && res.resseq <= h.last)
        s.flags[i] |= F_COLDSPOT;
    // the first chain keeps its own numbering; each later one is shifted
    // to start 49 past where the previous one ended
    s.residue_index[i] = (prev_last == 0)
        ? res.resseq
        : res.resseq - chain_first + prev_last + RECEPTOR_CHAIN_BREAK_GAP;
    s.asym_id[i] = tgt_asym;
    // ONE ENTITY PAST THE BINDER'S.  Copies are all entity 0, so the target
    // is 1; a Fab's L and H are entities 0 and 1, so the target is 2.
    // bc2's residue_entity_ids numbers the SORTED chain names, and
    // 'binder_0' < 'binder_1' < 'target_...', so the target is always last.
    s.entity_id[i] = scaf_chains > 1 ? scaf_chains : 1;
    s.interface_asym[i] = 1;
    s.seq_mask[i] = 1.f;
    if (res.aatype < NAA) s.sequence[(size_t)i * NAA + res.aatype] = 1.f;
    // TARGET FLEXIBILITY: a fraction of the template's residues give up
    // their SIDE CHAINS, so the model may repack them instead of being
    // held to the crystal rotamer.  The selection is bc2's and is
    // deterministic rather than random -- floor((idx+1)*f) > floor(idx*f),
    // i.e. every 1/f-th residue by index.  The backbone and CB stay.
    const bool flexible =
        flexibility > 0.f &&
        std::floor((res.resseq + 1) * (double)flexibility) >
            std::floor(res.resseq * (double)flexibility);
    for (int a = 0; a < ATOM; ++a) {
      // N, CA, C, CB, O are 0, 1, 2, 3, 4 in the 37-atom layout
      const bool backbone = (a <= 4);
      // NOT MASKED HERE ANY MORE.  Flexibility is a TEMPLATE feature --
      // bc2 applies it in target_template_features, selecting by the state's
      // residue index -- and the step and actConstFor apply it there
      // (plan.flexibility).  Masking the state's atoms selected by PDB number
      // instead, and fed every other reader of the target's atoms a
      // side-chain-less crystal structure bc2 never sees.
      (void)flexible; (void)backbone;
      s.atom_mask[(size_t)i * ATOM + a] = res.mask[a];
      for (int c = 0; c < 3; ++c)
        s.atoms[((size_t)i * ATOM + a) * 3 + c] = res.xyz[a][c];
    }
  }
  return s;
}

// THE BINDER ALONE: the binder chain and nothing else.
inline State buildBinderAlone(int binder_len, int nmax_floor = 0,
                              int copies = 1,
                              const std::vector<BinderRow>& scaffold = {},
                              int scaf_chains = 1) {
  State s;
  const int nb = binder_len * copies;
  s.resize(nb, std::max(padTo(nb), nmax_floor));
  // THE BINDER_ALONE STATE CARRIES THE SCAFFOLD TOO.  bc2's binder_alone
  // is {BINDER: binder} -- the same Protein object the complex holds -- so
  // a scaffold's framework is templated in the free fold exactly as it is
  // in the bound one.  Building it all-designed here would make the two
  // states disagree about what the binder IS, and every loss that reads
  // binder_alone would be reading a different molecule.
  for (int c = 0; c < copies; ++c)
    for (int k = 0; k < binder_len; ++k) {
      const int i = c * binder_len + k;
      if (!scaffold.empty()) {
        const BinderRow& b = scaffold[k];
        s.flags[i] = b.flags;
        s.residue_index[i] = b.residue_index;
        for (int a2 = 0; a2 < ATOM; ++a2) {
          s.atom_mask[(size_t)i * ATOM + a2] = b.mask[a2];
          for (int c2 = 0; c2 < 3; ++c2)
            s.atoms[((size_t)i * ATOM + a2) * 3 + c2] = b.xyz[a2][c2];
        }
        if (!(b.flags & F_DESIGN) && b.aatype < NAA)
          s.sequence[(size_t)i * NAA + b.aatype] = 1.f;
      } else {
        s.flags[i] = F_DESIGN;
        s.residue_index[i] = k + 1;
      }
      const int ch = scaffold.empty() ? 0 : scaffold[k].chain;
      s.asym_id[i] = c * scaf_chains + ch;
      s.entity_id[i] = scaf_chains > 1 ? ch : 0;
      s.interface_asym[i] = 0;
      s.seq_mask[i] = 1.f;
    }
  return s;
}

// The designed logits live on the BINDER rows of whichever state, and the
// same parameters drive both -- that is what ties the two folds together.
// ONE SEQUENCE INTO EVERY COPY.  The protomers are tied, so the logits
// array is one protomer long and is written into each copy's rows.
inline void scatterBinderLogits(State& s, const float* logits, int binder_len,
                                int copies = 1) {
  for (int c = 0; c < copies; ++c)
    for (int i = 0; i < binder_len; ++i) {
      // DESIGNED ROWS ONLY.  bc2 writes the logits under
      // `where(has_residue_flag(flags, DESIGN), ..., protein.sequence)` --
      // a scaffold's framework keeps its own identity, and a port that
      // scattered the logits across the whole chain would redesign the
      // framework it was given to preserve.
      if (!(s.flags[(size_t)c * binder_len + i] & F_DESIGN)) continue;
      for (int a = 0; a < NAA; ++a)
        s.sequence[((size_t)c * binder_len + i) * NAA + a] =
            logits[(size_t)i * NAA + a];
    }
}

// A HOTSPOT IS A CHAIN AND A SPAN, not a number.
//
// bc2 writes them "A67" or "B229-231", qualifies an unprefixed one with
// the FIRST receptor chain (chain_qualified_residue_span), and matches
// each against THAT CHAIN'S OWN residue numbering.
//
// This used to be `atoi(token)`, which reads "A67" as 0 and "B229-231" as
// 0 -- so every chain-qualified hotspot was silently dropped and every
// range collapsed.  The hIL2R target carries exactly those and the binary
// said so ("10 hotspots named but NONE matched"); interface_contacts then
// takes its SURFACE branch instead of its hotspot branch, and the campaign
// designs against the whole face rather than the epitope.
//
// Matched against the ORIGINAL PDB numbering, before the merge renumbers
// the later chains.

inline std::vector<Hotspot> parseHotspots(const std::string& spec,
                                          char default_chain = 0) {
  std::vector<Hotspot> v;
  std::string t;
  std::stringstream ss(spec);
  while (std::getline(ss, t, ',')) {
    // trim
    size_t a = t.find_first_not_of(" \t");
    size_t b = t.find_last_not_of(" \t");
    if (a == std::string::npos) continue;
    t = t.substr(a, b - a + 1);
    Hotspot h{default_chain, 0, 0};
    size_t i = 0;
    if (std::isalpha((unsigned char)t[0])) { h.chain = t[0]; i = 1; }
    const size_t dash = t.find('-', i);
    h.first = std::atoi(t.substr(i, dash == std::string::npos
                                        ? std::string::npos : dash - i).c_str());
    h.last = (dash == std::string::npos) ? h.first
                                         : std::atoi(t.substr(dash + 1).c_str());
    if (h.last < h.first) std::swap(h.first, h.last);
    v.push_back(h);
  }
  return v;
}

}  // namespace dsin
