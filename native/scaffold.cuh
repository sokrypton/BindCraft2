// mutate_positions -- bc2's scaffold edit spec, parsed.
//
// A scaffold campaign names which residues of a starting structure may be
// redesigned, and with what role.  bc2's _EDIT_RE:
//
//   (chain)(start)(-end)?(\(lengths\))?(flags)*
//
//   A27-33        a span, DEFAULT FLAGS = DESIGN | CONTACT  (paratope)
//   A35*          `*` CLEARS CONTACT  -> DESIGN only, a FRAMEWORK position
//   A35!          `!` CLEARS DESIGN   -> held unredesigned
//   A52-57(5-7)   the span is REPLACED by 5 to 7 residues, sampled --
//                 so the binder's LENGTH varies per trajectory
//   A35+COLDSPOT  set a named flag
//
// The two clearing forms are the ones to get right: `*` and `!` do not add
// anything, they remove from the default, and a parser that treated them as
// "set a flag" would make every framework position a paratope -- which is
// the difference between a VHH that keeps its framework and one that
// redesigns it.
//
// Residues NOT named by any span are not designed at all: they keep the
// scaffold's own identity and coordinates.
#pragma once
#include <string>
#include <vector>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "pdb_in.cuh"

namespace scaf {

// bc2's ResidueFlags, the bits the port already uses in design_inputs.cuh
enum Flag : uint8_t {
  F_DESIGN = 1u << 0, F_TEMPLATE = 1u << 1, F_SEQUENCE = 1u << 2,
  F_CONTACT = 1u << 3, F_HOTSPOT = 1u << 4, F_COLDSPOT = 1u << 5,
};

struct Span {
  char chain = 0;
  int first = 0, last = 0;          // inclusive, in the SCAFFOLD's numbering
  std::vector<int> lengths;         // empty = keep the span's own length
  uint8_t flags = F_DESIGN | F_CONTACT;
};

inline uint8_t namedFlag(const std::string& n) {
  if (n == "DESIGN") return F_DESIGN;
  if (n == "TEMPLATE") return F_TEMPLATE;
  if (n == "SEQUENCE") return F_SEQUENCE;
  if (n == "CONTACT") return F_CONTACT;
  if (n == "HOTSPOT") return F_HOTSPOT;
  if (n == "COLDSPOT") return F_COLDSPOT;
  return 0;                          // caller reports
}

// "5-7" or "3-25" or "6" -> every length in the range, as bc2 expands it
inline std::vector<int> parseLengths(const std::string& spec) {
  std::vector<int> out;
  size_t i = 0;
  while (i < spec.size()) {
    size_t j = spec.find(',', i);
    if (j == std::string::npos) j = spec.size();
    std::string tok = spec.substr(i, j - i);
    size_t a = tok.find_first_not_of(" \t");
    if (a != std::string::npos) {
      tok = tok.substr(a, tok.find_last_not_of(" \t") - a + 1);
      const size_t dash = tok.find('-');
      if (dash == std::string::npos) {
        out.push_back(atoi(tok.c_str()));
      } else {
        int lo = atoi(tok.substr(0, dash).c_str());
        int hi = atoi(tok.substr(dash + 1).c_str());
        if (hi < lo) { const int t = lo; lo = hi; hi = t; }
        for (int v = lo; v <= hi; ++v) out.push_back(v);
      }
    }
    i = j + 1;
  }
  return out;
}

// returns false and fills `bad` when text sits outside any span -- bc2
// RAISES on that, and a spec silently half-read is a scaffold half-designed
inline bool parseSpans(const std::string& spec, std::vector<Span>* out,
                       std::string* bad) {
  out->clear();
  size_t i = 0, consumed = 0;
  std::string unmatched;
  while (i < spec.size()) {
    if (!isalpha((unsigned char)spec[i])) { ++i; continue; }
    const size_t start = i;
    std::string chain;
    while (i < spec.size() && isalpha((unsigned char)spec[i])) chain += spec[i++];
    if (i >= spec.size() || !isdigit((unsigned char)spec[i])) {
      i = start + 1;                 // not a span after all
      continue;
    }
    // anything skipped between the last span and this one is unmatched
    { std::string gap = spec.substr(consumed, start - consumed);
      for (char c : gap) if (c != ',' && !isspace((unsigned char)c)) unmatched += c; }
    Span s;
    s.chain = chain.empty() ? 0 : chain[chain.size() - 1];
    std::string num;
    while (i < spec.size() && isdigit((unsigned char)spec[i])) num += spec[i++];
    s.first = s.last = atoi(num.c_str());
    if (i < spec.size() && spec[i] == '-' && i + 1 < spec.size() &&
        isdigit((unsigned char)spec[i + 1])) {
      ++i; std::string n2;
      while (i < spec.size() && isdigit((unsigned char)spec[i])) n2 += spec[i++];
      s.last = atoi(n2.c_str());
    }
    if (s.last < s.first) { const int t = s.first; s.first = s.last; s.last = t; }
    if (i < spec.size() && spec[i] == '(') {
      const size_t close = spec.find(')', i);
      if (close == std::string::npos) { *bad = "unclosed ( in " + spec; return false; }
      s.lengths = parseLengths(spec.substr(i + 1, close - i - 1));
      i = close + 1;
    }
    // THE FLAG DELTAS.  `*` and `!` CLEAR; +NAME sets, -NAME clears.
    while (i < spec.size()) {
      if (spec[i] == '*') { s.flags &= (uint8_t)~F_CONTACT; ++i; }
      else if (spec[i] == '!') { s.flags &= (uint8_t)~F_DESIGN; ++i; }
      else if (spec[i] == '+' || spec[i] == '-') {
        const bool set = spec[i] == '+';
        size_t j = i + 1;
        std::string name;
        while (j < spec.size() && isalpha((unsigned char)spec[j])) name += spec[j++];
        if (name.empty()) break;
        const uint8_t f = namedFlag(name);
        if (!f) { *bad = "unknown residue flag " + name; return false; }
        s.flags = set ? (uint8_t)(s.flags | f) : (uint8_t)(s.flags & ~f);
        i = j;
      } else break;
    }
    consumed = i;
    out->push_back(s);
  }
  { std::string tail = spec.substr(consumed);
    for (char c : tail) if (c != ',' && !isspace((unsigned char)c)) unmatched += c; }
  if (!unmatched.empty()) { *bad = "text outside any span: " + unmatched; return false; }
  return true;
}

// ---- THE SCAFFOLD BINDER ----------------------------------------------
//
// bc2's Protein.with_scaffold, which is not "mark some residues designed".
// It REBUILDS the chain:
//
//   residues before a span   kept -- scaffold coordinates, scaffold
//                            identity, TEMPLATE|SEQUENCE -- with their
//                            residue_index SHIFTED by the net length
//                            change of every span before them
//   a span                   DELETED and replaced by `replacement_length`
//                            new residues with NO COORDINATES and an
//                            all-false atom mask, carrying the span's
//                            flags and a fresh 0.01*normal logit row
//   residues after           kept, shifted
//
// So a designed position on a scaffold is exactly a de novo position: a
// hole.  The scaffold supplies the FRAMEWORK, not a starting point for the
// loops.  A port that kept the scaffold's coordinates on designed rows
// would hand the model a template for residues it is meant to invent --
// which folds, and folds to the scaffold.
//
// The residue numbering is the scaffold's own, shifted -- NOT 1..n.  Only
// differences reach the relative encoding, so the shift is what matters
// and a renumbering to 1..n would move every pair that crosses a
// length-changed span.
struct ScafRes {
  int aatype = 20;
  float xyz[37][3] = {};
  float mask[37] = {};
  uint8_t flags = 0;
  int residue_index = 0;
  bool designed = false;
  int chain = 0;        // which BINDER chain, 0-based; see buildScaffoldBinder
};

// `lengths_chosen` is one sampled length per span, in span order; pass the
// span's own length to keep it.  Explicit rather than sampled inside so a
// gate can ask the same question of bc2 with the same answer.
inline std::vector<ScafRes> buildScaffoldChain(
    const pdbin::Structure& scaffold, char chain,
    const std::vector<Span>& spans_in,
    const std::vector<int>& lengths_chosen) {
  std::vector<Span> spans;
  for (const Span& sp : spans_in) if (sp.chain == chain) spans.push_back(sp);
  std::vector<size_t> order(spans.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return spans[a].first < spans[b].first; });

  std::vector<const pdbin::Residue*> rows;
  for (const auto& r : scaffold.res) if (r.chain == chain) rows.push_back(&r);

  std::vector<ScafRes> out;
  size_t pos = 0;
  int shift = 0;
  for (size_t oi = 0; oi < order.size(); ++oi) {
    const Span& sp = spans[order[oi]];
    while (pos < rows.size() && rows[pos]->resseq < sp.first) {
      ScafRes r;
      r.aatype = rows[pos]->aatype;
      memcpy(r.xyz, rows[pos]->xyz, sizeof r.xyz);
      memcpy(r.mask, rows[pos]->mask, sizeof r.mask);
      r.flags = F_TEMPLATE | F_SEQUENCE;
      r.residue_index = rows[pos]->resseq + shift;
      out.push_back(r);
      ++pos;
    }
    const int start_shifted = sp.first + shift;
    // 1.0.4: A SPAN THAT DROPS DESIGN IS HELD, not rebuilt.  The scaffold's
    // own residues stay -- sequence, coordinates, mask -- with the span's
    // flags ORed onto theirs, and the span cannot change length (the
    // caller refuses a replacement length on one, as bc2 raises).  Before,
    // a `!` span was a hole of undesigned residues with no coordinates,
    // which is neither what the scaffold has nor what the edit asked for.
    if (!(sp.flags & F_DESIGN)) {
      while (pos < rows.size() && rows[pos]->resseq <= sp.last) {
        ScafRes r;
        r.aatype = rows[pos]->aatype;
        memcpy(r.xyz, rows[pos]->xyz, sizeof r.xyz);
        memcpy(r.mask, rows[pos]->mask, sizeof r.mask);
        r.flags = (uint8_t)(F_TEMPLATE | F_SEQUENCE | sp.flags);
        r.residue_index = rows[pos]->resseq + shift;
        out.push_back(r);
        ++pos;
      }
      continue;
    }
    while (pos < rows.size() && rows[pos]->resseq <= sp.last) ++pos;
    const int len = (oi < lengths_chosen.size() && lengths_chosen[order[oi]] > 0)
                        ? lengths_chosen[order[oi]]
                        : (sp.last - sp.first + 1);
    for (int k = 0; k < len; ++k) {
      ScafRes r;                       // no coordinates, no mask: a hole
      r.flags = sp.flags;
      r.residue_index = start_shifted + k;
      r.designed = (sp.flags & F_DESIGN) != 0;
      out.push_back(r);
    }
    shift += len - (sp.last - sp.first + 1);
  }
  while (pos < rows.size()) {
    ScafRes r;
    r.aatype = rows[pos]->aatype;
    memcpy(r.xyz, rows[pos]->xyz, sizeof r.xyz);
    memcpy(r.mask, rows[pos]->mask, sizeof r.mask);
    r.flags = F_TEMPLATE | F_SEQUENCE;
    r.residue_index = rows[pos]->resseq + shift;
    out.push_back(r);
    ++pos;
  }
  // 1.0.4's `coldspotted_framework_flags`: a chain that declares a
  // paratope (any CONTACT row) has said where the target belongs, which
  // says equally where it does not -- every other row of the chain is
  // COLDSPOT.  binder_coldspot repels exactly that set now, so the flags
  // are the one answer the loss, the campaign check and the output share.
  // A de novo chain declares no paratope and is left alone.
  bool declares = false;
  for (const ScafRes& r : out) declares = declares || (r.flags & F_CONTACT);
  if (declares)
    for (ScafRes& r : out)
      if (!(r.flags & F_CONTACT)) r.flags = (uint8_t)(r.flags | F_COLDSPOT);
  return out;
}

// THE WHOLE BINDER, every chain the scaffold holds, concatenated.
//
// A Fab or an scFv is ONE binder spanning two chains: bc2's
// resolve_binder_chains makes max(copies, scaffold chains) binder chains and
// prepare_binder_chains fills each from its own scaffold chain, so the
// binder is L followed by H and not a choice between them.
//
// `picks` is one sampled length per span IN SPEC ORDER -- the order
// `parseSpans` produced, which is the order a caller samples in.
// buildScaffoldChain takes them in the order of the spans it has FILTERED
// to one chain, so the re-indexing below is the whole reason this function
// exists rather than two calls at the call site: handing the global vector
// straight through is right for one chain and silently wrong for two, and
// what it produces is a plausible antibody wearing the light chain's loop
// lengths on the heavy chain.
inline std::vector<ScafRes> buildScaffoldBinder(
    const pdbin::Structure& scaffold, const std::vector<char>& chains,
    const std::vector<Span>& spans, const std::vector<int>& picks) {
  std::vector<ScafRes> out;
  for (size_t ci = 0; ci < chains.size(); ++ci) {
    std::vector<int> mine;
    for (size_t i = 0; i < spans.size(); ++i)
      if (spans[i].chain == chains[ci])
        mine.push_back(i < picks.size() ? picks[i] : 0);
    std::vector<ScafRes> rows =
        buildScaffoldChain(scaffold, chains[ci], spans, mine);
    for (ScafRes& r : rows) { r.chain = (int)ci; out.push_back(r); }
  }
  return out;
}

}  // namespace scaf
