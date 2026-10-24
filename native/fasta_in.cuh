// A TARGET READ FROM ITS SEQUENCE, and the crop that follows it.
//
// bc2's disordered-target modality points `target_path` at a fasta.
// `Protein.from_fasta` builds the chain with the SEQUENCE flag and
// nothing else -- no coordinates, no atom mask, no TEMPLATE -- so the
// model folds the target rather than being handed it, which is the whole
// point for an IDR.
//
// AND IT IS CROPPED.  `crop_fasta_sequence` defaults to (10, 40): every
// window of 10 to 40 residues is a candidate, a trajectory draws one it
// has not sampled, and the campaign designs against THAT crop.  A port
// that folded the whole sequence would be answering a different question
// -- and this port's first behaviour was worse than either, because
// `readPdb` keeps ATOM records and a fasta has none: it read zero
// residues and designed a binder against nothing.
#pragma once
#include "pdb_in.cuh"
#include "domains.cuh"
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fastain {

inline bool isFasta(const std::string& path) {
  auto ends = [&](const char* ext) {
    const size_t n = strlen(ext);
    return path.size() >= n && path.compare(path.size() - n, n, ext) == 0;
  };
  return ends(".fasta") || ends(".fa") || ends(".faa") || ends(".seq");
}

// `read_fasta_sequences` then `Protein.from_fasta`: the record whose
// header names the chain, or the only one.  The residues carry an
// aatype, a residue_index of 1..n and NO atoms.
inline pdbin::Structure readFasta(const std::string& path, char chain = 'A') {
  std::ifstream f(path);
  pdbin::Structure s;
  if (!f) {
    fprintf(stderr, "bindcraft: cannot open %s\n", path.c_str());
    return s;
  }
  std::vector<std::pair<std::string, std::string>> records;
  std::string line, name, seq;
  while (std::getline(f, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
      line.pop_back();
    if (!line.empty() && line[0] == '>') {
      if (!name.empty() || !seq.empty()) records.push_back({name, seq});
      name = line.substr(1);
      // the record's NAME is its first token, which is what a chain
      // letter is matched against
      const size_t sp = name.find_first_of(" \t");
      if (sp != std::string::npos) name = name.substr(0, sp);
      seq.clear();
      continue;
    }
    for (char c : line)
      if (!isspace((unsigned char)c)) seq += (char)toupper(c);
  }
  if (!name.empty() || !seq.empty()) records.push_back({name, seq});
  const std::string want(1, chain);
  std::string chosen;
  for (const auto& r : records)
    if (r.first == want) chosen = r.second;
  if (chosen.empty() && records.size() == 1) chosen = records[0].second;
  if (chosen.empty()) {
    fprintf(stderr, "bindcraft: %s has no record %c and more than one "
            "sequence\n", path.c_str(), chain);
    return s;
  }
  for (size_t i = 0; i < chosen.size(); ++i) {
    pdbin::Residue r{};
    r.chain = chain;
    r.resseq = (int)i + 1;
    // the one-letter alphabet AA1 is bc2's AMINO_ACIDS order
    r.aatype = (int)(strchr(pdbin::AA1, chosen[i]) ?
                     strchr(pdbin::AA1, chosen[i]) - pdbin::AA1 : 20);
    if (r.aatype < 0 || r.aatype >= 20) {
      fprintf(stderr, "bindcraft: %s has a non-standard amino acid '%c' at "
              "position %zu\n", path.c_str(), chosen[i], i + 1);
      return pdbin::Structure();
    }
    // unresolved_atom_arrays: no atom is present, so every mask is zero
    // and the coordinates are zero with it
    for (int a = 0; a < pdbin::ATOM37; ++a) { r.mask[a] = 0.f;
      for (int c = 0; c < 3; ++c) r.xyz[a][c] = 0.f; }
    s.res.push_back(r);
  }
  s.chains.clear();
  s.chains.push_back(chain);
  return s;
}

// every (start, length) `crop_fasta_sequence` allows, in bc2's order:
// lengths ascending, and starts ascending within a length
inline std::vector<std::pair<int, int>> cropWindows(int target_length,
                                                    int lo, int hi) {
  std::vector<int> lengths;
  for (int l = lo; l <= hi; ++l) {
    const int clipped = std::min(l, target_length);
    if (std::find(lengths.begin(), lengths.end(), clipped) == lengths.end())
      lengths.push_back(clipped);
  }
  std::sort(lengths.begin(), lengths.end());
  std::vector<std::pair<int, int>> windows;
  for (int l : lengths)
    for (int start = 0; start + l <= target_length; ++start)
      windows.push_back({start, l});
  return windows;
}

// `random_generator.choice(unsampled_epitopes)` with an INTEGER seed --
// the trajectory's.  `sampled_target_epitopes` is local to one
// prepare_targets call, so with a single crop every window is unsampled
// and the draw is over all of them.
inline std::pair<int, int> drawCrop(const std::vector<std::pair<int, int>>& windows,
                                    uint64_t trajectory_seed,
                                    const std::vector<std::pair<int, int>>& sampled =
                                        std::vector<std::pair<int, int>>()) {
  std::vector<std::pair<int, int>> unsampled;
  for (const auto& w : windows)
    if (std::find(sampled.begin(), sampled.end(), w) == sampled.end())
      unsampled.push_back(w);
  if (unsampled.empty()) return {0, 0};
  domains::MT rng;
  rng.seed(trajectory_seed);
  return unsampled[(size_t)rng.randbelow((int)unsampled.size())];
}

inline pdbin::Structure crop(const pdbin::Structure& full, int start, int length) {
  pdbin::Structure s;
  s.chains = full.chains;
  for (int i = start; i < start + length && i < (int)full.res.size(); ++i)
    s.res.push_back(full.res[(size_t)i]);
  return s;
}

}  // namespace fastain
