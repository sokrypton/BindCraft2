// READING A TARGET STRUCTURE, in C++.
//
// design.cu takes pre-dumped Oracle tensors, which is right for a kernel
// gate and useless for a product: a `bindcraft --target x.pdb` cannot ask
// the user to run a python script first.  This is the minimum that makes
// the CLI self-contained -- PDB ATOM records into the atom37 layout the
// design step wants.
//
// FIXED COLUMNS, NOT WHITESPACE.  The PDB spec puts the fields at fixed
// offsets and real files have names that run into their neighbours
// (" CA " vs "CA", four-character atom names like "HD11"), so splitting on
// spaces works until it meets a real structure.  Columns per the spec:
//   13-16 name, 18-20 resName, 22 chainID, 23-26 resSeq, 31-38/39-46/47-54 xyz
#pragma once
#include <cuda_fp16.h>
#include "common.cuh"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

namespace pdbin {

static constexpr int ATOM37 = 37;

// The atom37 order AlphaFold uses (residue_constants.atom_types).
static const char* ATOM_TYPES[ATOM37] = {
  "N","CA","C","CB","O","CG","CG1","CG2","OG","OG1","SG","CD","CD1","CD2",
  "ND1","ND2","OD1","OD2","SD","CE","CE1","CE2","CE3","NE","NE1","NE2","OE1",
  "OE2","CH2","NH1","NH2","OH","CZ","CZ2","CZ3","NZ","OXT"};

static const char* AA3[20] = {
  "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
  "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"};
// AlphaFold's restype order is ARNDCQEGHILKMFPSTWYV
static const char AA1[22] = "ARNDCQEGHILKMFPSTWYVX";

struct Residue {
  char chain;
  int resseq;
  int aatype;                  // 0..19, 20 = unknown
  float xyz[ATOM37][3];
  float mask[ATOM37];
};

struct Structure {
  std::vector<Residue> res;
  std::vector<char> chains;    // in order of first appearance
  int size() const { return (int)res.size(); }
};

// one float32 -> float16 -> float32 round trip, round-to-nearest-even,
// which is what numpy's astype(float16) does
inline float half2f(double v) {
  return __half2float(__float2half((float)v));
}

inline int atomIndex(const std::string& name) {
  for (int i = 0; i < ATOM37; ++i) if (name == ATOM_TYPES[i]) return i;
  return -1;
}
inline int aaIndex(const std::string& three) {
  for (int i = 0; i < 20; ++i) if (three == AA3[i]) return i;
  return 20;
}

// ONE MODEL ONLY, and the first.  A multi-model NMR file would otherwise
// stack every model into one chain, which presents as a target twice the
// length it should be rather than as an error.
inline Structure readPdb(const std::string& path) {
  Structure s;
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "bindcraft: cannot open %s\n", path.c_str()); exit(1); }
  std::string line;
  std::unordered_map<long long, int> seen;    // (chain, resseq) -> index
  while (std::getline(f, line)) {
    if (line.rfind("ENDMDL", 0) == 0) break;
    if (line.rfind("ATOM  ", 0) != 0) continue;
    if (line.size() < 54) continue;
    const char alt = line[16];
    if (alt != ' ' && alt != 'A') continue;   // one altloc
    std::string name = line.substr(12, 4);
    while (!name.empty() && name.front() == ' ') name.erase(name.begin());
    while (!name.empty() && name.back() == ' ') name.pop_back();
    const int ai = atomIndex(name);
    if (ai < 0) continue;                     // hydrogens and the rest
    const char chain = line[21];
    const int resseq = std::atoi(line.substr(22, 4).c_str());
    const long long key = ((long long)(unsigned char)chain << 32) | (unsigned int)resseq;
    auto it = seen.find(key);
    int ri;
    if (it == seen.end()) {
      Residue r{};
      r.chain = chain; r.resseq = resseq;
      r.aatype = aaIndex(line.substr(17, 3));
      ri = (int)s.res.size();
      s.res.push_back(r);
      seen[key] = ri;
      if (std::find(s.chains.begin(), s.chains.end(), chain) == s.chains.end())
        s.chains.push_back(chain);
    } else ri = it->second;
    Residue& r = s.res[ri];
    // ROUNDED TO float16, because that is what bc2 holds.
    //
    // Protein.atoms is float16 (and so is Protein.sequence), so the
    // template coordinates the model actually sees in bc2 have already
    // been through a half-precision round trip before anything reads
    // them.  Keeping full f32 here is "better" and it is a DIFFERENT
    // INPUT: up to 0.031 A at a 50 A coordinate, systematically, on the
    // template that drives act_const and the whole fold.  Same principle
    // as the bf16 narrowing the predict path keeps -- a port that removes
    // the reference's approximation answers a different question.
    //
    // pdb_gate.py reads this as an exact match against bc2's own reader;
    // without the rounding it was the only thing that differed, on every
    // structure.
    r.xyz[ai][0] = half2f(std::atof(line.substr(30, 8).c_str()));
    r.xyz[ai][1] = half2f(std::atof(line.substr(38, 8).c_str()));
    r.xyz[ai][2] = half2f(std::atof(line.substr(46, 8).c_str()));
    r.mask[ai] = 1.f;
  }
  return s;
}

inline char oneLetter(int aatype) { return AA1[aatype < 20 ? aatype : 20]; }

}  // namespace pdbin
