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
  // the residues `trimPartlyResolved` threw away, as bc2's
  // `partly_trimmed_residues` names them ("THR A/20") -- the campaign
  // prints them, because a target silently two residues shorter than the
  // file is the kind of input difference nothing downstream can report
  std::vector<std::string> dropped;
  int size() const { return (int)res.size(); }
};

// one float64 -> float16 -> float32 round trip, round-to-nearest-even and
// rounded ONCE, which is what numpy's astype(float16) and jnp.asarray(...,
// float16) do -- going through float32 first can flip a value that sits
// within a float ulp of a half midpoint
inline float half2f(double v) {
  return __half2float(__double2half(v));
}

inline int atomIndex(const std::string& name) {
  for (int i = 0; i < ATOM37; ++i) if (name == ATOM_TYPES[i]) return i;
  return -1;
}
// bc2's MODIFIED_RESIDUE_PARENTS: selenomethionine IS methionine, and it
// is common enough in crystal structures that reading it as "unknown"
// turns a MET into an X in the target sequence.  Its SE is the MET SD
// (MODIFIED_RESIDUE_ATOM_NAMES), which `modifiedAtom` renames.
inline int aaIndex(const std::string& three) {
  for (int i = 0; i < 20; ++i) if (three == AA3[i]) return i;
  if (three == "MSE") return 12;             // MET
  return 20;
}

inline std::string modifiedAtom(const std::string& resname,
                                const std::string& atom) {
  return (resname == "MSE" && atom == "SE") ? std::string("SD") : atom;
}

// `from_structure` keeps only the residues whose atoms cover the whole
// BACKBONE (N, CA, C) and DROPS the rest -- a partly resolved residue is
// not a residue with holes, it is gone, and every index downstream counts
// without it.  The port used to keep them, so a target with two trimmed
// residues was two residues longer here than in bc2, with the hole in the
// middle of the chain.
inline void trimPartlyResolved(Structure* s) {
  std::vector<Residue> kept;
  for (const Residue& r : s->res) {
    if (r.mask[0] > 0.f && r.mask[1] > 0.f && r.mask[2] > 0.f) {
      kept.push_back(r);
      continue;
    }
    // reported only for the residues bc2 reports: the standard twenty
    // (MSE already mapped onto MET above)
    if (r.aatype < 20)
      s->dropped.push_back(std::string(AA3[r.aatype]) + " " + r.chain + "/" +
                           std::to_string(r.resseq));
  }
  if (kept.size() == s->res.size()) return;
  s->res = kept;
  // a chain whose every residue was trimmed is no longer a chain
  std::vector<char> chains;
  for (char c : s->chains) {
    for (const Residue& r : s->res)
      if (r.chain == c) { chains.push_back(c); break; }
  }
  s->chains = chains;
}

// ONE MODEL ONLY, and the first.  A multi-model NMR file would otherwise
// stack every model into one chain, which presents as a target twice the
// length it should be rather than as an error.
// ONE PLACE A RESIDUE IS BUILT, shared by the PDB and the mmCIF readers.
//
// Factored out when the CIF reader arrived rather than transcribed into it:
// the float16 rounding below is not an optimisation, it is what bc2's
// Protein.atoms holds, and a second reader that forgot it would feed the
// model a DIFFERENT TEMPLATE -- correct-looking coordinates, 0.031 A out at
// 50 A, systematically.  pdb_gate.py catches that for the PDB path; a cif
// path with its own copy would have no such gate.
inline void addAtom(Structure* s, char chain, int resseq,
                    const std::string& resname, const std::string& atom,
                    double x, double y, double z) {
  const int ai = atomIndex(modifiedAtom(resname, atom));
  if (ai < 0) return;                       // hydrogens and the rest
  int ri = -1;
  for (int i = (int)s->res.size() - 1; i >= 0; --i)
    if (s->res[i].chain == chain && s->res[i].resseq == resseq) { ri = i; break; }
  if (ri < 0) {
    Residue r{};
    r.chain = chain; r.resseq = resseq; r.aatype = aaIndex(resname);
    ri = (int)s->res.size();
    s->res.push_back(r);
    if (std::find(s->chains.begin(), s->chains.end(), chain) == s->chains.end())
      s->chains.push_back(chain);
  }
  Residue& r = s->res[ri];
  r.xyz[ai][0] = half2f(x);
  r.xyz[ai][1] = half2f(y);
  r.xyz[ai][2] = half2f(z);
  r.mask[ai] = 1.f;
}

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
    const int ai = atomIndex(modifiedAtom(line.substr(17, 3), name));
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
  trimPartlyResolved(&s);
  return s;
}

inline char oneLetter(int aatype) { return AA1[aatype < 20 ? aatype : 20]; }

}  // namespace pdbin
