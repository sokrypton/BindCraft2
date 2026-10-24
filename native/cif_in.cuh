// A minimal mmCIF atom_site reader -- enough for bc2's scaffolds/*.cif.
//
// COLUMNS ARE RESOLVED BY NAME from the loop_ header, never by position.
// The order of _atom_site.* is a file's choice, not a standard, and a
// reader keyed on position reads coordinates out of the occupancy column
// the first time a file is written by a different program.
//
// AND THE CHAIN ID IS auth_asym_id, NOT label_asym_id.  In bc2's own
// scaffolds these DISAGREE -- VHH.cif is auth 'A' and label 'B' -- and
// `mutate_positions` names 'A'.  Read the label column and every span
// matches nothing: a scaffold with no designable positions, no error.
// Checked against what biotite hands bc2, which is the auth id.
#pragma once
#include "pdb_in.cuh"
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace cifin {

// splits an mmCIF value row, honouring the quoting the format allows
inline std::vector<std::string> splitRow(const std::string& line) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && isspace((unsigned char)line[i])) ++i;
    if (i >= line.size()) break;
    std::string tok;
    if (line[i] == '"' || line[i] == '\'') {
      const char q = line[i++];
      while (i < line.size() && line[i] != q) tok += line[i++];
      if (i < line.size()) ++i;
    } else {
      while (i < line.size() && !isspace((unsigned char)line[i])) tok += line[i++];
    }
    out.push_back(tok);
  }
  return out;
}

// Returns a pdbin::Structure so everything downstream -- the state
// builder, the filters, kabsch -- takes a scaffold exactly as it takes a
// target, with no second code path.
inline pdbin::Structure readCif(const std::string& path, bool* ok) {
  pdbin::Structure s;
  *ok = false;
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "cif: cannot open %s\n", path.c_str()); return s; }
  std::string line;
  std::map<std::string, int> col;
  bool in_loop = false, in_atom_site = false;
  int ncol = 0;
  std::vector<std::string> pending;
  while (std::getline(f, line)) {
    if (line.rfind("loop_", 0) == 0) {
      in_loop = true; in_atom_site = false; col.clear(); ncol = 0;
      continue;
    }
    if (in_loop && line.rfind("_atom_site.", 0) == 0) {
      std::string name = line.substr(11);
      while (!name.empty() && isspace((unsigned char)name.back())) name.pop_back();
      col[name] = ncol++;
      in_atom_site = true;
      continue;
    }
    if (in_loop && !line.empty() && line[0] == '_') { in_atom_site = false; continue; }
    if (!in_atom_site) continue;
    if (line.empty() || line[0] == '#') { in_atom_site = false; in_loop = false; continue; }
    const std::vector<std::string> v = splitRow(line);
    if ((int)v.size() < ncol) continue;
    auto at = [&](const char* k) -> const std::string& {
      static const std::string empty;
      auto it = col.find(k);
      return it == col.end() ? empty : v[it->second];
    };
    if (at("group_PDB") != "ATOM" && at("group_PDB") != "HETATM") continue;
    const std::string& atom = at("label_atom_id");
    const std::string& resn = at("label_comp_id");
    // auth, not label -- see the note at the top
    const std::string& ch = at("auth_asym_id").empty() ? at("label_asym_id")
                                                       : at("auth_asym_id");
    const std::string& seq = at("auth_seq_id").empty() ? at("label_seq_id")
                                                       : at("auth_seq_id");
    pdbin::addAtom(&s, ch.empty() ? 'A' : ch[0], atoi(seq.c_str()), resn,
                   atom, (float)atof(at("Cartn_x").c_str()),
                   (float)atof(at("Cartn_y").c_str()),
                   (float)atof(at("Cartn_z").c_str()));
  }
  // the same trim the PDB reader does: bc2's `from_structure` drops a
  // residue that does not carry N, CA and C, whichever format it came in
  pdbin::trimPartlyResolved(&s);
  *ok = !s.res.empty();
  if (!*ok) fprintf(stderr, "cif: %s has no atom_site rows this reader "
                    "understood\n", path.c_str());
  return s;
}

}  // namespace cifin
