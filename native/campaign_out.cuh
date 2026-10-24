// campaign_out.cuh -- bc2's CAMPAIGN TABLES, the schema and not a summary
// of it.
//
// `campaign_output.py` is 607 lines and almost none of it is arithmetic:
// it is a FILE FORMAT.  Three stage tables, a column order, a rounding
// rule, a summary, a metadata record and a progress file -- and every
// tool downstream of a campaign (`bindcraft rank`, `bindcraft filter`,
// `bindcraft campaign_output`, the parameter sweep, the desperation
// ladder) reads a campaign by reading those files.  A port that writes a
// table of its own with thirteen columns runs the design loop and leaves
// the campaign unreadable by everything bc2 ships to read it.
//
// So the test this file is written against is not "do the numbers agree".
// It is: **run bc2's own rank.py and campaign_filter.py on the port's
// output folder and have them work**, which is what
// `campaign_output_gate.py` does.  That is a stronger check than any
// value comparison, because those two modules read columns this file has
// never heard of and fail on a schema that is merely close.
//
// WHAT IS DELIBERATELY NOT HERE.  `trajectory_output.py`'s plots,
// animations and per-frame structures (545 lines of matplotlib and a
// packed-frame HTML player) are output for a human to look at, not a
// schema anything reads back -- except `losses.csv`, which
// `summarize_campaign` DOES read back, and which is here.
//
// THE ONE DIVERGENCE, STATED UP FRONT.  `design_hash` hashes the settings
// dict, the values DRAWN for the trajectory, and a fingerprint of the
// target conformation.  The port's draws are not bc2's draws (its RNG is
// not threefry and says so everywhere else), so the hash cannot be bc2's
// hash whatever basis it uses.  It is computed here over the port's own
// canonical settings text and serves the two jobs the column exists for
// -- de-duplicating a recipe within a folder, and naming a design
// reproducibly -- and it is NOT comparable across the two
// implementations.  See [[reproduce-a-draw-or-match-it]]: this is a draw
// to match in distribution, not one to reproduce.
#pragma once
#include "exact_stats.cuh"
#include "zipfile.cuh"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

namespace campout {

// ---- SHA-256, for the identity hash and the checkpoint digests ----------
struct Sha256 {
  uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                   0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
  uint8_t buf[64]; size_t len = 0; uint64_t total = 0;
  static uint32_t ror(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const uint8_t* p) {
    static const uint32_t k[64] = {
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
             ((uint32_t)p[i*4+2] << 8) | (uint32_t)p[i*4+3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = ror(w[i-15],7) ^ ror(w[i-15],18) ^ (w[i-15] >> 3);
      const uint32_t s1 = ror(w[i-2],17) ^ ror(w[i-2],19) ^ (w[i-2] >> 10);
      w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t S1 = ror(e,6) ^ ror(e,11) ^ ror(e,25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
      const uint32_t S0 = ror(a,2) ^ ror(a,13) ^ ror(a,22);
      const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = S0 + mj;
      hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
  }
  void update(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data; total += n;
    while (n) {
      const size_t take = std::min(n, (size_t)64 - len);
      memcpy(buf + len, p, take); len += take; p += take; n -= take;
      if (len == 64) { block(buf); len = 0; }
    }
  }
  std::string hex() {
    uint64_t bits = total * 8;
    uint8_t pad = 0x80; update(&pad, 1);
    uint8_t z = 0; while (len != 56) update(&z, 1);
    uint8_t be[8]; for (int i = 0; i < 8; ++i) be[i] = (uint8_t)(bits >> (56 - 8*i));
    update(be, 8);
    char out[65];
    for (int i = 0; i < 8; ++i) snprintf(out + i*8, 9, "%08x", h[i]);
    return std::string(out, 64);
  }
};

inline std::string sha256Hex(const std::string& s) {
  Sha256 d; d.update(s.data(), s.size()); return d.hex();
}

inline std::string sha256File(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return "";
  Sha256 d; std::vector<char> b(1 << 20); size_t n;
  while ((n = fread(b.data(), 1, b.size(), f)) > 0) d.update(b.data(), n);
  fclose(f);
  return d.hex();
}

// ---- recorded_number ----------------------------------------------------
//
// `round(number, 2 if abs(number) <= 1 else 1)`.  Every float a campaign
// records goes through it, so a table is readable and a diff of two runs
// is not a wall of float noise.  The CSV then holds `str(float)`, which
// for a value rounded to one or two places is the shortest decimal that
// round-trips -- so "0.5" and not "0.50", "13.0" and not "13".
inline double recordedNumber(double v) {
  if (!std::isfinite(v)) return v;
  // printf rounds the binary value half-to-even exactly as Python's round
  // does; going through a string is what keeps the two agreeing.
  char tmp[64];
  snprintf(tmp, sizeof tmp, std::fabs(v) <= 1.0 ? "%.2f" : "%.1f", v);
  return atof(tmp);
}

inline std::string recordedText(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  char tmp[64];
  snprintf(tmp, sizeof tmp, std::fabs(v) <= 1.0 ? "%.2f" : "%.1f", v);
  std::string s(tmp);
  // strip trailing zeros but keep one decimal: str(0.5) is "0.5",
  // str(1.0) is "1.0".  A float column that loses its ".0" reads back as
  // an int in pandas and as a different dtype in half the tools.
  const size_t dot = s.find('.');
  if (dot != std::string::npos) {
    size_t last = s.size();
    while (last > dot + 2 && s[last-1] == '0') --last;
    s = s.substr(0, last);
  }
  if (s == "-0.0") s = "-0.0";
  return s;
}

// ---- a row: names in insertion order, values as written -----------------
struct Row {
  std::vector<std::pair<std::string, std::string>> cells;
  void put(const std::string& name, const std::string& value) {
    for (auto& c : cells) if (c.first == name) { c.second = value; return; }
    cells.push_back({name, value});
  }
  void putNum(const std::string& name, double value) { put(name, recordedText(value)); }
  void putInt(const std::string& name, long value) {
    put(name, std::to_string(value));
  }
  bool has(const std::string& name) const {
    for (const auto& c : cells) if (c.first == name) return true;
    return false;
  }
  std::string get(const std::string& name) const {
    for (const auto& c : cells) if (c.first == name) return c.second;
    return "";
  }
  void erase(const std::string& name) {
    for (size_t i = 0; i < cells.size(); ++i)
      if (cells[i].first == name) { cells.erase(cells.begin() + i); return; }
  }
};

// ---- ONE COLUMN PER METRIC, WHATEVER THE TARGET COUNT -------------------
//
// bc2 does not widen the table when a campaign names several targets: it
// COLLAPSES `<metric>.<state>` into one `<metric>` cell holding the states'
// readings joined by ';', in weight-descending order, and names them in a
// `targets` column beside a `target_weights` one.  rank.py then derives
// `_mean`, `_worst`, `_best`, `_spread` and `_selectivity` from that single
// cell.  A port that wrote `i_pTM.hPDL1` and `i_pTM.mPDL1` as two columns
// would be unreadable to every tool bc2 ships.
//
// Inert below two targets, which is where target_ordered_row returns the
// row untouched -- so a de novo campaign's table is byte for byte what it
// was.
inline Row targetOrderedRow(const Row& row,
                            const std::vector<std::pair<std::string, double>>& targets) {
  if (targets.size() < 2) return row;
  std::vector<std::string> names;
  for (const auto& t : targets) names.push_back(t.first);
  Row collapsed;
  // base -> state -> text, in first-seen base order
  std::vector<std::string> bases;
  std::map<std::string, std::map<std::string, std::string>> per_target;
  for (const auto& c : row.cells) {
    const size_t dot = c.first.rfind('.');
    const std::string base = dot == std::string::npos ? std::string()
                                                      : c.first.substr(0, dot);
    const std::string state = dot == std::string::npos ? std::string()
                                                       : c.first.substr(dot + 1);
    if (!base.empty() && std::find(names.begin(), names.end(), state) != names.end()) {
      if (per_target.find(base) == per_target.end()) bases.push_back(base);
      per_target[base][state] = c.second;
    } else {
      collapsed.put(c.first, c.second);
    }
  }
  for (const std::string& base : bases) {
    std::map<std::string, std::string>& readings = per_target[base];
    // a metric written WITHOUT a state is the first target's reading
    // and it LEAVES ITS PLACE: bc2 pops it and re-inserts the collapsed
    // cell, so an undeclared column lands at the end of the row
    if (collapsed.has(base)) {
      if (readings.find(names[0]) == readings.end())
        readings[names[0]] = collapsed.get(base);
      collapsed.erase(base);
    }
    std::string joined;
    for (size_t i = 0; i < names.size(); ++i) {
      if (i) joined += ";";
      const auto it = readings.find(names[i]);
      if (it != readings.end()) joined += it->second;
    }
    collapsed.put(base, joined);
  }
  std::string tn, tw;
  for (size_t i = 0; i < targets.size(); ++i) {
    if (i) { tn += ";"; tw += ";"; }
    tn += targets[i].first;
    char buf[32]; snprintf(buf, sizeof buf, "%g", targets[i].second);
    tw += buf;
  }
  collapsed.put("targets", tn);
  collapsed.put("target_weights", tw);
  return collapsed;
}

// ---- the column order ---------------------------------------------------
//
// Six bands.  Identity first, then the sequence, then the confidence
// readings a human looks at, then everything else, then the metrics
// `ALWAYS_SUPPRESSED_METRICS` pushes to the back, then the metadata.
// Within a band the DECLARED order wins and anything undeclared keeps the
// order it was first written in -- which is why this takes the running
// column list and not a set.
static const char* DESIGN_IDENTITY_COLUMNS[] = {
  "rank", "trajectory", "design", "length", "outcome"};
static const char* LEADING_CONFIDENCE_COLUMNS[] = {
  "i_pDAE", "i_pTM", "pLDDT", "pTM", "i_pAE", "Unbound_Binder_pLDDT", "Target_pLDDT"};
static const char* SEQUENCE_COLUMNS[] = {
  "Binder_Sequence", "Interface_Binder_Residues", "Interface_Target_Residues"};
static const char* TRAILING_METADATA_COLUMNS[] = {
  "Timing", "failed_filters", "hash", "phase", "round", "terminated", "autotuned",
  "targets", "target_weights", "settings_core", "settings_modality",
  "settings_property", "settings_target", "settings_overrides"};
// campaign_log.ALWAYS_SUPPRESSED_METRICS -- transcribed, and the gate
// compares the tuple itself so a bc2 that adds one is caught.
static const char* ALWAYS_SUPPRESSED_METRICS[] = {
  "Unbound_Binder_pLDDT", "Protomer_Identity_Fraction", "Oligomer_Symmetry_RMSD",
  "Backbone_Clashes", "All_Atom_Clashes", "Epitope_Residues_Contacted",
  "Binder_Length", "Binder_Mass_kDa", "Binder_pI", "Binder_Net_Charge",
  "Binder_Extinction", "Binder_Cysteines", "Binder_Free_Cysteines",
  "Binder_Disulfides", "Interface_BuriedArea", "Interface_BuriedArea_Fraction",
  "Surface_Hydrophobicity", "Interface_Hydrophobicity", "SS_pLDDT", "Target_RMSD",
  "Binder_Helix_Fraction", "Binder_BetaSheet_Fraction", "Binder_Loop_Fraction",
  "Interface_Helix_Fraction", "Interface_BetaSheet_Fraction", "Interface_Loop_Fraction"};

template <size_t N>
inline bool inList(const char* const (&list)[N], const std::string& name) {
  for (size_t i = 0; i < N; ++i) if (name == list[i]) return true;
  return false;
}

inline std::string columnBase(const std::string& column) {
  const size_t dot = column.find('.');
  return dot == std::string::npos ? column : column.substr(0, dot);
}

inline int columnReadingOrder(const std::string& column) {
  const std::string base = columnBase(column);
  if (inList(DESIGN_IDENTITY_COLUMNS, column)) return 0;
  if (inList(TRAILING_METADATA_COLUMNS, column)) return 5;
  if (inList(SEQUENCE_COLUMNS, base)) return 1;
  if (inList(LEADING_CONFIDENCE_COLUMNS, base) ||
      inList(LEADING_CONFIDENCE_COLUMNS, column)) return 2;
  return inList(ALWAYS_SUPPRESSED_METRICS, base) ? 4 : 3;
}

inline int declaredColumnPosition(const std::string& column, int band) {
  const char* const* declared = nullptr; size_t n = 0;
  if (band == 0) { declared = DESIGN_IDENTITY_COLUMNS; n = sizeof(DESIGN_IDENTITY_COLUMNS)/sizeof(*DESIGN_IDENTITY_COLUMNS); }
  else if (band == 1) { declared = SEQUENCE_COLUMNS; n = sizeof(SEQUENCE_COLUMNS)/sizeof(*SEQUENCE_COLUMNS); }
  else if (band == 2) { declared = LEADING_CONFIDENCE_COLUMNS; n = sizeof(LEADING_CONFIDENCE_COLUMNS)/sizeof(*LEADING_CONFIDENCE_COLUMNS); }
  else if (band == 5) { declared = TRAILING_METADATA_COLUMNS; n = sizeof(TRAILING_METADATA_COLUMNS)/sizeof(*TRAILING_METADATA_COLUMNS); }
  if (!declared) return 0;
  std::string name = column; bool found = false;
  for (size_t i = 0; i < n; ++i) if (column == declared[i]) { found = true; break; }
  if (!found) name = columnBase(column);
  for (size_t i = 0; i < n; ++i) if (name == declared[i]) return (int)i;
  return (int)n;
}

inline std::vector<std::string> orderedCsvColumns(const std::vector<std::string>& names) {
  std::vector<std::string> ordered;            // dict.fromkeys: de-duped, order kept
  for (const std::string& name : names)
    if (std::find(ordered.begin(), ordered.end(), name) == ordered.end())
      ordered.push_back(name);
  std::vector<std::string> out;
  for (int band = 0; band < 6; ++band) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < ordered.size(); ++i)
      if (columnReadingOrder(ordered[i]) == band) idx.push_back(i);
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
      const int pa = declaredColumnPosition(ordered[a], band);
      const int pb = declaredColumnPosition(ordered[b], band);
      return pa != pb ? pa < pb : a < b;
    });
    for (size_t i : idx) out.push_back(ordered[i]);
  }
  return out;
}

// ---- CSV, as Python's csv module writes and reads it --------------------
inline std::string csvField(const std::string& v) {
  if (v.find_first_of(",\"\r\n") == std::string::npos) return v;
  std::string out = "\"";
  for (char c : v) { if (c == '"') out += '"'; out += c; }
  return out + "\"";
}

inline std::vector<std::string> csvSplit(const std::string& line) {
  std::vector<std::string> out; std::string cur; bool q = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (q) {
      if (c == '"') {
        if (i + 1 < line.size() && line[i+1] == '"') { cur += '"'; ++i; }
        else q = false;
      } else cur += c;
    } else if (c == '"') q = true;
    else if (c == ',') { out.push_back(cur); cur.clear(); }
    else cur += c;
  }
  out.push_back(cur);
  return out;
}

struct Table {
  std::vector<std::string> columns;
  std::vector<Row> rows;
};

inline Table parseMetricRows(std::istream& f) {
  Table t;
  std::string line;
  if (!std::getline(f, line)) return t;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  t.columns = csvSplit(line);
  while (std::getline(f, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    // a quoted field may hold a newline; join until the quotes balance
    while (std::count(line.begin(), line.end(), '"') % 2 == 1) {
      std::string more;
      if (!std::getline(f, more)) break;
      if (!more.empty() && more.back() == '\r') more.pop_back();
      line += "\n" + more;
    }
    const std::vector<std::string> cells = csvSplit(line);
    Row r;
    for (size_t i = 0; i < t.columns.size(); ++i)
      r.put(t.columns[i], i < cells.size() ? cells[i] : "");
    t.rows.push_back(r);
  }
  return t;
}

inline Table readMetricRows(const std::string& path) {
  std::ifstream f(path);
  if (!f) return Table();
  return parseMetricRows(f);
}

inline void makeDirs(const std::string& path) {
  if (path.empty()) return;
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    cur += path[i];
    if (path[i] == '/' || i + 1 == path.size())
      if (!cur.empty() && cur != "/") mkdir(cur.c_str(), 0755);
  }
}

inline std::string dirName(const std::string& path) {
  const size_t s = path.find_last_of('/');
  return s == std::string::npos ? std::string(".") : path.substr(0, s);
}

// 1.0.4: `<path>.<node>.<pid>.partial`, ONE PER WRITER.  Several workers
// (and several NODES on a shared filesystem) write the same tables, and a
// shared `.partial` name lets two of them interleave into one temporary
// file before either renames it -- a table that is neither writer's.
inline std::string writerPartial(const std::string& path) {
  char host[256] = {0};
  if (gethostname(host, sizeof host - 1) != 0) host[0] = 0;
  return path + "." + host + "." + std::to_string((long)getpid()) + ".partial";
}

// Written to a partial and renamed.  A campaign killed mid-write
// otherwise leaves a half table, and every reader of it -- including the
// port's own resume -- reads a truncated last row as a design.
inline bool writeCsvRows(const std::vector<Row>& rows, const std::string& path,
                         const std::vector<std::string>& columns) {
  makeDirs(dirName(path));
  const std::string partial = writerPartial(path);
  FILE* f = fopen(partial.c_str(), "w");
  if (!f) { fprintf(stderr, "bindcraft: cannot write %s\n", path.c_str()); return false; }
  for (size_t i = 0; i < columns.size(); ++i)
    fprintf(f, "%s%s", i ? "," : "", csvField(columns[i]).c_str());
  fputc('\n', f);
  for (const Row& r : rows) {
    for (size_t i = 0; i < columns.size(); ++i)
      fprintf(f, "%s%s", i ? "," : "", csvField(r.get(columns[i])).c_str());
    fputc('\n', f);
  }
  fclose(f);
  return rename(partial.c_str(), path.c_str()) == 0;
}

// bc2's `locked_campaign_folder`: flock the directory holding `path` across
// the read-modify-write so concurrent workers on one campaign folder cannot
// read the same row count and overwrite each other's newly appended row.
struct FolderLock {
  int fd = -1;
  explicit FolderLock(const std::string& path) {
    const std::string dir = dirName(path);
    makeDirs(dir);
    fd = open(dir.c_str(), O_RDONLY);
    if (fd >= 0) flock(fd, LOCK_EX);
  }
  ~FolderLock() {
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
  }
  FolderLock(const FolderLock&) = delete;
  FolderLock& operator=(const FolderLock&) = delete;
};

// bc2's `append_metric_row`: the new row's columns are UNIONED with the
// ones already in the file, the whole thing is reordered, and the table is
// rewritten.  Rewriting the file for every row is what lets a later
// trajectory introduce a metric the first one did not have -- which a
// campaign does constantly, because a filter that does not run records
// nothing.
inline void appendMetricRowUnlocked(const std::string& path, const Row& row) {
  Table t = readMetricRows(path);
  std::vector<std::string> names = t.columns;
  for (const auto& c : row.cells)
    if (std::find(names.begin(), names.end(), c.first) == names.end())
      names.push_back(c.first);
  const std::vector<std::string> columns = orderedCsvColumns(names);
  std::vector<Row> rows = t.rows;
  rows.push_back(row);
  writeCsvRows(rows, path, columns);
}

inline void appendMetricRow(const std::string& path, const Row& row) {
  const FolderLock hold(path);
  appendMetricRowUnlocked(path, row);
}

// ---- the stage layout ---------------------------------------------------
static const char* TRAJECTORY_STAGE = "1_Trajectories";
static const char* REFOLD_STAGE     = "2_Refolded";
static const char* RANK_STAGE       = "3_Ranked";

inline std::string stageFolder(const std::string& project, const char* stage) {
  return project + "/" + stage;
}
inline std::string stageTable(const std::string& project, const char* stage) {
  const std::string file = std::string(stage) == TRAJECTORY_STAGE ? "!_Trajectories.csv"
                         : std::string(stage) == REFOLD_STAGE     ? "!_Refolded.csv"
                                                                  : "!_Ranked.csv";
  return project + "/" + stage + "/" + file;
}
inline std::string acceptedTable(const std::string& project) {
  return stageTable(project, RANK_STAGE);
}

// ---- ranking ------------------------------------------------------------
//
// `on_target_mean(row, metric)`: a per-target metric is ONE cell holding
// "0.65;0.54", and the ranking averages only the attract targets -- a
// detarget reading is gated at acceptance and averaging it in would
// reward binding the off-target.  A missing reading ranks last
// (`-math.inf`), it does not rank zero.
inline std::vector<double> targetValues(const std::string& cell) {
  std::vector<double> out;
  std::stringstream ss(cell); std::string part;
  while (std::getline(ss, part, ';')) {
    if (part.empty()) continue;
    // THE WHOLE PART, as `float(part)` requires.  strtod stops at the
    // first character it cannot use and reports success: the design
    // identity `74611a93a194c3ec` parsed as 74611 and the campaign
    // summary carried a row averaging the hash column.
    const char* p = part.c_str(); char* end = nullptr;
    const double v = strtod(p, &end);
    while (end && (*end == ' ' || *end == '\t')) ++end;
    if (end != p && *end == '\0' && std::isfinite(v)) out.push_back(v);
  }
  return out;
}

inline bool onTargetMean(const Row& row, const std::string& metric, double& out) {
  const std::vector<double> values = targetValues(row.get(metric));
  if (values.empty()) return false;
  const std::vector<double> weights = targetValues(row.get("target_weights"));
  std::vector<double> kept;
  if (weights.size() == values.size()) {
    for (size_t i = 0; i < values.size(); ++i)
      if (weights[i] > 0) kept.push_back(values[i]);
  }
  const std::vector<double>& use = kept.empty() ? values : kept;
  double s = 0; for (double v : use) s += v;
  out = s / (double)use.size();
  return true;
}

inline void writeRankedDesignsUnlocked(const std::string& project, const std::string& metric) {
  Table t = readMetricRows(acceptedTable(project));
  if (t.rows.empty()) return;
  std::vector<size_t> idx(t.rows.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
    double va = -INFINITY, vb = -INFINITY;
    onTargetMean(t.rows[a], metric, va);
    onTargetMean(t.rows[b], metric, vb);
    return va > vb;
  });
  std::vector<Row> ranked; std::vector<std::string> names;
  int rank = 1;
  for (size_t i : idx) {
    Row r = t.rows[i];
    r.put("rank", std::to_string(rank++));
    ranked.push_back(r);
    for (const auto& c : r.cells)
      if (std::find(names.begin(), names.end(), c.first) == names.end())
        names.push_back(c.first);
  }
  writeCsvRows(ranked, stageTable(project, RANK_STAGE), orderedCsvColumns(names));
}

inline void writeRankedDesigns(const std::string& project, const std::string& metric) {
  const FolderLock hold(acceptedTable(project));
  writeRankedDesignsUnlocked(project, metric);
}

// `append_accepted_design`: the row goes into the accepted table and the
// WHOLE table is re-ranked, so `rank` is correct after every design and
// not only at the end of the campaign.  A campaign killed on trajectory
// 40 still has a ranked table.
inline void appendAcceptedDesign(const std::string& project, const Row& row,
                                 const std::string& metric = "i_pDAE") {
  const FolderLock hold(acceptedTable(project));
  appendMetricRowUnlocked(acceptedTable(project), row);
  writeRankedDesignsUnlocked(project, metric);
}

// ---- the summary --------------------------------------------------------
struct Samples { std::vector<double> v; };

// `statistics.fmean` is `math.fsum(data) / n`, and fsum is EXACT -- it
// carries a list of non-overlapping partials (Shewchuk) and rounds once.
// A naive accumulation differs from it in the last ulp, which is
// invisible until a value lands on a rounding boundary: the gate's first
// run disagreed on one summary cell, `max` 0.77 against bc2's 0.78,
// because one trajectory's mean crossed .775 by 1e-17.  A summary is
// rounded to two places, so every disagreement it can have is a boundary
// one.
inline double fsum(const std::vector<double>& values) {
  std::vector<double> partials;
  for (double x : values) {
    size_t i = 0;
    for (double y : partials) {
      if (std::fabs(x) < std::fabs(y)) std::swap(x, y);
      const double hi = x + y, lo = y - (hi - x);
      if (lo != 0.0) partials[i++] = lo;
      x = hi;
    }
    partials.resize(i);
    partials.push_back(x);
  }
  // ...AND CPython'S FINAL PASS, which is not a reverse sum.  It adds
  // partials from the top and STOPS at the first one that makes the sum
  // inexact, then applies a half-even fix-up when the next partial has the
  // same sign as the residue -- `math.fsum`'s own comment: "needed so that
  // sum([1e-16, 1, 1e16]) will round up the last digit to two instead of
  // down to zero".  Accumulating all the partials instead gives a
  // different last ulp: on six two-decimal readings the mean came out
  // 0.49499999999999994 against Python's 0.49500000000000005, which
  // `recorded_number` then rounds to 0.49 and 0.5.  An exact sum is not
  // enough; it has to be the SAME exact sum.
  size_t n = partials.size();
  double hi = 0.0, lo = 0.0;
  if (n > 0) {
    hi = partials[--n];
    while (n > 0) {
      const double x = hi, y = partials[--n];
      hi = x + y;
      const double yr = hi - x;
      lo = y - yr;
      if (lo != 0.0) break;
    }
    if (n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) ||
                  (lo > 0.0 && partials[n - 1] > 0.0))) {
      const double y = lo * 2.0;
      const double x = hi + y;
      const double yr = x - hi;
      if (y == yr) hi = x;
    }
  }
  return hi;
}

inline double fmean(const std::vector<double>& v) {
  return fsum(v) / (double)v.size();
}
// STATISTICS.PSTDEV IS EXACT, and this was not.  The formula was right --
// POPULATION, as summary_row asks for -- and the arithmetic was floating
// point: sum the squared deviations with fsum and take a sqrt.  That lands
// a hair off CPython's answer, and `recorded_number` turns a hair into a
// different cell whenever the true value is a two-decimal tie.  Two
// accepted designs at i_pDAE 0.04 and 0.01 have pstdev exactly 0.015: the
// float path printed 0.02 and bc2 printed 0.01.  See exact_stats.cuh.
inline double pstdev(const std::vector<double>& v) {
  return exstat::pstdev(v);
}

inline Row summaryRow(const std::string& campaign, const std::string& scope,
                      const std::string& metric, const std::vector<double>& v) {
  Row r;
  r.put("campaign", campaign);
  r.put("scope", scope);
  r.put("metric", metric);
  r.putInt("samples", (long)v.size());
  r.putNum("mean", fmean(v));
  r.putNum("std", pstdev(v));
  r.putNum("min", *std::min_element(v.begin(), v.end()));
  r.putNum("max", *std::max_element(v.begin(), v.end()));
  return r;
}

inline std::vector<std::string> listDir(const std::string& path) {
  std::vector<std::string> out;
  DIR* d = opendir(path.c_str());
  if (!d) return out;
  while (struct dirent* e = readdir(d)) {
    const std::string name = e->d_name;
    if (name != "." && name != "..") out.push_back(name);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

inline bool isDir(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
inline bool isFile(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// `trajectory_metric_means`: every numeric cell of a losses.csv row is
// recorded under BOTH its own phase and the pooled "trajectory" scope, and
// the per-trajectory mean is one sample of the campaign's distribution.
// `phase` and `round` are excluded by name -- `round` is a number and
// averaging it would report the mean step index as a campaign metric.
inline std::map<std::pair<std::string,std::string>, double>
trajectoryMetricMeans(const Table& t) {
  std::map<std::pair<std::string,std::string>, std::vector<double>> acc;
  for (const Row& row : t.rows) {
    std::string phase = row.get("phase");
    if (phase.empty()) phase = "trajectory";
    for (const auto& c : row.cells) {
      if (c.first == "phase" || c.first == "round") continue;
      const char* p = c.second.c_str(); char* end = nullptr;
      const double v = strtod(p, &end);
      if (p == end || *end != '\0' || !std::isfinite(v)) continue;
      acc[{phase, c.first}].push_back(v);
      if (phase != "trajectory") acc[{"trajectory", c.first}].push_back(v);
    }
  }
  std::map<std::pair<std::string,std::string>, double> out;
  for (const auto& kv : acc) out[kv.first] = fmean(kv.second);
  return out;
}

// ---- the archive ---------------------------------------------------------
//
// `archive_trajectories`: bc2 zips a finished trajectory folder and
// deletes it, and three readers then have to find the folder's losses.csv
// INSIDE the zip -- `summarize_campaign`, `recorded_trajectory_rows`
// (autotune's and the sweep's block review) and `trajectory_metric_rows`.
// The readers are the half that can fail quietly: an archived campaign
// whose summary simply has no trajectory scope, and an autotune review
// that measures an empty block and concludes nothing changed.
static const char* TRAJECTORY_ARCHIVE_SUFFIX = ".zip";

inline std::vector<std::string> trajectoryDirectories(const std::string& project) {
  std::vector<std::string> out;
  const std::string tdir = stageFolder(project, TRAJECTORY_STAGE);
  for (const std::string& name : listDir(tdir))
    if (isDir(tdir + "/" + name)) out.push_back(tdir + "/" + name);
  return out;
}

inline std::vector<std::string> trajectoryArchives(const std::string& project) {
  std::vector<std::string> out;
  const std::string tdir = stageFolder(project, TRAJECTORY_STAGE);
  const std::string suffix = TRAJECTORY_ARCHIVE_SUFFIX;
  for (const std::string& name : listDir(tdir)) {
    if (name.size() <= suffix.size()) continue;
    if (name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
      continue;
    if (isFile(tdir + "/" + name)) out.push_back(tdir + "/" + name);
  }
  return out;
}

// bc2 takes the FIRST member whose basename ends with the filename it was
// asked for, which is how `losses.csv` finds `<design>_losses.csv`.  The
// order is the archive's own, so the reader walks the central directory
// in order rather than sorting.
inline Table archivedMetricRows(const std::string& archive_path,
                                const std::string& filename) {
  for (const std::string& name : zipf::namelist(archive_path)) {
    const size_t slash = name.find_last_of('/');
    const std::string base =
        slash == std::string::npos ? name : name.substr(slash + 1);
    if (base.size() < filename.size() ||
        base.compare(base.size() - filename.size(), filename.size(),
                     filename) != 0)
      continue;
    std::string text;
    if (!zipf::readMember(archive_path, name, &text)) return Table();
    std::istringstream in(text);
    return parseMetricRows(in);
  }
  return Table();
}

// `recorded_trajectory_rows`: the folder if it is still there, the
// archive if it is not.
inline Table recordedTrajectoryRows(const std::string& project,
                                    const std::string& design,
                                    const std::string& filename = "losses.csv") {
  const std::string dir =
      stageFolder(project, TRAJECTORY_STAGE) + "/" + design;
  if (isDir(dir)) return readMetricRows(dir + "/" + design + "_" + filename);
  const std::string archive = dir + TRAJECTORY_ARCHIVE_SUFFIX;
  if (isFile(archive)) return archivedMetricRows(archive, filename);
  return Table();
}

inline void removeTree(const std::string& path) {
  for (const std::string& name : listDir(path)) {
    const std::string child = path + "/" + name;
    if (isDir(child)) removeTree(child);
    else remove(child.c_str());
  }
  rmdir(path.c_str());
}

// `os.walk` order, with `sorted(filenames)` inside each directory and the
// archive name relative to the trajectories folder -- so every member is
// `<design>/<file>` and an extraction puts the folder back where it was.
inline void collectFiles(const std::string& dir, const std::string& prefix,
                         std::vector<zipf::Entry>* out) {
  std::vector<std::string> subdirs;
  for (const std::string& name : listDir(dir)) {
    const std::string child = dir + "/" + name;
    if (isDir(child)) subdirs.push_back(name);
    else out->push_back({prefix + name, child});
  }
  for (const std::string& name : subdirs)
    collectFiles(dir + "/" + name, prefix + name + "/", out);
}

// `archive_trajectory_folder`: write `<folder>.zip.partial`, rename it
// over `<folder>.zip`, delete the folder.  The partial name is not
// decoration -- a campaign killed mid-archive must leave either the
// folder or a COMPLETE zip, never a truncated one that every reader
// above would silently read as a trajectory with no rows.
inline std::string archiveTrajectoryFolder(const std::string& trajectory_dir) {
  if (!isDir(trajectory_dir)) return "";
  std::string dir = trajectory_dir;
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  const size_t slash = dir.find_last_of('/');
  const std::string base =
      slash == std::string::npos ? dir : dir.substr(slash + 1);
  const std::string archive = dir + TRAJECTORY_ARCHIVE_SUFFIX;
  const std::string partial = writerPartial(archive);
  std::vector<zipf::Entry> entries;
  collectFiles(dir, base + "/", &entries);
  if (!zipf::write(partial, entries)) {
    remove(partial.c_str());
    fprintf(stderr, "  archive_trajectories: could not write %s\n",
            archive.c_str());
    return "";
  }
  if (rename(partial.c_str(), archive.c_str()) != 0) {
    remove(partial.c_str());
    return "";
  }
  removeTree(dir);
  return archive;
}

inline std::vector<Row> summarizeCampaign(const std::string& project,
                                          const std::string& campaign) {
  std::map<std::pair<std::string,std::string>, std::vector<double>> recorded;
  int recorded_trajectories = 0;
  // the folders first and then the ARCHIVES, as `trajectory_metric_rows`
  // concatenates them: an archived trajectory is still a sample of the
  // campaign's distribution, and before the archive reader existed this
  // loop would have dropped every one of them.
  std::vector<Table> measured;
  for (const std::string& dir : trajectoryDirectories(project)) {
    const size_t slash = dir.find_last_of('/');
    const std::string name = slash == std::string::npos ? dir : dir.substr(slash + 1);
    measured.push_back(readMetricRows(dir + "/" + name + "_losses.csv"));
  }
  for (const std::string& archive : trajectoryArchives(project))
    measured.push_back(archivedMetricRows(archive, "losses.csv"));
  for (const Table& t : measured) {
    if (t.rows.empty()) continue;
    ++recorded_trajectories;
    for (const auto& kv : trajectoryMetricMeans(t))
      recorded[kv.first].push_back(kv.second);
  }
  const char* scopes[3] = {"final", "candidate", "accepted"};
  const char* stages[3] = {TRAJECTORY_STAGE, REFOLD_STAGE, RANK_STAGE};
  for (int s = 0; s < 3; ++s) {
    const Table t = readMetricRows(stageTable(project, stages[s]));
    for (const Row& row : t.rows) {
      for (const auto& c : row.cells) {
        const std::string& metric = c.first;
        if (metric == "rank" || metric == "design" || metric == "terminated" ||
            metric == "failed_filters" || metric == "targets" ||
            metric == "target_weights") continue;
        const std::vector<double> values = targetValues(c.second);
        if (values.empty()) continue;
        std::vector<std::string> labels;
        const std::vector<std::string> declared = [&]{
          std::vector<std::string> out; std::stringstream ss(row.get("targets"));
          std::string part;
          while (std::getline(ss, part, ';')) if (!part.empty()) out.push_back(part);
          return out;
        }();
        if (declared.size() == values.size()) labels = declared;
        else if (values.size() > 1) { for (size_t i = 0; i < values.size(); ++i) labels.push_back(std::to_string(i)); }
        else labels.push_back("");
        for (size_t i = 0; i < values.size(); ++i) {
          const std::string scope = labels[i].empty() ? std::string(scopes[s])
                                  : std::string(scopes[s]) + ":" + labels[i];
          recorded[{scope, metric}].push_back(values[i]);
        }
      }
    }
  }
  // the campaign-status counts, which are rows of one sample each
  const Table traj = readMetricRows(stageTable(project, TRAJECTORY_STAGE));
  const Table cand = readMetricRows(stageTable(project, REFOLD_STAGE));
  const Table acc  = readMetricRows(acceptedTable(project));
  std::vector<Row> out;
  std::map<std::string, int> terminated;
  for (const Row& r : traj.rows) {
    std::string stage = r.get("terminated");
    if (stage.empty()) stage = "completed";
    terminated[stage]++;
  }
  std::vector<std::pair<std::string,double>> counts = {
    {"trajectories", (double)std::max((size_t)recorded_trajectories, traj.rows.size())},
    {"redesign_candidates", (double)cand.rows.size()},
    {"accepted_designs", (double)acc.rows.size()}};
  for (const auto& kv : terminated)
    counts.push_back({"terminated:" + kv.first, (double)kv.second});
  for (const auto& kv : counts)
    out.push_back(summaryRow(campaign, "campaign", kv.first, {kv.second}));
  for (const auto& kv : recorded)
    out.push_back(summaryRow(campaign, kv.first.first, kv.first.second, kv.second));
  return out;
}

inline std::string campaignName(const std::string& project) {
  std::string p = project;
  while (p.size() > 1 && p.back() == '/') p.pop_back();
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

inline void writeCampaignSummary(const std::string& project,
                                 const std::string& campaign = "") {
  const std::vector<Row> rows =
      summarizeCampaign(project, campaign.empty() ? campaignName(project) : campaign);
  if (rows.empty()) return;
  writeCsvRows(rows, project + "/summary.csv",
               {"campaign", "scope", "metric", "samples", "mean", "std", "min", "max"});
}

// ---- JSON out, for the metadata and the progress file -------------------
inline std::string jsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else if (c == '\r') out += "\\r";
    else out += c;
  }
  return out;
}

inline std::string jsonNumber(double v) {
  if (!std::isfinite(v)) return std::isnan(v) ? "\"NaN\"" : (v > 0 ? "\"Infinity\"" : "\"-Infinity\"");
  if (v == (double)(long long)v && std::fabs(v) < 1e15)
    return std::to_string((long long)v);
  // the SHORTEST text that reads back as the same double, which is what
  // Python's repr writes.  "%.17g" is always correct and always ugly:
  // 0.15 comes out 0.14999999999999999, and a metadata record a human is
  // meant to diff against the settings file is the one place that matters.
  char tmp[64];
  for (int digits = 15; digits <= 17; ++digits) {
    snprintf(tmp, sizeof tmp, "%.*g", digits, v);
    if (atof(tmp) == v) break;
  }
  return tmp;
}

}  // namespace campout
