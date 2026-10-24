// ---------------------------------------------------------------------
// progress.cuh -- `.campaign_state.json`, and the claim that makes two
// processes one campaign
// ---------------------------------------------------------------------
//
// bc2's `CampaignProgress` is what lets a campaign run as SEVERAL
// processes: every trajectory number, every recipe hash and every
// accepted design is taken under an exclusive lock on the campaign
// folder, so two workers never design the same trajectory and the budget
// (`number_of_final_designs`, `max_trajectories`) is counted once for all
// of them.  A single process needs none of it -- which is why this port
// counted the tables instead and got resume right without it -- and the
// moment there are two, counting the tables is a race: both read "4
// recorded" and both design trajectory 5.
//
// Three things here, and each is a decision rather than a transcription:
//
//   * THE LOCK IS ON THE DIRECTORY, not on the state file.  bc2 flocks
//     the folder's own descriptor, so the lock survives the state file
//     being replaced (which it is, every write: `.partial` then rename).
//     Locking the file would hand two writers two different inodes.
//   * THE STATE IS RECOVERED FROM THE TABLES when the file is missing
//     (`recovered_state`), so a campaign that crashed mid-run, or one
//     written by a port that never kept a journal, resumes to the same
//     numbers.  A journal and a table can disagree; a count cannot.
//   * THE BYTES ARE PYTHON'S.  `json.dumps(state, sort_keys=True)` --
//     sorted keys, ", " and ": " separators -- because bc2 reads this
//     file and writes it back, and a campaign can change hands between
//     the two implementations mid-run.
#pragma once
#include "campaign_out.cuh"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace progress {

struct Rejections {
  std::map<std::string, long> terminated;
  std::map<std::string, long> failed_filters;
  long candidates_scored = 0;
  long candidates_rejected = 0;
};

// 1.0.4 dropped `attempted` (the recipe hashes): a trajectory is claimed
// by CREATING ITS FOLDER now (`claimTrajectoryDirectory` below), which
// only one worker can win, so the state no longer carries a recipe list.
// An older file that still has one is read and the list ignored.
struct State {
  long trajectories = 0;
  long accepted = 0;
  Rejections rejections;
  // THE AUTOTUNE RECORD, kept as the JSON object it is.  bc2 keeps it in
  // this same file (`record = state.setdefault('autotune', ...)` inside
  // the same lock), so a writer that rewrote only the fields it knows
  // would delete the review's memory on every claim -- which is exactly
  // what happened: autotune reviewed nothing, because every trajectory's
  // claim wiped the record the last review wrote.  autotune.cuh owns the
  // contents; this only carries them.
  std::string autotune;          // "" when there is none
};

// the balanced {...} starting at `at` (which must be a '{')
inline size_t objectEnd(const std::string& t, size_t at) {
  int depth = 0; bool in_str = false;
  for (size_t i = at; i < t.size(); ++i) {
    const char c = t[i];
    if (in_str) { if (c == '\\') ++i; else if (c == '"') in_str = false; continue; }
    if (c == '"') in_str = true;
    else if (c == '{') ++depth;
    else if (c == '}') { if (--depth == 0) return i + 1; }
  }
  return std::string::npos;
}

// ---- the file ---------------------------------------------------------
inline std::string quoted(const std::string& s) {
  return "\"" + campout::jsonEscape(s) + "\"";
}

inline std::string mapJson(const std::map<std::string, long>& m) {
  std::string out = "{";
  bool first = true;
  for (const auto& kv : m) {                 // std::map is sorted, as is json
    if (!first) out += ", ";
    first = false;
    out += quoted(kv.first) + ": " + std::to_string(kv.second);
  }
  return out + "}";
}

// `json.dumps(state, sort_keys=True)`: accepted, rejections, trajectories
// -- in that order, which is alphabetical
inline std::string stateJson(const State& s) {
  std::string out = "{\"accepted\": " + std::to_string(s.accepted);
  if (!s.autotune.empty()) out += ", \"autotune\": " + s.autotune;
  out += ", \"rejections\": {\"candidates_rejected\": " +
         std::to_string(s.rejections.candidates_rejected) +
         ", \"candidates_scored\": " +
         std::to_string(s.rejections.candidates_scored) +
         ", \"failed_filters\": " + mapJson(s.rejections.failed_filters) +
         ", \"terminated\": " + mapJson(s.rejections.terminated) + "}";
  out += ", \"trajectories\": " + std::to_string(s.trajectories) + "}";
  return out;
}

// a reader for exactly the shape above -- and for the same shape written
// by Python, which may order its keys differently if a future bc2 drops
// sort_keys, so the parse is by NAME and not by position
inline size_t skipSpace(const std::string& t, size_t at) {
  while (at < t.size() && isspace((unsigned char)t[at])) ++at;
  return at;
}

inline size_t readString(const std::string& t, size_t at, std::string* out) {
  at = skipSpace(t, at);
  if (at >= t.size() || t[at] != '"') return at;
  ++at;
  out->clear();
  while (at < t.size() && t[at] != '"') {
    if (t[at] == '\\' && at + 1 < t.size()) ++at;
    *out += t[at++];
  }
  return at + 1;
}

inline std::map<std::string, long> readLongMap(const std::string& t,
                                               size_t* at) {
  std::map<std::string, long> out;
  *at = skipSpace(t, *at);
  if (*at >= t.size() || t[*at] != '{') return out;
  ++*at;
  while (true) {
    *at = skipSpace(t, *at);
    if (*at >= t.size() || t[*at] == '}') { ++*at; break; }
    if (t[*at] == ',') { ++*at; continue; }
    std::string key;
    *at = readString(t, *at, &key);
    *at = skipSpace(t, *at);
    if (*at < t.size() && t[*at] == ':') ++*at;
    *at = skipSpace(t, *at);
    out[key] = strtol(t.c_str() + *at, nullptr, 10);
    while (*at < t.size() && t[*at] != ',' && t[*at] != '}') ++*at;
  }
  return out;
}

inline State parseState(const std::string& text) {
  State s;
  size_t at = 0;
  while (at < text.size()) {
    at = skipSpace(text, at);
    if (at >= text.size()) break;
    if (text[at] != '"') { ++at; continue; }
    std::string key;
    at = readString(text, at, &key);
    at = skipSpace(text, at);
    if (at < text.size() && text[at] == ':') ++at;
    at = skipSpace(text, at);
    if (key == "trajectories" || key == "accepted") {
      long* slot = key == "accepted" ? &s.accepted : &s.trajectories;
      *slot = strtol(text.c_str() + at, nullptr, 10);
      while (at < text.size() && text[at] != ',' && text[at] != '}') ++at;
    } else if (key == "autotune") {
      if (at < text.size() && text[at] == '{') {
        const size_t e = objectEnd(text, at);
        if (e != std::string::npos) { s.autotune = text.substr(at, e - at); at = e; }
      }
    } else if (key == "attempted") {
      // a 1.0.3 file's recipe list: read past it
      if (at < text.size() && text[at] == '[') {
        const size_t close = text.find(']', at);
        at = close == std::string::npos ? text.size() : close + 1;
      }
    } else if (key == "rejections") {
      // one nested object, whose values are ints and two sub-objects
      ++at;                                  // the '{'
      while (at < text.size()) {
        at = skipSpace(text, at);
        if (at >= text.size() || text[at] == '}') { ++at; break; }
        if (text[at] == ',') { ++at; continue; }
        std::string field;
        at = readString(text, at, &field);
        at = skipSpace(text, at);
        if (at < text.size() && text[at] == ':') ++at;
        at = skipSpace(text, at);
        if (field == "terminated") s.rejections.terminated = readLongMap(text, &at);
        else if (field == "failed_filters") s.rejections.failed_filters = readLongMap(text, &at);
        else {
          const long v = strtol(text.c_str() + at, nullptr, 10);
          if (field == "candidates_scored") s.rejections.candidates_scored = v;
          else if (field == "candidates_rejected") s.rejections.candidates_rejected = v;
          while (at < text.size() && text[at] != ',' && text[at] != '}') ++at;
        }
      }
    } else {
      // an unknown key: skip its value, whatever shape it is
      int depth = 0;
      do {
        if (text[at] == '{' || text[at] == '[') ++depth;
        if (text[at] == '}' || text[at] == ']') --depth;
        ++at;
      } while (at < text.size() && (depth > 0 || (text[at] != ',' && text[at] != '}')));
    }
  }
  return s;
}

// ---- the campaign's own progress --------------------------------------
class Campaign {
 public:
  Campaign(const std::string& project, long requested_designs,
           long max_trajectories)
      : project_(project), requested_(requested_designs),
        max_trajectories_(max_trajectories),
        state_path_(project + "/.campaign_state.json") {
    campout::makeDirs(project);
  }

  // `recovered_state`: the tables, counted
  State recovered() const {
    State s;
    const campout::Table traj =
        campout::readMetricRows(campout::stageTable(project_,
                                                    campout::TRAJECTORY_STAGE));
    const campout::Table cand =
        campout::readMetricRows(campout::stageTable(project_,
                                                    campout::REFOLD_STAGE));
    s.trajectories = (long)traj.rows.size();
    s.accepted =
        (long)campout::readMetricRows(campout::acceptedTable(project_)).rows.size();
    for (const campout::Row& r : traj.rows) {
      std::string stage = r.get("terminated");
      if (stage.empty()) stage = "completed";
      s.rejections.terminated[stage]++;
    }
    for (const campout::Row& r : cand.rows) {
      s.rejections.candidates_scored++;
      const std::string failed = r.get("failed_filters");
      if (!failed.empty()) s.rejections.candidates_rejected++;
      std::stringstream ss(failed);
      std::string name;
      while (std::getline(ss, name, ','))
        if (!name.empty()) s.rejections.failed_filters[name]++;
    }
    return s;
  }

  // 1.0.4: a state file that will not parse is REBUILT from the tables,
  // and said -- a worker killed mid-write on a filesystem that does not
  // make the rename atomic, or a node writing while another reads, used to
  // take the whole campaign down on a JSONDecodeError.  The parse here is
  // lenient by construction, so "will not parse" is tested as what such a
  // file looks like: not a closed object holding the trajectory count.
  State read() const {
    std::ifstream f(state_path_);
    if (!f) return recovered();
    std::stringstream buf;
    buf << f.rdbuf();
    const std::string text = buf.str();
    size_t a = text.find_first_not_of(" \t\r\n"),
           b = text.find_last_not_of(" \t\r\n");
    const bool whole = a != std::string::npos && text[a] == '{' &&
                       text[b] == '}' &&
                       text.find("\"trajectories\"") != std::string::npos;
    if (!whole) {
      printf("campaign state rebuilt from the recorded tables: %s was not "
             "readable (not a complete JSON object)\n", state_path_.c_str());
      return recovered();
    }
    return parseState(text);
  }

  // claim_trajectory: the number, and how many designs are accepted so
  // far -- or nothing, when the campaign's budget is spent
  bool claimTrajectory(long* number, long* accepted) {
    const Lock hold(project_);
    State s = read();
    if (requested_ && s.accepted >= requested_) return false;
    if (max_trajectories_ && s.trajectories >= max_trajectories_) return false;
    s.trajectories += 1;
    *number = s.trajectories;
    *accepted = s.accepted;
    write(s);
    return true;
  }

  long recordAcceptedDesign() {
    const Lock hold(project_);
    State s = read();
    s.accepted += 1;
    write(s);
    return s.accepted;
  }

  void recordTrajectoryOutcome(const std::string& terminated_stage) {
    const Lock hold(project_);
    State s = read();
    s.rejections.terminated[terminated_stage.empty() ? "completed"
                                                     : terminated_stage]++;
    write(s);
  }

  void recordCandidateOutcome(const std::vector<std::string>& failed) {
    const Lock hold(project_);
    State s = read();
    s.rejections.candidates_scored++;
    if (!failed.empty()) s.rejections.candidates_rejected++;
    for (const std::string& name : failed)
      if (!name.empty()) s.rejections.failed_filters[name]++;
    write(s);
  }

  // the autotune review's own read and write, inside the same lock
  std::string autotuneRecord() {
    const Lock hold(project_);
    return read().autotune;
  }
  void setAutotuneRecord(const std::string& raw) {
    const Lock hold(project_);
    State s = read();
    s.autotune = raw;
    write(s);
  }

  void status(long* accepted, long* trajectories) {
    const Lock hold(project_);
    const State s = read();
    *accepted = s.accepted;
    *trajectories = s.trajectories;
  }

 private:
  // flock on the FOLDER, held for the read-modify-write
  class Lock {
   public:
    explicit Lock(const std::string& folder) {
      fd_ = open(folder.c_str(), O_RDONLY);
      if (fd_ >= 0) flock(fd_, LOCK_EX);
    }
    ~Lock() {
      if (fd_ >= 0) { flock(fd_, LOCK_UN); close(fd_); }
    }
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;
   private:
    int fd_ = -1;
  };

  void write(const State& s) const {
    const std::string partial = campout::writerPartial(state_path_);
    { std::ofstream f(partial);
      if (!f) return;
      f << stateJson(s); }
    rename(partial.c_str(), state_path_.c_str());
  }

  std::string project_;
  long requested_ = 0;
  long max_trajectories_ = 0;
  std::string state_path_;
};

// 1.0.4's `claimed_trajectory_directory`: a trajectory is CLAIMED by
// creating its folder, which only one worker can win -- mkdir fails with
// EEXIST for every other.  A folder that was archived away (a `.zip`
// beside it) is claimed already.  This replaces the recipe list the state
// file used to carry.
inline bool claimTrajectoryDirectory(const std::string& dir) {
  if (campout::isFile(dir + campout::TRAJECTORY_ARCHIVE_SUFFIX)) return false;
  const size_t slash = dir.find_last_of('/');
  if (slash != std::string::npos) campout::makeDirs(dir.substr(0, slash));
  return mkdir(dir.c_str(), 0755) == 0;    // EEXIST: another worker won
}

}  // namespace progress
