// ---------------------------------------------------------------------
// parameter_sweep: a campaign is ARMS, not one run
// ---------------------------------------------------------------------
//
// bc2's parameter_sweep.py, the half that decides WHAT to run.  A
// campaign that asks for a sweep does not run `max_trajectories`
// trajectories: it runs a BASELINE arm and one arm per swept level, each
// in its own subfolder with its own settings, splits the budget between
// them, and writes a record saying which arm won.  Running the baseline
// and reporting it under the campaign's name answers a question the
// campaign did not ask -- which is why the driver refused this until the
// arms existed.
//
// THE ARMS ARE INTERLEAVED, NOT RUN TO COMPLETION ONE AT A TIME.
// `sweep_block_budgets` gives (10, 20, ..., arm_trajectories) and the
// loop is `for budget: for arm:`, so every arm reaches trajectory 10
// before any reaches 20.  A campaign killed halfway then holds a
// comparison rather than one finished arm and nothing else -- and the
// sweep record is written after every block, so it is readable while the
// campaign runs.
//
// WHAT IS NOT HERE: the per-arm trajectory loop itself (the driver's),
// and `autotune`, which bc2 switches OFF inside an arm -- a sweep moves
// the settings on purpose and a tuner moving them underneath would be
// measuring two things at once.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "campaign_out.cuh"
#include "autotune.cuh"   // pyRound: bc2 rounds a stage length, and Python
                          // rounds halves to EVEN

namespace sweep {

// DEFAULT_PARAMETER_SWEEP
static const int   DEFAULT_MAX_ARMS = 5;
static const int   DEFAULT_BLOCK_TRAJECTORIES = 10;
static const double RESOLVING_ERRORS = 2.0;
static const int   MIN_PAIRED_TRAJECTORIES = 10;
static const char* BASELINE_ARM = "baseline";
static const char* SWEEP_FILENAME = "sweep.csv";
static const char* BEST_SETTINGS_FILENAME = "best_settings.json";
static const char* SWEEP_OBJECTIVE =
    "paired change in trajectory interface pTM, then accepted designs per "
    "trajectory, then mean accepted i_pTM";

// DEFAULT_SWEEP_AXES, in bc2's order -- the five a campaign gets when it
// names none, and only those it actually SETS to something non-zero
static const char* const DEFAULT_SWEEP_AXES[] = {
    "weights_binder_helicity", "weights_interface_contacts",
    "weights_compactness", "weights_binder_pae", "weights_interface_pae"};
// FILTER_STAGE_AXES
static const char* const FILTER_STAGE_AXES[] = {"screen_steps", "refine_steps"};
// MODALITY_SWITCH_AXES: a switch the campaign set brings in an axis that
// is not a weight at all
static const char* const SWITCH_AXIS[][2] = {
    {"forced_targeting", "forced_targeting_shell"}};
// AXIS_DEFAULTS -- EPITOPE_CUTOFF
static const double FORCED_TARGETING_SHELL_DEFAULT = 10.0;
// AXIS_SWEEP_VALUES: one axis is swept at ABSOLUTE values rather than by
// multiplying, because a negative weight times 0.5 is still negative and
// the question about helicity is which SIGN wins
static const double HELICITY_VALUES[2] = {-0.6, 0.3};
// SHARED_MODEL_SETTINGS: read by a model constructor every arm shares, so
// an arm cannot differ in them
static const char* const SHARED_MODEL_SETTINGS[] = {
    "design_models", "validation_models", "validation_model",
    "design_recycles", "validation_recycles", "subbatch_size",
    "attention_backend", "use_cueq", "length_bucket_size",
    "cyclic_offset_mode", "cyclize_peptide", "mpnn_model", "mpnn_variant",
    "aa_bias", "copies", "oligomer_tie"};

struct Options {
  std::vector<std::string> axes;        // explicit; empty means derived
  std::vector<double> levels;           // (0.5, 2.0) unless the campaign says
  double multiplier = 0.0;              // non-zero replaces the levels
  int max_arms = DEFAULT_MAX_ARMS;
  int block_trajectories = DEFAULT_BLOCK_TRAJECTORIES;
};

struct Arm {
  std::string name;
  std::vector<std::pair<std::string, double>> overrides;
};

// ---- the campaign's `parameter_sweep` value --------------------------
//
// A nested object the flat settings reader keeps as RAW TEXT (it skips
// nested objects, and what reaches `settings.str` is the braces and all),
// or a bare `true`.  Both mean a sweep; `false`, `0` and absent do not.
inline bool parseOptions(const std::string& raw, Options* out) {
  out->levels = {0.5, 2.0};
  if (raw.empty()) return true;            // `parameter_sweep: true`
  auto number = [&](const char* key, double* dst) {
    const std::string k = std::string("\"") + key + "\"";
    const size_t p = raw.find(k);
    if (p == std::string::npos) return;
    const size_t c = raw.find(':', p + k.size());
    if (c == std::string::npos) return;
    *dst = atof(raw.c_str() + c + 1);
  };
  double max_arms = out->max_arms, block = out->block_trajectories;
  number("max_arms", &max_arms);
  number("block_trajectories", &block);
  number("multiplier", &out->multiplier);
  out->max_arms = (int)max_arms;
  out->block_trajectories = (int)block;
  // the two lists
  auto list = [&](const char* key, std::vector<std::string>* names,
                  std::vector<double>* nums) {
    const std::string k = std::string("\"") + key + "\"";
    const size_t p = raw.find(k);
    if (p == std::string::npos) return false;
    const size_t open = raw.find('[', p);
    const size_t close = raw.find(']', open);
    if (open == std::string::npos || close == std::string::npos) return false;
    const std::string body = raw.substr(open + 1, close - open - 1);
    if (names) {
      for (size_t i = 0; i < body.size(); ++i)
        if (body[i] == '"') {
          const size_t e = body.find('"', i + 1);
          if (e == std::string::npos) break;
          names->push_back(body.substr(i + 1, e - i - 1));
          i = e;
        }
    }
    if (nums) {
      std::string cur;
      for (size_t i = 0; i <= body.size(); ++i) {
        if (i == body.size() || body[i] == ',') {
          if (cur.find_first_not_of(" \t\r\n") != std::string::npos)
            nums->push_back(atof(cur.c_str()));
          cur.clear();
        } else cur += body[i];
      }
    }
    return true;
  };
  list("axes", &out->axes, nullptr);
  std::vector<double> levels;
  if (list("levels", nullptr, &levels) && !levels.empty())
    out->levels = levels;
  return true;
}

inline std::vector<double> levels(const Options& o) {
  if (o.multiplier) return {o.multiplier};
  return o.levels;
}

// `axis_sweep_values`: the absolute-value table applies only when the
// campaign named NO axes of its own -- a campaign that asked for this
// axis by name gets the multipliers like any other
inline std::vector<double> axisValues(const std::string& name,
                                      const Options& o) {
  if (!o.axes.empty()) return {};
  if (name == "weights_binder_helicity")
    return {HELICITY_VALUES[0], HELICITY_VALUES[1]};
  return {};
}

inline int axisArmCount(const std::string& name, const Options& o) {
  const std::vector<double> v = axisValues(name, o);
  return (int)(v.empty() ? levels(o).size() : v.size());
}

// ---- which axes, and the three ways a campaign can name a bad one ----
//
// `error` is set to bc2's own message; the driver refuses on it rather
// than sweeping something else.
inline std::vector<std::string> axesFor(
    const std::map<std::string, double>& numeric_settings,
    const std::map<std::string, std::string>& text_settings,
    const std::map<std::string, std::vector<double>>& array_settings,
    const std::vector<std::string>& weight_axes,
    const std::map<std::string, double>& stage_rounds,
    bool (*known)(const std::string&), const Options& o,
    std::string* error) {
  std::vector<std::string> derived;
  for (const auto& sa : SWITCH_AXIS) {
    const auto it = numeric_settings.find(sa[0]);
    if (it != numeric_settings.end() && it->second != 0.0)
      derived.push_back(sa[1]);
  }
  for (const std::string& w : weight_axes) derived.push_back(w);
  for (const char* d : DEFAULT_SWEEP_AXES) {
    const auto it = numeric_settings.find(d);
    if (it != numeric_settings.end() && it->second != 0.0)
      derived.push_back(d);
  }
  for (const char* f : FILTER_STAGE_AXES)
    if (numeric_settings.count(f) && numeric_settings.at(f) != 0.0)
      derived.push_back(f);
  // dict.fromkeys: first occurrence wins, order kept
  std::vector<std::string> axes;
  { std::set<std::string> seen;
    for (const std::string& a : (o.axes.empty() ? derived : o.axes))
      if (seen.insert(a).second) axes.push_back(a); }
  // 1. an axis that names no setting this codebase reads
  { std::string bad;
    for (const std::string& a : axes)
      if (!known(a)) bad += (bad.empty() ? "" : ", ") + a;
    if (!bad.empty()) {
      *error = "parameter_sweep axes [" + bad +
               "] name no setting this codebase reads";
      return {};
    } }
  // 2. an axis every arm shares, because one model is built for all of them
  { std::string bad;
    for (const std::string& a : axes)
      for (const char* s : SHARED_MODEL_SETTINGS)
        if (a == s) bad += (bad.empty() ? "" : ", ") + a;
    if (!bad.empty()) {
      *error = "parameter_sweep axes [" + bad +
               "] are read by a model constructor that every arm shares";
      return {};
    } }
  // 3. an axis the campaign wrote as an ARRAY, which is a draw per
  //    trajectory and so has no single value for an arm to move
  { std::string bad;
    for (const std::string& a : axes)
      if (array_settings.count(a) ||
          (text_settings.count(a) && !text_settings.at(a).empty() &&
           text_settings.at(a)[0] == '['))
        bad += (bad.empty() ? "" : ", ") + a;
    if (!bad.empty()) {
      *error = "parameter_sweep axes [" + bad +
               "] are drawn per trajectory from the array the campaign "
               "wrote, so a sweep has no single value of them to move";
      return {};
    } }
  (void)stage_rounds;
  // and the budget: axes are kept while the arms they add, plus the
  // baseline, stay within max_arms
  std::vector<std::string> kept;
  int cum = 0;
  for (const std::string& a : axes) {
    cum += axisArmCount(a, o);
    if (1 + cum <= o.max_arms) kept.push_back(a);
  }
  return kept;
}

// `swept_value`: a stage length is an integer count and is ROUNDED (and
// never below 1); everything else is simply multiplied.  The stage's
// current value is design_stage_rounds', not the raw setting, because a
// campaign may not have set it at all.
inline double sweptValue(const std::string& name, double level,
                         const std::map<std::string, double>& numeric_settings,
                         const std::map<std::string, double>& stage_rounds) {
  const bool is_steps = name.size() > 6 &&
      name.compare(name.size() - 6, 6, "_steps") == 0;
  double current = 0.0;
  if (is_steps) {
    const std::string stage = name.substr(0, name.size() - 6);
    const auto it = stage_rounds.find(stage);
    if (it != stage_rounds.end()) current = it->second;
    else if (numeric_settings.count(name)) current = numeric_settings.at(name);
  } else if (numeric_settings.count(name)) {
    current = numeric_settings.at(name);
  } else if (name == "forced_targeting_shell") {
    current = FORCED_TARGETING_SHELL_DEFAULT;
  }
  if (!is_steps) return current * level;
  const double r = (double)autotune::pyRound(current * level);
  return r < 1.0 ? 1.0 : r;
}

// Python's `f'{value:g}'`
inline std::string gfmt(double v) {
  char buf[64];
  snprintf(buf, sizeof buf, "%g", v);
  return buf;
}

inline std::vector<Arm> armsFor(
    const std::vector<std::string>& axes, const Options& o,
    const std::map<std::string, double>& numeric_settings,
    const std::map<std::string, double>& stage_rounds) {
  std::vector<Arm> arms;
  arms.push_back({BASELINE_ARM, {}});
  for (const std::string& a : axes) {
    const std::vector<double> vals = axisValues(a, o);
    if (!vals.empty()) {
      for (double v : vals)
        arms.push_back({a + "_" + gfmt(v), {{a, v}}});
    } else {
      for (double lv : levels(o))
        arms.push_back({a + "_x" + gfmt(lv),
                        {{a, sweptValue(a, lv, numeric_settings, stage_rounds)}}});
    }
  }
  // a sweep with nothing to compare the baseline with is not a sweep
  if (arms.size() < 2) return {};
  return arms;
}

// `arm_trajectory_budget` and `sweep_block_budgets`
inline int armBudget(int max_trajectories, int arm_count) {
  if (max_trajectories <= 0) return 0;
  const int b = max_trajectories / arm_count;
  return b < 1 ? 1 : b;
}

inline std::vector<int> blockBudgets(int arm_trajectories, int block) {
  std::vector<int> out;
  const int step = block < 1 ? 1 : block;
  for (int b = step; b < arm_trajectories; b += step) out.push_back(b);
  out.push_back(arm_trajectories);
  return out;
}

// ---- THE RECORD: which arm won, and whether the answer is resolved ---
//
// The objective is PAIRED: trajectory N in one arm and trajectory N in
// another start from the same seed, so comparing them in pairs removes
// the variance between trajectories, which is far larger than the
// difference a sweep is looking for.  An arm's `resolved` flag is the
// honest part -- a change smaller than twice its own standard error over
// fewer than ten pairs is not an answer, and the record says so instead
// of ranking it anyway.
struct ArmResult {
  std::string arm;
  int trajectories = 0, accepted = 0;
  bool have_per_traj = false, have_acc_iptm = false, have_traj_iptm = false;
  double accepted_per_trajectory = 0, accepted_iptm = 0, trajectory_iptm = 0;
  int paired = 0;
  bool have_change = false, have_error = false;
  double change = 0, error = 0;
  bool resolved = false;
  std::map<int, double> scores;
  std::map<std::string, double> settings;
};

// `per_target_readings`, restricted to the states a sweep scores: the
// unbound fold has no interface and a detarget is meant not to have one
inline std::vector<double> interfaceReadings(
    const campout::Row& row, const std::string& metric,
    const std::set<std::string>& excluded) {
  const std::string raw = row.get(metric);
  if (raw.empty()) return {};
  std::vector<std::string> values, names;
  { std::string cur;
    for (size_t i = 0; i <= raw.size(); ++i) {
      if (i == raw.size() || raw[i] == ';') { values.push_back(cur); cur.clear(); }
      else cur += raw[i];
    } }
  { const std::string t = row.get("targets");
    std::string cur;
    for (size_t i = 0; i <= t.size(); ++i) {
      if (i == t.size() || t[i] == ';') {
        if (!cur.empty()) names.push_back(cur);
        cur.clear();
      } else cur += t[i];
    } }
  if (names.size() != values.size()) {
    names.clear();
    if (values.size() > 1)
      for (size_t i = 0; i < values.size(); ++i) names.push_back(std::to_string(i));
    else names.push_back("");
  }
  std::vector<double> out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (excluded.count(names[i])) continue;
    const char* p = values[i].c_str();
    char* end = nullptr;
    const double v = strtod(p, &end);
    if (end != p && std::isfinite(v)) out.push_back(v);
  }
  return out;
}

// bc2 takes the trajectory NUMBER off the row, falling back to the tail
// of the design name, and pairs the arms by it
inline int trajectoryNumber(const campout::Row& row) {
  std::string n = row.get("trajectory");
  if (n.empty()) {
    const std::string d = row.get("design");
    const size_t u = d.rfind('_');
    n = u == std::string::npos ? "" : d.substr(u + 1);
  }
  if (n.empty()) return -1;
  for (char c : n) if (!isdigit((unsigned char)c)) return -1;
  return atoi(n.c_str());
}

inline ArmResult armResult(const std::string& project, const Arm& arm,
                           const std::set<std::string>& excluded,
                           const std::vector<std::string>& setting_names,
                           const std::map<std::string, double>& campaign) {
  const std::string folder = project + "/" + arm.name;
  const campout::Table traj =
      campout::readMetricRows(campout::stageTable(folder, campout::TRAJECTORY_STAGE));
  const campout::Table acc =
      campout::readMetricRows(campout::acceptedTable(folder));
  ArmResult r;
  r.arm = arm.name;
  r.trajectories = (int)traj.rows.size();
  r.accepted = (int)acc.rows.size();
  if (r.trajectories) {
    r.have_per_traj = true;
    r.accepted_per_trajectory = (double)r.accepted / r.trajectories;
  }
  { std::vector<double> all;
    for (const campout::Row& row : acc.rows)
      for (double v : interfaceReadings(row, "i_pTM", excluded)) all.push_back(v);
    if (!all.empty()) { r.have_acc_iptm = true; r.accepted_iptm = campout::fmean(all); } }
  { std::vector<double> all;
    for (const campout::Row& row : traj.rows)
      for (double v : interfaceReadings(row, "iptm", excluded)) all.push_back(v);
    if (!all.empty()) { r.have_traj_iptm = true; r.trajectory_iptm = campout::fmean(all); } }
  for (const campout::Row& row : traj.rows) {
    const int n = trajectoryNumber(row);
    if (n < 0) continue;
    const std::vector<double> v = interfaceReadings(row, "iptm", excluded);
    if (!v.empty()) r.scores[n] = campout::fmean(v);
  }
  for (const std::string& name : setting_names) {
    double value = 0.0;
    bool have = false;
    for (const auto& ov : arm.overrides)
      if (ov.first == name) { value = ov.second; have = true; }
    if (!have) {
      const auto it = campaign.find(name);
      if (it != campaign.end()) { value = it->second; have = true; }
      else if (name == "forced_targeting_shell") {
        value = FORCED_TARGETING_SHELL_DEFAULT; have = true;
      }
    }
    if (have) r.settings[name] = value;
  }
  return r;
}

inline void pairedChange(ArmResult* r, const std::map<int, double>& baseline) {
  std::vector<double> changes;
  for (const auto& kv : r->scores) {
    const auto b = baseline.find(kv.first);
    if (b != baseline.end()) changes.push_back(kv.second - b->second);
  }
  r->paired = (int)changes.size();
  if (changes.empty()) return;
  r->have_change = true;
  r->change = campout::fmean(changes);
  if (changes.size() > 1) {
    // statistics.stdev -- the SAMPLE standard deviation (n-1), over the
    // same exact mean pstdev uses
    const double m = r->change;
    double ss = 0.0;
    for (double c : changes) ss += (c - m) * (c - m);
    r->have_error = true;
    r->error = std::sqrt(ss / (changes.size() - 1)) / std::sqrt((double)changes.size());
  }
  r->resolved = r->paired >= MIN_PAIRED_TRAJECTORIES && r->have_change &&
                r->have_error && std::fabs(r->change) > RESOLVING_ERRORS * r->error;
}

// ---- str(float), which is repr(), which is the SHORTEST round trip ---
//
// sweep.csv is not a metric table: nothing is rounded into it, so a cell
// is `str(value)` and a gate comparing text has to produce the same
// digits Python would.  recordedText above is the rounded form and is
// the wrong tool here.
inline std::string pyRepr(double v) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
  // the SHORTEST digit string that reads back as the same double, which
  // is what repr() produces...
  char buf[64];
  int p = 17;
  for (int q = 1; q <= 17; ++q) {
    snprintf(buf, sizeof buf, "%.*e", q - 1, v);
    if (strtod(buf, nullptr) == v) { p = q; break; }
  }
  snprintf(buf, sizeof buf, "%.*e", p - 1, v);
  // ...and then POSITIONAL unless the exponent is outside [-4, 16),
  // which is the rule repr() switches on.  `%g` has its own rule (it
  // leaves at p significant digits and goes exponential at p), so 100.0
  // came out as "1e+02" where Python writes "100.0".
  std::string m(buf);
  const size_t e = m.find('e');
  int exp10 = atoi(m.c_str() + e + 1);
  std::string digits;
  for (size_t i = 0; i < e; ++i) if (isdigit((unsigned char)m[i])) digits += m[i];
  const bool neg = v < 0;
  std::string out;
  if (exp10 < -4 || exp10 >= 16) {
    out = digits.substr(0, 1);
    out += ".";
    out += digits.size() > 1 ? digits.substr(1) : "0";
    char ebuf[16];
    snprintf(ebuf, sizeof ebuf, "e%c%02d", exp10 < 0 ? '-' : '+',
             exp10 < 0 ? -exp10 : exp10);
    out += ebuf;
  } else if (exp10 >= 0) {
    if ((int)digits.size() > exp10 + 1) {
      out = digits.substr(0, exp10 + 1) + "." + digits.substr(exp10 + 1);
    } else {
      out = digits;
      out.append(exp10 + 1 - digits.size(), '0');
      out += ".0";
    }
  } else {
    out = "0.";
    out.append(-exp10 - 1, '0');
    out += digits;
  }
  return neg ? "-" + out : out;
}

// AND THE AXES WHOSE VALUE IS AN INT.  `swept_value` returns
// `max(1, round(...))` for a stage length, which in Python is an int and
// prints as "25", not "25.0" -- in the arm's overrides, in sweep.csv and
// in best_settings.json.  Everything else is `current * level`, a float.
inline bool integerAxis(const std::string& name) {
  return name.size() > 6 && name.compare(name.size() - 6, 6, "_steps") == 0;
}

inline std::string valueText(const std::string& name, double v) {
  if (!integerAxis(name)) return pyRepr(v);
  char buf[32];
  snprintf(buf, sizeof buf, "%lld", (long long)llround(v));
  return buf;
}

inline std::string intText(long v) { return std::to_string(v); }

// ---- sweep.csv and best_settings.json --------------------------------
//
// Ranked by `arm_objective`: the paired change first, then accepted
// designs per trajectory, then the accepted designs' mean i_pTM -- with
// a missing value counting as 0.0, so an arm that produced nothing
// cannot win on an empty mean.  Python's `sorted(..., reverse=True)` is
// STABLE, so arms that tie keep the order the arms were built in, which
// puts the baseline first.
inline std::string writeRecord(const std::string& project,
                               const std::vector<Arm>& arms,
                               const std::set<std::string>& excluded,
                               const std::map<std::string, double>& campaign,
                               const std::string& campaign_name) {
  std::vector<std::string> setting_names;
  { std::set<std::string> seen;
    for (const Arm& a : arms)
      for (const auto& ov : a.overrides)
        if (seen.insert(ov.first).second) setting_names.push_back(ov.first); }
  std::vector<ArmResult> results;
  for (const Arm& a : arms)
    results.push_back(armResult(project, a, excluded, setting_names, campaign));
  bool any = false;
  for (const ArmResult& r : results) any |= r.trajectories > 0;
  if (!any) return "";
  std::map<int, double> baseline = results[0].scores;
  for (const ArmResult& r : results)
    if (r.arm == BASELINE_ARM) { baseline = r.scores; break; }
  for (ArmResult& r : results) pairedChange(&r, baseline);
  // the ranking, stable and descending on the three keys
  std::vector<int> order(results.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
  auto key = [&](int i) {
    const ArmResult& r = results[i];
    return std::vector<double>{r.have_change ? r.change : 0.0,
                               r.have_per_traj ? r.accepted_per_trajectory : 0.0,
                               r.have_acc_iptm ? r.accepted_iptm : 0.0};
  };
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return key(a) > key(b); });
  std::vector<int> rank(results.size(), 0);
  for (size_t i = 0; i < order.size(); ++i) rank[order[i]] = (int)i + 1;

  const std::string path = project + "/" + SWEEP_FILENAME;
  const std::string partial = path + ".partial";
  { FILE* f = fopen(partial.c_str(), "w");
    if (!f) return "";
    fprintf(f, "arm,rank,trajectories,accepted_designs,accepted_per_trajectory,"
               "accepted_interface_ptm,trajectory_interface_ptm,"
               "paired_trajectories,paired_change,paired_error,resolved");
    for (const std::string& n : setting_names) fprintf(f, ",%s", n.c_str());
    fprintf(f, "\r\n");
    for (size_t i = 0; i < results.size(); ++i) {
      const ArmResult& r = results[i];
      fprintf(f, "%s,%d,%d,%d,%s,%s,%s,%d,%s,%s,%s", r.arm.c_str(), rank[i],
              r.trajectories, r.accepted,
              r.have_per_traj ? pyRepr(r.accepted_per_trajectory).c_str() : "",
              r.have_acc_iptm ? pyRepr(r.accepted_iptm).c_str() : "",
              r.have_traj_iptm ? pyRepr(r.trajectory_iptm).c_str() : "",
              r.paired,
              r.have_change ? pyRepr(r.change).c_str() : "",
              r.have_error ? pyRepr(r.error).c_str() : "",
              r.resolved ? "True" : "False");
      for (const std::string& n : setting_names) {
        const auto it = r.settings.find(n);
        fprintf(f, ",%s", it == r.settings.end() ? ""
                                                 : valueText(n, it->second).c_str());
      }
      fprintf(f, "\r\n");
    }
    fclose(f); }
  rename(partial.c_str(), path.c_str());

  const ArmResult& best = results[order[0]];
  { FILE* f = fopen((project + "/" + BEST_SETTINGS_FILENAME).c_str(), "w");
    if (!f) return path;
    auto num = [&](bool have, double v) {
      return have ? pyRepr(v) : std::string("null");
    };
    fprintf(f, "{\n  \"campaign\": \"%s\",\n  \"arm\": \"%s\",\n"
               "  \"objective\": \"%s\",\n  \"resolved\": %s,\n"
               "  \"arms\": %d,\n  \"trajectories_per_arm\": %d,\n"
               "  \"paired_trajectories\": %d,\n  \"paired_change\": %s,\n"
               "  \"paired_error\": %s,\n  \"accepted_designs_in_arm\": %d,\n"
               "  \"accepted_per_trajectory\": %s,\n"
               "  \"accepted_interface_ptm\": %s,\n"
               "  \"trajectory_interface_ptm\": %s,\n"
               "  \"settings\": {",
            campaign_name.c_str(), best.arm.c_str(), SWEEP_OBJECTIVE,
            best.resolved ? "true" : "false", (int)results.size(),
            best.trajectories, best.paired,
            num(best.have_change, best.change).c_str(),
            num(best.have_error, best.error).c_str(), best.accepted,
            num(best.have_per_traj, best.accepted_per_trajectory).c_str(),
            num(best.have_acc_iptm, best.accepted_iptm).c_str(),
            num(best.have_traj_iptm, best.trajectory_iptm).c_str());
    bool first = true;
    for (const std::string& n : setting_names) {
      const auto it = best.settings.find(n);
      fprintf(f, "%s\n    \"%s\": %s", first ? "" : ",", n.c_str(),
              it == best.settings.end() ? "null"
                                        : valueText(n, it->second).c_str());
      first = false;
    }
    fprintf(f, first ? "}\n}" : "\n  }\n}");
    fclose(f); }
  return path;
}

}  // namespace sweep
