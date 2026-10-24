// autotune.cuh -- bc2's `autotuned_settings`, which is ON by default.
//
// Every ten trajectories, `autotune_review` looks at the block since the
// last review and decides whether a design stage is too short or too
// long.  It reads the campaign's own tables -- the trajectory table for
// `terminated`, and each trajectory's losses.csv for the loss curve -- so
// it is a feature the port could not have had before those files existed,
// and it is why `autotune` sat on the IGNORED list while being true in
// settings/core/default.json.
//
// WHAT IT ACTUALLY TUNES, and what it does not.  `FILTER_STAGE_AXES` is
// ('screen_steps', 'refine_steps') and the loop skips an axis the campaign
// has not SET (`if not configured: continue`).  default.json sets neither,
// so on a de novo campaign autotune reviews and changes nothing.  Four
// shipped modalities do set them -- peptide, cyclic_peptide, large_binder
// and multidomain -- and those are the campaigns where this is live.
// `autotune_loss_weights` is a separate setting, null in reference.json,
// and the weight-alternation half of the review is refused rather than
// half-implemented: it needs the loss weights to be mutable per
// trajectory, which they are not here.
//
// THE RULE, once a stage has at least five measured trajectories.  Split
// that stage's rows into thirds by step and take the mean weighted loss in
// each:
//
//   closing >= opening            the stage made the loss WORSE: leave it
//   rejecting, or still falling   +25%  (final third holds > 50% of the fall)
//   flattened                     -25%  (final third holds < 10% of it)
//   otherwise                     unchanged
//
// held within [0.5x, 2x] of what the campaign configured, so a review can
// never run away from the setting it is tuning.
#pragma once
#include "campaign_out.cuh"
#include "progress.cuh"
#include "domains.cuh"
#include "pyrandom.cuh"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace autotune {

static const int REVIEW_TRAJECTORIES = 10;
static const int MIN_MEASURED = 5;
static const double STAGE_STEP_CHANGE = 0.25;
static const double FINAL_THIRD_CONVERGED = 0.1;
static const double FINAL_THIRD_DESCENDING = 0.5;
static const double LENGTH_BAND_LOW = 0.5, LENGTH_BAND_HIGH = 2.0;

struct Record {
  int reviewed = 0;
  int accepted = 0;
  bool has_loss = false;
  double loss = 0.0;
  std::map<std::string, double> overrides;
  // the weight half: what the weights were before the last alternation,
  // and whether that alternation happened (bc2's `restores` and
  // `alternating`, which is `bool(drawn)`)
  std::map<std::string, double> restores;
  bool alternating = false;
};

inline std::string statePath(const std::string& project) {
  return project + "/.campaign_state.json";
}

// A deliberately small reader for the one block this writes.  bc2's
// `.campaign_state.json` also carries `trajectories`, `accepted`,
// `rejections` and `attempted`, all of which the port RECOVERS BY COUNTING
// THE TABLES instead -- a count of the table cannot disagree with the
// table, and a journal can.  Only the autotune record has no table to be
// recovered from, which is why it is the only thing here.
// The record is the `autotune` OBJECT of `.campaign_state.json`, read and
// written through progress::Campaign under the folder lock -- the same
// file and the same lock the trajectory claim uses, as bc2 has them.  This
// reader used to search the whole file for "accepted", which since the
// claim keeps a top-level `accepted` would have read the campaign's count
// as the review's.
inline Record readState(const std::string& project) {
  Record r;
  const std::string t = progress::Campaign(project, 0, 0).autotuneRecord();
  if (t.empty()) return r;
  auto num = [&](const char* key, double& slot, bool& seen) {
    const std::string pat = std::string("\"") + key + "\":";
    const size_t k = t.find(pat);
    if (k == std::string::npos) return;
    const size_t v = t.find_first_not_of(" \t\n", k + pat.size());
    if (v == std::string::npos || t.compare(v, 4, "null") == 0) return;
    slot = atof(t.c_str() + v);
    seen = true;
  };
  bool seen = false;
  double d = 0;
  num("reviewed", d, seen); r.reviewed = (int)d;
  seen = false; d = 0; num("accepted", d, seen); r.accepted = (int)d;
  seen = false; d = 0; num("loss", d, seen); r.loss = d; r.has_loss = seen;
  auto block = [&](const char* key, std::map<std::string, double>& into) {
    const size_t o = t.find(std::string("\"") + key + "\":");
    if (o == std::string::npos) return;
    const size_t a = t.find('{', o), b = t.find('}', o);
    if (a == std::string::npos || b == std::string::npos || b <= a) return;
    std::string body = t.substr(a + 1, b - a - 1);
    size_t i = 0;
    while (true) {
      const size_t q1 = body.find('"', i);
      if (q1 == std::string::npos) break;
      const size_t q2 = body.find('"', q1 + 1);
      if (q2 == std::string::npos) break;
      const size_t c = body.find(':', q2);
      if (c == std::string::npos) break;
      into[body.substr(q1 + 1, q2 - q1 - 1)] = atof(body.c_str() + c + 1);
      i = c + 1;
    }
  };
  block("overrides", r.overrides);
  block("restores", r.restores);
  { const size_t k = t.find("\"alternating\":");
    if (k != std::string::npos) {
      const size_t v = t.find_first_not_of(" \t\n", k + 14);
      r.alternating = v != std::string::npos && t.compare(v, 4, "true") == 0;
    } }
  return r;
}

// json.dumps(record, sort_keys=True): accepted, alternating, loss,
// overrides, restores, reviewed
inline void writeState(const std::string& project, const Record& r) {
  auto obj = [&](const std::map<std::string, double>& m) {
    std::string o = "{";
    bool first = true;
    for (const auto& kv : m) {
      o += (first ? "" : ", ");
      o += "\"" + kv.first + "\": " + campout::jsonNumber(kv.second);
      first = false;
    }
    return o + "}";
  };
  std::string js = "{\"accepted\": " + std::to_string(r.accepted);
  js += std::string(", \"alternating\": ") + (r.alternating ? "true" : "false");
  js += ", \"loss\": " + (r.has_loss ? campout::jsonNumber(r.loss) : std::string("null"));
  js += ", \"overrides\": " + obj(r.overrides);
  js += ", \"restores\": " + obj(r.restores);
  js += ", \"reviewed\": " + std::to_string(r.reviewed) + "}";
  progress::Campaign(project, 0, 0).setAutotuneRecord(js);
}

// `mean_design_loss`: the mean over EVERY cell of these rows whose column
// names a weighted loss.  Not a mean of per-row means -- bc2 collects the
// values and takes one fmean, so a row that recorded more losses than
// another counts for more.
// `weight_scales`: a loss recorded while its weight was ALTERNATED is
// scaled back to the campaign's own weight (settings[w] / override)
// before the thirds are compared, or the review would read its own
// alternation as the trajectory improving.  Empty until the weight half
// draws something, which is why the stage-length half never needed it.
inline bool meanDesignLoss(const std::vector<campout::Row>& rows,
                           const std::set<std::string>& loss_names,
                           double& out,
                           const std::map<std::string, double>& weight_scales =
                               std::map<std::string, double>()) {
  std::vector<double> values;
  for (const campout::Row& r : rows)
    for (const auto& c : r.cells) {
      const size_t dot = c.first.find('.');
      if (dot == std::string::npos) continue;
      std::string rest = c.first.substr(dot + 1);
      const size_t dot2 = rest.find('.');
      if (dot2 != std::string::npos) rest = rest.substr(0, dot2);
      if (!loss_names.count(rest)) continue;
      const char* p = c.second.c_str(); char* end = nullptr;
      const double v = strtod(p, &end);
      const auto sc = weight_scales.find(rest);
      const double scale = sc == weight_scales.end() ? 1.0 : sc->second;
      if (end != p && *end == '\0' && std::isfinite(v))
        values.push_back(v * scale);
    }
  if (values.empty()) return false;
  out = campout::fmean(values);
  return true;
}

// `stage_loss_thirds`: opening, middle, closing over the rows of ONE stage.
inline bool stageLossThirds(const campout::Table& losses,
                            const std::string& phase,
                            const std::set<std::string>& loss_names,
                            double thirds[3],
                            const std::map<std::string, double>& weight_scales =
                                std::map<std::string, double>()) {
  std::vector<campout::Row> stage_rows;
  for (const campout::Row& r : losses.rows)
    if (r.get("phase") == phase) stage_rows.push_back(r);
  const int ft = (int)stage_rows.size() / 3;
  if (ft == 0) return false;
  const int n = (int)stage_rows.size();
  const std::vector<campout::Row> a(stage_rows.begin(), stage_rows.begin() + ft);
  const std::vector<campout::Row> b(stage_rows.begin() + (n - 2 * ft),
                                    stage_rows.begin() + (n - ft));
  const std::vector<campout::Row> c(stage_rows.begin() + (n - ft), stage_rows.end());
  return meanDesignLoss(a, loss_names, thirds[0], weight_scales) &&
         meanDesignLoss(b, loss_names, thirds[1], weight_scales) &&
         meanDesignLoss(c, loss_names, thirds[2], weight_scales);
}

struct Measured {
  bool rejected = false;
  double thirds[3] = {0, 0, 0};
};

// Python's round(): half to EVEN, which `round(current * 1.25)` reaches
// whenever current*1.25 lands on a half.  std::round goes half AWAY from
// zero, so screen_steps 50 -> 62.5 would read 63 here and 62 in bc2.
inline long pyRound(double v) {
  const double f = std::floor(v), diff = v - f;
  if (diff > 0.5) return (long)f + 1;
  if (diff < 0.5) return (long)f;
  return ((long)f % 2 == 0) ? (long)f : (long)f + 1;
}

// `modality_loss_weight_axes`: every `weights_*` the campaign SETS to a
// non-zero value that differs from bc2's own default, ordered by
// magnitude (descending) and then by name.  A weight the campaign left
// alone is not the campaign's choice to alternate.
inline std::vector<std::string> weightAxes(
    const std::map<std::string, double>& campaign,
    const std::map<std::string, double>& defaults) {
  std::vector<std::pair<double, std::string>> keyed;
  for (const auto& kv : campaign) {
    if (kv.first.rfind("weights_", 0) != 0 || kv.second == 0.0) continue;
    // THE COMPARISON IS IN FLOAT.  bc2 compares the campaign's value with
    // DEFAULT_SETTINGS exactly, both doubles from the same JSON; the port
    // keeps its defaults as floats, so -0.3 read from a campaign is not
    // -0.3f widened and every core weight looked like an axis.  Five axes
    // where bc2 has one, and the extra four would have been alternated.
    const auto d = defaults.find(kv.first);
    if (d != defaults.end() && (float)d->second == (float)kv.second) continue;
    keyed.push_back({-std::fabs(kv.second), kv.first});
  }
  std::sort(keyed.begin(), keyed.end());
  std::vector<std::string> out;
  for (const auto& q : keyed) out.push_back(q.second);
  return out;
}

struct Review {
  Record record;
  std::vector<std::string> report;
};

// `trajectory_rows` is the whole trajectory table; `losses_for` returns one
// trajectory's losses.csv by design name.
// `weight_axes` is empty unless the campaign asked for
// `autotune_loss_weights`; `campaign_weights` holds the campaign's own
// value for each axis, which is what an alternation multiplies and what
// an acceptance restores.
inline Review review(const Record& previous,
                     const std::vector<campout::Row>& trajectory_rows,
                     int accepted_designs,
                     const std::set<std::string>& loss_names,
                     const std::map<std::string, double>& configured,
                     const std::function<campout::Table(const std::string&)>& losses_for,
                     const std::vector<std::string>& weight_axes =
                         std::vector<std::string>(),
                     const std::map<std::string, double>& campaign_weights =
                         std::map<std::string, double>(),
                     long campaign_seed = 0) {
  Review out;
  out.record.reviewed = (int)trajectory_rows.size();
  out.record.accepted = accepted_designs;
  out.record.overrides = previous.overrides;
  static const char* AXES[2] = {"screen_steps", "refine_steps"};
  // `configured_weight_scales`: the losses in the block were recorded at
  // the PREVIOUS review's weights, so they are scaled back to the
  // campaign's before the thirds are compared
  std::map<std::string, double> weight_scales;
  for (const auto& kv : previous.overrides) {
    if (kv.first.rfind("weights_", 0) != 0 || kv.second == 0.0) continue;
    const auto c = campaign_weights.find(kv.first);
    if (c == campaign_weights.end() || c->second == 0.0) continue;
    weight_scales[kv.first.substr(8)] = c->second / kv.second;
  }
  std::map<std::string, std::vector<Measured>> measured;
  for (size_t t = (size_t)previous.reviewed; t < trajectory_rows.size(); ++t) {
    const campout::Row& row = trajectory_rows[t];
    const campout::Table losses = losses_for(row.get("design"));
    for (const char* axis : AXES) {
      const std::string stage = std::string(axis).substr(
          0, std::string(axis).size() - 6);        // drop "_steps"
      Measured m;
      if (!stageLossThirds(losses, stage, loss_names, m.thirds, weight_scales))
        continue;
      m.rejected = row.get("terminated") == stage;
      measured[stage].push_back(m);
    }
  }
  // the block's end-of-screen loss, which the weight half reverts on
  { std::vector<double> closing;
    for (const Measured& m : measured["screen"]) closing.push_back(m.thirds[2]);
    if ((int)closing.size() >= MIN_MEASURED) {
      out.record.loss = campout::fmean(closing);
      out.record.has_loss = true;
    } }

  // ---- THE WEIGHT HALF: alternate, and undo what did not help ---------
  //
  // An override that is neither a stage axis nor a weight axis is dropped
  // (bc2 filters the record), and then:
  //
  //   a design was accepted since the last review  -> put the weights back
  //   nothing accepted and the block got WORSE     -> undo the alternation
  //   nothing accepted                             -> alternate again
  //
  // The multiplier is drawn from `random.Random(f"{seed}:{n}")`, which is
  // a STRING seed -- CPython hashes it with sha512 before init_by_array,
  // so pyrandom.cuh does too.
  { std::map<std::string, double> keep;
    const std::set<std::string> waxes(weight_axes.begin(), weight_axes.end());
    for (const auto& kv : out.record.overrides) {
      const bool is_stage = std::string(kv.first) == AXES[0] ||
                            std::string(kv.first) == AXES[1];
      if (is_stage || waxes.count(kv.first)) keep[kv.first] = kv.second;
    }
    out.record.overrides = keep;
    const bool accepted_since = accepted_designs > previous.accepted;
    const bool reverting = previous.alternating && !accepted_since &&
                           out.record.has_loss && previous.has_loss &&
                           out.record.loss >= previous.loss;
    if (reverting) {
      std::vector<std::string> reverted;
      for (const auto& kv : out.record.overrides) {
        if (!waxes.count(kv.first)) continue;
        const auto r = previous.restores.find(kv.first);
        if (r == previous.restores.end() || r->second != kv.second) {
          char line[160];
          snprintf(line, sizeof line, "%s %g -> %g", kv.first.c_str(), kv.second,
                   r != previous.restores.end()
                       ? r->second
                       : (campaign_weights.count(kv.first)
                              ? campaign_weights.at(kv.first) : kv.second));
          reverted.push_back(line);
        }
      }
      std::map<std::string, double> kept;
      for (const auto& kv : out.record.overrides)
        if (!waxes.count(kv.first)) kept[kv.first] = kv.second;
      for (const auto& kv : previous.restores) kept[kv.first] = kv.second;
      out.record.overrides = kept;
      if (!reverted.empty()) {
        std::string names;
        for (size_t i = 0; i < reverted.size(); ++i)
          names += (i ? ", " : "") + reverted[i];
        char line[512];
        snprintf(line, sizeof line,
                 "autotune: undid %s, mean end-of-screen design loss %.4f "
                 "against %.4f at the campaign's own weights over the %d "
                 "trajectories since", names.c_str(), out.record.loss,
                 previous.loss,
                 (int)trajectory_rows.size() - previous.reviewed);
        out.report.push_back(line);
      }
    }
    for (const auto& kv : out.record.overrides)
      if (waxes.count(kv.first)) out.record.restores[kv.first] = kv.second;
    std::vector<std::string> changes;
    if (accepted_since) {
      for (const std::string& name : weight_axes) {
        const auto it = out.record.overrides.find(name);
        if (it == out.record.overrides.end()) continue;
        char line[256];
        snprintf(line, sizeof line,
                 "autotune: %s %g -> %g, a design was accepted", name.c_str(),
                 it->second,
                 campaign_weights.count(name) ? campaign_weights.at(name) : 0.0);
        changes.push_back(line);
        out.record.overrides.erase(it);
      }
    } else if (!weight_axes.empty() && !reverting) {
      domains::MT rng;
      pyrandom::seedString(rng, std::to_string(campaign_seed) + ":" +
                                    std::to_string(trajectory_rows.size()));
      static const double MULTIPLIERS[2] = {0.5, 2.0};
      for (const std::string& name : weight_axes) {
        const double base = campaign_weights.count(name)
                          ? campaign_weights.at(name) : 0.0;
        const double drawn = base * MULTIPLIERS[rng.randbelow(2)];
        out.record.overrides[name] = drawn;
        char line[256];
        snprintf(line, sizeof line,
                 "autotune: %s %g -> %g, %d trajectories accepted nothing",
                 name.c_str(), base, drawn,
                 (int)trajectory_rows.size() - previous.reviewed);
        changes.push_back(line);
      }
      out.record.alternating = true;
    }
    for (const std::string& c : changes) out.report.push_back(c);
  }
  for (const char* axis : AXES) {
    const std::string stage = std::string(axis).substr(
        0, std::string(axis).size() - 6);
    auto cfg = configured.find(axis);
    if (cfg == configured.end() || cfg->second == 0) continue;
    const std::vector<Measured>& ms = measured[stage];
    if ((int)ms.size() < MIN_MEASURED) continue;
    double opening = 0, middle = 0, closing = 0;
    { std::vector<double> a, b, c;
      for (const Measured& m : ms) { a.push_back(m.thirds[0]);
                                     b.push_back(m.thirds[1]);
                                     c.push_back(m.thirds[2]); }
      opening = campout::fmean(a); middle = campout::fmean(b);
      closing = campout::fmean(c); }
    if (closing >= opening) continue;      // the stage made it worse
    const double share = (closing - middle) / (closing - opening);
    int rejections = 0;
    for (const Measured& m : ms) rejections += m.rejected ? 1 : 0;
    const bool rejecting = rejections * 2 > (int)ms.size();
    const long shortest = std::max(1L, pyRound(LENGTH_BAND_LOW * cfg->second));
    const long longest = std::max(1L, pyRound(LENGTH_BAND_HIGH * cfg->second));
    const double current = out.record.overrides.count(axis)
                         ? out.record.overrides[axis] : cfg->second;
    const double step = (rejecting || share > FINAL_THIRD_DESCENDING)
                            ? STAGE_STEP_CHANGE
                        : (share < FINAL_THIRD_CONVERGED) ? -STAGE_STEP_CHANGE
                                                          : 0.0;
    const long adjusted = std::min(longest, std::max(shortest,
                                   pyRound(current * (1.0 + step))));
    if ((double)adjusted != current) {
      out.record.overrides[axis] = (double)adjusted;
      char line[320];
      snprintf(line, sizeof line,
               "autotune: %s %g -> %ld (configured %g, held within %ld-%ld), %s",
               axis, current, adjusted, cfg->second, shortest, longest,
               rejecting
                   ? "the stage rejected more than half the trajectories that reached it"
               : share > FINAL_THIRD_DESCENDING
                   ? "the stage was still descending at its end"
                   : "the stage had flattened by its end");
      out.report.push_back(line);
    }
  }
  return out;
}

}  // namespace autotune
