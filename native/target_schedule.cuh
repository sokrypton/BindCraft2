// bc2's target_schedule.py: WHICH STATE GETS THE ROUND.
//
// A campaign with more than one prepared state does NOT score them all
// every round.  bc2 rotates: `select_protein_states` returns ONE state, the
// round's forward and backward are spent on it alone, and
// `losses_for_active_states` drops every loss whose required states are not
// in it.  The port used to run the complex AND every off-target every step
// and sum the gradients, which is a different optimiser on the same losses.
//
// Two transitions, chosen in `_select_target_transition`:
//
//   FIXED CADENCE    -- merged gradients, or an epitope rotation: swap
//                       after `rounds_before_target_swap()` rounds, whatever
//                       the prediction says.
//   OBJECTIVE GATED  -- the default: a POSITIVE target is left when its
//                       i_pTM clears `multitarget_swap_threshold` (or
//                       patience runs out), an OFF-TARGET when the binder
//                       has actually come off it -- quiet at the interface
//                       AND no longer touching it, because a short binder
//                       holds a whole epitope at a confidence no threshold
//                       rejects and the repulsion would keep the round
//                       forever.
//
// THE ROUND AFTER A SWAP IS NOT JUDGED.  `predictions` holds only the state
// that was just predicted, so the entry that has just become active is not
// in it and `select_protein_states` returns it untouched: no transition
// test, no `iterations_on_target` increment.  Every target therefore gets
// one free round on arrival.  Reproduced here by `have_prediction`.
#pragma once
#include "jaxrandom.cuh"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace tsched {

struct Entry {
  std::string name;
  bool detarget = false;
};

// what a round reports back about the state it just ran
struct Observed {
  bool have = false;        // the active entry was the one predicted
  float iptm = 0.f;
  float interface_residues = 0.f;
  float plddt = 0.f;        // carried for the stage check, not for the swap
};

// `stage_peak_predictions`: the best round this stage, per state, by the
// state's OWN direction -- highest i_pTM for a target, lowest for an
// off-target.  judge_stage filters on these rather than on the last round,
// because with a rotation the last round belongs to one state and the
// stage has to answer for all of them.
struct Peak {
  bool have = false;
  float iptm = 0.f;
  float plddt = 0.f;
};

class Multitarget {
 public:
  // bc2's defaults, from build_target_schedule's `settings.get` calls
  struct Params {
    int iterations = 1;
    float iptm_threshold = 0.5f;          // multitarget_swap_threshold
    int swap_patience = 20;               // multitarget_swap_patience
    int warmup_swap_patience = -1;        // multitarget_warmup_patience (-1: = swap_patience)
    int epitope_rotation_interval = -1;   // idr_crop_count > 1 -> design_model_count
    bool merged_target_gradients = false;
    float max_detarget_iptm = 0.4f;
    int detarget_check_interval = 1;      // DEFAULT_DETARGET_CHECK_INTERVAL
    int max_detarget_rounds = 10;
    // DEFAULT_DETARGET_INTERFACE_RESIDUES; a campaign writing it off keeps
    // the confidence criterion alone, so null becomes +inf, not a number
    float max_detarget_interface_residues = 3.f;
  };

  Multitarget(std::vector<Entry> entries, Params p)
      : entries_(std::move(entries)), p_(p) {
    if (p_.warmup_swap_patience < 0) p_.warmup_swap_patience = p_.swap_patience;
    if (p_.epitope_rotation_interval >= 0)
      p_.epitope_rotation_interval = std::max(1, p_.epitope_rotation_interval);
    p_.detarget_check_interval = std::max(1, p_.detarget_check_interval);
    p_.max_detarget_rounds = std::max(1, p_.max_detarget_rounds);
    peaks_.assign(entries_.size(), Peak());
    latest_.assign(entries_.size(), Peak());
    in_accumulated_gradient_.resize(entries_.size());
    std::vector<int> rotated, detargeted;
    rotatedAndDetargeted(rotated, detargeted);
    rotated_ = active_ = rotated.empty() ? 0 : rotated.front();
  }

  // ---- the two transitions ------------------------------------------
  bool fixedCadence() const {
    return p_.merged_target_gradients || p_.epitope_rotation_interval >= 0;
  }

  int roundsBeforeTargetSwap() const {
    if (p_.merged_target_gradients) return 1;
    const int configured = p_.epitope_rotation_interval >= 0
                               ? p_.epitope_rotation_interval
                               : (warmup_complete_ ? p_.swap_patience
                                                   : p_.warmup_swap_patience);
    std::vector<int> rotated, detargeted;
    rotatedAndDetargeted(rotated, detargeted);
    const int share = rotated.empty() ? p_.iterations
                                      : p_.iterations / (int)rotated.size();
    return std::max(1, std::min(configured, share));
  }

  bool reached(bool detarget, float iptm, float ires) const {
    if (fixedCadence())
      return iterations_on_target_ + 1 >= roundsBeforeTargetSwap();
    if (detarget)
      return (iptm <= p_.max_detarget_iptm &&
              ires <= p_.max_detarget_interface_residues) ||
             iterations_on_target_ + 1 >= p_.max_detarget_rounds;
    return iptm > p_.iptm_threshold ||
           iterations_on_target_ + 1 >= roundsBeforeTargetSwap();
  }

  // ---- the loop's three calls ---------------------------------------

  // select_protein_states: returns the index of the state to run this round
  int selectActive(const Observed& obs) {
    const bool detarget = entries_[active_].detarget;
    if (obs.have) {
      recordStagePeak(obs);
      const float ires = detarget ? obs.interface_residues : 0.f;
      // only an off-target is decided on contact, and only its own rounds
      // pay for measuring it
      if (!fixedCadence() && !detarget) ++rounds_since_detarget_check_;
      if (reached(detarget, obs.iptm, ires)) {
        iterations_on_target_ = 0;
        advanceActive(detarget);
      } else {
        ++iterations_on_target_;
      }
    }
    return active_;
  }

  bool shouldUpdateSequence(const Observed& obs) {
    recordStagePeak(obs);
    if (!p_.merged_target_gradients) return true;
    in_accumulated_gradient_.insert(active_);
    return in_accumulated_gradient_.size() >= entries_.size();
  }

  void recordSequenceUpdate() {
    in_accumulated_gradient_.clear();
    ++completed_sequence_updates_;
  }

  bool isComplete() const {
    return completed_sequence_updates_ >= p_.iterations;
  }

  // restart_for_stage: the stage boundary keeps the rotation state and the
  // warmup flag, and drops the peaks, the latest round and the merge set
  void restartForStage(int iterations, bool merged_target_gradients) {
    peaks_.assign(entries_.size(), Peak());
    latest_.assign(entries_.size(), Peak());
    in_accumulated_gradient_.clear();
    p_.merged_target_gradients = merged_target_gradients;
    p_.iterations = iterations;
    completed_sequence_updates_ = 0;
  }

  int active() const { return active_; }
  const Entry& entry(int i) const { return entries_[i]; }
  size_t size() const { return entries_.size(); }
  const std::vector<Peak>& peaks() const { return peaks_; }
  const std::vector<Peak>& latest() const { return latest_; }
  int iterationsOnTarget() const { return iterations_on_target_; }
  bool warmupComplete() const { return warmup_complete_; }

 private:
  void rotatedAndDetargeted(std::vector<int>& rotated,
                            std::vector<int>& detargeted) const {
    rotated.clear(); detargeted.clear();
    std::vector<int> positives, offs;
    for (size_t i = 0; i < entries_.size(); ++i)
      (entries_[i].detarget ? offs : positives).push_back((int)i);
    if (positives.empty() || p_.merged_target_gradients ||
        p_.epitope_rotation_interval >= 0) {
      for (size_t i = 0; i < entries_.size(); ++i) rotated.push_back((int)i);
      return;
    }
    rotated = positives; detargeted = offs;
  }

  void advanceActive(bool detarget) {
    std::vector<int> rotated, detargeted;
    rotatedAndDetargeted(rotated, detargeted);
    if (detarget && !detargeted.empty()) {
      rounds_since_detarget_check_ = 0;
    } else {
      size_t position = 0;
      const auto it = std::find(rotated.begin(), rotated.end(), rotated_);
      if (it != rotated.end())
        position = ((size_t)(it - rotated.begin()) + 1) % rotated.size();
      rotated_ = rotated[position];
      warmup_complete_ = warmup_complete_ || position == 0;
    }
    const bool detarget_due = !detargeted.empty() && !detarget &&
                              rounds_since_detarget_check_ >= p_.detarget_check_interval;
    active_ = detarget_due ? detargeted[detarget_position_] : rotated_;
    if (detarget_due)
      detarget_position_ = (detarget_position_ + 1) % (int)detargeted.size();
  }

  void recordStagePeak(const Observed& obs) {
    if (!obs.have) return;
    latest_[active_] = Peak{true, obs.iptm, obs.plddt};
    const Peak& peak = peaks_[active_];
    const bool better = entries_[active_].detarget ? obs.iptm < peak.iptm
                                                   : obs.iptm > peak.iptm;
    if (!peak.have || better) peaks_[active_] = Peak{true, obs.iptm, obs.plddt};
  }

  std::vector<Entry> entries_;
  Params p_;
  int active_ = 0, rotated_ = 0;
  int iterations_on_target_ = 0;
  int completed_sequence_updates_ = 0;
  int rounds_since_detarget_check_ = 0;
  int detarget_position_ = 0;
  bool warmup_complete_ = false;
  std::vector<Peak> peaks_, latest_;
  struct IndexSet {
    std::vector<char> in;
    size_t n = 0;
    void resize(size_t k) { in.assign(k, 0); n = 0; }
    void insert(int i) { if (i >= 0 && (size_t)i < in.size() && !in[i]) { in[i] = 1; ++n; } }
    void clear() { std::fill(in.begin(), in.end(), 0); n = 0; }
    size_t size() const { return n; }
  };
  IndexSet in_accumulated_gradient_;
};

}  // namespace tsched

// ---------------------------------------------------------------------
// InducedFitSchedule: WHICH TWO SHAPES THE ROUND COMPARES
// ---------------------------------------------------------------------
//
// The other schedule in target_schedule.py, and a different question
// from the rotation's.  `binder_shapes` is a list of CONFORMATION
// GROUPS, each holding one or more state names, and every round:
//
//   next_conformation = a uniform draw over the groups EXCLUDING the
//                       one the schedule is currently on
//   next_state        = a uniform draw inside that group
//   predicted         = {current state, next state}
//   current           = (next_conformation, next_state)
//
// Both draws come off `jax.random`, through two keys split from the
// schedule's own, so a trajectory repeats exactly.  With two groups of
// one state the draws have a single outcome each and the schedule is
// deterministic -- which is the shipped shape, and why this ran before
// the stream was ported.  With a group of several states it is a real
// draw, and reproducing it means reproducing threefry (jaxrandom.cuh).
//
// `_sample_conformation_index` falls back to [0] when excluding the
// current conformation leaves nothing -- a single group compares a
// state with itself rather than failing, and that branch is reproduced.
namespace tsched {

struct Compared {
  int current_group = 0, current_state = 0;
  int next_group = 0, next_state = 0;
};

class InducedFit {
 public:
  InducedFit(const std::vector<int>& group_sizes, const jrandom::Key& key)
      : sizes_(group_sizes), key_(key) {}

  // one round: the pair to predict, and the schedule moves on
  Compared select() {
    // `self.random_key, conformation_random_key, state_random_key =
    //  jax.random.split(self.random_key, 3)` -- the FIRST of the three
    // is the schedule's next key, which is why the stream advances even
    // when both draws have one outcome
    const std::vector<jrandom::Key> ks = jrandom::split(key_, 3);
    key_ = ks[0];
    Compared c;
    c.current_group = group_;
    c.current_state = state_;
    c.next_group = sample(ks[1], (int)sizes_.size(), group_);
    c.next_state = sample(ks[2], sizes_[(size_t)c.next_group], -1);
    group_ = c.next_group;
    state_ = c.next_state;
    return c;
  }

  int group() const { return group_; }
  int state() const { return state_; }

 private:
  // `_sample_conformation_index`: the available indices are every one
  // but `excluded`, or [0] when that leaves none
  static int sample(const jrandom::Key& key, int count, int excluded) {
    std::vector<int> available;
    for (int i = 0; i < count; ++i)
      if (i != excluded) available.push_back(i);
    if (available.empty()) available.push_back(0);
    return available[(size_t)jrandom::randint(key, 0,
                                              (int)available.size())];
  }

  std::vector<int> sizes_;
  jrandom::Key key_;
  int group_ = 0, state_ = 0;
};

}  // namespace tsched
