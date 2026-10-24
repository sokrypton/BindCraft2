// ---------------------------------------------------------------------
// workers.cuh -- bc2's design_workers.py: the plan, and the launch
// ---------------------------------------------------------------------
//
// A campaign is several PROCESSES in bc2: one per GPU, and several per
// GPU when the card has the memory for them.  They share nothing but the
// project folder, and `progress.cuh`'s claim is what keeps them from
// designing the same trajectory -- so this file is only the arithmetic
// of WHO RUNS WHERE, plus the launch that sets the three environment
// variables a worker reads.
//
// The arithmetic is bc2's, constant for constant, because every one of
// them is a judgement about a card this port does not get to re-make:
// 38000 bytes per residue pair, 3.4 GB resident per model, a safety
// factor of 2, 4 GB of headroom, at most 8 workers a GPU and 7 when the
// count is `auto`.  `workers_gate.py` compares them with bc2's own
// functions on injected memory readings, because the real ones depend on
// what else is on the card.
//
// WHAT THIS PORT CANNOT TAKE FROM bc2 IS THE PACKING ITSELF.  bc2's
// estimate is for a JAX process whose allocation is capped by
// XLA_PYTHON_CLIENT_MEM_FRACTION -- that is the knob the plan sets -- and
// this driver's step allocates what the model needs, around 15 GB at the
// lengths it ships with and no fraction to clamp it.  So the plan is
// computed exactly as bc2 computes it and then CLAMPED by this port's own
// footprint, out loud, in the driver.  Planning 7 workers a card and OOM
// at the second would be bc2-faithful arithmetic and a dead campaign.
#pragma once
#include "campaign_out.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <sched.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace workers {

static const double DESIGN_ACTIVATION_BYTES_PER_RESIDUE_PAIR = 38000.0;
static const double DESIGN_MODEL_RESIDENT_GB = 3.4;
static const double DESIGN_MEMORY_SAFETY_FACTOR = 2.0;
static const double GPU_MEMORY_HEADROOM_GB = 4.0;
static const int MAXIMUM_WORKERS_PER_GPU = 8;
static const int AUTOMATIC_WORKERS_PER_GPU = 7;
static const int TRAJECTORY_ONLY_WORKERS_PER_GPU = 1;
static const double HOST_MEMORY_PER_WORKER_GB = 4.0;

struct Worker {
  std::string gpu;               // 1.0.4: the card's UUID, GPU-xxxxxxxx-...
  int card = 0;                  // its ordinal among the visible cards
  double memory_fraction = 0.0;
  std::vector<int> lengths;      // when the campaign's lengths are split
  int residue_count = 0;
  std::vector<int> cpus;         // the host CPUs this worker is pinned to
};

inline std::vector<std::string> splitList(const std::string& text, char sep) {
  std::vector<std::string> out;
  std::string field;
  for (char c : text) {
    if (c == sep) { if (!field.empty()) out.push_back(field); field.clear(); }
    else if (!isspace((unsigned char)c)) field += c;
  }
  if (!field.empty()) out.push_back(field);
  return out;
}

inline std::string runCommand(const std::string& command) {
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) return "";
  std::string out;
  char buf[256];
  while (fgets(buf, sizeof buf, pipe)) out += buf;
  if (pclose(pipe) != 0) return "";
  return out;
}

// `visible_design_gpus`, 1.0.4: the UUID of every card the CUDA DRIVER
// shows this process -- which already honours CUDA_VISIBLE_DEVICES -- read
// through libcuda (cuDeviceGetUuid), formatted as nvidia-smi prints it.
// UUIDs, not ordinals, because an ordinal means a different card in every
// process whose CUDA_VISIBLE_DEVICES differs, and a worker handed "1"
// under a parent that could see only cards 2 and 3 got the wrong one.
//
// BC2_VISIBLE_GPUS (comma-separated) replaces the query, for the gate:
// a plan cannot be compared with anything if it depends on the box.
inline std::vector<std::string> visibleGpus() {
  if (const char* fake = getenv("BC2_VISIBLE_GPUS")) return splitList(fake, ',');
  std::vector<std::string> out;
  void* lib = dlopen("libcuda.so.1", RTLD_NOW);
  if (!lib) return out;
  typedef int (*InitF)(unsigned);
  typedef int (*CountF)(int*);
  typedef int (*GetF)(int*, int);
  typedef int (*UuidF)(char*, int);
  InitF cuInit_ = (InitF)dlsym(lib, "cuInit");
  CountF cuCount = (CountF)dlsym(lib, "cuDeviceGetCount");
  GetF cuGet = (GetF)dlsym(lib, "cuDeviceGet");
  UuidF cuUuid = (UuidF)dlsym(lib, "cuDeviceGetUuid_v2");
  if (!cuUuid) cuUuid = (UuidF)dlsym(lib, "cuDeviceGetUuid");
  int count = 0;
  if (!cuInit_ || !cuCount || !cuGet || !cuUuid || cuInit_(0) || cuCount(&count))
    return out;
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    int device = 0;
    char blob[16] = {0};
    if (cuGet(&device, ordinal) || cuUuid(blob, device)) return std::vector<std::string>();
    char hex[33];
    for (int i = 0; i < 16; ++i) snprintf(hex + 2 * i, 3, "%02x", (unsigned char)blob[i]);
    const std::string d(hex);
    out.push_back("GPU-" + d.substr(0, 8) + "-" + d.substr(8, 4) + "-" +
                  d.substr(12, 4) + "-" + d.substr(16, 4) + "-" + d.substr(20));
  }
  return out;
}

// `selected_design_gpus`: a requested card is an ORDINAL or a UUID, both
// resolved to the UUID, duplicates dropped in order; one the process
// cannot see is a refusal rather than a silent drop
inline bool selectedGpus(const std::string& gpu_ids,
                         std::vector<std::string>* out, std::string* missing) {
  const std::vector<std::string> visible = visibleGpus();
  if (gpu_ids.empty() || gpu_ids == "all") { *out = visible; return true; }
  for (const std::string& want : splitList(gpu_ids, ',')) {
    std::string uuid;
    for (size_t k = 0; k < visible.size(); ++k)
      if (want == std::to_string(k) || want == visible[k]) uuid = visible[k];
    if (uuid.empty()) {
      *missing += (missing->empty() ? "" : ", ") + std::string("'") + want + "'";
      continue;
    }
    if (std::find(out->begin(), out->end(), uuid) == out->end())
      out->push_back(uuid);
  }
  return missing->empty();
}

struct Memory { double free_gb = 0.0, total_gb = 0.0; };

inline std::map<std::string, Memory> gpuMemory() {
  std::map<std::string, Memory> out;
  for (const std::string& line :
       splitList(runCommand("nvidia-smi --query-gpu=uuid,memory.free,"
                            "memory.total --format=csv,noheader,nounits "
                            "2>/dev/null"), '\n')) {
    const std::vector<std::string> fields = splitList(line, ',');
    if (fields.size() != 3) continue;
    // 1.0.4: one unreadable card ("[N/A]") does not end the campaign --
    // its row is skipped, and the card reads as "no constraint"
    char* e1 = nullptr; char* e2 = nullptr;
    const double free_mib = strtod(fields[1].c_str(), &e1);
    const double total_mib = strtod(fields[2].c_str(), &e2);
    if (e1 == fields[1].c_str() || *e1 || e2 == fields[2].c_str() || *e2) continue;
    Memory m;
    m.free_gb = free_mib / 1024.0;
    m.total_gb = total_mib / 1024.0;
    out[fields[0]] = m;
  }
  return out;
}

inline double estimateDesignMemoryGb(long residue_count) {
  return DESIGN_MEMORY_SAFETY_FACTOR *
         (DESIGN_MODEL_RESIDENT_GB +
          DESIGN_ACTIVATION_BYTES_PER_RESIDUE_PAIR *
              (double)residue_count * (double)residue_count / 1e9);
}

// `design_worker_memory_fractions`: the share of the CARD each packed
// worker is allowed, which is what XLA_PYTHON_CLIENT_MEM_FRACTION caps.
// One worker gets nothing set at all -- the list is all zeros below two.
inline std::vector<double> memoryFractions(double free_gb, double total_gb,
                                           const std::vector<double>& need) {
  std::vector<double> out(need.size(), 0.0);
  if (total_gb == 0.0 || need.size() < 2) return out;
  double sum = 0.0;
  for (double g : need) sum += g;
  if (sum == 0.0) return out;
  const double packed_share = std::max(0.0, free_gb - GPU_MEMORY_HEADROOM_GB) / sum;
  for (size_t i = 0; i < need.size(); ++i) {
    const double v = packed_share * need[i] / total_gb;
    // Python's round(x, 3), which is round-half-to-EVEN
    double scaled = v * 1000.0;
    double floor_v = std::floor(scaled);
    const double frac = scaled - floor_v;
    if (frac > 0.5) floor_v += 1.0;
    else if (frac == 0.5 && std::fmod(floor_v, 2.0) != 0.0) floor_v += 1.0;
    out[i] = floor_v / 1000.0;
  }
  return out;
}

// `design_workers_per_gpu`: the requested count (or `auto`), then the
// memory, then the ceiling
inline int workersPerGpu(const std::string& requested_setting,
                         int ceiling_setting, bool trajectory_only,
                         double free_gb, long residue_count) {
  std::string requested = requested_setting;
  if (const char* env = getenv("BINDCRAFT_WORKERS_PER_GPU")) requested = env;
  for (char& c : requested) c = (char)tolower((unsigned char)c);
  int ceiling = ceiling_setting > 0 ? ceiling_setting : MAXIMUM_WORKERS_PER_GPU;
  if (const char* env = getenv("BINDCRAFT_MAX_WORKERS_PER_GPU"))
    ceiling = std::max(1, atoi(env));
  int packed;
  if (requested.empty() || requested == "auto")
    packed = (trajectory_only || !(residue_count && free_gb > 0.0))
                 ? TRAJECTORY_ONLY_WORKERS_PER_GPU
                 : AUTOMATIC_WORKERS_PER_GPU;
  else
    packed = atoi(requested.c_str());
  if (residue_count && free_gb > 0.0) {
    const double room = std::max(0.0, free_gb - GPU_MEMORY_HEADROOM_GB);
    packed = std::min(packed,
                      (int)(room / estimateDesignMemoryGb(residue_count)));
  }
  return std::max(1, std::min(ceiling, packed));
}

inline double availableHostMemoryGb() {
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return 0.0;
  char line[256];
  double gb = 0.0;
  while (fgets(line, sizeof line, f))
    if (strncmp(line, "MemAvailable:", 13) == 0) {
      gb = atof(line + 13) / 1e6;
      break;
    }
  fclose(f);
  return gb;
}

inline int hostMemoryWorkerCeiling(int gpu_count) {
  const double available = availableHostMemoryGb();
  if (available == 0.0 || gpu_count < 1) return MAXIMUM_WORKERS_PER_GPU;
  return std::max(1, (int)(available / HOST_MEMORY_PER_WORKER_GB) / gpu_count);
}

inline int paddedPredictionLength(int residues, int bucket) {
  if (bucket <= 0) bucket = 32;
  return ((residues + bucket - 1) / bucket) * bucket;
}

// `campaign_length_buckets`: the campaign's binder lengths, grouped by
// the padded length they compile at -- a scaffolded campaign has none,
// because its length is the scaffold's
inline std::vector<std::vector<int>> lengthBuckets(const std::vector<int>& lengths,
                                                   int bucket) {
  std::map<int, std::vector<int>> grouped;
  std::set<int> unique(lengths.begin(), lengths.end());
  for (int length : unique)
    grouped[paddedPredictionLength(length, bucket)].push_back(length);
  std::vector<std::vector<int>> out;
  for (const auto& kv : grouped) out.push_back(kv.second);
  return out;
}

// `assign_worker_length_buckets`: with more than one bucket, each worker
// takes a share, so a card is not compiling every length
inline void assignLengthBuckets(std::vector<Worker>* plan,
                                const std::vector<std::vector<int>>& buckets) {
  if (buckets.size() < 2 || plan->size() < 2) return;
  std::vector<std::vector<int>> widest_first = buckets;
  std::stable_sort(widest_first.begin(), widest_first.end(),
                   [](const std::vector<int>& a, const std::vector<int>& b) {
                     return a.size() > b.size();
                   });
  for (size_t w = 0; w < plan->size(); ++w) {
    std::vector<int> share;
    if (plan->size() >= buckets.size()) {
      for (int length : widest_first[w % buckets.size()]) share.push_back(length);
    } else {
      for (size_t b = w; b < buckets.size(); b += plan->size())
        for (int length : buckets[b]) share.push_back(length);
    }
    (*plan)[w].lengths = share;
  }
}

// ---- the host CPUs, sliced between packed workers (1.0.4) -------------
//
// `cgroup_cpu_quota` (a container's limit, cgroup v2 then v1) caps the
// affinity set; with at least HOST_CPUS_PER_WORKER = 2 a worker, each
// worker gets an even contiguous share, which it pins itself to through
// BINDCRAFT_WORKER_CPUS.  Fewer than that and nobody is pinned.
static const int HOST_CPUS_PER_WORKER = 2;

inline int cgroupCpuQuota() {
  if (FILE* f = fopen("/sys/fs/cgroup/cpu.max", "r")) {
    char q[64] = {0}; long period = 0;
    const int got = fscanf(f, "%63s %ld", q, &period);
    fclose(f);
    if (got == 2 && std::string(q) != "max" && period > 0)
      return std::max(1, (int)(atol(q) / period));
  }
  long quota = 0, period = 0;
  if (FILE* f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
    if (fscanf(f, "%ld", &quota) != 1) quota = 0;
    fclose(f);
  }
  if (FILE* f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
    if (fscanf(f, "%ld", &period) != 1) period = 0;
    fclose(f);
  }
  if (quota > 0 && period > 0) return std::max(1L, quota / period);
  return 0;
}

inline std::vector<int> allocatedHostCpus() {
  std::vector<int> cpus;
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) == 0)
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &set)) cpus.push_back(c);
  const int quota = cgroupCpuQuota();
  if (quota && quota < (int)cpus.size()) cpus.resize((size_t)quota);
  return cpus;
}

inline std::vector<std::vector<int>> workerCpuSlices(int workers) {
  std::vector<std::vector<int>> out((size_t)std::max(0, workers));
  const std::vector<int> cpus = allocatedHostCpus();
  if (workers < 2 || (int)cpus.size() < HOST_CPUS_PER_WORKER * workers) return out;
  const int share = (int)cpus.size() / workers;
  for (int w = 0; w < workers; ++w)
    out[(size_t)w].assign(cpus.begin() + w * share, cpus.begin() + (w + 1) * share);
  return out;
}

// the worker's side: BINDCRAFT_WORKER_CPUS, applied before anything runs
// (bc2 does it on `import bindcraft`)
inline void pinToWorkerCpus() {
  const char* list = getenv("BINDCRAFT_WORKER_CPUS");
  if (!list || !*list) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (const std::string& c : splitList(list, ',')) CPU_SET(atoi(c.c_str()), &set);
  sched_setaffinity(0, sizeof set, &set);
}

struct Plan {
  std::vector<Worker> plan;
  std::string refusal;                 // a GPU that is not visible
};

// `plan_design_workers`, with the campaign's inputs passed in rather than
// read out of a settings dict: the GPUs, how many fit on each, the
// trajectory budget, and the length buckets to divide.
// `design_residue_count` for ONE worker's share of the lengths: the
// target, plus its own longest binder padded, the whole thing padded
// again.  A worker that was given four of the campaign's lengths folds
// at ITS longest, not at the campaign's, and its memory estimate (and so
// its share of the card) is smaller for it.
inline int workerResidueCount(int target_residues, int copies,
                              const std::vector<int>& lengths, int bucket) {
  if (lengths.empty()) return 0;
  const int longest = *std::max_element(lengths.begin(), lengths.end());
  const int binder = paddedPredictionLength(longest, bucket) *
                     (copies > 0 ? copies : 1);
  return paddedPredictionLength(target_residues + binder, bucket);
}

inline Plan planWorkers(const std::string& gpu_ids,
                        const std::string& workers_per_gpu_setting,
                        int max_workers_per_gpu, int design_workers,
                        bool trajectory_only, long residue_count,
                        int trajectory_budget,
                        const std::vector<int>& binder_lengths,
                        int length_bucket,
                        const std::map<std::string, Memory>& memory,
                        int host_ceiling, int target_residues = 0,
                        int copies = 1) {
  Plan out;
  std::vector<std::string> gpus;
  std::string missing;
  std::string ids = gpu_ids;
  if (const char* env = getenv("BINDCRAFT_GPU_IDS")) ids = env;
  if (!selectedGpus(ids, &gpus, &missing)) {
    out.refusal = "requested GPUs [" + missing +
                  "] are not visible to this process";
    return out;
  }
  int limit = design_workers;
  if (const char* env = getenv("BINDCRAFT_DESIGN_WORKERS")) limit = atoi(env);
  for (size_t card = 0; card < gpus.size(); ++card) {
    const std::string& gpu = gpus[card];
    const auto it = memory.find(gpu);
    const double free_gb = it == memory.end() ? 0.0 : it->second.free_gb;
    const int packed = std::min(host_ceiling,
                                workersPerGpu(workers_per_gpu_setting,
                                              max_workers_per_gpu,
                                              trajectory_only, free_gb,
                                              residue_count));
    for (int i = 0; i < packed; ++i) {
      Worker w;
      w.gpu = gpu;
      w.card = (int)card;
      out.plan.push_back(w);
    }
  }

  int cap = 0;
  for (int one : {limit, trajectory_budget})
    if (one > 0) cap = cap ? std::min(cap, one) : one;
  if (cap && (int)out.plan.size() > cap) out.plan.resize((size_t)cap);
  assignLengthBuckets(&out.plan, lengthBuckets(binder_lengths, length_bucket));
  for (Worker& w : out.plan)
    w.residue_count = w.lengths.empty()
                          ? (int)residue_count
                          : workerResidueCount(target_residues, copies,
                                               w.lengths, length_bucket);
  // the fractions are per CARD, over the workers packed onto it
  for (const std::string& gpu : gpus) {
    std::vector<size_t> packed;
    std::vector<double> need;
    for (size_t i = 0; i < out.plan.size(); ++i)
      if (out.plan[i].gpu == gpu) {
        packed.push_back(i);
        need.push_back(estimateDesignMemoryGb(out.plan[i].residue_count));
      }
    const auto it = memory.find(gpu);
    const Memory m = it == memory.end() ? Memory() : it->second;
    const std::vector<double> fractions =
        memoryFractions(m.free_gb, m.total_gb, need);
    for (size_t k = 0; k < packed.size(); ++k)
      out.plan[packed[k]].memory_fraction = fractions[k];
  }
  // the CPU slices, over the plan AFTER the budget has trimmed it (bc2
  // assigns them inside its per-card fractions loop, over the whole plan,
  // so each card's pass writes the same slices)
  if (!out.plan.empty()) {
    const std::vector<std::vector<int>> slices = workerCpuSlices((int)out.plan.size());
    for (size_t i = 0; i < out.plan.size(); ++i) out.plan[i].cpus = slices[i];
  }
  return out;
}

inline bool runningAsWorker() { return getenv("BINDCRAFT_WORKER_ID") != nullptr; }

// `launch_design_workers`: one child per worker, each with its own card,
// its own id and its own log, started a RUNG at a time -- the first
// worker on every card, then the second on every card -- because
// `worker_launch_stagger` is there to keep N processes from compiling on
// one card at the same instant.
inline int launchWorkers(const std::vector<Worker>& plan,
                         const std::string& log_directory,
                         const std::vector<std::string>& command,
                         double stagger_seconds) {
  campout::makeDirs(log_directory);
  std::map<std::string, int> rung_of;
  std::vector<std::pair<int, size_t>> order;         // (rung, worker)
  for (size_t i = 0; i < plan.size(); ++i) {
    const int rung = rung_of[plan[i].gpu]++;
    order.push_back({rung, i});
  }
  std::sort(order.begin(), order.end());
  std::vector<pid_t> children;
  std::vector<std::string> logs;
  std::vector<int> ids;
  int launched_rung = 0;
  for (const auto& entry : order) {
    const Worker& w = plan[entry.second];
    if (entry.first != launched_rung && stagger_seconds > 0)
      usleep((useconds_t)(stagger_seconds * 1e6));
    launched_rung = entry.first;
    // 1.0.4: named after the LAUNCHING process, so two campaigns started
    // into one folder keep separate logs, and after the card's ordinal
    char name[96];
    snprintf(name, sizeof name, "/%ld_worker_%02d_gpu_%d.log", (long)getpid(),
             (int)entry.second, w.card);
    const std::string log_path = log_directory + name;
    // FLUSH BEFORE THE FORK.  A child inherits the parent's stdio buffer,
    // and `freopen` flushes what is in it to the OLD destination -- so
    // every line the parent had buffered came out again, once per child,
    // on the terminal.  The campaign printed its own header three times
    // before anything had run.
    fflush(nullptr);
    const pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
      setenv("CUDA_VISIBLE_DEVICES", w.gpu.c_str(), 1);
      setenv("BINDCRAFT_WORKER_ID", std::to_string(entry.second).c_str(), 1);
      setenv("BINDCRAFT_WORKER_COUNT", std::to_string(plan.size()).c_str(), 1);
      if (w.memory_fraction > 0) {
        char fraction[32];
        snprintf(fraction, sizeof fraction, "%g", w.memory_fraction);
        setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", fraction, 1);
      }
      if (!w.cpus.empty()) {
        std::string cpus;
        for (size_t i = 0; i < w.cpus.size(); ++i)
          cpus += (i ? "," : "") + std::to_string(w.cpus[i]);
        setenv("BINDCRAFT_WORKER_CPUS", cpus.c_str(), 1);
      }
      if (!w.lengths.empty()) {
        std::string lengths;
        for (size_t i = 0; i < w.lengths.size(); ++i)
          lengths += (i ? "," : "") + std::to_string(w.lengths[i]);
        setenv("BINDCRAFT_BINDER_LENGTHS", lengths.c_str(), 1);
      }
      // the worker's output is its log, and the log is what the campaign
      // keeps: a child writing to the shared terminal interleaves lines
      // mid-sentence with every other child
      FILE* log = freopen(log_path.c_str(), "a", stdout);
      if (log) dup2(fileno(stdout), fileno(stderr));
      std::vector<char*> argv;
      for (const std::string& part : command)
        argv.push_back(const_cast<char*>(part.c_str()));
      argv.push_back(nullptr);
      execvp(argv[0], argv.data());
      perror("execvp");
      _exit(127);
    }
    children.push_back(pid);
    logs.push_back(log_path);
    ids.push_back((int)entry.second);
    printf("worker=%d gpu=%d log=%s\n", (int)entry.second, w.card,
           log_path.c_str());
    fflush(stdout);
  }
  // 1.0.4's supervision: a worker that stops early is NAMED, with how it
  // stopped and where its log is, and the campaign says at the end how many
  // it closed short -- their trajectories are abandoned, which an exit code
  // alone does not say.  A worker killed by a signal makes the campaign's
  // status 128 + the signal, as a shell reports it.
  int failure = 0, lost = 0;
  for (size_t k = 0; k < children.size(); ++k) {
    int status = 0;
    waitpid(children[k], &status, 0);
    int code = 0;
    std::string how;
    if (WIFSIGNALED(status)) {
      const int sig = WTERMSIG(status);
      code = 128 + sig;
      // Python's signal.Signals(...).name: SIGKILL, not "Killed"
      static const std::pair<int, const char*> NAMES[] = {
          {SIGHUP, "SIGHUP"}, {SIGINT, "SIGINT"}, {SIGQUIT, "SIGQUIT"},
          {SIGILL, "SIGILL"}, {SIGABRT, "SIGABRT"}, {SIGBUS, "SIGBUS"},
          {SIGFPE, "SIGFPE"}, {SIGKILL, "SIGKILL"}, {SIGSEGV, "SIGSEGV"},
          {SIGPIPE, "SIGPIPE"}, {SIGALRM, "SIGALRM"}, {SIGTERM, "SIGTERM"},
          {SIGUSR1, "SIGUSR1"}, {SIGUSR2, "SIGUSR2"}};
      how = "killed by signal " + std::to_string(sig);
      for (const auto& nm : NAMES)
        if (nm.first == sig) how = std::string("killed by ") + nm.second;
    } else {
      code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
      how = "exited " + std::to_string(code);
    }
    if (code) {
      ++lost;
      printf("campaign lost design worker %d, %s | %s\n", ids[k], how.c_str(),
             logs[k].c_str());
      if (!failure) failure = code;
    }
  }
  if (lost)
    printf("campaign closed %d of %zu design worker(s) short: their "
           "trajectories are abandoned, and each is named above with its "
           "log\n", lost, children.size());
  fflush(stdout);
  return failure;
}

}  // namespace workers
