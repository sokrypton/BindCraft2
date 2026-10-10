// A CUPTI PROFILE OF THE STACK, borrowed from sokrypton/localfold's
// `opus-55-opt` branch (cuda/af3/src/profile.cuh), which is:
//
//   ----------------------------------------------------------------------
//   "THE BEER-WARE LICENSE" (Revision 42):
//
//   Sergey Ovchinnikov <https://github.com/sokrypton> wrote this. As long as
//   you retain this notice you can do whatever you want with this stuff. If
//   we meet some day, and you think this stuff is worth it, you can buy me a
//   beer in return.
//   ----------------------------------------------------------------------
//
// WHY THIS IS WORTH HAVING HERE.  Every per-kernel number in this directory
// comes from BLK_PROF, which brackets each step with events and SYNCHRONISES
// to read them -- so it serialises what the real run overlaps, and its own
// note says the total it reports is larger than the real one.  It also only
// sees what someone wrapped: BLK_PROF=1 accounts 603.4 ms of a 606 ms
// backward while BLK_PROF=2 accounts 511.6, and the 92 ms between them is
// kernels inside modules that nobody bracketed.
//
// CUPTI's activity API records every kernel's start and end ON THE DEVICE,
// so it needs no brackets, perturbs nothing, and misses nothing.  It also
// answers the question this directory has only ever answered indirectly --
// IS THE DEVICE EVER IDLE -- by sorting the stretches between kernels.
//
// This box has no ncu and no nsys, which is why the question kept being
// answered indirectly.  It does have libcupti.so.12.
//
// NE_CUPTI=1 turns it on; NE_CUPTI_GAPS=<n> also prints the n longest idle
// stretches with the kernels either side.  CUPTI is dlopen'd, never linked,
// so a build without it is unaffected.
#pragma once
#include <cxxabi.h>
#include <dlfcn.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cupti.h>
namespace cuprof {
inline CUptiResult (*getNextRecord)(uint8_t*, size_t, CUpti_Activity**) = nullptr;
inline CUptiResult (*registerCallbacks)(CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc) = nullptr;
inline CUptiResult (*enable)(CUpti_ActivityKind) = nullptr;
inline CUptiResult (*flushAll)(uint32_t) = nullptr;
inline std::map<std::string, std::pair<double, int>> byName;   // name -> (ns, calls)
inline uint64_t first = UINT64_MAX, last = 0; inline double busy = 0;
inline bool on = false;
// LOCALFOLD_PROF_GAPS=<n>: every kernel's span kept too, and the n longest stretches the device sat idle printed with
// the kernels either side - where a wall clock's time goes that no kernel's does (host work, a synchronisation)
struct Span { uint64_t start, end; std::string name; };
inline std::vector<Span> spans;
inline const int GAPS = getenv("NE_CUPTI_GAPS") ? atoi(getenv("NE_CUPTI_GAPS")) : 0;
inline void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxRecords) {
  *size = 8 << 20; *buffer = (uint8_t*)aligned_alloc(8, *size); *maxRecords = 0;
}
inline std::string shortName(const char* mangled) {
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> d(abi::__cxa_demangle(mangled, nullptr, nullptr, &status), std::free);
  std::string n = status == 0 ? d.get() : mangled;
  size_t p = n.find('(');            // drop the argument list; keep template arguments
  return p == std::string::npos ? n : n.substr(0, p);
}
inline void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  CUpti_Activity* record = nullptr;
  while (getNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = (CUpti_ActivityKernel5*)record;
      if (!on) continue;
      auto& e = byName[shortName(k->name)];
      e.first += (double)(k->end - k->start); e.second += 1;
      busy += (double)(k->end - k->start);
      if (GAPS) spans.push_back({k->start, k->end, shortName(k->name)});
      first = std::min<uint64_t>(first, k->start); last = std::max<uint64_t>(last, k->end);
    }
  }
  free(buffer);
}
inline void init() {
  // the CUPTI of the toolkit this was compiled with (12 here, 13 on Colab): its activity records are read through that
  // toolkit's own struct layouts, which another major version need not keep
  const std::string name = "libcupti.so." + std::to_string(CUDART_VERSION / 1000);
  void* lib = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!lib) lib = dlopen("libcupti.so", RTLD_NOW | RTLD_GLOBAL);
  if (!lib) { fprintf(stderr, "NE_CUPTI needs CUPTI (%s, the CUDA toolkit's extras/CUPTI/lib64): %s\n", name.c_str(), dlerror()); exit(1); }
  getNextRecord = (decltype(getNextRecord))dlsym(lib, "cuptiActivityGetNextRecord");
  registerCallbacks = (decltype(registerCallbacks))dlsym(lib, "cuptiActivityRegisterCallbacks");
  enable = (decltype(enable))dlsym(lib, "cuptiActivityEnable");
  flushAll = (decltype(flushAll))dlsym(lib, "cuptiActivityFlushAll");
  if (!getNextRecord || !registerCallbacks || !enable || !flushAll) { fprintf(stderr, "NE_CUPTI: this CUPTI lacks the activity API\n"); exit(1); }
  registerCallbacks(bufferRequested, bufferCompleted);
  enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}
inline void start() { flushAll(0); byName.clear(); spans.clear(); busy = 0; first = UINT64_MAX; last = 0; on = true; }
inline void stop(int top = 25) {
  CU(cudaDeviceSynchronize());
  flushAll(0);
  on = false;
  std::vector<std::pair<std::string, std::pair<double, int>>> rows(byName.begin(), byName.end());
  std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.first > b.second.first; });
  double span = last > first ? (double)(last - first) : 1;
  printf("GPU kernel time %.1f ms over a %.1f ms span (%.0f%% busy):\n", busy / 1e6, span / 1e6, 100 * busy / span);
  for (int i = 0; i < (int)rows.size() && i < top; ++i)
    printf("  %8.2f ms %5.1f%% %7d  %s\n", rows[i].second.first / 1e6, 100 * rows[i].second.first / busy,
           rows[i].second.second, rows[i].first.substr(0, 90).c_str());
  if (GAPS && spans.size() > 1) {
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.start < b.start; });
    struct Gap { double ns; size_t at; };
    std::vector<Gap> gaps; uint64_t reach = spans[0].end; double idle = 0;
    for (size_t i = 1; i < spans.size(); ++i) {
      if (spans[i].start > reach) { gaps.push_back({(double)(spans[i].start - reach), i}); idle += spans[i].start - reach; }
      reach = std::max(reach, spans[i].end);
    }
    std::sort(gaps.begin(), gaps.end(), [](const Gap& a, const Gap& b) { return a.ns > b.ns; });
    double small = 0; for (auto& g : gaps) if (g.ns < 100000) small += g.ns;
    printf("idle %.1f ms in %zu gaps (%.1f ms of them under 0.1 ms each); the longest:\n", idle / 1e6, gaps.size(), small / 1e6);
    for (int i = 0; i < (int)gaps.size() && i < GAPS; ++i)
      printf("  %7.2f ms at +%7.1f ms  after %s  before %s\n", gaps[i].ns / 1e6, (spans[gaps[i].at].start - first) / 1e6,
             spans[gaps[i].at - 1].name.substr(0, 50).c_str(), spans[gaps[i].at].name.substr(0, 50).c_str());
  }
}
}  // namespace cuprof
