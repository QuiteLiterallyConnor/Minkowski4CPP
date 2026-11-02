// CPU phase benchmark; synthetic sorted keys exercise the internal splitting
// algorithm without LibTorch work. Whole-map performance is measured separately
// with adaptive_kmap_bench on valid adaptive tensors.
#include "MinkowskiEngine/adaptive/part_split.hpp"
#include "part_split_reference.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

int main(int argc, char **argv) {
  try {
    size_t n = 262144;
    int repeats = 25, warmup = 5, arm = 0;
    std::string layout = "clustered";
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::invalid_argument("Missing option value");
      const std::string key = argv[i], value = argv[i + 1];
      if (key == "--n") n = std::stoull(value);
      else if (key == "--reps") repeats = std::stoi(value);
      else if (key == "--warmup") warmup = std::stoi(value);
      else if (key == "--arm") arm = std::stoi(value);
      else if (key == "--layout") layout = value;
      else throw std::invalid_argument("Unknown option " + key);
    }
    if (n == 0 || repeats < 1 || warmup < 0 || (arm != 0 && arm != 1))
      throw std::invalid_argument("Invalid benchmark control");
    if (layout != "clustered" && layout != "unique" && layout != "single")
      throw std::invalid_argument("Unknown layout");
    std::vector<split_reference::Key> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const uint32_t level = layout == "single" ? 3 : uint32_t(i % 4);
      const int64_t cell = layout == "unique" ? int64_t(i * 3) - int64_t(n) :
                          int64_t((i / 4) % 64) * 6 - 192;
      keys.push_back({{cell * (int64_t(1) << level), int64_t(i / 256), int64_t(i)}, level});
    }
    std::sort(keys.begin(), keys.end(), split_reference::less);
    namespace split = minkowski::adaptive::detail;
    // Correctness outside the measurement, including exact per-part leaf order.
    const auto first = split_reference::starts(keys, 1);
    if (split::part_starts(keys, 1) != first ||
        split::part_leaves(first, keys) != split_reference::leaves(first, keys))
      throw std::runtime_error("Reference disagreement");
    std::vector<double> times;
    uint64_t checksum = 0;
    for (int r = -warmup; r < repeats; ++r) {
      const auto begin = std::chrono::steady_clock::now();
      auto starts = arm ? split::part_starts(keys, 1) : split_reference::starts(keys, 1);
      auto leaves = arm ? split::part_leaves(starts, keys) : split_reference::leaves(starts, keys);
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
      checksum += starts.size() + leaves.size() + leaves.front().size() + leaves.back().back();
      if (r >= 0) times.push_back(ms);
    }
    auto sorted = times;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 ? sorted[sorted.size() / 2] :
                          (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2;
    double peak_mb = 0;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS info{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)))
      peak_mb = double(info.PeakWorkingSetSize) / 1048576;
#else
    rusage info{};
    if (getrusage(RUSAGE_SELF, &info) == 0) peak_mb = double(info.ru_maxrss) / 1024;
#endif
    std::cout << std::setprecision(12) << "{\"arm\":" << arm << ",\"n\":" << n
              << ",\"layout\":\"" << layout << "\",\"median_ms\":" << median
              << ",\"min_ms\":" << sorted.front() << ",\"max_ms\":" << sorted.back()
              << ",\"peak_process_mib\":" << peak_mb << ",\"checksum\":" << checksum
              << ",\"samples_ms\":[";
    for (size_t i = 0; i < times.size(); ++i) std::cout << (i ? "," : "") << times[i];
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
