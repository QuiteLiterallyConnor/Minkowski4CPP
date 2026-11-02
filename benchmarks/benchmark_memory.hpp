#pragma once

// Process high-water resident memory. This is a lifetime peak, not a resettable
// per-phase allocation counter. Keep it independent of LibTorch for fast tests.
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace mink_bench {
inline double peak_rss_mb() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
#elif defined(__linux__) || defined(__APPLE__)
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
    // Darwin reports ru_maxrss in bytes; Linux reports KiB.
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
  }
#endif
  return -1.0; // unavailable, never a measured zero
}

inline const char *peak_rss_source() {
#ifdef _WIN32
  return "GetProcessMemoryInfo.PeakWorkingSetSize";
#elif defined(__linux__) || defined(__APPLE__)
  return "getrusage(RUSAGE_SELF).ru_maxrss";
#else
  return "unavailable";
#endif
}
} // namespace mink_bench
