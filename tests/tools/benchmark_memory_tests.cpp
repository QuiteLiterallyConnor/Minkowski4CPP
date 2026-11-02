#include "benchmarks/benchmark_memory.hpp"
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

int main() {
  const double before = mink_bench::peak_rss_mb();
  if (before == -1.0) {
    std::cout << "Process peak RSS unavailable on this platform\n";
    return 77;
  }
  auto require = [](bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
  };
  require(std::isfinite(before) && before > 0 && before < 8192,
          "Invalid initial RSS or incorrect RSS units");
  constexpr size_t bytes = 64 * 1024 * 1024;
  auto allocation = std::make_unique<unsigned char[]>(bytes);
  volatile unsigned char *touch = allocation.get();
  for (size_t i = 0; i < bytes; i += 4096) touch[i] = static_cast<unsigned char>(i / 4096);
  const double allocated = mink_bench::peak_rss_mb();
  require(std::isfinite(allocated) && allocated >= before + 32 && allocated < before + 128,
          "Touching 64 MiB did not produce a plausible high-water RSS increase");
  allocation.reset();
  const double released = mink_bench::peak_rss_mb();
  require(released >= allocated, "Process peak RSS must not decrease after deallocation");
  std::cout << "peak RSS MiB before=" << before << " allocated=" << allocated
            << " after_free=" << released << " source=" << mink_bench::peak_rss_source()
            << "; lifetime high-water and unit checks passed\n";
}
