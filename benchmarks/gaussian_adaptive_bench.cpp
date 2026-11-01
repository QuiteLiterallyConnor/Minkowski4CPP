// Conversion cost of the Gaussian <-> adaptive bridge (gaussian_adaptive.hpp)
// versus the number of Gaussians.
//
//   gaussian_adaptive_bench [--device cpu|cuda|both] [--counts 1000,10000,...]
//                           [--repeats 5] [--warmup 2] [--box-max-n 10000]
//                           [--splat center|box|both] [--output results.csv]
//
// Scene: N Gaussians, positions uniform in [0, 64)^3, log-uniform axis scales in
// [0.02, 2], random rotations, float32, seed 1234; config base 64, max_depth 8.
// Timed (median, min, max over --repeats after --warmup):
//   to_adaptive      gaussians_to_adaptive (no grad)
//   to_adaptive_bwd  gaussians_to_adaptive + backward of sum(features * W) to
//                    opacities/colors
//   to_gaussians     adaptive_to_gaussians
//   find_leaves      find_leaves for the N Gaussian means
// for Center and BoxIntegral splats. CUDA timings synchronize before reading
// the clock. Peak memory: CUDA caching-allocator max allocated (reset per row);
// process peak working set (Windows) at the end of each row.
#include "gaussian/gaussian_adaptive.hpp"

#include <torch/torch.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace minkowski;
using namespace minkowski::gaussian;
using Clock = std::chrono::steady_clock;

namespace {

struct Stat {
  double median = 0, min = 0, max = 0;
};

Stat summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  Stat s;
  s.median = v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
  s.min = v.front();
  s.max = v.back();
  return s;
}

void sync(const torch::Device &d) {
  if (d.is_cuda())
    torch::cuda::synchronize();
}

double peak_rss_mb() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
#endif
  return -1.0;
}

void reset_cuda_peak() {
#ifdef USE_CUDA
  if (torch::cuda::is_available())
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
#endif
}

double cuda_peak_mb() {
#ifdef USE_CUDA
  if (torch::cuda::is_available()) {
    auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    return static_cast<double>(
               st.allocated_bytes[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)]
                   .peak) /
           (1024.0 * 1024.0);
  }
#endif
  return -1.0;
}

GaussianTensor scene(int64_t n, const torch::Device &dev) {
  torch::manual_seed(1234);
  const auto o = torch::TensorOptions().dtype(torch::kFloat32);
  auto pos = torch::rand({n, 3}, o) * 64.0;
  auto logs = torch::rand({n, 3}, o) * std::log(100.0) + std::log(0.02);
  auto rot = torch::randn({n, 4}, o);
  rot = rot / rot.norm(2, 1, true);
  auto op = torch::rand({n, 1}, o) * 0.9 + 0.05;
  auto col = torch::rand({n, 3}, o);
  return GaussianTensor(pos, rot, logs.exp(), op, col).to(dev);
}

Stat time_it(const torch::Device &dev, int warmup, int repeats, const std::function<void()> &f) {
  for (int i = 0; i < warmup; ++i) {
    f();
    sync(dev);
  }
  std::vector<double> ms;
  for (int i = 0; i < repeats; ++i) {
    sync(dev);
    const auto t0 = Clock::now();
    f();
    sync(dev);
    ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  return summarize(ms);
}

std::vector<int64_t> parse_counts(const std::string &s) {
  std::vector<int64_t> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    v.push_back(std::stoll(item));
  return v;
}

} // namespace

int main(int argc, char **argv) {
  std::string device = "both", output = "gaussian_adaptive_bench.csv";
  // 1e6 is opt-in: with mixed scales a coarse Gaussian gets one entry per fine
  // leaf it covers (1e6 Gaussians -> ~1.2e8 entries, > 100 s and ~8 GB on CPU).
  std::vector<int64_t> counts{1000, 10000, 100000};
  int repeats = 5, warmup = 2;
  int64_t box_max_n = 10000;
  std::string splats = "both";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--device")
      device = v;
    else if (a == "--counts")
      counts = parse_counts(v);
    else if (a == "--repeats")
      repeats = std::stoi(v);
    else if (a == "--warmup")
      warmup = std::stoi(v);
    else if (a == "--output")
      output = v;
    else if (a == "--box-max-n")
      box_max_n = std::stoll(v);
    else if (a == "--splat")
      splats = v;
  }
  std::vector<torch::Device> devices;
  if (device == "cpu" || device == "both")
    devices.emplace_back(torch::kCPU);
  if ((device == "cuda" || device == "both") && torch::cuda::is_available())
    devices.emplace_back(torch::kCUDA);

  adaptive::AdaptiveVoxelConfig cfg;
  cfg.base_voxel_size = {64.0, 64.0, 64.0};
  cfg.max_depth = 8;

  bool fresh = true;
  {
    std::ifstream existing(output);
    fresh = !existing || existing.peek() == std::ifstream::traits_type::eof();
  }
  std::ofstream csv(output, std::ios::app); // one file across separate CPU / CUDA runs
  const std::string header =
      "device,splat,n,leaves,entries,op,median_ms,min_ms,max_ms,gaussians_per_s,"
      "cuda_peak_alloc_mb,peak_rss_mb";
  if (fresh)
    csv << header << "\n";
  std::cout << header << "\n";
  for (const auto &dev : devices) {
    for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
      for (int64_t n : counts) {
        if (splat == GaussianSplat::BoxIntegral && (n > box_max_n || splats == "center"))
          continue; // tens of entries per Gaussian; keep each run short
        if (splat == GaussianSplat::Center && splats == "box")
          continue;
        auto g = scene(n, dev);
        GaussianToAdaptiveOptions o;
        o.splat = splat;
        GaussianAdaptiveMap map;
        auto t = gaussians_to_adaptive(g, cfg, o, &map);
        const int64_t leaves = t.size(), entries = map.gaussian_index.numel();
        torch::manual_seed(7);
        auto W = torch::randn(t.features().sizes(), t.features().options());
        struct Op {
          const char *name;
          std::function<void()> f;
        };
        std::vector<Op> ops{
            {"to_adaptive",
             [&] {
               torch::NoGradGuard ng;
               auto r = gaussians_to_adaptive(g, cfg, o);
             }},
            {"to_adaptive_bwd",
             [&] {
               auto op = g.opacities().detach().clone().requires_grad_(true);
               auto col = g.colors().detach().clone().requires_grad_(true);
               GaussianTensor gg(g.positions(), g.rotations(), g.scales(), op, col);
               auto r = gaussians_to_adaptive(gg, cfg, o);
               (r.features() * W).sum().backward();
             }},
            {"to_gaussians",
             [&] {
               torch::NoGradGuard ng;
               auto b = adaptive_to_gaussians(t);
             }},
            {"find_leaves", [&] { auto l = find_leaves(t, g.positions()); }},
        };
        for (auto &op : ops) {
          reset_cuda_peak();
          const Stat s = time_it(dev, warmup, repeats, op.f);
          std::ostringstream row;
          row << (dev.is_cuda() ? "cuda" : "cpu") << ","
              << (splat == GaussianSplat::Center ? "center" : "box") << "," << n << ","
              << leaves << "," << entries << "," << op.name << "," << std::fixed
              << std::setprecision(3) << s.median << "," << s.min << "," << s.max << ","
              << std::setprecision(0) << (1000.0 * static_cast<double>(n) / s.median) << ","
              << std::setprecision(1) << cuda_peak_mb() << "," << peak_rss_mb();
          csv << row.str() << std::endl; // flushed: partial runs keep their rows
          std::cout << row.str() << std::endl;
        }
      }
    }
  }
  return 0;
}
