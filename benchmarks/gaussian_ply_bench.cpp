#include "gaussian/gaussian_optimizer.hpp"
#include "gaussian/gaussian_ply.hpp"

#include <torch/torch.h>
#include <torch/version.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MINK_BUILD_TYPE
#define MINK_BUILD_TYPE "unknown"
#endif
#ifndef MINK_TEST_DATA_DIR
#define MINK_TEST_DATA_DIR "tests/data"
#endif

using minkowski::gaussian::GaussianOptimizer;
using minkowski::gaussian::GaussianTensor;
using minkowski::gaussian::load_gaussian_ply;
using Clock = std::chrono::steady_clock;

namespace {

struct Stats {
  double median_ms = 0, p95_ms = 0, stddev_ms = 0;
};

struct Row {
  std::string name, device;
  Stats stats;
};

Stats summarize(std::vector<double> s) {
  std::sort(s.begin(), s.end());
  const size_t mid = s.size() / 2;
  const double median =
      s.size() % 2 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
  const size_t p95 = std::min(
      s.size() - 1, static_cast<size_t>(std::ceil(0.95 * s.size()) - 1));
  const double mean = std::accumulate(s.begin(), s.end(), 0.0) / s.size();
  double var = 0;
  for (double v : s)
    var += (v - mean) * (v - mean);
  return {median, s[p95], std::sqrt(var / s.size())};
}

void sync(bool cuda) {
#ifdef USE_CUDA
  if (cuda)
    cudaDeviceSynchronize();
#else
  (void)cuda;
#endif
}

// Wall-clock timing with device synchronization; fine for ms-scale work.
template <typename F> Stats measure(F &&f, bool cuda, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i)
    f();
  sync(cuda);
  std::vector<double> samples;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = Clock::now();
    f();
    sync(cuda);
    samples.push_back(
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  return summarize(std::move(samples));
}

std::string json_escape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '\\' || c == '"')
      out += '\\';
    out += c;
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::vector<std::string> inputs;
    std::string output = "gaussian_ply_benchmark_results.json";
    int iterations = 10;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--input" && i + 1 < argc)
        inputs.push_back(argv[++i]);
      else if (a == "--output" && i + 1 < argc)
        output = argv[++i];
      else if (a == "--iterations" && i + 1 < argc)
        iterations = std::max(1, std::stoi(argv[++i]));
      else {
        std::cerr << "usage: gaussian_ply_bench [--input file.ply ...] "
                     "[--output results.json] [--iterations N]\n";
        return 2;
      }
    }
    if (inputs.empty())
      for (const char *name : {"test.ply", "test_compressed.ply"})
        inputs.push_back(
            (std::filesystem::path(MINK_TEST_DATA_DIR) / name).string());

    std::vector<Row> rows;
    std::ostringstream files_json;
    files_json << std::fixed << std::setprecision(4);
    std::optional<GaussianTensor> first_scene;

    // The loader detects standard vs. compressed PLY from the header. Cold is
    // the first call (includes OS page-cache effects); warm is the median.
    for (size_t k = 0; k < inputs.size(); ++k) {
      const auto &input = inputs[k];
      const auto file_bytes = std::filesystem::file_size(input);
      const std::string label = std::filesystem::path(input).filename().string();
      const auto c0 = Clock::now();
      GaussianTensor loaded = load_gaussian_ply(input);
      const double cold_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - c0).count();
      rows.push_back({"load_cold:" + label, "cpu", {cold_ms, cold_ms, 0.0}});
      const auto load_stats = measure(
          [&] { auto g = load_gaussian_ply(input); }, false, 1, iterations);
      rows.push_back({"load_warm:" + label, "cpu", load_stats});
      files_json << (k ? ",\n" : "") << "    {\"path\": \""
                 << json_escape(input) << "\", \"file_bytes\": " << file_bytes
                 << ", \"gaussians\": " << loaded.size()
                 << ", \"load_throughput_mb_s\": "
                 << (file_bytes / 1.0e6) / (load_stats.median_ms / 1000.0)
                 << ", \"load_gaussians_per_s\": "
                 << loaded.size() / (load_stats.median_ms / 1000.0) << "}";
      if (k == 0)
        first_scene.emplace(std::move(loaded));
    }

    // Operations are benchmarked on the first input.
    GaussianTensor &scene = *first_scene;
    const int64_t n = scene.size();
    const auto lo = std::get<0>(scene.positions().min(0));
    const auto hi = std::get<0>(scene.positions().max(0));
    const auto q = at::tensor({0.9238795f, 0.f, 0.3826834f, 0.f});
    const auto t = at::tensor({0.5f, 1.0f, -1.0f});

    auto run_ops = [&](const GaussianTensor &g, const at::Tensor &qq,
                       const at::Tensor &tt, const at::Tensor &l,
                       const at::Tensor &h, const std::string &dev) {
      const bool cuda = dev == "cuda";
      rows.push_back({"covariances", dev,
                      measure([&] { g.covariances(); }, cuda, 3, iterations)});
      rows.push_back({"transform", dev,
                      measure([&] { g.transform(qq, tt); }, cuda, 3,
                              iterations)});
      rows.push_back({"filter_opacity", dev,
                      measure([&] { g.filter_opacity(0.5); }, cuda, 3,
                              iterations)});
      rows.push_back({"filter_bounds", dev,
                      measure([&] { g.filter_bounds(l, h); }, cuda, 3,
                              iterations)});
      GaussianTensor work = g.clone();
      GaussianOptimizer optimizer(work, 1e-3);
      const auto target = at::zeros_like(work.colors());
      rows.push_back(
          {"optimizer_step", dev,
           measure(
               [&] {
                 optimizer.step((work.colors() - target).pow(2).mean());
               },
               cuda, 2, std::max(3, iterations / 2))});
    };

    run_ops(scene, q, t, lo, hi, "cpu");

#ifdef USE_CUDA
    if (at::cuda::is_available()) {
      rows.push_back({"host_to_device", "cuda",
                      measure([&] { scene.to(at::kCUDA); }, true, 2,
                              iterations)});
      const auto gpu = scene.to(at::kCUDA);
      run_ops(gpu, q.cuda(), t.cuda(), lo.cuda(), hi.cuda(), "cuda");
    }
#endif

    std::ostringstream js;
    js << std::fixed << std::setprecision(4);
    js << "{\n  \"operations_input\": \"" << json_escape(inputs[0]) << "\",\n"
       << "  \"operations_gaussians\": " << n << ",\n"
       << "  \"build_type\": \"" << MINK_BUILD_TYPE << "\",\n"
       << "  \"torch_version\": \"" << TORCH_VERSION << "\",\n"
       << "  \"iterations\": " << iterations << ",\n"
       << "  \"files\": [\n" << files_json.str() << "\n  ],\n"
       << "  \"results\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto &r = rows[i];
      js << "    {\"name\": \"" << r.name << "\", \"device\": \"" << r.device
         << "\", \"median_ms\": " << r.stats.median_ms
         << ", \"p95_ms\": " << r.stats.p95_ms
         << ", \"stddev_ms\": " << r.stats.stddev_ms << "}"
         << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    js << "  ]\n}\n";

    std::ofstream(output) << js.str();
    std::cout << js.str() << "wrote " << output << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
