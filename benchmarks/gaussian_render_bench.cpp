// Latency, (tile, Gaussian) pair count and peak memory of the Gaussian
// rasterizer (render_gaussians) forward and backward, over a grid of splat
// counts and image sizes, on CPU, the CUDA kernels and the LibTorch-on-CUDA
// path. Every timed sample is written to the JSON output so that runs of two
// builds can be interleaved and aggregated outside the process.
//
//   gaussian_render_bench [--input scene.ply | --synthetic]
//       [--counts 5000,20000,0] (0 = all splats) [--sizes 128,512]
//       [--devices cpu,cuda,cuda-torch] [--warmup 2] [--iterations 5]
//       [--train] [--label name] [--output results.json]
//       [--dump-grads prefix] [--compare-grads prefix]
//
// --dump-grads saves the parameter gradients of one backward per
// configuration (prefix + "<device>_<count>_<size>.pt"); --compare-grads
// reports the max difference to such a dump from another build.

#include "gaussian/gaussian_model.hpp"
#include "gaussian/gaussian_ply.hpp"
#include "gaussian/gaussian_render.hpp"
#include "gaussian/gaussian_trainer.hpp"

#include <ATen/CPUGeneratorImpl.h>
#include <torch/torch.h>
#include <torch/version.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#endif

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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

using namespace minkowski::gaussian;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<std::string> split(const std::string &text) {
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ','))
    if (!item.empty())
      out.push_back(item);
  return out;
}

void sync(const torch::Device &device) {
#ifdef USE_CUDA
  if (device.is_cuda())
    cudaDeviceSynchronize();
#else
  (void)device;
#endif
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

// Process peak working set (CPU memory high-water mark; never decreases).
uint64_t process_peak_bytes() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
    return static_cast<uint64_t>(counters.PeakWorkingSetSize);
  return 0;
#else
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

void reset_cuda_peak(const torch::Device &device) {
#ifdef USE_CUDA
  if (device.is_cuda())
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
#else
  (void)device;
#endif
}

// {peak allocated, peak reserved} since the last reset.
std::pair<uint64_t, uint64_t> cuda_peak(const torch::Device &device) {
#ifdef USE_CUDA
  if (device.is_cuda()) {
    const auto stats =
        c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    // c10::CachingDeviceAllocator names StatType in LibTorch 2.5 (CUDA 12.4
    // profile) and re-exports c10::CachingAllocator's in 2.15.
    const auto all = static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
    return {static_cast<uint64_t>(stats.allocated_bytes[all].peak),
            static_cast<uint64_t>(stats.reserved_bytes[all].peak)};
  }
#else
  (void)device;
#endif
  return {0, 0};
}

// Pair count reported by the renderer when RenderOutput has tile_pairs;
// otherwise the square 3-sigma box implied by radii (the upstream bounds).
template <typename T>
auto reported_pairs(const T &out, int) -> decltype(out.tile_pairs, int64_t()) {
  return out.tile_pairs.defined() ? out.tile_pairs.template item<int64_t>()
                                  : -1;
}
template <typename T> int64_t reported_pairs(const T &, long) { return -1; }

int64_t square_box_pairs(const RenderOutput &out, int64_t tile, int64_t width,
                         int64_t height) {
  torch::NoGradGuard guard;
  const auto keep = out.visible;
  const auto r = out.radii.index({keep}).to(at::kDouble);
  const auto m = out.means2d.detach().index({keep}).to(at::kDouble);
  if (r.numel() == 0)
    return 0;
  const double t = static_cast<double>(tile);
  const int64_t cols = (width + tile - 1) / tile - 1;
  const int64_t rows = (height + tile - 1) / tile - 1;
  const auto x = m.select(1, 0), y = m.select(1, 1);
  const auto x0 = at::floor((x - r) / t).clamp(0, cols);
  const auto x1 = at::floor((x + r) / t).clamp(0, cols);
  const auto y0 = at::floor((y - r) / t).clamp(0, rows);
  const auto y1 = at::floor((y + r) / t).clamp(0, rows);
  return ((x1 - x0 + 1) * (y1 - y0 + 1)).sum().to(at::kLong).item<int64_t>();
}

struct Samples {
  std::vector<double> forward, backward, forward_backward, train;
};

struct Record {
  std::string device;
  int64_t count = 0, size = 0;
  int64_t visible = 0, pairs = 0;
  std::string pairs_source;
  double mean_alpha = 0.0;
  uint64_t cuda_peak_allocated_forward = 0, cuda_peak_allocated_backward = 0;
  uint64_t cuda_peak_reserved = 0, process_peak = 0;
  bool compared = false, grad_bit_identical = true;
  double grad_max_abs_diff = 0.0, grad_max_rel_diff = 0.0;
  Samples samples;
};

double median(std::vector<double> v) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  const size_t mid = v.size() / 2;
  return v.size() % 2 ? v[mid] : 0.5 * (v[mid - 1] + v[mid]);
}

void write_samples(std::ostream &out, const char *name,
                   const std::vector<double> &v) {
  out << "\"" << name << "_ms\": [";
  for (size_t i = 0; i < v.size(); ++i)
    out << (i ? ", " : "") << v[i];
  out << "], \"" << name << "_median_ms\": " << median(v);
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string input =
        (std::filesystem::path(MINK_TEST_DATA_DIR) / "test.ply").string();
    bool synthetic = false, train = false;
    std::vector<int64_t> counts{5000, 20000, 0};
    std::vector<int64_t> sizes{128, 512};
    std::vector<std::string> devices{"cpu", "cuda"};
    int warmup = 2, iterations = 5;
    std::string label = "run", output = "gaussian_render_bench.json";
    std::string dump_prefix, compare_prefix;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc)
          throw std::invalid_argument("missing value for " + a);
        return argv[++i];
      };
      if (a == "--input")
        input = next();
      else if (a == "--synthetic")
        synthetic = true;
      else if (a == "--counts") {
        counts.clear();
        for (const auto &c : split(next()))
          counts.push_back(std::stoll(c));
      } else if (a == "--sizes") {
        sizes.clear();
        for (const auto &s : split(next()))
          sizes.push_back(std::stoll(s));
      } else if (a == "--devices")
        devices = split(next());
      else if (a == "--warmup")
        warmup = std::max(0, std::stoi(next()));
      else if (a == "--iterations")
        iterations = std::max(1, std::stoi(next()));
      else if (a == "--train")
        train = true;
      else if (a == "--label")
        label = next();
      else if (a == "--output")
        output = next();
      else if (a == "--dump-grads")
        dump_prefix = next();
      else if (a == "--compare-grads")
        compare_prefix = next();
      else {
        std::cerr << "usage: gaussian_render_bench [--input scene.ply | "
                     "--synthetic] [--counts 5000,0] [--sizes 128,512] "
                     "[--devices cpu,cuda,cuda-torch] [--warmup N] "
                     "[--iterations N] [--train] [--label name] "
                     "[--output file.json]\n";
        return 2;
      }
    }

    // Scene: the PLY (all SH bands) or a synthetic cloud of 200k splats.
    GaussianModel full = [&] {
      if (!synthetic)
        return load_gaussian_model_ply(input);
      const auto gen = at::make_generator<at::CPUGeneratorImpl>(7);
      const int64_t n = 200000;
      return GaussianModel(at::randn({n, 3}, gen, at::kFloat),
                           at::randn({n, 3}, gen, at::kFloat) * 0.5,
                           at::zeros({n, 15, 3}),
                           at::randn({n, 1}, gen, at::kFloat),
                           at::randn({n, 3}, gen, at::kFloat) * 0.4 - 4.0,
                           at::randn({n, 4}, gen, at::kFloat));
    }();
    const auto center = full.xyz().mean(0);
    const double radius = full.xyz().std(0).max().item<double>() * 2.0;
    const std::array<double, 3> c{center[0].item<double>(),
                                  center[1].item<double>(),
                                  center[2].item<double>()};
    // Same camera as gaussian_ply_bench.
    auto make_camera = [&](int64_t size) {
      return Camera::look_at({c[0], c[1], c[2] - radius}, c, {0, -1, 0}, size,
                             size, 1.0);
    };
    // Deterministic random subsets (the PLY order is spatially coherent).
    const auto perm =
        at::randperm(full.size(), at::make_generator<at::CPUGeneratorImpl>(0),
                     at::kLong);

    std::vector<Record> records;
    for (const auto &device_name : devices) {
      const bool cuda = device_name.rfind("cuda", 0) == 0;
#ifdef USE_CUDA
      if (cuda && !at::cuda::is_available()) {
        std::cerr << "skipping " << device_name << ": CUDA not available\n";
        continue;
      }
#else
      if (cuda) {
        std::cerr << "skipping " << device_name << ": CPU-only build\n";
        continue;
      }
#endif
      const torch::Device device =
          cuda ? torch::Device(torch::kCUDA, 0) : torch::Device(torch::kCPU);
      RenderOptions options;
      options.use_cuda_kernels = device_name != "cuda-torch";
      options.background =
          at::tensor({0.1f, 0.2f, 0.3f}).to(device);
      for (const int64_t requested : counts) {
        const int64_t count =
            requested <= 0 ? full.size() : std::min(requested, full.size());
        const auto ids = std::get<0>(perm.slice(0, 0, count).sort());
        const auto base = full.select(ids).to(device);
        for (const int64_t size : sizes) {
          const auto camera = make_camera(size);
          const auto float_options =
              at::TensorOptions().dtype(at::kFloat).device(device);
          const auto gen = at::make_generator<at::CPUGeneratorImpl>(1);
          const auto w_image =
              at::randn({size, size, 3}, gen, at::kFloat).to(device);
          const auto w_depth =
              at::randn({size, size}, gen, at::kFloat).to(device) * 0.01;
          const auto w_alpha =
              at::randn({size, size}, gen, at::kFloat).to(device);

          Record record;
          record.device = device_name;
          record.count = count;
          record.size = size;
          sync(device);
          reset_cuda_peak(device);
          {
            torch::NoGradGuard guard;
            const auto out = render(base, camera, options);
            sync(device);
            record.cuda_peak_allocated_forward = cuda_peak(device).first;
            record.visible = out.visible.sum().item<int64_t>();
            record.mean_alpha = out.alpha.mean().item<double>();
            record.pairs = reported_pairs(out, 0);
            record.pairs_source = "renderer";
            if (record.pairs < 0) {
              record.pairs = square_box_pairs(
                  out, cuda && options.use_cuda_kernels ? 16 : options.tile_size,
                  size, size);
              record.pairs_source = "square_box_from_radii";
            }
          }
          for (int it = 0; it < warmup + iterations; ++it) {
            const bool timed = it >= warmup;
            {
              torch::NoGradGuard guard;
              sync(device);
              const auto t0 = Clock::now();
              const auto out = render(base, camera, options);
              sync(device);
              if (timed)
                record.samples.forward.push_back(elapsed_ms(t0));
            }
            auto model = base.clone();
            model.enable_gradients();
            sync(device);
            const auto t0 = Clock::now();
            const auto out = render(model, camera, options);
            const auto loss = (out.image * w_image).sum() +
                              (out.depth * w_depth).sum() +
                              (out.alpha * w_alpha).sum();
            sync(device);
            const auto t1 = Clock::now();
            loss.backward();
            sync(device);
            if (timed) {
              record.samples.backward.push_back(elapsed_ms(t1));
              record.samples.forward_backward.push_back(elapsed_ms(t0));
            }
          }
          record.cuda_peak_allocated_backward = cuda_peak(device).first;
          record.cuda_peak_reserved = cuda_peak(device).second;
          if (!dump_prefix.empty() || !compare_prefix.empty()) {
            // Gradients of one deterministic backward for cross-build checks.
            auto model = base.clone();
            model.enable_gradients();
            const auto out = render(model, camera, options);
            ((out.image * w_image).sum() + (out.depth * w_depth).sum() +
             (out.alpha * w_alpha).sum())
                .backward();
            std::vector<at::Tensor> grads;
            for (const auto &p : model.parameters())
              grads.push_back(p.grad().defined() ? p.grad().cpu()
                                                 : at::zeros_like(p).cpu());
            const std::string name = device_name + "_" +
                                     std::to_string(count) + "_" +
                                     std::to_string(size) + ".pt";
            if (!dump_prefix.empty())
              torch::save(grads, dump_prefix + name);
            if (!compare_prefix.empty()) {
              std::vector<at::Tensor> other;
              torch::load(other, compare_prefix + name);
              for (size_t k = 0; k < grads.size(); ++k) {
                if (grads[k].numel() == 0)
                  continue;
                const double diff =
                    (grads[k] - other[k]).abs().max().item<double>();
                const double scale = other[k].abs().max().item<double>();
                record.grad_max_abs_diff =
                    std::max(record.grad_max_abs_diff, diff);
                record.grad_max_rel_diff = std::max(
                    record.grad_max_rel_diff, diff / std::max(scale, 1e-30));
                record.grad_bit_identical =
                    record.grad_bit_identical && at::equal(grads[k], other[k]);
              }
              record.compared = true;
            }
          }
          if (train) {
            auto model = base.clone();
            GaussianTrainerConfig config;
            config.densify = false;
            config.scene_extent = radius;
            config.render = options;
            GaussianTrainer trainer(model, config);
            const auto target = at::full({size, size, 3}, 0.5f, float_options);
            for (int it = 0; it < warmup + iterations; ++it) {
              sync(device);
              const auto t0 = Clock::now();
              trainer.step(camera, target);
              sync(device);
              if (it >= warmup)
                record.samples.train.push_back(elapsed_ms(t0));
            }
          }
          record.process_peak = process_peak_bytes();
          std::cerr << label << " " << device_name << " n=" << count << " "
                    << size << "px: fwd "
                    << median(record.samples.forward) << " ms, bwd "
                    << median(record.samples.backward) << " ms, pairs "
                    << record.pairs << "\n";
          records.push_back(std::move(record));
        }
      }
    }

    std::ofstream file(output, std::ios::out | std::ios::trunc);
    if (!file)
      throw std::runtime_error("could not open " + output);
    file << std::setprecision(6);
    file << "{\n  \"label\": \"" << label << "\",\n  \"input\": \""
         << (synthetic ? std::string("synthetic")
                       : std::filesystem::path(input).filename().string())
         << "\",\n  \"scene_gaussians\": " << full.size()
         << ",\n  \"build_type\": \"" << MINK_BUILD_TYPE
         << "\",\n  \"torch_version\": \"" << TORCH_VERSION
         << "\",\n  \"warmup\": " << warmup
         << ",\n  \"iterations\": " << iterations << ",\n  \"records\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
      const auto &r = records[i];
      file << "    {\"device\": \"" << r.device << "\", \"count\": " << r.count
           << ", \"size\": " << r.size << ", \"visible\": " << r.visible
           << ", \"pairs\": " << r.pairs << ", \"pairs_source\": \""
           << r.pairs_source << "\", \"mean_alpha\": " << r.mean_alpha
           << ", \"cuda_peak_allocated_forward\": "
           << r.cuda_peak_allocated_forward
           << ", \"cuda_peak_allocated_backward\": "
           << r.cuda_peak_allocated_backward
           << ", \"cuda_peak_reserved\": " << r.cuda_peak_reserved
           << ", \"process_peak_working_set\": " << r.process_peak << ", ";
      if (r.compared)
        file << "\"grad_bit_identical\": "
             << (r.grad_bit_identical ? "true" : "false")
             << ", \"grad_max_abs_diff\": " << r.grad_max_abs_diff
             << ", \"grad_max_rel_diff\": " << r.grad_max_rel_diff << ", ";
      write_samples(file, "forward", r.samples.forward);
      file << ", ";
      write_samples(file, "backward", r.samples.backward);
      file << ", ";
      write_samples(file, "forward_backward", r.samples.forward_backward);
      file << ", ";
      write_samples(file, "train", r.samples.train);
      file << "}" << (i + 1 < records.size() ? "," : "") << "\n";
    }
    file << "  ]\n}\n";
    std::cerr << "wrote " << output << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "gaussian_render_bench: " << e.what() << "\n";
    return 1;
  }
}
