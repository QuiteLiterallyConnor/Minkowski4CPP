#include "gaussian/gaussian_optimizer.hpp"

#include <torch/torch.h>
#include <torch/version.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using minkowski::gaussian::GaussianOptimizer;
using minkowski::gaussian::GaussianTensor;
using Clock = std::chrono::steady_clock;

namespace {

struct Statistics {
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double stddev_ms = 0.0;
};

struct Record {
  int64_t count = 0;
  std::string device;
  Statistics covariance;
  Statistics transform;
  Statistics optimization;
  double initial_loss = 0.0;
  double final_loss = 0.0;
  int optimization_steps = 0;
  uint64_t estimated_peak_tensor_bytes = 0;
};

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

#ifdef USE_CUDA
void check_cuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
}
#endif

Statistics summarize(std::vector<double> samples) {
  check(!samples.empty(), "benchmark produced no samples");
  std::sort(samples.begin(), samples.end());
  const auto middle = samples.size() / 2;
  const double median =
      samples.size() % 2 == 0
          ? 0.5 * (samples[middle - 1] + samples[middle])
          : samples[middle];
  const auto p95_index =
      std::min(samples.size() - 1,
               static_cast<size_t>(std::ceil(0.95 * samples.size()) - 1));
  const double mean =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  double variance = 0.0;
  for (double value : samples)
    variance += (value - mean) * (value - mean);
  variance /= samples.size();
  return {median, samples[p95_index], std::sqrt(variance)};
}

template <typename Function>
Statistics measure(Function &&function, bool use_cuda, int warmup = 3,
                   int iterations = 15) {
  for (int i = 0; i < warmup; ++i)
    function();
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iterations));
#ifdef USE_CUDA
  if (use_cuda) {
    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    for (int i = 0; i < iterations; ++i) {
      check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
      function();
      check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
      check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");
      float elapsed = 0.0f;
      check_cuda(cudaEventElapsedTime(&elapsed, start, stop),
                 "cudaEventElapsedTime");
      samples.push_back(static_cast<double>(elapsed));
    }
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return summarize(std::move(samples));
  }
#else
  (void)use_cuda;
#endif
  for (int i = 0; i < iterations; ++i) {
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
  }
  return summarize(std::move(samples));
}

GaussianTensor make_gaussians(int64_t count, const at::TensorOptions &options) {
  auto positions = at::randn({count, 3}, options);
  auto rotations = at::zeros({count, 4}, options);
  rotations.select(1, 0).fill_(1.0);
  auto scales = at::rand({count, 3}, options) + 0.05;
  auto opacities = at::rand({count, 1}, options);
  auto colors = at::rand({count, 3}, options);
  return GaussianTensor(std::move(positions), std::move(rotations),
                        std::move(scales), std::move(opacities),
                        std::move(colors));
}

Record benchmark(int64_t count, const torch::Device &device) {
  const bool use_cuda = device.is_cuda();
  const auto options =
      at::TensorOptions().device(device).dtype(at::kFloat);
  auto gaussians = make_gaussians(count, options);
  auto rotation = at::tensor({1.0f, 0.0f, 0.0f, 0.0f}, options);
  auto translation = at::tensor({0.1f, -0.2f, 0.3f}, options);

  Record record;
  record.count = count;
  record.device = use_cuda ? "cuda:0" : "cpu";
  record.covariance = measure([&] { (void)gaussians.covariances(); }, use_cuda);
  record.transform =
      measure([&] { (void)gaussians.transform(rotation, translation); }, use_cuda);

  auto target = at::zeros_like(gaussians.colors());
  GaussianOptimizer optimizer(gaussians, 1e-2);
  record.initial_loss =
      GaussianOptimizer::rgb_reconstruction_loss(gaussians.colors(), target)
          .item<double>();
  record.optimization = measure(
      [&] {
        auto loss =
            GaussianOptimizer::objective(gaussians, gaussians.colors(), target,
                                         1e-4, 1e-4);
        optimizer.step(loss);
      },
      use_cuda, 2, 10);
  record.optimization_steps = 12;
  record.final_loss =
      GaussianOptimizer::rgb_reconstruction_loss(gaussians.colors(), target)
          .item<double>();
  record.estimated_peak_tensor_bytes =
      static_cast<uint64_t>(count) * 23 * sizeof(float);
  return record;
}

std::string json_escape(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    if (c == '"' || c == '\\')
      escaped.push_back('\\');
    if (c == '\n')
      escaped += "\\n";
    else if (c != '\r')
      escaped.push_back(c);
  }
  return escaped;
}

void write_statistics(std::ostream &out, const Statistics &stats,
                      int64_t gaussian_count) {
  out << "{\"median_ms\":" << stats.median_ms << ",\"p95_ms\":" << stats.p95_ms
      << ",\"stddev_ms\":" << stats.stddev_ms
      << ",\"gaussians_per_second\":"
      << (static_cast<double>(gaussian_count) * 1000.0 / stats.median_ms) << '}';
}

void write_json(std::ostream &out, const std::vector<Record> &records,
                bool cuda_available, const std::string &cuda_name) {
  const char *cpu_identifier = std::getenv("PROCESSOR_IDENTIFIER");
  out << "{\n  \"schema_version\": 1,\n"
      << "  \"torch_version\": \"" << TORCH_VERSION << "\",\n"
      << "  \"build_type\": \"" << MINK_BUILD_TYPE << "\",\n"
      << "  \"hardware_threads\": " << std::thread::hardware_concurrency()
      << ",\n"
      << "  \"cpu_identifier\": \""
      << json_escape(cpu_identifier == nullptr ? "unknown" : cpu_identifier)
      << "\",\n"
#ifdef _MSC_VER
      << "  \"compiler\": \"MSVC " << _MSC_VER << "\",\n"
#elif defined(__clang__)
      << "  \"compiler\": \"Clang " << __clang_major__ << '.' << __clang_minor__
      << "\",\n"
#elif defined(__GNUC__)
      << "  \"compiler\": \"GCC " << __GNUC__ << '.' << __GNUC_MINOR__ << "\",\n"
#else
      << "  \"compiler\": \"unknown\",\n"
#endif
      << "  \"cuda_available\": " << (cuda_available ? "true" : "false")
      << ",\n"
      << "  \"cuda_device\": \"" << json_escape(cuda_name) << "\",\n"
#ifdef USE_CUDA
      << "  \"cuda_runtime_version\": " << CUDART_VERSION << ",\n"
#else
      << "  \"cuda_runtime_version\": null,\n"
#endif
      << "  \"timing_note\": \"CUDA operations use events on the current "
         "stream; CPU uses steady_clock. Timings include output allocation; "
         "estimated_peak_tensor_bytes covers input fields and largest output "
         "tensor only.\",\n"
      << "  \"records\": [\n";
  for (size_t i = 0; i < records.size(); ++i) {
    const auto &record = records[i];
    out << "    {\"count\":" << record.count << ",\"device\":\""
        << record.device << "\",\"estimated_peak_tensor_bytes\":"
        << record.estimated_peak_tensor_bytes << ",\"covariance\":";
    write_statistics(out, record.covariance, record.count);
    out << ",\"transform\":";
    write_statistics(out, record.transform, record.count);
    out << ",\"optimization\":";
    write_statistics(out, record.optimization, record.count);
    out << ",\"optimization_initial_loss\":" << record.initial_loss
        << ",\"optimization_final_loss\":" << record.final_loss
        << ",\"optimization_steps\":" << record.optimization_steps << '}';
    if (i + 1 != records.size())
      out << ',';
    out << '\n';
  }
  out << "  ]\n}\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::vector<int64_t> counts{1000, 10000, 100000};
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--count" && i + 1 < argc) {
        counts = {std::stoll(argv[++i])};
      } else if (arg == "--output" && i + 1 < argc) {
        output_path = argv[++i];
      } else {
        throw std::invalid_argument(
            "usage: gaussian_bench [--count N] [--output path.json]");
      }
    }
    for (const auto count : counts)
      check(count > 0, "Gaussian count must be positive");

    at::manual_seed(7);
    std::vector<Record> records;
    for (const auto count : counts)
      records.push_back(benchmark(count, torch::Device(torch::kCPU)));

    bool cuda_available = false;
    std::string cuda_name;
#ifdef USE_CUDA
    cuda_available = at::cuda::is_available();
    if (cuda_available) {
      cudaDeviceProp properties{};
      check_cuda(cudaGetDeviceProperties(&properties, 0),
                 "cudaGetDeviceProperties");
      cuda_name = properties.name;
      for (const auto count : counts)
        records.push_back(
            benchmark(count, torch::Device(torch::kCUDA, 0)));
    }
#endif

    std::ofstream file;
    std::ostream *out = &std::cout;
    if (!output_path.empty()) {
      file.open(output_path, std::ios::out | std::ios::trunc);
      check(file.is_open(), "could not open output file: " + output_path);
      out = &file;
    }
    *out << std::fixed << std::setprecision(6);
    write_json(*out, records, cuda_available, cuda_name);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gaussian_bench: " << error.what() << '\n';
    return 1;
  }
}
