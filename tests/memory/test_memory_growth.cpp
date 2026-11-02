// Memory-growth (leak) tests: run representative forward + backward steps many times and check that process
// memory and CUDA allocated memory stop growing after a warm-up.
//
// Each case runs MINK_MEMORY_WARMUP (default 30) warm-up iterations, then MINK_MEMORY_ITERS (default 120)
// measured iterations, sampling after every iteration (CUDA synchronized first):
//   - CPU: process private bytes and working set (Windows GetProcessMemoryInfo; Linux /proc/self/statm).
//   - CUDA: c10 caching-allocator allocated bytes (current) of the device.
// Growth = median of the last kWindow samples - median of the first kWindow measured samples. Tolerances:
//   - CUDA allocated bytes      <= 16 KiB  (steady state is exact: every tensor of an iteration is freed and the
//                                           allocator rounds blocks to >= 512 B, so a leak of one tensor per
//                                           iteration over ~100 iterations is >= 50 KiB)
//   - CPU private bytes         <= 4 MiB   (heap/allocator noise; a leak >= ~40 KiB per iteration is detected
//                                           at the default count, >= ~4 KiB with MINK_MEMORY_ITERS=1000)
//   - CPU working set           <= 16 MiB  (the OS may trim/reload pages, so it is checked more loosely)
// memory_growth_detects_planted_*_leak plant a leak per iteration and require the detector to flag it, so the
// tolerances are known to be tight enough. Each case prints one "MEMORY ..." line with the raw numbers.
// The CUDA caching allocator must be on (PYTORCH_NO_CUDA_MEMORY_CACHING unset): its stats are the measurement.
#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/minkowski.hpp"
#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_render.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

namespace {

namespace ad = minkowski::adaptive;
namespace gs = minkowski::gaussian;

constexpr int64_t kWindow = 10;
constexpr double kCudaTolBytes = 16.0 * 1024;
constexpr double kPrivateTolBytes = 4.0 * 1024 * 1024;
constexpr double kWorkingSetTolBytes = 16.0 * 1024 * 1024;

int64_t env_int(const char *name, int64_t fallback) {
    const char *v = std::getenv(name);
    return v && *v ? std::atoll(v) : fallback;
}

struct Sample {
    double private_bytes = 0, working_set = 0, cuda_allocated = 0;
};

Sample sample(bool cuda) {
    Sample s;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&pmc),
                             sizeof(pmc))) {
        s.private_bytes = double(pmc.PrivateUsage);
        s.working_set = double(pmc.WorkingSetSize);
    }
#else
    long pages_total = 0, pages_resident = 0;
    if (FILE *f = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(f, "%ld %ld", &pages_total, &pages_resident) != 2) pages_resident = 0;
        std::fclose(f);
    }
    s.working_set = double(pages_resident) * double(sysconf(_SC_PAGESIZE));
    s.private_bytes = s.working_set;
#endif
#ifdef USE_CUDA
    if (cuda) {
        const auto dev = c10::cuda::current_device();
        // Index 0 = StatType::AGGREGATE (literal: the enum's namespace differs across LibTorch versions).
        s.cuda_allocated =
            double(c10::cuda::CUDACachingAllocator::getDeviceStats(dev).allocated_bytes[0].current);
    }
#else
    (void)cuda;
#endif
    return s;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

struct Growth {
    double private_bytes = 0, working_set = 0, cuda_allocated = 0;
    double private_start = 0, working_set_start = 0, cuda_start = 0;
    bool exceeds(bool cuda) const {
        return private_bytes > kPrivateTolBytes || working_set > kWorkingSetTolBytes ||
               (cuda && cuda_allocated > kCudaTolBytes);
    }
};

// Runs `step` warm-up + measured times and returns the growth between the first and last sample windows.
Growth measure(const std::string &name, bool cuda, const std::function<void()> &step) {
    const int64_t warmup = std::max<int64_t>(env_int("MINK_MEMORY_WARMUP", 30), 0);
    const int64_t iters = std::max<int64_t>(env_int("MINK_MEMORY_ITERS", 120), 3 * kWindow);
    auto sync = [&] {
#ifdef USE_CUDA
        if (cuda) c10::cuda::device_synchronize();
#endif
    };
    for (int64_t i = 0; i < warmup; ++i) step();
    sync();
    std::vector<Sample> samples;
    samples.reserve(size_t(iters));
    for (int64_t i = 0; i < iters; ++i) {
        step();
        sync();
        samples.push_back(sample(cuda));
    }
    auto window = [&](bool last, double Sample::*field) {
        std::vector<double> v;
        const size_t n = samples.size(), w = size_t(kWindow);
        for (size_t i = last ? n - w : 0; i < (last ? n : w); ++i) v.push_back(samples[i].*field);
        return median(v);
    };
    Growth g;
    g.private_start = window(false, &Sample::private_bytes);
    g.working_set_start = window(false, &Sample::working_set);
    g.cuda_start = window(false, &Sample::cuda_allocated);
    g.private_bytes = window(true, &Sample::private_bytes) - g.private_start;
    g.working_set = window(true, &Sample::working_set) - g.working_set_start;
    g.cuda_allocated = window(true, &Sample::cuda_allocated) - g.cuda_start;
    std::cout << "MEMORY case=" << name << " device=" << (cuda ? "cuda" : "cpu") << " warmup=" << warmup
              << " iters=" << iters << " private_start=" << g.private_start
              << " private_growth=" << g.private_bytes << " ws_start=" << g.working_set_start
              << " ws_growth=" << g.working_set << " cuda_alloc_start=" << g.cuda_start
              << " cuda_alloc_growth=" << g.cuda_allocated << "\n";
    return g;
}

void expect_no_growth(const std::string &name, bool cuda, const std::function<void()> &step) {
    const Growth g = measure(name, cuda, step);
    if (g.exceeds(cuda))
        FAIL_AT(name << " memory grows after warm-up: private +" << g.private_bytes << " B (tol "
                     << kPrivateTolBytes << "), working set +" << g.working_set << " B (tol "
                     << kWorkingSetTolBytes << "), CUDA allocated +" << g.cuda_allocated << " B (tol "
                     << kCudaTolBytes << ")");
}

torch::Device require_device(bool cuda) {
    if (!cuda) return torch::kCPU;
#ifdef USE_CUDA
    if (at::cuda::is_available()) return torch::kCUDA;
    throw mink_test::Skip("CUDA not available");
#else
    throw mink_test::Skip("built without CUDA");
#endif
}

// ---- workloads ------------------------------------------------------------------------------------------

// Uniform sparse convolution: a new SparseTensor (and coordinate manager / kernel maps) every iteration, a
// stride-1 and a stride-2 convolution, backward to features and weights.
void sparse_conv_case(bool cuda) {
    const auto device = require_device(cuda);
    torch::manual_seed(11);
    auto coords = std::get<0>(torch::unique_dim(torch::randint(0, 24, {3000, 3}, torch::kInt32), 0));
    coords = torch::cat({torch::zeros({coords.size(0), 1}, torch::kInt32), coords}, 1).to(device);
    const auto feats = torch::randn({coords.size(0), 4}).to(device);
    minkowski::MinkowskiConvolution conv1(4, 8, 3, 1, 1, false, false, minkowski::ConvolutionMode::DEFAULT, 3);
    minkowski::MinkowskiConvolution conv2(8, 8, 2, 2, 1, false, false, minkowski::ConvolutionMode::DEFAULT, 3);
    conv1->to(device);
    conv2->to(device);
    expect_no_growth("sparse_conv", cuda, [&] {
        auto f = feats.clone().requires_grad_(true);
        minkowski::SparseTensor x(f, coords);
        auto y = conv2->forward(conv1->forward(x));
        y.F().square().sum().backward();
        conv1->zero_grad();
        conv2->zero_grad();
    });
}

ad::AdaptiveVoxelConfig adaptive_config() {
    ad::AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 8.0, 8.0};
    c.min_voxel_size = {1.0, 1.0, 1.0};
    c.max_depth = 3;
    return c;
}

at::Tensor adaptive_points() {
    torch::manual_seed(5);
    return torch::cat({torch::rand({1500, 3}, torch::kFloat64) * 64.0,
                       torch::rand({1500, 3}, torch::kFloat64) * 12.0});
}

// Adaptive convolution: voxelize + refine a fresh topology every iteration (kernel-map cache misses and LRU
// evictions; capacity lowered to 4 so the cache is full after the warm-up), conv fwd + bwd.
void adaptive_conv_case(bool cuda) {
    const auto device = require_device(cuda);
    const auto pts = adaptive_points();
    const auto feats = torch::randn({pts.size(0), 4}, torch::kFloat64);
    const auto weight = (torch::randn({27, 4, 6}, torch::kFloat64) * 0.1).to(device);
    ad::AdaptiveKernelConfig kc;
    kc.mode = ad::AdaptiveConvMode::OverlapWeighted;
    auto &cache = ad::KernelMapCache::instance();
    cache.set_capacity(4);
    try {
        expect_no_growth("adaptive_conv", cuda, [&] {
            auto t = ad::AdaptiveSparseTensor::from_points(pts, feats, adaptive_config());
            t.refine(ad::AxisAlignedRegion{{0, 0, 0}, {12, 12, 12}}, 3);
            t = t.to(device);
            auto w = weight.clone().requires_grad_(true);
            auto f = t.features().clone().requires_grad_(true);
            auto y = ad::adaptive_convolution(t.with_features(f), w, kc);
            y.features().square().sum().backward();
        });
    } catch (...) {
        cache.set_capacity(64);
        throw;
    }
    cache.set_capacity(64);
}

ad::AdaptiveSparseTensor fixed_adaptive_tensor(torch::Device device) {
    const auto pts = adaptive_points();
    auto t = ad::AdaptiveSparseTensor::from_points(pts, torch::randn({pts.size(0), 4}, torch::kFloat64),
                                                   adaptive_config());
    t.refine(ad::AxisAlignedRegion{{0, 0, 0}, {12, 12, 12}}, 3);
    return t.to(device);
}

// Interpolation (nearest + multilinear) on one fixed topology: fwd + bwd to the leaf features.
void adaptive_interp_case(bool cuda) {
    const auto device = require_device(cuda);
    const auto t = fixed_adaptive_tensor(device);
    torch::manual_seed(9);
    const auto queries = (torch::rand({4000, 3}, torch::kFloat64) * 64.0).to(device);
    expect_no_growth("adaptive_interp", cuda, [&] {
        auto f = t.features().clone().requires_grad_(true);
        auto x = t.with_features(f);
        auto a = ad::interpolate(x, queries, ad::InterpMode::Multilinear);
        auto b = ad::interpolate(x, queries, ad::InterpMode::Nearest);
        (a.square().sum() + b.sum()).backward();
    });
}

// Local pooling to level 0 (mean + max) and volume-weighted global average pooling, fwd + bwd.
void adaptive_pool_case(bool cuda) {
    const auto device = require_device(cuda);
    const auto t = fixed_adaptive_tensor(device);
    expect_no_growth("adaptive_pool", cuda, [&] {
        auto f = t.features().clone().requires_grad_(true);
        auto x = t.with_features(f);
        auto mean = ad::adaptive_pool(x, 0, ad::Reduction::Mean);
        auto mx = ad::adaptive_pool(x, 1, ad::Reduction::Max);
        auto g = ad::adaptive_global_avg_pool(x);
        (mean.features().square().sum() + mx.features().sum() + g.sum()).backward();
    });
}

// Gaussian splatting render (fused CPU / CUDA rasterizer) fwd + bwd to every input.
void gaussian_render_case(bool cuda) {
    const auto device = require_device(cuda);
    const auto gen = at::make_generator<at::CPUGeneratorImpl>(23);
    const int64_t n = 2000;
    const auto pos = (at::randn({n, 3}, gen, at::kFloat) * 0.8).to(device);
    const auto rot = at::randn({n, 4}, gen, at::kFloat).to(device);
    const auto scl = at::exp(at::randn({n, 3}, gen, at::kFloat) * 0.4 - 3.5).to(device);
    const auto opa = at::sigmoid(at::randn({n, 1}, gen, at::kFloat)).to(device);
    const auto col = at::rand({n, 3}, gen, at::kFloat).to(device);
    const auto camera = gs::Camera::look_at({0.3, -0.2, -3.0}, {0, 0, 0}, {0, -1, 0}, 77, 45, 1.0);
    expect_no_growth("gaussian_render", cuda, [&] {
        auto p = pos.clone().requires_grad_(true);
        auto r = rot.clone().requires_grad_(true);
        auto s = scl.clone().requires_grad_(true);
        auto o = opa.clone().requires_grad_(true);
        auto c = col.clone().requires_grad_(true);
        auto out = gs::render_gaussians(p, r, s, o, c, camera);
        (out.image.square().sum() + 0.1 * out.depth.sum() + out.alpha.sum()).backward();
    });
}

// The detector itself: a planted per-iteration leak must be flagged.
void planted_leak_case(bool cuda, int64_t floats_per_iteration) {
    const auto device = require_device(cuda);
    std::vector<at::Tensor> retained;
    const Growth g = measure(std::string("planted_leak"), cuda, [&] {
        retained.push_back(torch::ones({floats_per_iteration}, torch::TensorOptions().device(device)));
    });
    CHECK(g.exceeds(cuda));
}

}  // namespace

TEST(memory_growth_sparse_conv_cpu) { sparse_conv_case(false); }
TEST(memory_growth_sparse_conv_cuda) { sparse_conv_case(true); }
TEST(memory_growth_adaptive_conv_cpu) { adaptive_conv_case(false); }
TEST(memory_growth_adaptive_conv_cuda) { adaptive_conv_case(true); }
TEST(memory_growth_adaptive_interp_cpu) { adaptive_interp_case(false); }
TEST(memory_growth_adaptive_interp_cuda) { adaptive_interp_case(true); }
TEST(memory_growth_adaptive_pool_cpu) { adaptive_pool_case(false); }
TEST(memory_growth_adaptive_pool_cuda) { adaptive_pool_case(true); }
TEST(memory_growth_gaussian_render_cpu) { gaussian_render_case(false); }
TEST(memory_growth_gaussian_render_cuda) { gaussian_render_case(true); }
// 128 KiB per iteration on the CPU (~14 MiB over the default run; 64 KiB measured only +5.0 MB of private bytes,
// too close to the 4 MiB tolerance), 1 KiB per iteration on CUDA (+110 KiB measured vs 16 KiB tolerance).
TEST(memory_growth_detects_planted_cpu_leak) { planted_leak_case(false, 32 * 1024); }
TEST(memory_growth_detects_planted_cuda_leak) { planted_leak_case(true, 256); }
