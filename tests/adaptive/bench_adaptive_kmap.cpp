// Kernel-map, convolution and interpolation benchmark (not part of the correctness suite).
// Run: adaptive_kmap_bench [--n 20000] [--depth 4] [--channels 8] [--warmup 2] [--reps 5]
//                          [--modes center,volume,overlap] [--shift 0] [--interp 20000]
//                          [--conv none|cpu|cuda] [--cout 16] [--dtype f32|f64]
//                          [--region 16] [--phases cold,warm]
// Workload: half the points uniform in [0,256)^3, half in [0,16)^3; voxelized at level 0 and
// the region [0,R)^3 refined to `depth`, R = --region (16 = the adaptive_bench workload).
// --phases picks which kernel-map timings run (cold, warm or both).
// Prints "key   value" lines; times are median / min / max over --reps after --warmup runs.
//  kmap_<mode>_cold_ms : first map build on a freshly created topology (lazy indices included)
//  kmap_<mode>_warm_ms : map build on a topology whose lazy indices already exist
//  interp_multilinear_ms / interp_nearest_ms : interpolate() at --interp random points
// With --conv cpu|cuda the kernel-map/interp parts are skipped and adaptive_convolution is
// timed instead, with the kernel map already cached (the steady state of a training loop):
//  conv_<mode>_first_ms   : first convolution after the map was built (uploads included)
//  conv_<mode>_fwd_ms     : forward only, synchronized
//  conv_<mode>_fwdbwd_ms  : forward + backward (features and weight), synchronized
//  conv_<mode>_cuda_ms    : forward time between CUDA events (cuda only)
//  cuda_max_alloc_mb / cuda_max_reserved_mb : caching-allocator peaks over the conv runs
// Always: peak_working_set_mb (process peak, Windows) at exit.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace minkowski::adaptive;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    int64_t n = 20000;
    uint32_t depth = 4;
    int64_t channels = 8;
    int warmup = 2;
    int reps = 5;
    std::string modes = "center,volume,overlap";
    int shift = 0;
    int64_t interp = 20000;
    std::string conv = "none";
    int64_t cout = 16;
    std::string dtype = "f32";
    double region = 16.0;
    std::string phases = "cold,warm";
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string k = argv[i], v = argv[i + 1];
        if (k == "--n") a.n = std::atoll(v.c_str());
        else if (k == "--depth") a.depth = uint32_t(std::atoi(v.c_str()));
        else if (k == "--channels") a.channels = std::atoll(v.c_str());
        else if (k == "--warmup") a.warmup = std::atoi(v.c_str());
        else if (k == "--reps") a.reps = std::atoi(v.c_str());
        else if (k == "--modes") a.modes = v;
        else if (k == "--shift") a.shift = std::atoi(v.c_str());
        else if (k == "--interp") a.interp = std::atoll(v.c_str());
        else if (k == "--conv") a.conv = v;
        else if (k == "--cout") a.cout = std::atoll(v.c_str());
        else if (k == "--dtype") a.dtype = v;
        else if (k == "--region") a.region = std::atof(v.c_str());
        else if (k == "--phases") a.phases = v;
        else {
            std::cerr << "unknown argument " << k << "\n";
            std::exit(2);
        }
    }
    return a;
}

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void print(const std::string& key, double v) {
    std::cout << std::left << std::setw(34) << key << "  " << v << "\n";
}

void print_stats(const std::string& key, std::vector<double> t) {
    std::sort(t.begin(), t.end());
    const double med = t.size() % 2 ? t[t.size() / 2] : 0.5 * (t[t.size() / 2 - 1] + t[t.size() / 2]);
    print(key + "_ms", med);
    print(key + "_min_ms", t.front());
    print(key + "_max_ms", t.back());
}

// setup() runs untimed before every call of body().
template <class S, class B>
void report(const std::string& key, const Args& a, S&& setup, B&& body) {
    for (int i = 0; i < a.warmup; ++i) {
        setup();
        body();
    }
    std::vector<double> t;
    for (int i = 0; i < a.reps; ++i) {
        setup();
        auto t0 = Clock::now();
        body();
        t.push_back(ms_since(t0));
    }
    print_stats(key, t);
}

bool parse_mode(const std::string& mode, AdaptiveKernelConfig& kc) {
    if (mode == "center") kc.mode = AdaptiveConvMode::CenterSampled;
    else if (mode == "volume") kc.mode = AdaptiveConvMode::VolumeWeighted;
    else if (mode == "overlap") kc.mode = AdaptiveConvMode::OverlapWeighted;
    else if (mode == "averaged") kc.mode = AdaptiveConvMode::CenterAveraged;
    else return false;
    return true;
}

void sync(const torch::Device& dev) {
#ifdef USE_CUDA
    if (dev.is_cuda()) torch::cuda::synchronize();
#else
    (void)dev;
#endif
}

int run_conv(const Args& a, const AdaptiveSparseTensor& t_cpu) {
    const torch::Device dev = a.conv == "cuda" ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    if (dev.is_cuda() && !torch::cuda::is_available()) {
        std::cerr << "CUDA not available\n";
        return 3;
    }
    const auto dtype = a.dtype == "f64" ? torch::kFloat64 : torch::kFloat32;
    auto t = t_cpu.with_features(t_cpu.features().to(dtype)).to(dev);
    print("leaves", static_cast<double>(t.size()));
    print("channels_in", static_cast<double>(a.channels));
    print("channels_out", static_cast<double>(a.cout));
#ifdef USE_CUDA
    if (dev.is_cuda()) {
        sync(dev);
        c10::cuda::CUDACachingAllocator::emptyCache();
        c10::cuda::CUDACachingAllocator::resetPeakStats(dev.index() < 0 ? 0 : dev.index());
    }
#endif
    std::stringstream ms(a.modes);
    std::string mode;
    while (std::getline(ms, mode, ',')) {
        AdaptiveKernelConfig kc;
        kc.output_level_shift = a.shift;
        if (!parse_mode(mode, kc)) {
            std::cerr << "unknown mode " << mode << "\n";
            return 2;
        }
        torch::manual_seed(1);
        auto w = torch::randn({27, a.channels, a.cout}, torch::TensorOptions().dtype(dtype)).to(dev);
        KernelMapCache::instance().clear();
        auto map = build_adaptive_kernel_map(t, kc);  // map built and cached, not timed
        print("conv_" + mode + "_pairs", static_cast<double>(map.in_index.numel()));
        {
            sync(dev);
            auto t0 = Clock::now();
            auto y = adaptive_convolution(t, w, kc);
            sync(dev);
            print("conv_" + mode + "_first_ms", ms_since(t0));
        }
        report("conv_" + mode + "_fwd", a, [&] { sync(dev); },
               [&] {
                   torch::NoGradGuard ng;
                   auto y = adaptive_convolution(t, w, kc);
                   sync(dev);
               });
        report("conv_" + mode + "_fwdbwd", a, [&] { sync(dev); },
               [&] {
                   auto f = t.features().detach().requires_grad_(true);
                   auto wg = w.detach().requires_grad_(true);
                   auto y = adaptive_convolution(t.with_features(f), wg, kc);
                   y.features().sum().backward();
                   sync(dev);
               });
#ifdef USE_CUDA
        if (dev.is_cuda()) {
            std::vector<double> ev;
            for (int i = 0; i < a.warmup + a.reps; ++i) {
                at::cuda::CUDAEvent e0(cudaEventDefault), e1(cudaEventDefault);
                torch::NoGradGuard ng;
                auto stream = c10::cuda::getCurrentCUDAStream();
                e0.record(stream);
                auto y = adaptive_convolution(t, w, kc);
                e1.record(stream);
                e1.synchronize();
                if (i >= a.warmup) ev.push_back(e0.elapsed_time(e1));
            }
            print_stats("conv_" + mode + "_cuda", ev);
        }
#endif
    }
#ifdef USE_CUDA
    if (dev.is_cuda()) {
        const auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(dev.index() < 0 ? 0 : dev.index());
        using c10::CachingAllocator::StatType;
        const auto agg = static_cast<size_t>(StatType::AGGREGATE);
        print("cuda_max_alloc_mb", st.allocated_bytes[agg].peak / 1048576.0);
        print("cuda_max_reserved_mb", st.reserved_bytes[agg].peak / 1048576.0);
    }
#endif
    return 0;
}

void print_peak_ws() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        print("peak_working_set_mb", pmc.PeakWorkingSetSize / 1048576.0);
#endif
}

}  // namespace

int main(int argc, char** argv) {
    const Args a = parse(argc, argv);
    torch::manual_seed(0);
    auto pts = torch::cat({torch::rand({a.n / 2, 3}, torch::kFloat64) * 256.0,
                           torch::rand({a.n - a.n / 2, 3}, torch::kFloat64) * 16.0});
    auto feats = torch::randn({a.n, a.channels}, torch::kFloat64);
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {8.0, 8.0, 8.0};
    const double m = 8.0 / double(1u << a.depth);
    cfg.min_voxel_size = {m, m, m};
    cfg.max_depth = a.depth;
    auto t = AdaptiveSparseTensor::from_points(pts, feats, cfg);
    t.refine(AxisAlignedRegion{{0, 0, 0}, {a.region, a.region, a.region}}, a.depth);
    const auto coords = t.coordinates();
    const auto levels = t.levels();
    const auto f = t.features();

    std::cout << std::fixed << std::setprecision(3);
    print("points", static_cast<double>(a.n));
    print("depth", a.depth);
    print("shift", a.shift);
    if (a.conv != "none") {
        const int rc = run_conv(a, t);
        print_peak_ws();
        return rc;
    }
    print("leaves", static_cast<double>(t.size()));

    std::stringstream ms(a.modes);
    std::string mode;
    while (std::getline(ms, mode, ',')) {
        AdaptiveKernelConfig kc;
        kc.output_level_shift = a.shift;
        if (!parse_mode(mode, kc)) {
            std::cerr << "unknown mode " << mode << "\n";
            return 2;
        }
        {
            KernelMapCache::instance().clear();
            auto map = build_adaptive_kernel_map(t, kc);
            print("kmap_" + mode + "_pairs", static_cast<double>(map.in_index.numel()));
        }
        AdaptiveSparseTensor fresh = t;
        if (a.phases.find("cold") != std::string::npos) {
            report("kmap_" + mode + "_cold", a,
                   [&] {
                       KernelMapCache::instance().clear();
                       fresh = AdaptiveSparseTensor::from_voxels(coords, levels, f, cfg);
                   },
                   [&] { build_adaptive_kernel_map(fresh, kc); });
        }
        if (a.phases.find("warm") != std::string::npos) {
            report("kmap_" + mode + "_warm", a, [&] { KernelMapCache::instance().clear(); },
                   [&] { build_adaptive_kernel_map(t, kc); });
        }
    }
    KernelMapCache::instance().clear();

    if (a.interp > 0) {
        auto q = torch::cat({torch::rand({a.interp / 2, 3}, torch::kFloat64) * 256.0,
                             torch::rand({a.interp - a.interp / 2, 3}, torch::kFloat64) * 16.0});
        report("interp_multilinear", a, [] {},
               [&] { interpolate(t, q, InterpMode::Multilinear); });
        report("interp_nearest", a, [] {}, [&] { interpolate(t, q, InterpMode::Nearest); });
    }
    print_peak_ws();
    return 0;
}
