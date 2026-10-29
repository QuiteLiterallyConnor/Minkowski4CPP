// Benchmarks (not part of the correctness suite). Prints timings; makes no speed claims.
// Run: adaptive_bench [occupied_voxels] [max_depth]
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "MinkowskiEngine/adaptive/adaptive.hpp"

using namespace minkowski::adaptive;
using Clock = std::chrono::steady_clock;

namespace {

template <typename F>
double time_ms(F&& f, int reps = 5) {
    f();  // warm-up
    auto t0 = Clock::now();
    for (int i = 0; i < reps; ++i) f();
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / reps;
}

AdaptiveVoxelConfig config(uint32_t depth) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 8.0, 8.0};
    double m = 8.0 / double(1u << depth);
    c.min_voxel_size = {m, m, m};
    c.max_depth = depth;
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    int64_t n = argc > 1 ? std::atoll(argv[1]) : 20000;
    uint32_t depth = argc > 2 ? uint32_t(std::atoi(argv[2])) : 4;
    torch::manual_seed(0);

    // Sparse workload: points concentrated near a small region of interest.
    auto pts = torch::cat({torch::rand({n / 2, 3}, torch::kFloat64) * 256.0,
                           torch::rand({n / 2, 3}, torch::kFloat64) * 16.0});
    auto feats = torch::randn({n, 8}, torch::kFloat64);
    auto cfg = config(depth);
    auto region = AxisAlignedRegion{{0, 0, 0}, {16, 16, 16}};
    auto weight = torch::randn({27, 8, 8}, torch::kFloat64);
    AdaptiveKernelConfig kc;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "points=" << n << " depth=" << depth << "\n";

    std::cout << "voxelization_ms        "
              << time_ms([&] { AdaptiveSparseTensor::from_points(pts, feats, cfg); }) << "\n";

    auto t = AdaptiveSparseTensor::from_points(pts, feats, cfg);
    std::cout << "refine_roi_ms          "
              << time_ms([&] {
                     auto c = t;
                     c.refine(region, depth);
                 })
              << "\n";
    t.refine(region, depth);

    std::cout << "adaptive_voxels        " << t.size() << "\n";
    auto counts = t.counts_by_level();
    for (size_t l = 0; l < counts.size(); ++l)
        std::cout << "  level " << l << "            " << counts[l] << "\n";

    std::cout << "kernel_map_ms          "
              << time_ms([&] {
                     KernelMapCache::instance().clear();
                     build_adaptive_kernel_map(t, kc);
                 })
              << "\n";
    std::cout << "forward_ms             "
              << time_ms([&] { adaptive_convolution(t, weight, kc); }) << "\n";

    auto f = t.features().clone().set_requires_grad(true);
    std::cout << "forward_backward_ms    "
              << time_ms([&] {
                     auto o = adaptive_convolution(t.with_features(f), weight, kc);
                     o.features().sum().backward();
                 })
              << "\n";

    // Equivalent uniform representation over the same physical domain (skipped when huge).
    double uniform_est = 0;
    for (size_t l = 0; l < counts.size(); ++l)
        uniform_est += double(counts[l]) * std::pow(8.0, double(depth) - double(l));
    std::cout << "uniform_voxels_est     " << uniform_est << "\n";
    if (uniform_est > 5e6) {
        std::cout << "uniform comparison skipped (> 5M voxels)\n";
        std::cout << "feature_bytes_adaptive " << t.features().nbytes() << "\n";
        std::cout << "feature_bytes_uniform_est " << uniform_est * t.features().size(1) * 8 << "\n";
        return 0;
    }
    auto u = t.resample_uniform(depth, Reduction::Mean);
    std::cout << "uniform_voxels         " << u.size() << "\n";
    std::cout << "uniform_forward_ms     "
              << time_ms([&] { adaptive_convolution(u, weight, kc); }) << "\n";
    std::cout << "feature_bytes_adaptive " << t.features().nbytes() << "\n";
    std::cout << "feature_bytes_uniform  " << u.features().nbytes() << "\n";
    return 0;
}
