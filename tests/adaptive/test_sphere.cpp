// voxelize_sphere: accuracy target, placement, and error handling.
#include <cmath>
#include <stdexcept>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/utils/sphere.hpp"
#include "test_framework.hpp"

using namespace minkowski::adaptive;

namespace {
constexpr double kPi = 3.14159265358979323846;
double exact(double r) { return 4.0 / 3.0 * kPi * r * r * r; }
}  // namespace

TEST(sphere_reaches_target_accuracy) {
    for (double target : {90.0, 99.0, 99.9}) {
        double achieved = 0.0;
        auto t = voxelize_sphere(100.0, target, {0.0, 0.0, 0.0}, 10'000'000, 10, &achieved);
        CHECK(achieved >= target);
        const double est = sphere_volume_estimate(t);
        const double acc = 100.0 * (1.0 - std::abs(est - exact(100.0)) / exact(100.0));
        CHECK_NEAR(acc, achieved, 1e-6);
    }
}

TEST(sphere_known_voxel_counts) {
    CHECK(voxelize_sphere(100.0, 90.0).size() == 57);
    CHECK(voxelize_sphere(100.0, 99.0).size() == 204);
}

TEST(sphere_center_sets_origin) {
    auto t = voxelize_sphere(5.0, 95.0, {10.0, -20.0, 30.0});
    auto b = t.bounds();
    CHECK_NEAR(b.first.min().item<double>(), -20.0 - 5.0, 1e-9);
    CHECK_NEAR(b.second.max().item<double>(), 30.0 + 5.0, 1e-9);
}

TEST(sphere_invalid_arguments) {
    bool threw = false;
    try { voxelize_sphere(-1.0, 90.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { voxelize_sphere(1.0, 100.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { voxelize_sphere(1.0, 99.99, {0, 0, 0}, 20); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

#ifndef CPU_ONLY
TEST(sphere_cuda_matches_cpu) {
    if (!torch::cuda::is_available()) return;
    for (double target : {90.0, 99.0, 99.9, 99.99}) {
        double cpu_acc = 0.0, gpu_acc = 0.0;
        auto cpu = voxelize_sphere(100.0, target, {1.0, 2.0, 3.0}, 10'000'000, 10, &cpu_acc);
        auto gpu = voxelize_sphere_cuda(100.0, target, {1.0, 2.0, 3.0}, 10'000'000, 10, &gpu_acc);
        CHECK(gpu.size() == cpu.size());
        CHECK_NEAR(gpu_acc, cpu_acc, 1e-9);
        CHECK(gpu.features().is_cuda());
        CHECK(torch::equal(gpu.coordinates(), cpu.coordinates()));
        CHECK(torch::equal(gpu.levels(), cpu.levels()));
        CHECK(torch::equal(gpu.features().cpu(), cpu.features()));
        CHECK_NEAR(sphere_volume_estimate(gpu), sphere_volume_estimate(cpu), 1e-6);
    }
}

TEST(sphere_cuda_errors) {
    if (!torch::cuda::is_available()) return;
    bool threw = false;
    try { voxelize_sphere_cuda(1.0, 100.0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { voxelize_sphere_cuda(1.0, 99.99, {0, 0, 0}, 20); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}
#endif