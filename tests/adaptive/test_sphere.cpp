// voxelize_sphere: accuracy target, placement, and error handling.
#include <cmath>
#include <stdexcept>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/utils/sphere.hpp"
#include "MinkowskiEngine/utils/sphere_geometry.hpp"
#include "utils/test_framework.hpp"

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

// ---------------------------------------------------------------------------------------------
// SphereErrorMetric: the default (SignedVolume) lets over- and under-estimates cancel;
// Geometric bounds the misclassified (symmetric-difference) volume.
// ---------------------------------------------------------------------------------------------
namespace {
// Monte-Carlo symmetric difference between the occupied leaves and the ball (centered at
// the origin), in percent of the ball volume. Deterministic seed; ~0.02 % standard error at
// 1 % error with the default sample count.
double mc_symmetric_difference_pct(const AdaptiveSparseTensor& s, double r, int64_t n = 400000) {
    torch::manual_seed(1234);
    auto pts = (torch::rand({n, 3}, torch::kFloat64) * 2.0 - 1.0) * r;
    auto inside = pts.pow(2).sum(1) <= r * r;
    auto occ = interpolate(s.with_features(s.features().cpu()), pts).squeeze(1) > 0.5;
    const double frac = (occ != inside).to(torch::kFloat64).mean().item<double>();
    return 100.0 * frac * 8.0 * r * r * r / exact(r);
}
}  // namespace

TEST(sphere_signed_volume_default_is_unchanged) {
    // Known (legacy) behavior: both targets stop at the same lucky cancellation.
    for (double target : {99.5, 99.9}) {
        double achieved = 0.0;
        auto a = voxelize_sphere(100.0, target, {0.0, 0.0, 0.0}, 10'000'000, 10, &achieved);
        CHECK(a.size() == 232);
        CHECK(achieved > 99.96);
        auto b = voxelize_sphere(100.0, target, {0.0, 0.0, 0.0}, 10'000'000, 10, nullptr,
                                 SphereErrorMetric::SignedVolume);
        CHECK(torch::equal(a.coordinates(), b.coordinates()));
        CHECK(torch::equal(a.levels(), b.levels()));
        CHECK(torch::equal(a.features(), b.features()));
        // The reported 99.97 % hides a ~24 % shape error.
        CHECK(mc_symmetric_difference_pct(a, 100.0) > 20.0);
    }
}

TEST(sphere_geometric_metric_bounds_shape_error) {
    int64_t previous = 0;
    for (double target : {90.0, 95.0, 98.0, 99.0}) {
        double achieved = 0.0;
        auto t = voxelize_sphere(100.0, target, {0.0, 0.0, 0.0}, 10'000'000, 10, &achieved,
                                 SphereErrorMetric::Geometric);
        CHECK(achieved >= target);
        CHECK(t.size() >= previous);
        previous = t.size();
        // The volume accuracy is reached too ...
        const double est = sphere_volume_estimate(t);
        CHECK(100.0 * (1.0 - std::abs(est - exact(100.0)) / exact(100.0)) >= target);
        // ... and the shape error really is within the target (Monte Carlo, > 6 sigma margin).
        const double sym = mc_symmetric_difference_pct(t, 100.0);
        CHECK(sym <= 100.0 - target + 0.15);
        // The reported error is an estimate of the same quantity.
        CHECK_NEAR(100.0 - achieved, sym, 0.05 * sym + 0.1);
    }
}

TEST(sphere_geometric_metric_unreachable_throws) {
    // 99.9 % shape accuracy needs cells below 2r / 2^10; the signed metric "reaches" it at 232.
    CHECK_THROWS(voxelize_sphere(100.0, 99.9, {0.0, 0.0, 0.0}, 10'000'000, 6, nullptr,
                                 SphereErrorMetric::Geometric));
    CHECK(voxelize_sphere(100.0, 99.9, {0.0, 0.0, 0.0}, 10'000'000, 6).size() == 232);
    CHECK_THROWS(voxelize_sphere(100.0, 99.0, {0.0, 0.0, 0.0}, 1000, 10, nullptr,
                                 SphereErrorMetric::Geometric));
}

TEST(sphere_geometry_inside_volume_and_misclassified_volume) {
    using minkowski::adaptive::sphere_detail::SphereGeometry;
    const SphereGeometry g{1.0};
    // A cell fully inside: inside volume = cell volume, nothing misclassified.
    CHECK_NEAR(g.inside_volume(1, 1, 1, 2), 0.125, 1e-15);
    CHECK_NEAR(g.misclassified_volume(1, 1, 1, 2), 0.0, 0.0);
    // A cell fully outside (corner of the bounding cube at level 3).
    CHECK_NEAR(g.inside_volume(0, 0, 0, 3), 0.0, 0.0);
    CHECK_NEAR(g.misclassified_volume(0, 0, 0, 3), 0.0, 0.0);
    // Inside volumes of one level add up to the ball volume.
    for (int level : {2, 4}) {
        const int64_t n = int64_t(1) << level;
        double sum = 0.0;
        for (int64_t x = 0; x < n; ++x)
            for (int64_t y = 0; y < n; ++y)
                for (int64_t z = 0; z < n; ++z) {
                    const double v = g.inside_volume(x, y, z, level);
                    const double cell = std::pow(g.cell_size(level), 3);
                    CHECK(v >= 0.0 && v <= cell * (1.0 + 1e-12));
                    sum += v;
                }
        CHECK_NEAR(sum / exact(1.0), 1.0, 2e-3);
    }
    // Uniform level-5 grid: the summed misclassified volume matches a Monte-Carlo symmetric
    // difference of the center-sampled voxelization.
    const SphereGeometry h{100.0};
    const int level = 5;
    const int64_t n = int64_t(1) << level;
    double wrong = 0.0;
    for (int64_t x = 0; x < n; ++x)
        for (int64_t y = 0; y < n; ++y)
            for (int64_t z = 0; z < n; ++z)
                wrong += h.misclassified_volume(x, y, z, level);
    const double estimate_pct = 100.0 * wrong / h.exact_volume();
    torch::manual_seed(99);
    auto pts = (torch::rand({400000, 3}, torch::kFloat64) * 2.0 - 1.0) * 100.0;
    const double size = h.cell_size(level);
    auto centre = (torch::floor((pts + 100.0) / size) + 0.5) * size - 100.0;
    auto occ = centre.pow(2).sum(1) <= 100.0 * 100.0;
    auto inside = pts.pow(2).sum(1) <= 100.0 * 100.0;
    const double mc_pct = 100.0 * (occ != inside).to(torch::kFloat64).mean().item<double>() *
                          8e6 / h.exact_volume();
    CHECK(estimate_pct > 4.0 && estimate_pct < 5.0);
    CHECK_NEAR(estimate_pct, mc_pct, 0.05 * mc_pct);
}

#ifndef CPU_ONLY
TEST(sphere_cuda_matches_cpu_geometric_metric) {
    if (!torch::cuda::is_available()) return;
    for (double target : {90.0, 95.0, 98.0, 99.0}) {
        double cpu_acc = 0.0, gpu_acc = 0.0;
        auto cpu = voxelize_sphere(100.0, target, {1.0, 2.0, 3.0}, 10'000'000, 10, &cpu_acc,
                                   SphereErrorMetric::Geometric);
        auto gpu = voxelize_sphere_cuda(100.0, target, {1.0, 2.0, 3.0}, 10'000'000, 10, &gpu_acc,
                                        SphereErrorMetric::Geometric);
        CHECK(gpu.size() == cpu.size());
        CHECK_NEAR(gpu_acc, cpu_acc, 1e-9);
        CHECK(torch::equal(gpu.coordinates(), cpu.coordinates()));
        CHECK(torch::equal(gpu.levels(), cpu.levels()));
        CHECK(torch::equal(gpu.features().cpu(), cpu.features()));
    }
}
#endif
