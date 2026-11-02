// InterpMode::ContinuousMultilinear: normalized tent partition over the leaves.
// Checked against a brute-force oracle over every leaf (candidate search and pruning), for
// continuity across leaf boundaries (incl. level changes, where Multilinear jumps), partition
// of unity, equality with Multilinear where all contributing leaves share one level,
// autograd (finite differences), empty / outside inputs (zeros, still connected) and CPU vs
// CUDA tensors.
//
// Tolerances: weights are computed in float64 from dyadic geometry; the oracle uses different
// but equivalent arithmetic (|q - c| / s vs frac((q - o) / s - 1/2)), so float64 results agree
// to ~1e-15 relative -> 1e-12. Continuity: the two points of a crossing are 2e-10 apart and the
// result's gradient is below ~3e4 * max|v| on these trees (<= 32 tents of slope <= 16 over a
// denominator >= 1/8), so a continuous result moves < 1e-5 there; checked against 1e-4.
#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "interp_eval_util.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;
using namespace interp_eval;

namespace {

constexpr InterpMode kCont = InterpMode::ContinuousMultilinear;

AdaptiveSparseTensor from_keys2(const std::vector<std::array<int64_t, 3>>& keys,
                                const at::Tensor& features, std::vector<double> base,
                                uint32_t depth) {
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = std::move(base);
    cfg.max_depth = depth;
    const int64_t n = static_cast<int64_t>(keys.size());
    auto c = torch::zeros({n, 2}, torch::kInt64);
    auto l = torch::zeros({n}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        c[i][0] = keys[i][0];
        c[i][1] = keys[i][1];
        l[i] = static_cast<int32_t>(keys[i][2]);
    }
    return AdaptiveSparseTensor::from_voxels(c, l, features, cfg);
}

// Values at a batch of points through interpolate().
at::Tensor interp_at(const AdaptiveSparseTensor& x, const std::vector<DVec>& pts, InterpMode m) {
    return interpolate(x, to_tensor(pts, static_cast<int>(x.dimension())), m);
}

// Oracle result for every point: sum of brute-force weights times features.
at::Tensor oracle(const Topology& t, const at::Tensor& feats, const std::vector<DVec>& pts) {
    const int64_t C = feats.size(1);
    auto f = feats.contiguous();
    const double* fp = f.data_ptr<double>();
    auto out = torch::zeros({static_cast<int64_t>(pts.size()), C}, torch::kFloat64);
    double* op = out.data_ptr<double>();
    for (size_t m = 0; m < pts.size(); ++m)
        for (const auto& [i, w] : brute_tent_weights(t, pts[m]))
            for (int64_t c = 0; c < C; ++c) op[m * C + c] += w * fp[i * C + c];
    return out;
}

at::Tensor numerical_grad(const std::function<double(const at::Tensor&)>& loss, at::Tensor x,
                          double eps = 1e-6) {
    auto g = torch::zeros_like(x);
    auto flat = x.view(-1);
    auto gflat = g.view(-1);
    for (int64_t i = 0; i < flat.numel(); ++i) {
        const double orig = flat[i].item<double>();
        flat[i] = orig + eps;
        const double up = loss(x);
        flat[i] = orig - eps;
        const double down = loss(x);
        flat[i] = orig;
        gflat[i] = (up - down) / (2 * eps);
    }
    return g;
}

}  // namespace

// The Multilinear characterization case of test_coverage (coarse [0,2)^2 = 0, fine
// [2,3)x[0,1) = 0, fine [2,3)x[1,2) = 1): Multilinear jumps 1/6 -> 0 across x = 2; the tent
// partition is continuous there, with the hand-computed value at (2, 0.9):
// coarse tent 0.5 * 0.95 = 0.475, fine (2,0) 0.5 * 0.6 = 0.3, fine (2,1) 0.5 * 0.4 = 0.2.
TEST(continuous_interp_level_change_example) {
    auto t = from_keys2({{0, 0, 0}, {2, 0, 1}, {2, 1, 1}},
                        torch::tensor({{0.0}, {0.0}, {1.0}}, torch::kFloat64), {2.0, 2.0}, 1);
    const double e = 1e-9;
    auto at = [&](double x, double y, InterpMode m) {
        return interpolate(t, torch::tensor({{x, y}}, torch::kFloat64), m)[0][0].item<double>();
    };
    CHECK_NEAR(at(2.0 - e, 0.5, InterpMode::Multilinear), 1.0 / 6.0, 1e-6);
    CHECK_NEAR(at(2.0 + e, 0.5, InterpMode::Multilinear), 0.0, 1e-6);
    CHECK_NEAR(at(2.0 - e, 0.5, kCont), at(2.0 + e, 0.5, kCont), 1e-8);
    CHECK_NEAR(at(2.0, 0.5, kCont), 0.0, 1e-15);
    const double v = 0.2 / 0.975;
    CHECK_NEAR(at(2.0 - e, 0.9, kCont), v, 1e-8);
    CHECK_NEAR(at(2.0, 0.9, kCont), v, 1e-15);
    CHECK_NEAR(at(2.0 + e, 0.9, kCont), v, 1e-8);
    CHECK_NEAR(at(2.5, 1.0 - e, kCont), at(2.5, 1.0 + e, kCont), 1e-8);
}

// The node-table candidate search with pruning returns exactly the brute-force weights, for
// D = 1..4, anisotropic random bases/origins, sparse trees and points inside, near and outside
// the leaves.
TEST(continuous_interp_matches_brute_force_oracle) {
    std::mt19937_64 rng(4242);
    struct Case {
        int D;
        uint32_t depth;
        int extent;
        int64_t points;
    };
    for (const Case c : {Case{1, 6, 4, 3000}, Case{2, 4, 3, 3000}, Case{3, 3, 2, 2000},
                         Case{4, 2, 1, 1000}}) {
        for (int rep = 0; rep < 2; ++rep) {
            auto topo = random_tree(c.D, c.depth, c.extent, 0.4, 0.2, rng);
            auto feats = torch::randn({topo->size(), 3}, torch::kFloat64);
            auto x = AdaptiveSparseTensor::from_topology(topo, feats);
            auto pts = random_points(*topo, c.points, rng, 0.25);
            auto got = interp_at(x, pts, kCont);
            auto want = oracle(*topo, feats, pts);
            CHECK_ALLCLOSE(got, want, 1e-12, 1e-12);
            // Points outside every leaf give exact zeros.
            for (size_t m = 0; m < pts.size(); ++m)
                if (brute_containing(*topo, pts[m]) < 0)
                    CHECK_EQ(got[static_cast<int64_t>(m)].abs().sum().item<double>(), 0.0);
        }
    }
}

// Many points (several worker threads): same result as the oracle and bit-identical between
// runs (deterministic block order).
TEST(continuous_interp_parallel_deterministic) {
    std::mt19937_64 rng(77);
    auto topo = random_tree(3, 4, 2, 0.3, 0.2, rng);
    auto feats = torch::randn({topo->size(), 2}, torch::kFloat64);
    auto x = AdaptiveSparseTensor::from_topology(topo, feats);
    auto pts = random_points(*topo, 60000, rng);
    auto q = to_tensor(pts, 3);
    auto a = interpolate(x, q, kCont);
    auto b = interpolate(x, q, kCont);
    CHECK(torch::equal(a, b));
    std::vector<DVec> sub(pts.begin(), pts.begin() + 3000);
    CHECK_ALLCLOSE(a.slice(0, 0, 3000), oracle(*topo, feats, sub), 1e-12, 1e-12);
}

// Constant features are reproduced exactly inside the leaves (partition of unity); outside,
// zeros.
TEST(continuous_interp_reproduces_constants) {
    std::mt19937_64 rng(9);
    for (int D : {2, 3}) {
        auto topo = random_tree(D, 4, 2, 0.45, 0.25, rng);
        auto feats = torch::tensor({3.5, -1.25}, torch::kFloat64).repeat({topo->size(), 1});
        auto x = AdaptiveSparseTensor::from_topology(topo, feats);
        auto pts = random_points(*topo, 4000, rng, 0.2);
        auto got = interp_at(x, pts, kCont);
        int inside = 0;
        for (size_t m = 0; m < pts.size(); ++m) {
            const bool in = topo->find_containing(pts[m]) >= 0;
            inside += in;
            CHECK_NEAR(got[static_cast<int64_t>(m)][0].item<double>(), in ? 3.5 : 0.0, 1e-12);
            CHECK_NEAR(got[static_cast<int64_t>(m)][1].item<double>(), in ? -1.25 : 0.0, 1e-12);
        }
        CHECK(inside > 1000);
    }
}

// Tents of equal width sum like Multilinear's box overlaps, so a single-level topology (sparse,
// anisotropic) gives Multilinear exactly, at every point incl. the outer boundary.
TEST(continuous_interp_equals_multilinear_single_level) {
    std::mt19937_64 rng(31);
    for (int D : {1, 2, 3}) {
        auto topo = random_tree(D, 1, 3, 1.0, 0.3, rng);  // every leaf at level 1
        auto feats = torch::randn({topo->size(), 2}, torch::kFloat64);
        auto x = AdaptiveSparseTensor::from_topology(topo, feats);
        auto q = to_tensor(random_points(*topo, 5000, rng, 0.2), D);
        CHECK_ALLCLOSE(interpolate(x, q, kCont), interpolate(x, q, InterpMode::Multilinear), 1e-12,
                       1e-12);
    }
}

// On mixed-level trees: wherever every leaf with a positive tent and every leaf overlapping
// Multilinear's box has the containing leaf's level, the two modes agree.
TEST(continuous_interp_equals_multilinear_on_uniform_neighbourhoods) {
    std::mt19937_64 rng(1234);
    int64_t uniform = 0, mixed = 0;
    for (int D : {2, 3}) {
        auto topo = random_tree(D, 3, 3, 0.35, 0.1, rng);
        auto feats = torch::randn({topo->size(), 2}, torch::kFloat64);
        auto x = AdaptiveSparseTensor::from_topology(topo, feats);
        auto pts = random_points(*topo, 4000, rng);
        auto cont = interp_at(x, pts, kCont);
        auto ml = interp_at(x, pts, InterpMode::Multilinear);
        for (size_t m = 0; m < pts.size(); ++m) {
            const int64_t j = topo->find_containing(pts[m]);
            if (j < 0) continue;
            const uint32_t l = topo->keys[j].level;
            bool same = true;
            for (const auto& [i, w] : brute_tent_weights(*topo, pts[m]))
                same = same && topo->keys[i].level == l;
            const DVec s = topo->size_at(l);
            DVec lo{}, hi{};
            for (int d = 0; d < D; ++d) {
                lo[d] = pts[m][d] - 0.5 * s[d];
                hi[d] = pts[m][d] + 0.5 * s[d];
            }
            topo->for_each_overlap_scan(lo, hi, [&](int64_t i, double) {
                same = same && topo->keys[i].level == l;
            });
            if (!same) {
                ++mixed;
                continue;
            }
            ++uniform;
            const auto r = static_cast<int64_t>(m);
            CHECK_ALLCLOSE(cont[r], ml[r], 1e-12, 1e-12);
        }
    }
    CHECK(uniform > 1000);
    CHECK(mixed > 100);
}

// Leaf-boundary crossings along random lines: the tent partition never jumps; Multilinear is
// continuous across equal-level faces and jumps at some level changes (characterization).
TEST(continuous_interp_continuous_across_levels) {
    std::mt19937_64 rng(2026);
    for (int D : {2, 3}) {
        auto topo = random_tree(D, 3, 2, 0.45, 0.05, rng);
        auto feats = torch::randn({topo->size(), 2}, torch::kFloat64);
        auto x = AdaptiveSparseTensor::from_topology(topo, feats);
        std::vector<DVec> before, after;
        std::vector<char> level_change;
        auto ends = random_points(*topo, 400, rng);
        for (size_t i = 0; i + 1 < ends.size(); i += 2)
            for (const auto& c : line_crossings(*topo, ends[i], ends[i + 1], 1e-10)) {
                before.push_back(c.before);
                after.push_back(c.after);
                level_change.push_back(c.level_change);
            }
        int64_t n_change = 0;
        for (char c : level_change) n_change += c;
        CHECK(n_change > 100);
        CHECK(static_cast<int64_t>(level_change.size()) - n_change > 100);
        auto jc = (interp_at(x, before, kCont) - interp_at(x, after, kCont)).abs().amax(1);
        auto jm = (interp_at(x, before, InterpMode::Multilinear) -
                   interp_at(x, after, InterpMode::Multilinear))
                      .abs()
                      .amax(1);
        auto change = torch::tensor(std::vector<int64_t>(level_change.begin(), level_change.end()))
                          .to(torch::kBool);
        CHECK(jc.max().item<double>() < 1e-4);
        CHECK(jm.masked_select(change.logical_not()).max().item<double>() < 1e-4);
        CHECK(jm.masked_select(change).max().item<double>() > 1e-2);
    }
}

// Linear in the features: autograd equals central finite differences (float64).
TEST(continuous_interp_gradcheck) {
    std::mt19937_64 rng(55);
    for (int D : {2, 3}) {
        auto topo = random_tree(D, 2, 1, 0.5, 0.1, rng);
        auto f0 = torch::randn({topo->size(), 2}, torch::kFloat64);
        auto base = AdaptiveSparseTensor::from_topology(topo, f0);
        auto q = to_tensor(random_points(*topo, 64, rng, 0.1), D);
        auto probe = torch::randn({q.size(0), 2}, torch::kFloat64);
        auto f = f0.clone().set_requires_grad(true);
        (interpolate(base.with_features(f), q, kCont) * probe).sum().backward();
        auto num = numerical_grad(
            [&](const at::Tensor& xf) {
                return (interpolate(base.with_features(xf), q, kCont) * probe)
                    .sum()
                    .item<double>();
            },
            f0.clone());
        CHECK_ALLCLOSE(f.grad(), num, 1e-5, 1e-6);
        CHECK(f.grad().abs().sum().item<double>() > 0.0);
    }
}

// Empty input, no points and points outside every leaf: zeros of shape [M, C] that are still
// part of the autograd graph (backward gives zero gradients instead of throwing). Bad shapes
// and non-finite points behave like the other modes.
TEST(continuous_interp_empty_and_outside) {
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {1.0, 1.0, 1.0};
    cfg.max_depth = 2;
    auto empty = AdaptiveSparseTensor::from_voxels(torch::zeros({0, 3}, torch::kInt64),
                                                   torch::zeros({0}, torch::kInt32),
                                                   torch::zeros({0, 4}, torch::kFloat64), cfg);
    auto fe = empty.features().clone().set_requires_grad(true);
    auto ye = interpolate(empty.with_features(fe), torch::rand({5, 3}, torch::kFloat64), kCont);
    CHECK_EQ(ye.size(0), int64_t(5));
    CHECK_EQ(ye.size(1), int64_t(4));
    CHECK_EQ(ye.abs().sum().item<double>(), 0.0);
    CHECK(ye.requires_grad());
    ye.sum().backward();
    CHECK(fe.grad().defined());

    auto t = from_keys2({{0, 0, 0}, {2, 0, 1}}, torch::randn({2, 3}, torch::kFloat64), {2.0, 2.0},
                        1);
    auto f = t.features().clone().set_requires_grad(true);
    auto far = torch::tensor({{-5.0, -5.0}, {100.0, 1.0}, {std::nan(""), 0.5}}, torch::kFloat64);
    auto yo = interpolate(t.with_features(f), far, kCont);
    CHECK_EQ(yo.abs().sum().item<double>(), 0.0);
    CHECK(yo.requires_grad());
    yo.sum().backward();
    CHECK(f.grad().defined());
    CHECK_EQ(f.grad().abs().sum().item<double>(), 0.0);

    auto y0 = interpolate(t, torch::zeros({0, 2}, torch::kFloat64), kCont);
    CHECK_EQ(y0.size(0), int64_t(0));
    CHECK_EQ(y0.size(1), int64_t(3));
    CHECK_THROWS(interpolate(t, torch::zeros({4, 3}, torch::kFloat64), kCont));
    // float32 features stay float32.
    auto t32 = t.with_features(t.features().to(torch::kFloat32));
    CHECK(interpolate(t32, torch::tensor({{1.0, 1.0}}), kCont).scalar_type() == torch::kFloat32);
}

// CUDA feature tensors give the CPU result (float64 tightly, float32 to rounding) and
// backpropagate on the device; empty/outside stays connected. Skips without CUDA.
TEST(continuous_interp_cpu_cuda_agreement) {
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA not available");
    std::mt19937_64 rng(8080);
    auto topo = random_tree(3, 3, 2, 0.4, 0.2, rng);
    auto feats = torch::randn({topo->size(), 4}, torch::kFloat64);
    auto x = AdaptiveSparseTensor::from_topology(topo, feats);
    auto q = to_tensor(random_points(*topo, 20000, rng, 0.1), 3);
    auto cpu = interpolate(x, q, kCont);
    auto dev = torch::Device(torch::kCUDA);
    auto g64 = interpolate(x.with_features(feats.to(dev)), q.to(dev), kCont);
    CHECK(g64.is_cuda());
    CHECK_ALLCLOSE(g64.cpu(), cpu, 1e-12, 1e-12);
    auto g32 = interpolate(x.with_features(feats.to(torch::kFloat32).to(dev)), q, kCont);
    CHECK_ALLCLOSE(g32.cpu().to(torch::kFloat64), cpu, 1e-5, 1e-5);
    // Gradients on the device equal the CPU gradients.
    auto probe = torch::randn({q.size(0), 4}, torch::kFloat64);
    auto fc = feats.clone().set_requires_grad(true);
    (interpolate(x.with_features(fc), q, kCont) * probe).sum().backward();
    auto fd = feats.to(dev).set_requires_grad(true);
    (interpolate(x.with_features(fd), q, kCont) * probe.to(dev)).sum().backward();
    CHECK_ALLCLOSE(fd.grad().cpu(), fc.grad(), 1e-10, 1e-10);
    // Outside every leaf on the device: zeros, connected.
    auto fo = feats.to(dev).set_requires_grad(true);
    auto yo = interpolate(x.with_features(fo), torch::full({3, 3}, 1e6, torch::kFloat64), kCont);
    CHECK(yo.is_cuda());
    CHECK_EQ(yo.abs().sum().item<double>(), 0.0);
    yo.sum().backward();
    CHECK(fo.grad().defined());
}
