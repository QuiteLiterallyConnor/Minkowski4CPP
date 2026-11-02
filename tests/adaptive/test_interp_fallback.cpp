// interpolate(..., InterpFallback::CoarserAncestor): uncovered points take the volume-weighted
// mean of the leaves inside their finest occupied ancestor cell instead of zeros.
#include <algorithm>
#include <cmath>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveVoxelConfig fb_config(uint32_t depth = 3, double base = 8.0, int dim = 3) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size.assign(size_t(dim), base);
    c.min_voxel_size.assign(size_t(dim), base / double(1u << depth));
    c.max_depth = depth;
    return c;
}

// Level-0 cell (0,0,0) of size 8 holds leaf A = level-1 (0,0,0) [0,4)^3, value 1, volume 64,
// and leaf B = level-2 (2,2,2) [4,6)^3, value 10, volume 8. Level-0 cell (1,0,0) holds
// leaf C = level-0 (1,0,0) itself, value 100. Everything else is uncovered.
AdaptiveSparseTensor two_level_fixture(torch::Device dev = torch::kCPU) {
    auto coords = torch::tensor({{0, 0, 0}, {2, 2, 2}, {1, 0, 0}}, torch::kInt64);
    auto levels = torch::tensor({1, 2, 0}, torch::kInt32);
    auto feats = torch::tensor({{1.0, -1.0}, {10.0, -10.0}, {100.0, -100.0}}, torch::kFloat64);
    return AdaptiveSparseTensor::from_voxels(coords, levels, feats.to(dev), fb_config());
}

// Clustered points (two tight clusters + sparse background) and a smooth 2-channel feature.
std::pair<at::Tensor, at::Tensor> clustered_points(int64_t n, uint64_t seed) {
    torch::manual_seed(seed);
    auto a = torch::randn({n / 2, 3}, torch::kFloat64) * 1.5 + 5.0;
    auto b = torch::randn({n / 4, 3}, torch::kFloat64) * 3.0 + 20.0;
    auto bg = torch::rand({n - n / 2 - n / 4, 3}, torch::kFloat64) * 32.0 - 4.0;
    auto pts = torch::cat({a, b, bg});
    auto feats = torch::stack({torch::sin(pts.select(1, 0) / 3.0), torch::cos(pts.select(1, 1) / 5.0)}, 1);
    return {pts, feats};
}

// Brute force: for every query, scan levels from max_depth down to 0 for the finest cell that
// contains the query and strictly contains at least one leaf; volume-weighted mean of those
// leaves. Covered queries and queries without such a cell get NaN (not compared).
at::Tensor brute_force_fallback(const AdaptiveSparseTensor &t, const at::Tensor &q,
                                const AdaptiveVoxelConfig &cfg) {
    const auto bounds = t.bounds();
    const auto &lo = bounds.first;  // [N, D]
    auto lv = t.levels().to(torch::kInt64);
    auto vol = t.volumes();
    auto f = t.features().to(torch::kCPU).to(torch::kFloat64);
    auto out = torch::full({q.size(0), f.size(1)}, NAN, torch::kFloat64);
    for (int64_t m = 0; m < q.size(0); ++m) {
        auto qm = q[m];
        auto inside = ((lo <= qm) & (qm < bounds.second)).all(1);
        if (inside.any().item<bool>())
            continue;
        for (int64_t l = int64_t(cfg.max_depth); l >= 0; --l) {
            const double s = cfg.base_voxel_size[0] / double(int64_t(1) << l);
            auto qcell = torch::floor(qm / s);
            // Cell (at level l) of every leaf's lower corner; strictly inside = finer leaf.
            auto lcell = torch::floor((lo + 1e-9 * s) / s);
            auto in_cell = (lcell == qcell).all(1) & (lv > l);
            if (!in_cell.any().item<bool>())
                continue;
            auto w = vol.index({in_cell});
            out[m] = (f.index({in_cell}) * w.unsqueeze(1)).sum(0) / w.sum();
            break;
        }
    }
    return out;
}

}  // namespace

TEST(interp_fallback_default_is_zero_and_unchanged) {
    auto t = two_level_fixture();
    auto q = torch::tensor({{1.0, 1.0, 1.0}, {6.5, 1.0, 1.0}, {4.5, 4.5, 4.5}, {5.0, 4.5, 6.5},
                            {12.0, 3.0, 3.0}, {-3.0, 1.0, 1.0}},
                           torch::kFloat64);
    for (auto mode : {InterpMode::Nearest, InterpMode::Multilinear}) {
        auto a = interpolate(t, q, mode);
        auto b = interpolate(t, q, mode, InterpFallback::Zero);
        CHECK(torch::equal(a, b));
    }
    auto a = interpolate(t, q);
    CHECK_NEAR(a[1][0].item<double>(), 0.0, 0.0);  // uncovered -> zeros by default
    CHECK_NEAR(a[3][0].item<double>(), 0.0, 0.0);
}

TEST(interp_fallback_hand_computed_values) {
    auto t = two_level_fixture();
    auto q = torch::tensor({{1.0, 1.0, 1.0},    // inside A
                            {6.5, 1.0, 1.0},    // uncovered, finest occupied cell = level-0 (0,0,0)
                            {4.5, 4.5, 4.5},    // inside B
                            {5.0, 4.5, 6.5},    // uncovered, level-1 cell (1,1,1) holds only B
                            {12.0, 3.0, 3.0},   // inside C
                            {-3.0, 1.0, 1.0},   // level-0 cell (-1,0,0) is empty -> 0
                            {3.0, 20.0, 1.0}},  // level-0 cell (0,2,0) is empty -> 0
                           torch::kFloat64);
    auto y = interpolate(t, q, InterpMode::Nearest, InterpFallback::CoarserAncestor);
    const double root_mean = (64.0 * 1.0 + 8.0 * 10.0) / 72.0;  // = 2
    const std::vector<double> expect = {1.0, root_mean, 10.0, 10.0, 100.0, 0.0, 0.0};
    for (size_t i = 0; i < expect.size(); ++i) {
        CHECK_NEAR(y[int64_t(i)][0].item<double>(), expect[i], 1e-12);
        CHECK_NEAR(y[int64_t(i)][1].item<double>(), -expect[i], 1e-12);
    }
    // Multilinear: covered points are unchanged by the fallback, uncovered points get the same
    // ancestor mean as Nearest.
    auto ml0 = interpolate(t, q, InterpMode::Multilinear);
    auto ml1 = interpolate(t, q, InterpMode::Multilinear, InterpFallback::CoarserAncestor);
    for (int64_t i : {0, 2, 4})
        CHECK(torch::equal(ml0[i], ml1[i]));
    for (int64_t i : {1, 3, 5, 6})
        CHECK_NEAR(ml1[i][0].item<double>(), expect[size_t(i)], 1e-12);
}

TEST(interp_fallback_gradients) {
    auto t = two_level_fixture();
    auto f = t.features().detach().clone().set_requires_grad(true);
    auto u = t.with_features(f);
    auto q = torch::tensor({{6.5, 1.0, 1.0}, {5.0, 4.5, 6.5}, {12.0, 3.0, 3.0}}, torch::kFloat64);
    interpolate(u, q, InterpMode::Nearest, InterpFallback::CoarserAncestor).select(1, 0).sum().backward();
    auto g = f.grad().select(1, 0);
    const int64_t a = t.index_of_key({0, 0, 0}, 1), b = t.index_of_key({2, 2, 2}, 2),
                  c = t.index_of_key({1, 0, 0}, 0);
    CHECK(a >= 0 && b >= 0 && c >= 0);
    CHECK_NEAR(g[a].item<double>(), 64.0 / 72.0, 1e-12);        // A via the level-0 mean
    CHECK_NEAR(g[b].item<double>(), 8.0 / 72.0 + 1.0, 1e-12);   // B via both cells
    CHECK_NEAR(g[c].item<double>(), 1.0, 1e-12);                // C (covered)
    CHECK_NEAR(f.grad().select(1, 1).abs().sum().item<double>(), 0.0, 0.0);
}

TEST(interp_fallback_matches_brute_force_after_drop_empty) {
    auto [pts, feats] = clustered_points(3000, 11);
    const auto cfg = fb_config(4);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, cfg);
    t.refine_by_density(4, /*drop_empty=*/true);
    torch::manual_seed(12);
    auto q = torch::rand({600, 3}, torch::kFloat64) * 36.0 - 6.0;
    auto y0 = interpolate(t, q);
    auto y1 = interpolate(t, q, InterpMode::Nearest, InterpFallback::CoarserAncestor);
    auto ref = brute_force_fallback(t, q, cfg);
    auto has_ref = ~torch::isnan(ref.select(1, 0));
    CHECK(has_ref.sum().item<int64_t>() > 50);  // the fixture really has filled holes
    CHECK_ALLCLOSE(y1.index({has_ref}), ref.index({has_ref}), 1e-12, 1e-12);
    // Covered points and points without an occupied ancestor are untouched.
    auto covered = (interpolate(t.with_features(torch::ones({t.size(), 1}, torch::kFloat64)), q)
                        .squeeze(1) > 0.5);
    CHECK(torch::equal(y1.index({covered}), y0.index({covered})));
    auto none = ~covered & ~has_ref;
    CHECK((y1.index({none}) == 0).all().item<bool>());
    // Multilinear agrees with the brute force on uncovered points too.
    auto ml = interpolate(t, q, InterpMode::Multilinear, InterpFallback::CoarserAncestor);
    CHECK_ALLCLOSE(ml.index({has_ref}), ref.index({has_ref}), 1e-12, 1e-12);
}

TEST(interp_fallback_independent_of_batching) {
    auto [pts, feats] = clustered_points(4000, 13);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, fb_config(4));
    t.refine_by_density(4, true);
    torch::manual_seed(14);
    auto q = torch::rand({40000, 3}, torch::kFloat64) * 36.0 - 6.0;
    auto all = interpolate(t, q, InterpMode::Nearest, InterpFallback::CoarserAncestor);
    for (int64_t s = 0; s < q.size(0); s += 997) {
        const int64_t e = std::min<int64_t>(s + 997, q.size(0));
        auto part = interpolate(t, q.slice(0, s, e), InterpMode::Nearest, InterpFallback::CoarserAncestor);
        CHECK(torch::equal(part, all.slice(0, s, e)));
    }
}

TEST(interp_fallback_2d_negative_coordinates) {
    // D = 2, base 8, depth 2: leaves at level 2 (size 2) in level-0 cell (-1,-1) = [-8,0)^2.
    auto coords = torch::tensor({{-4, -4}, {-1, -1}}, torch::kInt64);  // [-8,-6)^2 and [-2,0)^2
    auto levels = torch::tensor({2, 2}, torch::kInt32);
    auto feats = torch::tensor({{3.0}, {5.0}}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_voxels(coords, levels, feats, fb_config(2, 8.0, 2));
    auto q = torch::tensor({{-5.0, -7.0},   // level-1 cell (-2,-2) = [-8,-4)^2 holds only (-4,-4)
                            {-3.0, -3.0},   // level-1 cell (-1,-1) = [-4,0)^2 holds only (-1,-1)
                            {-7.0, -1.0},   // level-1 cell (-2,-1) is empty -> level-0 mean 4
                            {1.0, 1.0}},    // level-0 cell (0,0) empty -> 0
                           torch::kFloat64);
    auto y = interpolate(t, q, InterpMode::Nearest, InterpFallback::CoarserAncestor);
    CHECK_NEAR(y[0][0].item<double>(), 3.0, 1e-12);
    CHECK_NEAR(y[1][0].item<double>(), 5.0, 1e-12);
    CHECK_NEAR(y[2][0].item<double>(), 4.0, 1e-12);
    CHECK_NEAR(y[3][0].item<double>(), 0.0, 0.0);
}

TEST(interp_fallback_cuda_matches_cpu) {
    if (!torch::cuda::is_available())
        return;
    auto [pts, feats] = clustered_points(3000, 15);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, fb_config(4));
    t.refine_by_density(4, true);
    auto tc = t.with_features(t.features().to(torch::kCUDA));
    torch::manual_seed(16);
    auto q = torch::rand({2000, 3}, torch::kFloat64) * 36.0 - 6.0;
    for (auto mode : {InterpMode::Nearest, InterpMode::Multilinear}) {
        auto a = interpolate(t, q, mode, InterpFallback::CoarserAncestor);
        auto b = interpolate(tc, q, mode, InterpFallback::CoarserAncestor);
        CHECK(b.is_cuda());
        CHECK_ALLCLOSE(b.cpu(), a, 1e-12, 1e-12);
    }
}

TEST(interp_fallback_continuous_mode) {
    // ContinuousMultilinear + CoarserAncestor: covered points keep the tent result exactly,
    // uncovered points get the same ancestor mean as in Nearest mode.
    auto [pts, feats] = clustered_points(3000, 21);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, fb_config(4));
    t.refine_by_density(4, true);
    torch::manual_seed(22);
    auto q = torch::rand({2000, 3}, torch::kFloat64) * 36.0 - 6.0;
    auto c0 = interpolate(t, q, InterpMode::ContinuousMultilinear);
    auto c1 = interpolate(t, q, InterpMode::ContinuousMultilinear, InterpFallback::CoarserAncestor);
    auto n1 = interpolate(t, q, InterpMode::Nearest, InterpFallback::CoarserAncestor);
    auto ones = t.with_features(torch::ones({t.size(), 1}, torch::kFloat64));
    auto covered = interpolate(ones, q, InterpMode::Nearest).squeeze(1) > 0.5;
    auto uncovered = covered.logical_not();
    CHECK(covered.any().item<bool>());
    CHECK(uncovered.any().item<bool>());
    CHECK_ALLCLOSE(c1.index({covered}), c0.index({covered}), 0.0, 0.0);
    CHECK_ALLCLOSE(c1.index({uncovered}), n1.index({uncovered}), 1e-12, 1e-12);
    CHECK((c1.index({uncovered}).abs().sum() > 0).item<bool>());
}
