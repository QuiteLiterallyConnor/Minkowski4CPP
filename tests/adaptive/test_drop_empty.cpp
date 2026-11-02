// drop_empty refinement: only children that contain source points are created.
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveVoxelConfig make_config(uint32_t depth = 3, double base = 8.0, int dim = 3) {
    AdaptiveVoxelConfig c;
    const double m = base / double(1u << depth);
    c.base_voxel_size.assign(size_t(dim), base);
    c.min_voxel_size.assign(size_t(dim), m);
    c.max_depth = depth;
    return c;
}

AxisAlignedRegion box(std::vector<double> lo, std::vector<double> hi) {
    return AxisAlignedRegion{std::move(lo), std::move(hi)};
}

// Clustered points (two tight clusters + sparse background) and a smooth 2-channel feature.
std::pair<at::Tensor, at::Tensor> clustered(int64_t n, uint64_t seed) {
    torch::manual_seed(seed);
    auto a = torch::randn({n / 2, 3}, torch::kFloat64) * 1.5 + 5.0;
    auto b = torch::randn({n / 4, 3}, torch::kFloat64) * 3.0 + 20.0;
    auto bg = torch::rand({n - n / 2 - n / 4, 3}, torch::kFloat64) * 32.0 - 4.0;
    auto pts = torch::cat({a, b, bg});
    auto feats = torch::stack({torch::sin(pts.select(1, 0) / 3.0), torch::cos(pts.select(1, 1) / 5.0)}, 1);
    return {pts, feats};
}

// Number of source points inside every leaf.
at::Tensor point_counts(const AdaptiveSparseTensor &t, const at::Tensor &pts) {
    auto ids = torch::arange(1, t.size() + 1, torch::kFloat64).unsqueeze(1);
    auto leaf = interpolate(t.with_features(ids), pts, InterpMode::Nearest).squeeze(1).round().to(torch::kInt64);
    return torch::bincount(leaf, {}, t.size() + 1).slice(0, 1);
}

// `dropped` must equal `full` restricted to its leaves that contain at least one point.
void check_equals_filtered(const AdaptiveSparseTensor &full, const AdaptiveSparseTensor &dropped,
                           const at::Tensor &pts) {
    auto keep = point_counts(full, pts) > 0;
    CHECK_EQ(dropped.size(), keep.sum().item<int64_t>());
    CHECK(torch::equal(dropped.coordinates(), full.coordinates().index({keep})));
    CHECK(torch::equal(dropped.levels(), full.levels().index({keep})));
    CHECK_ALLCLOSE(dropped.features(), full.features().index({keep}), 0, 0);
    CHECK((point_counts(dropped, pts) > 0).all().item<bool>());
    CHECK(!dropped.has_overlapping_leaves());
}

}  // namespace

TEST(drop_empty_refine_region_keeps_only_occupied_children) {
    auto pts = torch::tensor({{0.5, 0.5, 0.5}, {5.0, 5.0, 5.0}, {5.5, 5.0, 5.0}}, torch::kFloat64);
    auto feats = torch::ones({3, 1}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config());
    auto full = t;
    full.refine(box({0, 0, 0}, {8, 8, 8}), 1);
    CHECK_EQ(full.size(), int64_t(8));
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1, RefinePolicy::Replicate, /*drop_empty=*/true);
    CHECK_EQ(t.size(), int64_t(2));
    CHECK(t.index_of({0, 0, 0}, 1) >= 0);
    CHECK(t.index_of({1, 1, 1}, 1) >= 0);
    CHECK(t.index_of({0, 0, 0}, 0) < 0);
}

TEST(drop_empty_region_equals_default_without_empty_leaves) {
    auto [pts, feats] = clustered(3000, 1);
    auto base = AdaptiveSparseTensor::from_points(pts, feats, make_config(4));
    auto full = base, dropped = base;
    full.refine(box({-4, -4, -4}, {14, 14, 14}), 4);
    dropped.refine(box({-4, -4, -4}, {14, 14, 14}), 4, RefinePolicy::Replicate, true);
    CHECK(dropped.size() < full.size());
    check_equals_filtered(full, dropped, pts);
}

TEST(drop_empty_density_equals_default_without_empty_leaves) {
    auto [pts, feats] = clustered(4000, 2);
    for (int64_t k : {0, 4, 16, 64}) {
        auto full = AdaptiveSparseTensor::from_points(pts, feats, make_config(4));
        auto dropped = full;
        full.refine_by_density(k);
        dropped.refine_by_density(k, true);
        check_equals_filtered(full, dropped, pts);
    }
}

TEST(drop_empty_variance_equals_default_without_empty_leaves) {
    auto [pts, feats] = clustered(4000, 3);
    auto full = AdaptiveSparseTensor::from_points(pts, feats, make_config(4));
    auto dropped = full;
    full.refine_by_variance(0.01);
    dropped.refine_by_variance(0.01, true);
    CHECK(dropped.size() <= full.size());
    check_equals_filtered(full, dropped, pts);
}

TEST(drop_empty_refine_if_never_creates_point_free_leaves) {
    auto [pts, feats] = clustered(2000, 4);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config(3));
    int calls = 0;
    t.refine_if([&](int64_t, const std::array<double, 4> &lo, const std::array<double, 4> &,
                    uint32_t) {
        ++calls;
        return lo[0] < 12.0;  // geometric criterion
    },
                true);
    CHECK(calls > 0);
    CHECK((point_counts(t, pts) > 0).all().item<bool>());
    CHECK(!t.has_overlapping_leaves());
    // Every leaf whose box starts below x = 12 reached max_depth.
    auto [lo, hi] = t.bounds();
    auto left = lo.select(1, 0) < 12.0;
    CHECK((t.levels().index({left}) == 3).all().item<bool>());
}

TEST(drop_empty_split_conserves_sums) {
    auto [pts, feats] = clustered(2000, 5);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config(3));
    const auto total = t.features().sum(0);
    t.refine(box({-8, -8, -8}, {40, 40, 40}), 3, RefinePolicy::Split, true);
    CHECK_ALLCLOSE(t.features().sum(0), total, 1e-9, 1e-9);
    CHECK((point_counts(t, pts) > 0).all().item<bool>());
}

TEST(drop_empty_handles_negative_coordinates_and_2d) {
    // 2-D, points on both sides of the origin.
    auto pts = torch::tensor({{-0.5, -7.5}, {-7.9, -0.1}, {3.0, 1.0}}, torch::kFloat64);
    auto feats = torch::tensor({{1.0}, {2.0}, {3.0}}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config(2, 8.0, 2));
    t.refine(box({-8, -8}, {8, 8}), 2, RefinePolicy::Replicate, true);
    CHECK_EQ(t.size(), int64_t(3));
    // Level-2 cells have size 2: (-0.5,-7.5) -> (-1,-4); (-7.9,-0.1) -> (-4,-1); (3,1) -> (1,0).
    CHECK(t.index_of_key({-1, -4}, 2) >= 0);
    CHECK(t.index_of_key({-4, -1}, 2) >= 0);
    CHECK(t.index_of_key({1, 0}, 2) >= 0);
    // Replicate copies the level-0 mean of the two points in cell (-1,-1).
    CHECK_NEAR(t.features()[t.index_of_key({-4, -1}, 2)][0].item<double>(), 1.5, 1e-12);
}

TEST(drop_empty_gradients_reach_parent_once_per_kept_child) {
    auto pts = torch::tensor({{0.5, 0.5, 0.5}, {5.0, 5.0, 5.0}, {9.0, 1.0, 1.0}}, torch::kFloat64);
    auto feats = torch::ones({3, 1}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config());
    auto f = t.features().detach().clone().set_requires_grad(true);
    auto u = t.with_features(f);
    u.refine(box({0, 0, 0}, {8, 8, 8}), 1, RefinePolicy::Replicate, true);
    u.features().sum().backward();
    // Leaf (0,0,0) kept 2 children, leaf (1,0,0) is outside the region.
    CHECK_NEAR(f.grad()[t.index_of({0, 0, 0}, 0)][0].item<double>(), 2.0, 1e-12);
    CHECK_NEAR(f.grad()[t.index_of({1, 0, 0}, 0)][0].item<double>(), 1.0, 1e-12);
}

TEST(drop_empty_requires_source_points) {
    auto coords = torch::zeros({1, 3}, torch::kInt64);
    auto levels = torch::zeros({1}, torch::kInt32);
    auto t = AdaptiveSparseTensor::from_voxels(coords, levels, torch::ones({1, 1}, torch::kFloat64),
                                               make_config());
    CHECK_THROWS(t.refine(box({0, 0, 0}, {8, 8, 8}), 1, RefinePolicy::Replicate, true));
    CHECK_THROWS(t.refine_if([](int64_t, const std::array<double, 4> &,
                                const std::array<double, 4> &, uint32_t) { return true; },
                             true));
    CHECK_THROWS(t.refine_by_density(0, true));
    // The default keeps working without source points.
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1);
    CHECK_EQ(t.size(), int64_t(8));
}

TEST(drop_empty_default_is_unchanged) {
    auto [pts, feats] = clustered(2000, 6);
    auto a = AdaptiveSparseTensor::from_points(pts, feats, make_config(3));
    auto b = a;
    a.refine_by_density(8);
    b.refine_by_density(8, false);
    CHECK(torch::equal(a.coordinates(), b.coordinates()));
    CHECK(torch::equal(a.levels(), b.levels()));
    CHECK(torch::equal(a.features(), b.features()));
    // Default refinement creates point-free children.
    CHECK((point_counts(a, pts) == 0).any().item<bool>());
}
