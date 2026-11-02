// Topology tests: refinement, coarsening, overlap invariants, feature policies, cache invalidation.
#include <array>
#include <functional>
#include <random>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveVoxelConfig make_config(uint32_t depth = 3, double base = 8.0) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {base, base, base};
    c.min_voxel_size = {base / double(1u << depth), base / double(1u << depth),
                        base / double(1u << depth)};
    c.max_depth = depth;
    return c;
}

// n level-0 voxels along x, one feature channel with value i + 1.
AdaptiveSparseTensor make_row(int n, uint32_t depth = 3) {
    auto coords = torch::zeros({n, 3}, torch::kInt64);
    auto levels = torch::zeros({n}, torch::kInt32);
    auto feats = torch::zeros({n, 1}, torch::kFloat64);
    for (int i = 0; i < n; ++i) {
        coords[i][0] = i;
        feats[i][0] = double(i + 1);
    }
    return AdaptiveSparseTensor::from_voxels(coords, levels, feats, make_config(depth));
}

AxisAlignedRegion box(std::vector<double> lo, std::vector<double> hi) {
    return AxisAlignedRegion{std::move(lo), std::move(hi)};
}

}  // namespace

TEST(topology_empty_tensor) {
    auto t = AdaptiveSparseTensor::from_points(torch::zeros({0, 3}, torch::kFloat64),
                                               torch::zeros({0, 2}, torch::kFloat64),
                                               make_config());
    CHECK_EQ(t.size(), int64_t(0));
    CHECK_EQ(t.coordinates().size(0), int64_t(0));
    for (auto n : t.counts_by_level()) CHECK_EQ(n, int64_t(0));
    t.refine(box({0, 0, 0}, {8, 8, 8}), 2);
    t.coarsen(box({0, 0, 0}, {8, 8, 8}), 0, Reduction::Mean);
    CHECK_EQ(t.size(), int64_t(0));
    CHECK(!t.has_overlapping_leaves());
}

TEST(topology_from_points_quantizes_to_base_level) {
    auto pts = torch::tensor({{1.0, 1.0, 1.0}, {7.9, 0.1, 0.1}, {8.0, 0.0, 0.0}},
                             torch::kFloat64);
    auto feats = torch::tensor({{1.0}, {3.0}, {10.0}}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config());
    CHECK_EQ(t.size(), int64_t(2));
    auto counts = t.counts_by_level();
    CHECK_EQ(counts[0], int64_t(2));
    int64_t i0 = t.index_of({0, 0, 0}, 0);
    int64_t i1 = t.index_of({1, 0, 0}, 0);
    CHECK(i0 >= 0 && i1 >= 0);
    CHECK_NEAR(t.features()[i0][0].item<double>(), 2.0, 1e-12);  // mean of 1 and 3
    CHECK_NEAR(t.features()[i1][0].item<double>(), 10.0, 1e-12);
}

TEST(topology_refine_creates_all_children_at_next_level) {
    auto t = make_row(2);
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1);  // only voxel (0,0,0)
    auto counts = t.counts_by_level();
    CHECK_EQ(counts[0], int64_t(1));
    CHECK_EQ(counts[1], int64_t(8));
    CHECK(t.index_of({0, 0, 0}, 0) < 0);  // parent is no longer an active leaf
    CHECK(t.index_of({1, 0, 0}, 0) >= 0);
    CHECK(!t.has_overlapping_leaves());
}

TEST(topology_refine_region_boundary_is_half_open) {
    auto t = make_row(2);
    // Region [0,8) must not touch the voxel starting at x = 8.
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1);
    CHECK(t.index_of({1, 0, 0}, 0) >= 0);
}

TEST(topology_refine_to_target_level_and_idempotence) {
    auto t = make_row(1);
    t.refine(box({0, 0, 0}, {8, 8, 8}), 2);
    auto counts = t.counts_by_level();
    CHECK_EQ(counts[0], int64_t(0));
    CHECK_EQ(counts[1], int64_t(0));
    CHECK_EQ(counts[2], int64_t(64));
    uint64_t v = t.topology_version();
    t.refine(box({0, 0, 0}, {8, 8, 8}), 2);  // already refined: must be a no-op
    CHECK_EQ(t.size(), int64_t(64));
    CHECK_EQ(t.topology_version(), v);
}

TEST(topology_refine_beyond_max_depth_throws_or_clamps) {
    auto t = make_row(1, 2);
    CHECK_THROWS(t.refine(box({0, 0, 0}, {8, 8, 8}), 3));
}

TEST(topology_refine_replicate_policy_copies_parent_feature) {
    auto t = make_row(1);
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1, RefinePolicy::Replicate);
    CHECK_EQ(t.size(), int64_t(8));
    CHECK_ALLCLOSE(t.features(), torch::ones({8, 1}, torch::kFloat64), 0, 1e-12);
}

TEST(topology_refine_split_policy_conserves_extensive_features) {
    auto t = make_row(1);
    t.features().mul_(8.0);  // parent feature 8
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1, RefinePolicy::Split);
    CHECK_NEAR(t.features().sum().item<double>(), 8.0, 1e-12);
    CHECK_ALLCLOSE(t.features(), torch::ones({8, 1}, torch::kFloat64), 0, 1e-12);
}

TEST(topology_coarsen_mean_sum_max_use_only_occupied_children) {
    // Build a level-1 parent (0,0,0) with only two occupied children: values 2 and 6.
    auto coords = torch::tensor({{0, 0, 0}, {1, 0, 0}}, torch::kInt64);
    auto levels = torch::tensor({1, 1}, torch::kInt32);
    auto feats = torch::tensor({{2.0}, {6.0}}, torch::kFloat64);

    auto run = [&](Reduction r) {
        auto t = AdaptiveSparseTensor::from_voxels(coords, levels, feats, make_config());
        t.coarsen(box({0, 0, 0}, {8, 8, 8}), 0, r);
        CHECK_EQ(t.size(), int64_t(1));
        int64_t i = t.index_of({0, 0, 0}, 0);
        CHECK(i >= 0);
        return t.features()[i][0].item<double>();
    };
    CHECK_NEAR(run(Reduction::Mean), 4.0, 1e-12);  // empty cells do not dilute the mean
    CHECK_NEAR(run(Reduction::Sum), 8.0, 1e-12);
    CHECK_NEAR(run(Reduction::Max), 6.0, 1e-12);
}

TEST(topology_coarsen_only_inside_region) {
    auto t = make_row(2);
    t.refine(box({0, 0, 0}, {16, 8, 8}), 1);  // both voxels -> 16 leaves
    CHECK_EQ(t.size(), int64_t(16));
    t.coarsen(box({0, 0, 0}, {8, 8, 8}), 0, Reduction::Sum);
    auto counts = t.counts_by_level();
    CHECK_EQ(counts[0], int64_t(1));
    CHECK_EQ(counts[1], int64_t(8));
    CHECK(!t.has_overlapping_leaves());
}

TEST(topology_coarsen_requires_full_region_coverage_of_parent) {
    auto t = make_row(1);
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1);
    // The region covers only part of the parent; the parent must stay refined.
    t.coarsen(box({0, 0, 0}, {2, 2, 2}), 0, Reduction::Mean);
    CHECK_EQ(t.counts_by_level()[1], int64_t(8));
}

TEST(topology_refine_then_coarsen_restores_original) {
    auto t = make_row(3);
    auto before_features = t.features().clone();
    auto before_coords = t.coordinates().clone();
    t.refine(box({0, 0, 0}, {24, 8, 8}), 2, RefinePolicy::Replicate);
    t.coarsen(box({0, 0, 0}, {24, 8, 8}), 0, Reduction::Mean);
    CHECK_EQ(t.size(), int64_t(3));
    CHECK(torch::equal(t.coordinates(), before_coords));
    CHECK_ALLCLOSE(t.features(), before_features, 0, 1e-12);
}

TEST(topology_split_then_sum_conserves_total) {
    auto t = make_row(4);
    double total = t.features().sum().item<double>();
    t.refine(box({0, 0, 0}, {16, 8, 8}), 2, RefinePolicy::Split);
    CHECK_NEAR(t.features().sum().item<double>(), total, 1e-9);
    t.coarsen(box({0, 0, 0}, {32, 8, 8}), 0, Reduction::Sum);
    CHECK_NEAR(t.features().sum().item<double>(), total, 1e-9);
}

TEST(topology_repeated_random_modifications_keep_invariants) {
    std::mt19937 rng(1234);
    auto t = make_row(6, 3);
    const double total = t.features().sum().item<double>();
    std::uniform_real_distribution<double> pos(0.0, 48.0);
    std::uniform_int_distribution<uint32_t> lvl(0, 3);
    for (int iter = 0; iter < 200; ++iter) {
        double x = pos(rng), y = pos(rng) / 6.0, z = pos(rng) / 6.0;
        double w = 4.0 + pos(rng) / 4.0;
        auto r = box({x, y, z}, {x + w, y + w, z + w});
        if (iter % 2 == 0)
            t.refine(r, lvl(rng), RefinePolicy::Split);
        else
            t.coarsen(r, lvl(rng), Reduction::Sum);
        CHECK(!t.has_overlapping_leaves());
        CHECK_NEAR(t.features().sum().item<double>(), total, 1e-6);
    }
    // Total physical volume of the leaves is invariant under topology changes.
    auto [lo, hi] = t.bounds();
    double volume = (hi - lo).prod(1).sum().item<double>();
    CHECK_NEAR(volume, 6.0 * 512.0, 1e-6);
}

TEST(topology_refine_by_density_splits_dense_cells_only) {
    // 5 points in cell (0,0,0), 1 point in cell (1,0,0).
    auto pts = torch::tensor({{1.0, 1.0, 1.0}, {1.5, 1.0, 1.0}, {2.0, 1.0, 1.0},
                              {2.5, 1.0, 1.0}, {3.0, 1.0, 1.0}, {9.0, 1.0, 1.0}},
                             torch::kFloat64);
    auto feats = torch::ones({6, 1}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config());
    t.refine_by_density(3);
    CHECK(t.index_of({1, 0, 0}, 0) >= 0);  // sparse cell untouched
    CHECK(t.index_of({0, 0, 0}, 0) < 0);   // dense cell refined
    CHECK(!t.has_overlapping_leaves());
}

TEST(topology_refine_by_density_requires_source_points) {
    auto t = make_row(2);
    CHECK_THROWS(t.refine_by_density(1));
}

TEST(topology_refine_by_variance_splits_high_variance_cells) {
    auto pts = torch::tensor({{1.0, 1.0, 1.0}, {2.0, 1.0, 1.0},   // cell (0,0,0): 0 and 100
                              {9.0, 1.0, 1.0}, {10.0, 1.0, 1.0}},  // cell (1,0,0): 5 and 5
                             torch::kFloat64);
    auto feats = torch::tensor({{0.0}, {100.0}, {5.0}, {5.0}}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config());
    t.refine_by_variance(1.0);
    CHECK(t.index_of({0, 0, 0}, 0) < 0);
    CHECK(t.index_of({1, 0, 0}, 0) >= 0);
}

TEST(topology_revoxelize_from_points_uses_original_points) {
    auto pts = torch::tensor({{0.5, 0.5, 0.5}, {3.5, 0.5, 0.5}}, torch::kFloat64);
    auto feats = torch::tensor({{1.0}, {5.0}}, torch::kFloat64);
    auto t = AdaptiveSparseTensor::from_points(pts, feats, make_config());
    t.refine(box({0, 0, 0}, {8, 8, 8}), 2, RefinePolicy::Replicate);
    t.revoxelize_from_points();
    // Points land in different level-2 cells (size 2), so the result is exact, not replicated.
    CHECK(t.index_of({0, 0, 0}, 2) >= 0);
    CHECK(t.index_of({1, 0, 0}, 2) >= 0);
    CHECK_NEAR(t.features()[t.index_of({0, 0, 0}, 2)][0].item<double>(), 1.0, 1e-12);
    CHECK_NEAR(t.features()[t.index_of({1, 0, 0}, 2)][0].item<double>(), 5.0, 1e-12);
}

TEST(topology_version_changes_only_on_modification) {
    auto t = make_row(2);
    uint64_t v0 = t.topology_version();
    (void)t.coordinates();
    CHECK_EQ(t.topology_version(), v0);
    t.refine(box({0, 0, 0}, {8, 8, 8}), 1);
    uint64_t v1 = t.topology_version();
    CHECK(v1 != v0);
    t.coarsen(box({0, 0, 0}, {8, 8, 8}), 0, Reduction::Mean);
    CHECK(t.topology_version() != v1);
}

TEST(topology_kernel_map_cache_hits_and_invalidation) {
    KernelMapCache::instance().clear();
    auto t = make_row(3);
    AdaptiveKernelConfig kc;
    kc.kernel_size = 3;

    auto m1 = build_adaptive_kernel_map(t, kc);
    auto s1 = KernelMapCache::instance().stats();
    auto m2 = build_adaptive_kernel_map(t, kc);
    auto s2 = KernelMapCache::instance().stats();
    CHECK_EQ(s2.hits, s1.hits + 1);
    CHECK_EQ(s2.misses, s1.misses);
    CHECK_EQ(m1.in_index.numel(), m2.in_index.numel());

    t.refine(box({0, 0, 0}, {8, 8, 8}), 1);
    auto m3 = build_adaptive_kernel_map(t, kc);
    auto s3 = KernelMapCache::instance().stats();
    CHECK_EQ(s3.misses, s2.misses + 1);
    CHECK(m3.in_index.numel() != m1.in_index.numel());

    AdaptiveKernelConfig kc2 = kc;
    kc2.kernel_size = 5;  // different kernel config must not reuse the entry
    build_adaptive_kernel_map(t, kc2);
    CHECK_EQ(KernelMapCache::instance().stats().misses, s3.misses + 1);
}

TEST(topology_addition_requires_matching_topology) {
    auto a = make_row(3);
    auto b = make_row(3);
    auto sum = a + b;
    CHECK_ALLCLOSE(sum.features(), a.features() * 2, 0, 1e-12);

    b.refine(box({0, 0, 0}, {8, 8, 8}), 1);
    CHECK_THROWS(a + b);
    CHECK_THROWS(concatenate(a, b));

    auto aligned = b.resample_to(a);  // explicit alignment operation
    auto ok = a + aligned;
    CHECK_EQ(ok.size(), a.size());
}

TEST(topology_counts_by_level_has_depth_plus_one_entries) {
    auto t = make_row(2, 4);
    CHECK_EQ(t.counts_by_level().size(), size_t(5));
}

namespace {

AdaptiveSparseTensor from_keys(const std::vector<std::array<int64_t, 4>>& keys,
                               const std::vector<double>& values, AdaptiveVoxelConfig cfg) {
    const int64_t n = static_cast<int64_t>(keys.size());
    auto c = torch::zeros({n, 3}, torch::kInt64);
    auto l = torch::zeros({n}, torch::kInt32);
    auto f = torch::zeros({n, 1}, torch::kFloat64);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) c[i][d] = keys[i][d];
        l[i] = int(keys[i][3]);
        f[i][0] = values[i];
    }
    return AdaptiveSparseTensor::from_voxels(c, l, f, cfg);
}

// Random non-overlapping leaves below a 3x2x2 block of level-0 cells: each cell is a leaf, empty,
// or split into its children (recursively up to max_depth). Empty cells make occupancy partial.
AdaptiveSparseTensor random_tree(uint32_t depth, uint32_t seed, int64_t channels) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<std::array<int64_t, 4>> keys;
    std::function<void(std::array<int64_t, 3>, uint32_t)> visit = [&](std::array<int64_t, 3> c,
                                                                       uint32_t level) {
        const double r = u(rng);
        if (level > 0 && r < 0.2) return;  // empty
        if (level == depth || r < 0.5) {
            keys.push_back({c[0], c[1], c[2], int64_t(level)});
            return;
        }
        for (int i = 0; i < 8; ++i)
            visit({2 * c[0] + (i & 1), 2 * c[1] + ((i >> 1) & 1), 2 * c[2] + ((i >> 2) & 1)},
                  level + 1);
    };
    for (int64_t x = 0; x < 3; ++x)
        for (int64_t y = 0; y < 2; ++y)
            for (int64_t z = 0; z < 2; ++z) visit({x, y, z}, 0);
    const int64_t n = static_cast<int64_t>(keys.size());
    auto c = torch::zeros({n, 3}, torch::kInt64);
    auto l = torch::zeros({n}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) c[i][d] = keys[i][d];
        l[i] = int(keys[i][3]);
    }
    torch::manual_seed(seed);
    return AdaptiveSparseTensor::from_voxels(c, l, torch::randn({n, channels}, torch::kFloat64),
                                             make_config(depth));
}

}  // namespace

TEST(topology_resample_to_rejects_a_different_grid) {
    auto a = make_row(3);
    auto shifted = make_config();
    shifted.origin = {1.0, 0.0, 0.0};
    auto b = from_keys({{0, 0, 0, 0}}, {1.0}, shifted);
    CHECK_THROWS(b.resample_to(a));
    CHECK_THROWS(a.resample_to(b));

    auto finer_base = make_config(3, 4.0);
    CHECK_THROWS(from_keys({{0, 0, 0, 0}}, {1.0}, finer_base).resample_to(a));

    // Same grid: accepted.
    CHECK_EQ(from_keys({{0, 0, 0, 1}}, {1.0}, make_config()).resample_to(a).size(), a.size());

    // A different max_depth on the same grid is the same set of cells: accepted, and a
    // level-4 leaf (deeper than a's max_depth) reduces into the level-0 leaf containing it.
    auto deep = from_keys({{0, 0, 0, 4}, {16, 0, 0, 0}}, {2.0, 5.0}, make_config(4));
    auto onto_a = deep.resample_to(a, Reduction::Sum);
    CHECK_NEAR(onto_a.features()[a.index_of({0, 0, 0}, 0)][0].item<double>(), 2.0, 0.0);
    CHECK_NEAR(onto_a.features()[a.index_of({1, 0, 0}, 0)][0].item<double>(), 0.0, 0.0);
    auto back = a.resample_to(deep, Reduction::Mean);
    CHECK_NEAR(back.features()[deep.index_of({0, 0, 0}, 4)][0].item<double>(), 1.0, 0.0);
    CHECK_NEAR(back.features()[deep.index_of({16, 0, 0}, 0)][0].item<double>(), 0.0, 0.0);
}

TEST(topology_resample_to_mean_matches_coarsen_on_mixed_depths) {
    // Inside the level-0 cell: one level-1 leaf with value 0 and the 8 level-2 children of the
    // level-1 cell next to it, value 8. coarsen() averages occupied children per level: 4.
    std::vector<std::array<int64_t, 4>> keys = {{0, 0, 0, 1}};
    std::vector<double> values = {0.0};
    for (int i = 0; i < 8; ++i) {
        keys.push_back({2 + (i & 1), (i >> 1) & 1, (i >> 2) & 1, 2});
        values.push_back(8.0);
    }
    auto t = from_keys(keys, values, make_config());
    auto ref = from_keys({{0, 0, 0, 0}}, {0.0}, make_config());

    auto coarsened = adaptive_pool(t, 0, Reduction::Mean);
    CHECK_EQ(coarsened.size(), int64_t(1));
    CHECK_NEAR(coarsened.features()[0][0].item<double>(), 4.0, 1e-12);
    CHECK_NEAR(t.resample_to(ref, Reduction::Mean).features()[0][0].item<double>(), 4.0, 1e-12);
    CHECK_NEAR(t.resample_to(ref, Reduction::Sum).features()[0][0].item<double>(), 64.0, 1e-12);
    CHECK_NEAR(t.resample_to(ref, Reduction::Max).features()[0][0].item<double>(), 8.0, 1e-12);
}

TEST(topology_resample_to_coarser_reference_equals_adaptive_pool) {
    for (uint32_t seed : {1u, 2u, 3u, 4u, 5u, 6u}) {
        auto t = random_tree(3, seed, 2);
        for (uint32_t level : {0u, 1u, 2u})
            for (auto r : {Reduction::Mean, Reduction::Sum, Reduction::Max}) {
                auto pooled = adaptive_pool(t, level, r);
                auto resampled = t.resample_to(pooled, r);
                CHECK_EQ(resampled.size(), pooled.size());
                CHECK_ALLCLOSE(resampled.features(), pooled.features(), 1e-12, 1e-12);
            }
    }
}

TEST(topology_resample_to_replicates_coarser_and_zero_fills_uncovered) {
    auto t = from_keys({{0, 0, 0, 0}}, {3.0}, make_config());
    // Reference: the 8 children of the source leaf plus a far-away leaf the source does not cover.
    std::vector<std::array<int64_t, 4>> keys;
    for (int i = 0; i < 8; ++i) keys.push_back({i & 1, (i >> 1) & 1, (i >> 2) & 1, 1});
    keys.push_back({5, 0, 0, 0});
    auto ref = from_keys(keys, std::vector<double>(keys.size(), 0.0), make_config());
    for (auto r : {Reduction::Mean, Reduction::Sum, Reduction::Max}) {
        auto out = t.resample_to(ref, r);
        CHECK_EQ(out.size(), int64_t(9));
        CHECK_NEAR(out.features()[ref.index_of({5, 0, 0}, 0)][0].item<double>(), 0.0, 0.0);
        for (int i = 0; i < 8; ++i)
            CHECK_NEAR(out.features()[ref.index_of({i & 1, (i >> 1) & 1, (i >> 2) & 1}, 1)][0]
                           .item<double>(),
                       3.0, 1e-12);
    }
}
