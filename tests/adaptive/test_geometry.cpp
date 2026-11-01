// Geometry tests: keys, hashing, physical bounds, parent/child relations, config validation.
#include <array>
#include <limits>
#include <unordered_set>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveVoxelGeometry<3> make_geometry(std::array<double, 3> base = {8.0, 8.0, 8.0},
                                       std::array<double, 3> origin = {0.0, 0.0, 0.0},
                                       uint32_t depth = 6) {
    AdaptiveVoxelGeometry<3> g;
    g.origin = origin;
    g.base_voxel_size = base;
    g.max_depth = depth;
    return g;
}

}  // namespace

TEST(geometry_voxel_size_halves_per_level) {
    auto g = make_geometry();
    for (uint32_t l = 0; l <= 6; ++l) {
        auto s = g.voxel_size(l);
        for (int d = 0; d < 3; ++d) CHECK_NEAR(s[d], 8.0 / double(1u << l), 0.0);
    }
}

TEST(geometry_anisotropic_voxel_size) {
    auto g = make_geometry({8.0, 4.0, 2.0});
    auto s = g.voxel_size(3);
    CHECK_NEAR(s[0], 1.0, 0.0);
    CHECK_NEAR(s[1], 0.5, 0.0);
    CHECK_NEAR(s[2], 0.25, 0.0);
}

TEST(geometry_bounds_and_center) {
    auto g = make_geometry({8.0, 4.0, 2.0}, {10.0, -5.0, 1.0});
    AdaptiveVoxelKey<3> k{{1, -1, 0}, 1};  // size (4, 2, 1)
    auto [lo, hi] = g.bounds(k);
    CHECK_NEAR(lo[0], 14.0, 1e-12);
    CHECK_NEAR(hi[0], 18.0, 1e-12);
    CHECK_NEAR(lo[1], -7.0, 1e-12);
    CHECK_NEAR(hi[1], -5.0, 1e-12);
    CHECK_NEAR(lo[2], 1.0, 1e-12);
    CHECK_NEAR(hi[2], 2.0, 1e-12);
    auto c = g.center(k);
    CHECK_NEAR(c[0], 16.0, 1e-12);
    CHECK_NEAR(c[1], -6.0, 1e-12);
    CHECK_NEAR(c[2], 1.5, 1e-12);
}

TEST(geometry_half_open_ownership) {
    auto g = make_geometry();
    // Lower boundary belongs to the voxel above it, upper boundary does not.
    auto a = g.locate({8.0, 0.0, 0.0}, 0);
    CHECK_EQ(a.coordinates[0], 1);
    CHECK_EQ(a.coordinates[1], 0);
    auto b = g.locate({std::nextafter(8.0, 0.0), 0.0, 0.0}, 0);
    CHECK_EQ(b.coordinates[0], 0);
    // Negative coordinates floor rather than truncate.
    auto c = g.locate({-0.001, -8.0, -8.001}, 0);
    CHECK_EQ(c.coordinates[0], -1);
    CHECK_EQ(c.coordinates[1], -1);
    CHECK_EQ(c.coordinates[2], -2);
}

TEST(geometry_locate_is_consistent_with_bounds) {
    auto g = make_geometry({3.0, 5.0, 7.0}, {0.5, -0.25, 2.0});
    for (uint32_t l = 0; l <= 6; ++l) {
        for (double x : {-13.37, -0.01, 0.0, 0.5, 4.2, 99.9}) {
            std::array<double, 3> p = {x, x * 0.5, -x};
            auto k = g.locate(p, l);
            auto [lo, hi] = g.bounds(k);
            for (int d = 0; d < 3; ++d) {
                CHECK(p[d] >= lo[d]);
                CHECK(p[d] < hi[d]);
            }
        }
    }
}

TEST(geometry_parent_child_roundtrip) {
    AdaptiveVoxelKey<3> k{{-3, 5, 0}, 2};
    auto kids = AdaptiveVoxelGeometry<3>::children(k);
    CHECK_EQ(kids.size(), size_t(8));
    std::unordered_set<AdaptiveVoxelKey<3>, AdaptiveVoxelKeyHash<3>> unique;
    for (auto& c : kids) {
        CHECK_EQ(c.level, 3u);
        CHECK(AdaptiveVoxelGeometry<3>::parent(c) == k);
        CHECK(AdaptiveVoxelGeometry<3>::is_ancestor(k, c));
        CHECK(!AdaptiveVoxelGeometry<3>::is_ancestor(c, k));
        unique.insert(c);
    }
    CHECK_EQ(unique.size(), size_t(8));
}

TEST(geometry_parent_of_negative_coordinates_floors) {
    AdaptiveVoxelKey<3> k{{-1, -2, -3}, 4};
    auto p = AdaptiveVoxelGeometry<3>::parent(k);
    CHECK_EQ(p.level, 3u);
    CHECK_EQ(p.coordinates[0], -1);
    CHECK_EQ(p.coordinates[1], -1);
    CHECK_EQ(p.coordinates[2], -2);
}

TEST(geometry_children_tile_parent_exactly) {
    auto g = make_geometry({8.0, 4.0, 2.0}, {1.0, 2.0, 3.0});
    AdaptiveVoxelKey<3> parent{{2, -1, 3}, 1};
    auto [plo, phi] = g.bounds(parent);
    double parent_vol = 1.0, child_vol = 0.0;
    for (int d = 0; d < 3; ++d) parent_vol *= phi[d] - plo[d];
    for (auto& c : AdaptiveVoxelGeometry<3>::children(parent)) {
        auto [lo, hi] = g.bounds(c);
        double v = 1.0;
        for (int d = 0; d < 3; ++d) {
            CHECK(lo[d] >= plo[d] - 1e-12);
            CHECK(hi[d] <= phi[d] + 1e-12);
            v *= hi[d] - lo[d];
        }
        child_vol += v;
    }
    CHECK_NEAR(child_vol, parent_vol, 1e-9);
}

TEST(geometry_keys_at_different_levels_never_collide) {
    std::unordered_set<AdaptiveVoxelKey<3>, AdaptiveVoxelKeyHash<3>> set;
    size_t expected = 0;
    for (uint32_t l = 0; l <= 5; ++l)
        for (int64_t x = -3; x <= 3; ++x)
            for (int64_t y = -3; y <= 3; ++y)
                for (int64_t z = -3; z <= 3; ++z) {
                    set.insert({{x, y, z}, l});
                    ++expected;
                }
    CHECK_EQ(set.size(), expected);
}

TEST(geometry_large_coordinates_roundtrip) {
    auto g = make_geometry({1.0, 1.0, 1.0}, {0.0, 0.0, 0.0}, 5);
    const int64_t big = int64_t(1) << 40;
    AdaptiveVoxelKey<3> k{{big, -big, big + 7}, 5};
    auto c = g.center(k);
    auto back = g.locate(c, 5);
    CHECK(back == k);
    auto p = AdaptiveVoxelGeometry<3>::parent(k);
    CHECK_EQ(p.coordinates[0], big / 2);
    CHECK_EQ(p.coordinates[1], -big / 2);
}

TEST(geometry_two_and_four_dimensional) {
    AdaptiveVoxelGeometry<2> g2;
    g2.origin = {0, 0};
    g2.base_voxel_size = {2.0, 1.0};
    g2.max_depth = 3;
    CHECK_EQ(AdaptiveVoxelGeometry<2>::children({{0, 0}, 0}).size(), size_t(4));

    AdaptiveVoxelGeometry<4> g4;
    g4.origin = {0, 0, 0, 0};
    g4.base_voxel_size = {1, 1, 1, 1};
    g4.max_depth = 2;
    CHECK_EQ(AdaptiveVoxelGeometry<4>::children({{0, 0, 0, 0}, 0}).size(), size_t(16));
}

TEST(config_accepts_consistent_resolution) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 8.0, 8.0};
    c.min_voxel_size = {0.125, 0.125, 0.125};
    c.max_depth = 6;
    c.validate();
}

TEST(config_rejects_inconsistent_resolution) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 8.0, 8.0};
    c.min_voxel_size = {0.125, 0.125, 0.125};

    c.max_depth = 5;  // 8 / 2^5 = 0.25 != 0.125
    CHECK_THROWS(c.validate());

    c.max_depth = 7;
    CHECK_THROWS(c.validate());

    c.max_depth = 6;
    c.min_voxel_size = {0.125, 0.125};  // dimension mismatch
    CHECK_THROWS(c.validate());

    c.min_voxel_size = {0.3, 0.3, 0.3};  // ratio is not a power of two
    CHECK_THROWS(c.validate());
}

TEST(config_rejects_nonpositive_and_unsupported_depth) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {1.0, 0.0, 1.0};
    c.min_voxel_size = {1.0, 1.0, 1.0};
    c.max_depth = 0;
    CHECK_THROWS(c.validate());

    c.base_voxel_size = {1.0, 1.0, 1.0};
    c.max_depth = AdaptiveVoxelConfig::kMaxSupportedDepth + 1;
    c.min_voxel_size = {};
    CHECK_THROWS(c.validate());
}

namespace {

AdaptiveVoxelConfig cube_config(uint32_t depth) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 8.0, 8.0};
    c.max_depth = depth;
    return c;
}

AdaptiveSparseTensor one_leaf(int64_t x, int32_t level, uint32_t depth) {
    auto c = torch::tensor({{x, int64_t(0), int64_t(0)}}, torch::kInt64);
    auto l = torch::tensor({level}, torch::kInt32);
    return AdaptiveSparseTensor::from_voxels(c, l, torch::ones({1, 1}, torch::kFloat64),
                                             cube_config(depth));
}

}  // namespace

// A level-l coordinate c covers finest-level coordinates c * 2^(max_depth - l); they must stay
// below 2^52 in magnitude so 2*c+bit cannot overflow and bounds/centres are exact doubles.
TEST(topology_rejects_coordinates_outside_the_exact_range) {
    const int64_t lim0 = int64_t(1) << 49;  // level 0 with max_depth 3
    CHECK_EQ(one_leaf(lim0 - 1, 0, 3).size(), int64_t(1));
    CHECK_EQ(one_leaf(-lim0, 0, 3).size(), int64_t(1));
    CHECK_THROWS(one_leaf(lim0, 0, 3));
    CHECK_THROWS(one_leaf(-lim0 - 1, 0, 3));

    const int64_t lim3 = int64_t(1) << 52;  // finest level
    CHECK_EQ(one_leaf(lim3 - 1, 3, 3).size(), int64_t(1));
    CHECK_THROWS(one_leaf(lim3, 3, 3));

    // Far outside: 2*c would overflow int64 on refinement.
    CHECK_THROWS(one_leaf(int64_t(1) << 62, 0, 3));
    CHECK_THROWS(one_leaf(std::numeric_limits<int64_t>::min(), 0, 3));
    CHECK_THROWS(one_leaf(std::numeric_limits<int64_t>::max(), 3, 3));

    // At the deepest supported hierarchy level 0 keeps 2^22 cells per side and direction.
    const uint32_t deepest = AdaptiveVoxelConfig::kMaxSupportedDepth;
    const int64_t lim_deep = int64_t(1) << (52 - deepest);
    CHECK_EQ(one_leaf(lim_deep - 1, 0, deepest).size(), int64_t(1));
    CHECK_THROWS(one_leaf(lim_deep, 0, deepest));
}

TEST(topology_rejects_out_of_range_coordinates_in_any_dimension) {
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {1.0, 1.0};
    cfg.max_depth = 2;
    const int64_t lim = int64_t(1) << 50;  // level 0, max_depth 2
    auto make = [&](int64_t y) {
        return AdaptiveSparseTensor::from_voxels(
            torch::tensor({{int64_t(0), y}, {int64_t(1), int64_t(0)}}, torch::kInt64),
            torch::tensor({0, 0}, torch::kInt32), torch::ones({2, 1}, torch::kFloat64), cfg);
    };
    CHECK_EQ(make(lim - 1).size(), int64_t(2));
    CHECK_THROWS(make(lim));
    CHECK_THROWS(make(-lim - 1));
}

TEST(topology_from_points_rejects_points_outside_the_exact_range) {
    auto cfg = cube_config(3);
    // 1e17 / 8 = 1.25e16 > 2^49 cells at level 0.
    auto far = torch::tensor({{1e17, 0.0, 0.0}}, torch::kFloat64);
    CHECK_THROWS(AdaptiveSparseTensor::from_points(far, torch::ones({1, 1}, torch::kFloat64), cfg));
    auto near = torch::tensor({{1e15, 0.0, 0.0}}, torch::kFloat64);
    CHECK_EQ(AdaptiveSparseTensor::from_points(near, torch::ones({1, 1}, torch::kFloat64), cfg).size(),
             int64_t(1));
}

TEST(topology_refinement_at_the_coordinate_limit_stays_exact) {
    const int64_t c0 = (int64_t(1) << 49) - 1;
    auto t = one_leaf(c0, 0, 3);
    t.refine(AxisAlignedRegion::everything(3), 3);
    CHECK_EQ(t.size(), int64_t(512));
    CHECK(!t.has_overlapping_leaves());
    // The last child of the last child... has finest coordinate 2^52 - 1.
    const int64_t cmax = (int64_t(1) << 52) - 1;
    const int64_t j = t.index_of({cmax, 7, 7}, 3);
    CHECK(j >= 0);
    // Its centre is an exact double and locates back to the same leaf.
    auto centre = t.centers()[j].unsqueeze(0);
    CHECK_EQ(centre[0][0].item<double>(), (double(cmax) + 0.5) * 1.0);
    t.set_features(torch::arange(t.size(), torch::kFloat64).unsqueeze(1));
    CHECK_EQ(interpolate(t, centre, InterpMode::Nearest)[0][0].item<double>(), double(j));
}

TEST(config_supports_anisotropic_resolution) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 4.0, 2.0};
    c.min_voxel_size = {1.0, 0.5, 0.25};
    c.max_depth = 3;
    c.validate();
}
