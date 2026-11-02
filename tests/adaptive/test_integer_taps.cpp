// CenterSampled / VolumeWeighted taps resolve on the integer grid. A tap of an output leaf is
// the centre of the neighbouring output-level cell; where that cell is split into finer
// leaves, the centre is a corner they share, and the tap reads the leaf whose lower corner it
// is (cells are half-open). The answer depends only on the keys, so it is the same wherever
// the grid sits in space, also when the voxel size and origin are not binary fractions (the
// float point location these modes used before picked between the touching leaves by
// rounding there).
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveVoxelConfig make_config(double base, double origin) {
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size.assign(3, base);
    cfg.origin.assign(3, origin);
    cfg.max_depth = 3;
    return cfg;
}

// Leaves at levels 0-3: about 4 level-0 cells per axis, refined in two corners.
AdaptiveSparseTensor make_sample(int seed, const AdaptiveVoxelConfig &cfg) {
    torch::manual_seed(seed);
    const double b = cfg.base_voxel_size[0];
    const double o = cfg.origin[0];
    auto pts = torch::cat({torch::rand({150, 3}, torch::kFloat64) * (4 * b),
                           torch::rand({100, 3}, torch::kFloat64) * b}) + o;
    auto t = AdaptiveSparseTensor::from_points(pts, torch::ones({250, 1}, torch::kFloat64), cfg);
    t.refine(AxisAlignedRegion{{o, o, o}, {o + b, o + b, o + b}}, 3);
    t.refine(AxisAlignedRegion{{o + 2 * b, o + 2 * b, o + 2 * b}, {o + 3 * b, o + 3 * b, o + 3 * b}},
             2);
    return t;
}

// The same leaves moved by `cells` level-0 cells along x.
AdaptiveSparseTensor moved(const AdaptiveSparseTensor &s, int64_t cells,
                           const AdaptiveVoxelConfig &cfg) {
    auto c = s.coordinates().clone();
    c.select(1, 0).add_(
        (torch::pow(2.0, s.levels().to(torch::kFloat64)) * static_cast<double>(cells))
            .to(torch::kInt64));
    return AdaptiveSparseTensor::from_voxels(c, s.levels(), s.features(), cfg);
}

AdaptiveKernelMap fresh_map(const AdaptiveSparseTensor &t, const AdaptiveKernelConfig &kc) {
    KernelMapCache::instance().clear();
    return build_adaptive_kernel_map(t, kc);
}

} // namespace

// A topology moved by whole level-0 cells gets the same map, on binary and non-binary grids.
TEST(integer_taps_translation_invariant) {
    for (double base : {2.0, 0.1, 0.05}) {
        const auto cfg = make_config(base, base == 2.0 ? 0.0 : 0.3);
        auto s = make_sample(77, cfg);
        for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted}) {
            for (int shift : {0, -1, -2}) {
                AdaptiveKernelConfig kc;
                kc.mode = mode;
                kc.output_level_shift = shift;
                const auto a = fresh_map(s, kc);
                for (int64_t cells : {int64_t(1), int64_t(37), int64_t(-1000003)}) {
                    const auto b = fresh_map(moved(s, cells, cfg), kc);
                    CHECK(a.kernel_offsets == b.kernel_offsets);
                    CHECK(torch::equal(a.in_index, b.in_index));
                    CHECK(torch::equal(a.out_index, b.out_index));
                    CHECK(torch::equal(a.weight, b.weight));
                }
            }
        }
    }
    KernelMapCache::instance().clear();
}

// A level-0 leaf next to a refined cell: the +x tap is the centre of the refined cell, the
// corner shared by its 8 children; it reads the child whose lower corner that is, (3, 1, 1)
// at level 1 - also on a grid with voxel size 0.1 and origin 0.3, at any position.
TEST(integer_taps_coarse_to_fine_reads_upper_child) {
    for (double base : {1.0, 0.1}) {
        const auto cfg = make_config(base, base == 1.0 ? 0.0 : 0.3);
        for (int64_t x0 : {int64_t(0), int64_t(5), int64_t(123457)}) {
            // Leaf (x0, 0, 0) at level 0, and every child of cell (x0 + 1, 0, 0) at level 1.
            std::vector<int64_t> coords{x0, 0, 0};
            std::vector<int32_t> levels{0};
            for (int i = 0; i < 8; ++i) {
                coords.insert(coords.end(),
                              {2 * (x0 + 1) + (i & 1), (i >> 1) & 1, (i >> 2) & 1});
                levels.push_back(1);
            }
            auto t = AdaptiveSparseTensor::from_voxels(
                torch::tensor(coords, torch::kInt64).view({9, 3}),
                torch::tensor(levels, torch::kInt32), torch::zeros({9, 1}, torch::kFloat64), cfg);
            const int64_t coarse = t.index_of({x0, 0, 0}, 0);
            const int64_t upper = t.index_of({2 * (x0 + 1) + 1, 1, 1}, 1);
            CHECK(coarse >= 0 && upper >= 0);
            AdaptiveKernelConfig kc; // CenterSampled 3^3
            const auto m = fresh_map(t, kc);
            const int64_t slot = kernel_index({1, 0, 0}, 3);
            int64_t hits = 0;
            for (int64_t q = m.kernel_offsets[slot]; q < m.kernel_offsets[slot + 1]; ++q)
                if (m.out_index[q].item<int64_t>() == coarse) {
                    CHECK_EQ(m.in_index[q].item<int64_t>(), upper);
                    ++hits;
                }
            CHECK_EQ(hits, int64_t(1));
        }
    }
    KernelMapCache::instance().clear();
}
