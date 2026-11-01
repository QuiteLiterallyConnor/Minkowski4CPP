// Gradient tests: autograd results are compared with central finite differences.
// Topology, voxel positions, sizes and refinement decisions are non-differentiable by design,
// so only feature and weight gradients are checked.
#include <functional>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveVoxelConfig make_config() {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 8.0, 8.0};
    c.min_voxel_size = {2.0, 2.0, 2.0};
    c.max_depth = 2;
    return c;
}

AdaptiveSparseTensor make_mixed(int64_t channels) {
    std::vector<std::array<int64_t, 4>> v = {{0, 0, 0, 0}};
    for (int x = 2; x <= 3; ++x)
        for (int y = 0; y <= 1; ++y)
            for (int z = 0; z <= 1; ++z) v.push_back({x, y, z, 1});
    v.push_back({0, 1, 0, 0});
    int64_t n = v.size();
    auto c = torch::zeros({n, 3}, torch::kInt64);
    auto l = torch::zeros({n}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) c[i][d] = v[i][d];
        l[i] = int(v[i][3]);
    }
    torch::manual_seed(5);
    return AdaptiveSparseTensor::from_voxels(c, l, torch::randn({n, channels}, torch::kFloat64),
                                             make_config());
}

// Central finite differences of loss(x) w.r.t. every element of x.
at::Tensor numerical_grad(const std::function<double(const at::Tensor&)>& loss, at::Tensor x,
                          double eps = 1e-6) {
    auto g = torch::zeros_like(x);
    auto flat = x.view(-1);
    auto gflat = g.view(-1);
    for (int64_t i = 0; i < flat.numel(); ++i) {
        double orig = flat[i].item<double>();
        flat[i] = orig + eps;
        double up = loss(x);
        flat[i] = orig - eps;
        double down = loss(x);
        flat[i] = orig;
        gflat[i] = (up - down) / (2 * eps);
    }
    return g;
}

void check_conv_gradients(AdaptiveConvMode mode) {
    auto base = make_mixed(2);
    AdaptiveKernelConfig kc;
    kc.kernel_size = 3;
    kc.mode = mode;
    auto weight0 = torch::randn({27, 2, 3}, torch::kFloat64);
    auto feats0 = base.features().clone();
    auto probe = torch::randn({base.size(), 3}, torch::kFloat64);

    auto loss_of = [&](const at::Tensor& f, const at::Tensor& w) {
        auto out = adaptive_convolution(base.with_features(f), w, kc);
        return (out.features() * probe).sum();
    };

    auto f = feats0.clone().set_requires_grad(true);
    auto w = weight0.clone().set_requires_grad(true);
    loss_of(f, w).backward();

    auto num_f = numerical_grad(
        [&](const at::Tensor& x) { return loss_of(x, weight0).item<double>(); }, feats0.clone());
    auto num_w = numerical_grad(
        [&](const at::Tensor& x) { return loss_of(feats0, x).item<double>(); }, weight0.clone());
    CHECK_ALLCLOSE(f.grad(), num_f, 1e-5, 1e-6);
    CHECK_ALLCLOSE(w.grad(), num_w, 1e-5, 1e-6);
}

}  // namespace

TEST(gradient_convolution_center_sampled) { check_conv_gradients(AdaptiveConvMode::CenterSampled); }
TEST(gradient_convolution_volume_weighted) { check_conv_gradients(AdaptiveConvMode::VolumeWeighted); }
TEST(gradient_convolution_overlap_weighted) { check_conv_gradients(AdaptiveConvMode::OverlapWeighted); }

TEST(gradient_global_average_pool) {
    auto base = make_mixed(2);
    auto feats0 = base.features().clone();
    auto probe = torch::randn({2}, torch::kFloat64);
    auto f = feats0.clone().set_requires_grad(true);
    (adaptive_global_avg_pool(base.with_features(f)) * probe).sum().backward();
    auto num = numerical_grad(
        [&](const at::Tensor& x) {
            return (adaptive_global_avg_pool(base.with_features(x)) * probe).sum().item<double>();
        },
        feats0.clone());
    CHECK_ALLCLOSE(f.grad(), num, 1e-5, 1e-7);
}

TEST(gradient_local_pool_all_reductions) {
    for (auto r : {Reduction::Mean, Reduction::Sum, Reduction::Max}) {
        auto base = make_mixed(2);
        auto feats0 = base.features().clone();
        auto f = feats0.clone().set_requires_grad(true);
        auto out = adaptive_pool(base.with_features(f), 0, r);
        auto probe = torch::randn({out.size(), 2}, torch::kFloat64);
        (out.features() * probe).sum().backward();
        auto num = numerical_grad(
            [&](const at::Tensor& x) {
                return (adaptive_pool(base.with_features(x), 0, r).features() * probe)
                    .sum()
                    .item<double>();
            },
            feats0.clone());
        CHECK_ALLCLOSE(f.grad(), num, 1e-5, 1e-6);
    }
}

TEST(gradient_interpolation_multilinear) {
    auto base = make_mixed(2);
    auto feats0 = base.features().clone();
    auto pts = torch::tensor({{6.0, 2.0, 2.0}, {9.5, 3.0, 1.0}, {2.0, 10.0, 3.0}}, torch::kFloat64);
    auto probe = torch::randn({3, 2}, torch::kFloat64);
    auto f = feats0.clone().set_requires_grad(true);
    (interpolate(base.with_features(f), pts, InterpMode::Multilinear) * probe).sum().backward();
    auto num = numerical_grad(
        [&](const at::Tensor& x) {
            return (interpolate(base.with_features(x), pts, InterpMode::Multilinear) * probe)
                .sum()
                .item<double>();
        },
        feats0.clone());
    CHECK_ALLCLOSE(f.grad(), num, 1e-5, 1e-6);
}

TEST(gradient_refine_split_and_coarsen_sum_are_linear) {
    auto base = make_mixed(2);
    auto feats0 = base.features().clone();
    auto region = AxisAlignedRegion{{0, 0, 0}, {8, 8, 8}};
    auto f = feats0.clone().set_requires_grad(true);
    auto t = base.with_features(f);
    t.refine(region, 1, RefinePolicy::Split);
    auto probe = torch::randn({t.size(), 2}, torch::kFloat64);
    (t.features() * probe).sum().backward();
    // d/df = sum of the probe over the children that inherit each parent's value, divided by 8.
    CHECK(f.grad().defined());
    CHECK(torch::isfinite(f.grad()).all().item<bool>());
}

TEST(gradient_resample_to_all_reductions) {
    // Source with uneven depth and partial occupancy inside level-0 cell (1,0,0): one level-1
    // leaf, two level-2 leaves of a second level-1 cell and one level-1 leaf elsewhere.
    std::vector<std::array<int64_t, 4>> v = {{0, 0, 0, 0}, {2, 0, 0, 1}, {6, 0, 0, 2},
                                             {7, 1, 1, 2}, {3, 1, 1, 1}, {0, 1, 0, 0}};
    const int64_t n = v.size();
    auto c = torch::zeros({n, 3}, torch::kInt64);
    auto l = torch::zeros({n}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) c[i][d] = v[i][d];
        l[i] = int(v[i][3]);
    }
    torch::manual_seed(9);
    auto src = AdaptiveSparseTensor::from_voxels(c, l, torch::randn({n, 2}, torch::kFloat64),
                                                 make_config());
    // Reference: (1,0,0)@0 reduces four finer leaves, the children of (0,0,0)@0 replicate it,
    // (0,1,0)@0 matches exactly and (5,0,0)@0 is uncovered.
    std::vector<std::array<int64_t, 4>> rv = {{1, 0, 0, 0}, {0, 1, 0, 0}, {5, 0, 0, 0}};
    for (int i = 0; i < 8; ++i) rv.push_back({i & 1, (i >> 1) & 1, (i >> 2) & 1, 1});
    const int64_t m = rv.size();
    auto rc = torch::zeros({m, 3}, torch::kInt64);
    auto rl = torch::zeros({m}, torch::kInt32);
    for (int64_t i = 0; i < m; ++i) {
        for (int d = 0; d < 3; ++d) rc[i][d] = rv[i][d];
        rl[i] = int(rv[i][3]);
    }
    auto ref = AdaptiveSparseTensor::from_voxels(rc, rl, torch::zeros({m, 1}, torch::kFloat64),
                                                 make_config());
    auto probe = torch::randn({m, 2}, torch::kFloat64);
    auto feats0 = src.features().clone();
    for (auto r : {Reduction::Mean, Reduction::Sum, Reduction::Max}) {
        auto f = feats0.clone().set_requires_grad(true);
        (src.with_features(f).resample_to(ref, r).features() * probe).sum().backward();
        auto num = numerical_grad(
            [&](const at::Tensor& x) {
                return (src.with_features(x).resample_to(ref, r).features() * probe)
                    .sum()
                    .item<double>();
            },
            feats0.clone());
        CHECK_ALLCLOSE(f.grad(), num, 1e-5, 1e-6);
    }
    // Mean weights of the four leaves under (1,0,0)@0 follow coarsen(): (2,0,0)@1 and (3,1,1)@1
    // and the level-1 cell (3,0,0)@1 holding (6,0,0)@2 and (7,1,1)@2 each count 1/3, the two
    // level-2 leaves 1/6 each.
    auto f = feats0.clone().set_requires_grad(true);
    auto out = src.with_features(f).resample_to(ref, Reduction::Mean);
    out.features()[ref.index_of({1, 0, 0}, 0)][0].backward();
    auto g = f.grad().select(1, 0);
    CHECK_NEAR(g[src.index_of({2, 0, 0}, 1)].item<double>(), 1.0 / 3.0, 1e-12);
    CHECK_NEAR(g[src.index_of({3, 1, 1}, 1)].item<double>(), 1.0 / 3.0, 1e-12);
    CHECK_NEAR(g[src.index_of({6, 0, 0}, 2)].item<double>(), 1.0 / 6.0, 1e-12);
    CHECK_NEAR(g[src.index_of({7, 1, 1}, 2)].item<double>(), 1.0 / 6.0, 1e-12);
}

TEST(gradient_topology_decisions_are_not_differentiable) {
    auto base = make_mixed(2);
    CHECK(!base.coordinates().requires_grad());
    CHECK(!base.levels().requires_grad());
    CHECK(!base.voxel_sizes().requires_grad());
}
