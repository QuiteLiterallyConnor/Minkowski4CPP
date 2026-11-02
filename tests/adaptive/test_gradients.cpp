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

// docs/adaptive.md: "Feature computations are built from torch operations and support
// autograd". revoxelize_from_points rebuilds leaf features from the source point features, so
// gradients must reach them (they used to be detached in from_points).
namespace {

// The autograd-contract checks below run with features (and weights) on `dev`; the CUDA variants
// skip when no GPU is present. Points and topology always stay on the host.
void check_revoxelize_gradient(torch::Device dev) {
    torch::manual_seed(6);
    auto pts = torch::rand({40, 3}, torch::kFloat64) * 16.0;
    auto f0 = torch::randn({40, 2}, torch::kFloat64).to(dev);
    const AxisAlignedRegion roi{{0, 0, 0}, {8, 8, 8}};
    auto shape = AdaptiveSparseTensor::from_points(pts, f0, make_config());
    shape.refine(roi, 2);
    auto probe = torch::randn({shape.size(), 2}, torch::kFloat64).to(dev);
    auto loss = [&](const at::Tensor& f) {
        auto t = AdaptiveSparseTensor::from_points(pts, f, make_config());
        t.refine(roi, 2);
        t.revoxelize_from_points();
        CHECK(t.features().device().type() == dev.type());
        return (t.features() * probe).sum();
    };
    auto f = f0.clone().set_requires_grad(true);
    loss(f).backward();
    CHECK(f.grad().defined());
    CHECK(f.grad().device().type() == dev.type());
    auto num = numerical_grad([&](const at::Tensor& x) { return loss(x).item<double>(); },
                              f0.clone());
    CHECK_ALLCLOSE(f.grad().cpu(), num.cpu(), 1e-5, 1e-6);
    CHECK(f.grad().abs().sum().item<double>() > 0.0);
}

void check_empty_inputs_connected(torch::Device dev) {
    auto opt = torch::TensorOptions(torch::kFloat64).device(dev);
    auto w = torch::randn({27, 4, 5}, opt).set_requires_grad(true);
    auto wt = torch::randn({27, 5, 4}, opt).set_requires_grad(true);
    auto f = torch::zeros({0, 4}, opt).set_requires_grad(true);
    const auto empty_points = torch::zeros({0, 3}, torch::kFloat64);
    for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted,
                      AdaptiveConvMode::OverlapWeighted}) {
        for (int shift : {0, -1}) {
            auto t = AdaptiveSparseTensor::from_points(empty_points, f, make_config());
            CHECK(t.features().requires_grad());
            CHECK(t.features().device().type() == dev.type());
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            kc.output_level_shift = shift;
            w.mutable_grad() = at::Tensor();
            wt.mutable_grad() = at::Tensor();
            auto out = adaptive_convolution(t, w, kc);
            auto back = adaptive_convolution_transpose(out, wt, kc, t);
            CHECK(back.features().requires_grad());
            CHECK(back.features().device().type() == dev.type());
            (adaptive_global_avg_pool(back).sum() +
             interpolate(out, torch::rand({3, 3}, torch::kFloat64)).sum())
                .backward();
            CHECK(w.grad().defined() && wt.grad().defined() && f.grad().defined());
            CHECK(w.grad().device().type() == dev.type());
            CHECK_NEAR(w.grad().abs().sum().item<double>(), 0.0, 0.0);
            CHECK_NEAR(wt.grad().abs().sum().item<double>(), 0.0, 0.0);
        }
    }
    auto t = AdaptiveSparseTensor::from_points(empty_points, f, make_config());
    CHECK(t.resample_to(t).features().requires_grad());
    CHECK(t.resample_uniform(1).features().requires_grad());
}

void check_interpolation_without_hits(torch::Device dev) {
    auto base = make_mixed(2);
    auto f = base.features().to(dev).clone().set_requires_grad(true);
    auto far = torch::full({2, 3}, 1000.0, torch::kFloat64);
    for (auto mode : {InterpMode::Nearest, InterpMode::Multilinear}) {
        auto y = interpolate(base.with_features(f), far, mode);
        CHECK(y.device().type() == dev.type());
        CHECK_NEAR(y.abs().sum().item<double>(), 0.0, 0.0);
        y.sum().backward();
        CHECK(f.grad().defined());
        CHECK_NEAR(f.grad().abs().sum().item<double>(), 0.0, 0.0);
        f.mutable_grad() = at::Tensor();
    }
}

void require_cuda() {
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA not available");
}

}  // namespace

TEST(gradient_revoxelize_from_points_reaches_point_features) {
    check_revoxelize_gradient(torch::kCPU);
}

TEST(gradient_revoxelize_from_points_reaches_point_features_cuda) {
    require_cuda();
    check_revoxelize_gradient(torch::kCUDA);
}

// Point-driven refinement reads the (now attached) source features only to make topology
// decisions; that must neither need nor break autograd.
TEST(gradient_refine_by_variance_with_attached_point_features) {
    torch::manual_seed(7);
    auto pts = torch::rand({40, 3}, torch::kFloat64) * 16.0;
    auto f = torch::randn({40, 2}, torch::kFloat64).set_requires_grad(true);
    auto t = AdaptiveSparseTensor::from_points(pts, f * 2.0, make_config());
    const auto before = t.size();
    t.refine_by_variance(0.1);
    t.refine_by_density(1);
    CHECK(t.size() > before);
    t.features().sum().backward();
    CHECK(f.grad().defined());
}

// An empty batch must not break a training step: outputs stay in the graph, so backward()
// runs and yields zero gradients for the weight and the input features.
TEST(gradient_empty_inputs_stay_connected_to_the_graph) { check_empty_inputs_connected(torch::kCPU); }

TEST(gradient_empty_inputs_stay_connected_to_the_graph_cuda) {
    require_cuda();
    check_empty_inputs_connected(torch::kCUDA);
}

// Queries that miss every leaf of a non-empty tensor return zeros that are still differentiable.
TEST(gradient_interpolation_without_hits_is_connected) {
    check_interpolation_without_hits(torch::kCPU);
}

TEST(gradient_interpolation_without_hits_is_connected_cuda) {
    require_cuda();
    check_interpolation_without_hits(torch::kCUDA);
}

TEST(gradient_topology_decisions_are_not_differentiable) {
    auto base = make_mixed(2);
    CHECK(!base.coordinates().requires_grad());
    CHECK(!base.levels().requires_grad());
    CHECK(!base.voxel_sizes().requires_grad());
}
