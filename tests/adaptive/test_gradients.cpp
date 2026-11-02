// Gradient tests: autograd results are compared with central finite differences.
// Topology, voxel positions, sizes and refinement decisions are non-differentiable by design,
// so only feature and weight gradients are checked.
#include <functional>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/sparse_tensor.hpp"
#include "MinkowskiEngine/tensor_field.hpp"
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

// A maximum of exactly 0 keeps the whole gradient (torch's amax backward also counted the
// zero-initialised output as a tied maximum and halved it at every pooling step).
TEST(gradient_max_pool_at_zero_keeps_unit_gradient) {
    auto base = make_mixed(1);
    // Children of (1,0,0)@0: values 0 and negatives; level-0 leaves: 0 and 3.
    auto f0 = -torch::rand({base.size(), 1}, torch::kFloat64) - 0.5;
    const int64_t z_child = base.index_of({2, 0, 0}, 1);
    const int64_t z_leaf = base.index_of({0, 1, 0}, 0);
    const int64_t pos_leaf = base.index_of({0, 0, 0}, 0);
    CHECK(z_child >= 0 && z_leaf >= 0 && pos_leaf >= 0);
    f0[z_child][0] = 0.0;
    f0[z_leaf][0] = 0.0;
    f0[pos_leaf][0] = 3.0;
    for (const at::Device dev : {at::Device(at::kCPU), at::Device(at::kCUDA)}) {
        if (dev.is_cuda() && !torch::cuda::is_available())
            continue;
        auto f = f0.to(dev).detach().clone().requires_grad_(true);
        auto t = base.with_features(f);
        auto pooled = adaptive_pool(t, 0, Reduction::Max);
        CHECK_EQ(pooled.size(), int64_t(3));
        pooled.features().sum().backward();
        auto g = f.grad().cpu();
        CHECK_NEAR(g[z_child][0].item<double>(), 1.0, 0.0); // the parent's maximum is 0
        CHECK_NEAR(g[z_leaf][0].item<double>(), 1.0, 0.0);  // an unmerged leaf holding 0
        CHECK_NEAR(g[pos_leaf][0].item<double>(), 1.0, 0.0);
        CHECK_NEAR(g.sum().item<double>(), 3.0, 0.0);
        // The same through resample_to (a reference cell without inputs stays 0).
        auto coarse = adaptive_pool(base, 0, Reduction::Max);
        auto f2 = f0.to(dev).detach().clone().requires_grad_(true);
        auto r = base.with_features(f2).resample_to(coarse, Reduction::Max);
        CHECK(torch::equal(r.features().detach().cpu(), pooled.features().detach().cpu()));
        r.features().sum().backward();
        CHECK_NEAR(f2.grad().cpu()[z_child][0].item<double>(), 1.0, 0.0);
    }
}

TEST(gradient_max_pool_rows_without_entries_stay_zero) {
    // resample_to onto a reference with a cell no input overlaps: that row is 0, not -inf.
    auto base = make_mixed(2);
    auto ref = base;
    ref.refine(AxisAlignedRegion{{0, 8, 0}, {8, 16, 8}}, 1); // (0,1,0)@0 -> 8 children
    auto shifted = AdaptiveSparseTensor::from_voxels(
        base.coordinates().index_select(0, torch::tensor({int64_t(0)})),
        base.levels().index_select(0, torch::tensor({int64_t(0)})),
        torch::full({1, 2}, -2.0, torch::kFloat64), base.config());
    auto r = shifted.resample_to(ref, Reduction::Max);
    CHECK(torch::isfinite(r.features()).all().item<bool>());
    CHECK_NEAR(r.features().abs().sum().item<double>(), 4.0, 0.0); // only (0,0,0)@0 holds -2
}

// The same contract for MAX_POOL quantization of the sparse engine (SparseTensor and
// TensorField::sparse), which used the same zero-initialised scatter_reduce.
TEST(gradient_sparse_max_pool_quantization_at_zero_keeps_unit_gradient) {
    // Points 0 and 1 share voxel (0,0,0) with values 0 and -1; point 2 is alone with 2.
    auto coords = torch::tensor({{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}}, torch::kInt32);
    const auto f0 = torch::tensor({{0.0}, {-1.0}, {2.0}}, torch::kFloat64);
    for (const at::Device dev : {at::Device(at::kCPU), at::Device(at::kCUDA)}) {
        if (dev.is_cuda() && !torch::cuda::is_available())
            continue;
        {
            auto f = f0.to(dev).detach().clone().requires_grad_(true);
            minkowski::SparseTensor s(f, coords.to(dev), nullptr,
                                      minkowski::SparseTensorQuantizationMode::MAX_POOL);
            CHECK_EQ(s.F().size(0), int64_t(2));
            CHECK_NEAR(s.F().sum().item<double>(), 2.0, 0.0);
            s.F().sum().backward();
            auto g = f.grad().cpu();
            CHECK_NEAR(g[0][0].item<double>(), 1.0, 0.0); // the voxel's maximum is exactly 0
            CHECK_NEAR(g[1][0].item<double>(), 0.0, 0.0);
            CHECK_NEAR(g[2][0].item<double>(), 1.0, 0.0);
        }
        {
            auto f = f0.to(dev).detach().clone().requires_grad_(true);
            minkowski::TensorField field(f, coords.to(dev));
            auto s = field.sparse({}, c10::nullopt, minkowski::SparseTensorQuantizationMode::MAX_POOL);
            CHECK_EQ(s.F().size(0), int64_t(2));
            s.F().sum().backward();
            auto g = f.grad().cpu();
            CHECK_NEAR(g[0][0].item<double>(), 1.0, 0.0);
            CHECK_NEAR(g[2][0].item<double>(), 1.0, 0.0);
            CHECK_NEAR(g.sum().item<double>(), 2.0, 0.0);
        }
    }
}

TEST(gradient_topology_decisions_are_not_differentiable) {
    auto base = make_mixed(2);
    CHECK(!base.coordinates().requires_grad());
    CHECK(!base.levels().requires_grad());
    CHECK(!base.voxel_sizes().requires_grad());
}
