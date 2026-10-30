// Regression tests: the existing uniform sparse tensor APIs must keep working unchanged
// once adaptive support is added.
#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/minkowski.hpp"
#include "utils/test_framework.hpp"

namespace {

minkowski::SparseTensor make_legacy(int64_t n, int64_t channels) {
    torch::manual_seed(3);
    auto coords = std::get<0>(torch::unique_dim(torch::randint(0, 6, {n, 3}, torch::kInt32), 0));
    auto batch = torch::zeros({coords.size(0), 1}, torch::kInt32);
    return minkowski::SparseTensor(torch::randn({coords.size(0), channels}),
                                   torch::cat({batch, coords}, 1));
}

}  // namespace

TEST(regression_sparse_tensor_roundtrip) {
    auto t = make_legacy(100, 4);
    CHECK_EQ(t.dimension(), 3);
    CHECK_EQ(t.F().size(0), t.C().size(0));
    CHECK_EQ(t.F().size(1), int64_t(4));
}

TEST(regression_uniform_convolution_shapes_and_determinism) {
    auto t = make_legacy(150, 3);
    minkowski::MinkowskiConvolution conv(3, 5, 3, 1, 1, false, false,
                                         minkowski::ConvolutionMode::DEFAULT, 3);
    auto a = conv->forward(t);
    auto b = conv->forward(t);
    CHECK_EQ(a.F().size(1), int64_t(5));
    CHECK_EQ(a.F().size(0), t.F().size(0));  // stride 1, no coordinate expansion
    CHECK_ALLCLOSE(a.F(), b.F(), 1e-6, 1e-6);
}

TEST(regression_uniform_tensor_arithmetic) {
    auto t = make_legacy(80, 2);
    auto doubled = t + t;
    CHECK_ALLCLOSE(doubled.F(), t.F() * 2, 1e-6, 1e-6);
}

TEST(regression_adaptive_is_opt_in) {
    // Constructing an adaptive tensor must not perturb the uniform coordinate manager state.
    auto t = make_legacy(60, 2);
    auto before = t.C().clone();
    auto c = minkowski::adaptive::AdaptiveVoxelConfig();
    c.base_voxel_size = {1.0, 1.0, 1.0};
    c.max_depth = 0;
    auto a = minkowski::adaptive::AdaptiveSparseTensor::from_voxels(
        torch::zeros({1, 3}, torch::kInt64), torch::zeros({1}, torch::kInt32),
        torch::ones({1, 2}, torch::kFloat64), c);
    CHECK_EQ(a.size(), int64_t(1));
    CHECK(torch::equal(before, t.C()));
}
