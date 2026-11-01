// fp64 central finite-difference gradient checks of convolution, transposed convolution, local
// pooling and global pooling on CPU.
#include "core_test_utils.hpp"

using namespace minkowski;

namespace {

struct Input {
    at::Tensor feats;  // fp64 leaf, rows in the order of the coordinate map
    CoordinateMapKey key;
    std::shared_ptr<CoordinateManager> manager;
    SparseTensor tensor() const { return SparseTensor(feats, key, manager); }
};

Input make_input(int64_t channels, uint64_t seed, int64_t points = 24, int64_t extent = 5) {
    auto coords = core_test::random_coords(points, extent, 2, 3, seed);
    auto x = SparseTensor(torch::randn({coords.size(0), channels}, torch::kFloat64), coords);
    return {x.F().detach().clone().contiguous().requires_grad_(true), x.coordinate_map_key(),
            x.coordinate_manager()};
}

template <typename Module>
void gradcheck_conv(Module conv, const Input& in) {
    conv->to(torch::kFloat64);
    auto W = conv->base()->kernel_;
    core_test::gradcheck([&] { return conv->forward(in.tensor()).F(); }, {in.feats, W});
}

template <typename Module>
void gradcheck_features(Module module, const Input& in) {
    core_test::gradcheck([&] { return module->forward(in.tensor()).F(); }, {in.feats});
}

}  // namespace

TEST(gradcheck_conv_stride1) {
    gradcheck_conv(MinkowskiConvolution(2, 3, 3, 1, 1, false, false, ConvolutionMode::DEFAULT, 3),
                   make_input(2, 21));
}

TEST(gradcheck_conv_stride2) {
    gradcheck_conv(MinkowskiConvolution(3, 2, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 3),
                   make_input(3, 22));
}

TEST(gradcheck_conv_transpose_stride2) {
    auto base = make_input(2, 23);
    MinkowskiConvolution down(2, 2, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 3);
    down->to(torch::kFloat64);
    auto y = down->forward(base.tensor());
    Input in{torch::randn({y.F().size(0), 2}, torch::kFloat64).requires_grad_(true),
             y.coordinate_map_key(), y.coordinate_manager()};
    gradcheck_conv(
        MinkowskiConvolutionTranspose(2, 3, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 3), in);
}

TEST(gradcheck_avg_pooling_kernel3_stride1) {
    gradcheck_features(MinkowskiAvgPooling(3, 1, 1, 3), make_input(2, 24));
}

TEST(gradcheck_sum_pooling_kernel2_stride2) {
    gradcheck_features(MinkowskiSumPooling(2, 2, 1, 3), make_input(2, 25));
}

TEST(gradcheck_max_pooling_kernel3_stride2) {
    gradcheck_features(MinkowskiMaxPooling(3, 2, 1, 3), make_input(2, 26));
}

TEST(gradcheck_global_sum_pooling) { gradcheck_features(MinkowskiGlobalSumPooling(), make_input(3, 27)); }

TEST(gradcheck_global_avg_pooling) { gradcheck_features(MinkowskiGlobalAvgPooling(), make_input(3, 28)); }

TEST(gradcheck_global_max_pooling) { gradcheck_features(MinkowskiGlobalMaxPooling(), make_input(3, 29)); }
