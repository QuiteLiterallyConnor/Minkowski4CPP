// Sparse convolution and transposed convolution (D = 3, kernel 3, stride 1 and 2) against a
// brute-force dense reference (torch conv3d / conv_transpose3d on a scattered grid), forward and
// backward (input and kernel gradients), fp32 and fp64.
#include <string>

#include "core_test_utils.hpp"

using namespace minkowski;

namespace {

constexpr int64_t kExtent = 8;   // coordinates in [0, 8)^3
constexpr int64_t kBatches = 2;
constexpr int64_t kPoints = 70;  // per batch, out of 512 cells

void check_conv_against_dense(int64_t cin, int64_t cout, int64_t stride, bool transpose,
                              c10::ScalarType dtype, uint64_t seed) {
    auto coords = core_test::random_coords(kPoints, kExtent, kBatches, 3, seed);
    auto x = SparseTensor(torch::randn({coords.size(0), cin}, dtype), coords);

    // A transposed convolution reads a strided tensor: create it with a strided convolution on
    // the same coordinate manager, then give it fresh features.
    SparseTensor input = x;
    if (transpose && stride > 1) {
        MinkowskiConvolution down(cin, cin, 3, stride, 1, false, false, ConvolutionMode::DEFAULT, 3);
        down->to(dtype);
        auto y = down->forward(x);
        input = SparseTensor(torch::randn({y.F().size(0), cin}, dtype), y.coordinate_map_key(),
                             y.coordinate_manager());
    }
    auto feats = input.F().detach().clone().requires_grad_(true);
    SparseTensor in(feats, input.coordinate_map_key(), input.coordinate_manager());

    torch::manual_seed(seed + 100);
    std::shared_ptr<MinkowskiConvolutionBaseImpl> base;
    SparseTensor out = in;
    if (transpose) {
        MinkowskiConvolutionTranspose conv(cin, cout, 3, stride, 1, false, false,
                                           ConvolutionMode::DEFAULT, 3);
        conv->to(dtype);
        base = conv->base();
        out = conv->forward(in);
    } else {
        MinkowskiConvolution conv(cin, cout, 3, stride, 1, false, false, ConvolutionMode::DEFAULT, 3);
        conv->to(dtype);
        base = conv->base();
        out = conv->forward(in);
    }
    auto W = base->kernel_;
    CHECK_EQ(W.size(0), 27);

    // Dense reference with independent leaves.
    const int64_t s_in = in.tensor_stride()[0];
    const int64_t s_out = out.tensor_stride()[0];
    CHECK_EQ(s_out, transpose ? s_in / stride : s_in * stride);
    auto feats_ref = feats.detach().clone().requires_grad_(true);
    auto W_ref = W.detach().clone().requires_grad_(true);
    auto dense_in = core_test::to_dense(in.C(), feats_ref, kBatches, kExtent / s_in, s_in);
    at::Tensor dense_out;
    if (transpose) {
        dense_out = at::conv_transpose3d(dense_in, core_test::me_kernel_to_conv_transpose3d(W_ref),
                                         {}, stride, 1, stride - 1);
    } else {
        dense_out = at::conv3d(dense_in, core_test::me_kernel_to_conv3d(W_ref), {}, stride, 1);
    }
    auto ref = core_test::gather_dense(dense_out, out.C(), s_out);

    const double tol = dtype == torch::kFloat64 ? 1e-10 : 1e-4;
    CHECK_EQ(out.F().size(0), ref.size(0));
    CHECK_EQ(out.F().size(1), cout);
    CHECK_ALLCLOSE(out.F(), ref, tol, tol);

    torch::manual_seed(seed + 200);
    auto R = torch::randn_like(ref).detach();
    (out.F() * R).sum().backward();
    (ref * R).sum().backward();
    CHECK_ALLCLOSE(feats.grad(), feats_ref.grad(), tol, tol);
    CHECK_ALLCLOSE(W.grad(), W_ref.grad(), tol, tol);
}

}  // namespace

TEST(conv_reference_self_check_dense_layout) {
    // Guards the reference itself: a single point and a one-hot kernel move the value by the
    // kernel offset, k = ix + 3 * iy + 9 * iz with offset (ix - 1, iy - 1, iz - 1).
    auto coords = torch::tensor({{0, 3, 3, 3}}, torch::kInt32);
    auto feats = torch::ones({1, 1}, torch::kFloat64);
    auto W = torch::zeros({27, 1, 1}, torch::kFloat64);
    W[1 + 3 * 0 + 9 * 2] = 1.0;  // offset (0, -1, +1): out[o] = in[o + (0, -1, 1)]
    auto dense = at::conv3d(core_test::to_dense(coords, feats, 1, 8, 1),
                            core_test::me_kernel_to_conv3d(W), {}, 1, 1);
    auto at_o = core_test::gather_dense(dense, torch::tensor({{0, 3, 4, 2}}, torch::kInt32), 1);
    CHECK_NEAR(at_o.item<double>(), 1.0, 0.0);
    CHECK_NEAR(dense.sum().item<double>(), 1.0, 0.0);
}

TEST(conv_stride1_matches_dense_fp64) { check_conv_against_dense(3, 5, 1, false, torch::kFloat64, 11); }
TEST(conv_stride1_matches_dense_fp32) { check_conv_against_dense(4, 2, 1, false, torch::kFloat32, 12); }
TEST(conv_stride2_matches_dense_fp64) { check_conv_against_dense(2, 7, 2, false, torch::kFloat64, 13); }
TEST(conv_stride2_matches_dense_fp32) { check_conv_against_dense(5, 3, 2, false, torch::kFloat32, 14); }
TEST(conv_transpose_stride1_matches_dense_fp64) { check_conv_against_dense(3, 4, 1, true, torch::kFloat64, 15); }
TEST(conv_transpose_stride2_matches_dense_fp64) { check_conv_against_dense(6, 2, 2, true, torch::kFloat64, 16); }
TEST(conv_transpose_stride2_matches_dense_fp32) { check_conv_against_dense(3, 5, 2, true, torch::kFloat32, 17); }
TEST(conv_square_channels_matches_dense_fp32) { check_conv_against_dense(4, 4, 1, false, torch::kFloat32, 18); }
