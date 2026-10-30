#include "gaussian/gaussian_tensor.hpp"
#include "gaussian/gaussian_optimizer.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

using minkowski::gaussian::GaussianTensor;
using minkowski::gaussian::GaussianOptimizer;

namespace {

GaussianTensor make_tensors() {
  auto positions = at::tensor({1.0, 0.0, 0.0, 0.0, 2.0, 0.0},
                              at::TensorOptions().dtype(at::kDouble))
                       .reshape({2, 3});
  auto rotations =
      at::tensor({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
                 positions.options())
          .reshape({2, 4});
  auto scales = at::ones({2, 3}, positions.options());
  auto opacities = at::tensor({0.25, 0.75}, positions.options()).reshape({2, 1});
  auto colors = at::zeros({2, 3}, positions.options());
  auto features =
      at::tensor({4.0, 5.0, 6.0, 7.0}, positions.options()).reshape({2, 2});
  auto velocities =
      at::tensor({1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, positions.options())
          .reshape({2, 3});
  return GaussianTensor(positions, rotations, scales, opacities, colors,
                        features, velocities);
}

} // namespace

TEST(gaussian_tensor_covariance_and_shape) {
  auto gaussians = make_tensors();
  const auto covariance = gaussians.covariances();
  CHECK(covariance.sizes() == at::IntArrayRef({2, 3, 3}));
  CHECK_ALLCLOSE(covariance, at::eye(3, gaussians.positions().options())
                                 .unsqueeze(0)
                                 .expand({2, 3, 3}),
                 1e-12, 1e-12);
}

TEST(gaussian_tensor_select_remove_and_filter) {
  auto gaussians = make_tensors();
  auto selected =
      gaussians.select(at::tensor({1}, at::TensorOptions().dtype(at::kLong)));
  CHECK_EQ(selected.size(), 1);
  CHECK_EQ(selected.feature_channels(), 2);
  CHECK_NEAR(selected.positions()[0][1].item<double>(), 2.0, 0.0);
  CHECK_EQ(gaussians
               .remove(at::tensor({0}, at::TensorOptions().dtype(at::kLong)))
               .size(),
           1);
  CHECK_EQ(gaussians.filter_opacity(0.5).size(), 1);
  auto lower = at::tensor({-1.0, -1.0, -1.0}, gaussians.positions().options());
  auto upper = at::tensor({1.5, 1.0, 1.0}, gaussians.positions().options());
  CHECK_EQ(gaussians.filter_bounds(lower, upper).size(), 1);
}

TEST(gaussian_tensor_concat_clone_and_empty) {
  auto gaussians = make_tensors();
  auto joined = gaussians.append(gaussians);
  CHECK_EQ(joined.size(), 4);
  CHECK_EQ(joined.feature_channels(), 2);
  CHECK(joined.has_velocities());
  CHECK_EQ(joined.clone().size(), 4);
  auto empty = GaussianTensor::empty(0, gaussians.positions().options(), 2, true);
  CHECK_EQ(empty.size(), 0);
  CHECK_EQ(gaussians.append(empty).size(), 2);
}

TEST(gaussian_tensor_transform_and_gradients) {
  auto gaussians = make_tensors();
  auto angle = at::tensor(3.14159265358979323846 / 2.0,
                          gaussians.positions().options());
  auto rotation = at::stack({at::cos(angle * 0.5), at::zeros_like(angle),
                             at::zeros_like(angle), at::sin(angle * 0.5)})
                      .reshape({4});
  auto translation =
      at::tensor({2.0, 3.0, 4.0}, gaussians.positions().options());
  auto transformed = gaussians.transform(rotation, translation);
  CHECK_NEAR(transformed.positions()[0][0].item<double>(), 2.0, 1e-12);
  CHECK_NEAR(transformed.positions()[0][1].item<double>(), 4.0, 1e-12);
  CHECK_NEAR(transformed.positions()[0][2].item<double>(), 4.0, 1e-12);
  CHECK_NEAR(transformed.velocities()[0][0].item<double>(), 0.0, 1e-12);
  CHECK_NEAR(transformed.velocities()[0][1].item<double>(), 1.0, 1e-12);

  auto scales = gaussians.scales().clone();
  scales.set_requires_grad(true);
  GaussianTensor differentiable(
      gaussians.positions(), gaussians.rotations(), scales,
      gaussians.opacities(), gaussians.colors(), gaussians.features());
  differentiable.covariances().sum().backward();
  CHECK(scales.grad().defined());
  CHECK(scales.grad().abs().sum().item<double>() > 0.0);
}

TEST(gaussian_optimizer_reduces_rgb_reconstruction_loss) {
  auto gaussians = make_tensors();
  gaussians.scales().fill_(-1.0);
  gaussians.opacities().fill_(2.0);
  auto target = at::ones_like(gaussians.colors());
  GaussianOptimizer optimizer(gaussians, 0.1);
  const auto initial =
      GaussianOptimizer::rgb_reconstruction_loss(gaussians.colors(), target)
          .item<double>();
  for (int iteration = 0; iteration < 35; ++iteration) {
    auto loss =
        GaussianOptimizer::objective(gaussians, gaussians.colors(), target,
                                     0.0, 0.0);
    optimizer.step(loss);
  }
  const auto final_loss =
      GaussianOptimizer::rgb_reconstruction_loss(gaussians.colors(), target)
          .item<double>();
  CHECK(final_loss < initial * 0.1);
  CHECK(gaussians.scales().min().item<double>() >= 1e-6);
  CHECK(gaussians.opacities().max().item<double>() <= 1.0);
  CHECK(GaussianOptimizer::depth_reconstruction_loss(
            at::ones({2}, gaussians.positions().options()),
            at::zeros({2}, gaussians.positions().options()))
            .item<double>() > 0.0);
}

#ifdef USE_CUDA
TEST(gaussian_cuda_kernels_match_tensor_reference) {
  if (!at::cuda::is_available())
    throw mink_test::Skip("CUDA runtime or device is unavailable");

  auto cpu = make_tensors();
  auto gpu = cpu.to(torch::Device(torch::kCUDA, 0));
  const auto cpu_covariance = cpu.covariances();
  const auto gpu_covariance = gpu.covariances().to(torch::Device(torch::kCPU));
  CHECK_ALLCLOSE(gpu_covariance, cpu_covariance, 1e-5, 1e-6);

  auto rotation =
      at::tensor({0.9238795325, 0.0, 0.0, 0.3826834324},
                 cpu.positions().options());
  auto translation = at::tensor({2.0, 3.0, 4.0}, cpu.positions().options());
  const auto expected = cpu.transform(rotation, translation);
  const auto actual =
      gpu.transform(rotation.to(torch::Device(torch::kCUDA, 0)),
                    translation.to(torch::Device(torch::kCUDA, 0)))
          .to(torch::Device(torch::kCPU));
  CHECK_ALLCLOSE(actual.positions(), expected.positions(), 1e-5, 1e-6);
  CHECK_ALLCLOSE(actual.rotations(), expected.rotations(), 1e-5, 1e-6);
  CHECK_ALLCLOSE(actual.velocities(), expected.velocities(), 1e-5, 1e-6);
}
#endif
