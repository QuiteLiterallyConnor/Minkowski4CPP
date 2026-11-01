// SSIM loss of the trainer (separable batched filter on CPU, dense on CUDA)
// against a
// float64 reference that filters each moment with the dense window x window
// Gaussian (the original formulation), for values and gradients, on CPU and
// CUDA, including windows larger than the image.

#include "gaussian/gaussian_trainer.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace minkowski::gaussian;

namespace {

// Dense reference: five depthwise window x window convolutions in float64.
at::Tensor ssim_dense_reference(const at::Tensor &prediction,
                                const at::Tensor &target, int64_t window,
                                double sigma) {
  const auto p = prediction.to(at::kDouble), t = target.to(at::kDouble);
  const auto options = p.options();
  auto coordinates =
      at::arange(window, options) - static_cast<double>(window / 2);
  auto kernel = at::exp(-(coordinates * coordinates) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();
  const auto weight = (kernel.unsqueeze(1) * kernel.unsqueeze(0))
                          .expand({3, 1, window, window})
                          .contiguous();
  const auto x = p.permute({2, 0, 1}).unsqueeze(0);
  const auto y = t.permute({2, 0, 1}).unsqueeze(0);
  auto filter = [&](const at::Tensor &v) {
    return at::conv2d(v, weight, {}, 1, window / 2, 1, 3);
  };
  const auto mu_x = filter(x), mu_y = filter(y);
  const auto mu_xx = mu_x * mu_x, mu_yy = mu_y * mu_y, mu_xy = mu_x * mu_y;
  const auto sigma_xx = filter(x * x) - mu_xx;
  const auto sigma_yy = filter(y * y) - mu_yy;
  const auto sigma_xy = filter(x * y) - mu_xy;
  constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
  return (((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) /
          ((mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)))
      .mean();
}

std::pair<at::Tensor, at::Tensor> random_pair(int64_t h, int64_t w,
                                              uint64_t seed) {
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
  // A structured target plus noise, so SSIM is neither ~0 nor ~1.
  auto target = at::rand({h, w, 3}, gen, at::kFloat);
  auto prediction =
      (0.7 * target + 0.3 * at::rand({h, w, 3}, gen, at::kFloat)).clamp(0, 1);
  return {prediction, target};
}

void check_against_reference(const torch::Device &device, int64_t h,
                             int64_t w, int64_t window, double sigma,
                             uint64_t seed) {
  auto [prediction, target] = random_pair(h, w, seed);
  const auto ref_input = prediction.to(at::kDouble).requires_grad_(true);
  const auto ref = ssim_dense_reference(ref_input, target, window, sigma);
  ref.backward();

  const auto input = prediction.to(device).requires_grad_(true);
  const auto value = ssim(input, target.to(device), window, sigma);
  value.backward();

  const double dv = std::abs(value.item<double>() - ref.item<double>());
  const auto grad = input.grad().cpu().to(at::kDouble);
  const double scale = ref_input.grad().abs().max().item<double>();
  const double dg = (grad - ref_input.grad()).abs().max().item<double>();
  const std::string label = std::string(device.is_cuda() ? "cuda" : "cpu") +
                            " " + std::to_string(h) + "x" + std::to_string(w) +
                            " window " + std::to_string(window);
  std::cout << "         " << label << ": ssim " << value.item<double>()
            << " |d value| " << dv << " max |d grad| / max |grad| "
            << dg / scale << "\n";
  // float32 filtering of values in [0, 1]: ~1e-7 relative per tap.
  if (!(dv < 2e-6 && dg <= 1e-4 * scale))
    FAIL_AT(label << " differs from the dense float64 reference");
}

bool ssim_cuda_available() {
#ifdef USE_CUDA
  return at::cuda::is_available();
#else
  return false;
#endif
}

} // namespace

TEST(gaussian_ssim_identical_images_is_one) {
  auto [prediction, target] = random_pair(24, 31, 1);
  CHECK_NEAR(ssim(target, target).item<double>(), 1.0, 1e-6);
  CHECK(ssim(prediction, target).item<double>() < 0.99);
}

TEST(gaussian_ssim_matches_dense_reference_cpu) {
  const torch::Device cpu(torch::kCPU);
  // Non-square sizes, odd windows from the default 11 down to 1, and an image
  // smaller than the window (zero padding on every side).
  check_against_reference(cpu, 37, 53, 11, 1.5, 2);
  check_against_reference(cpu, 64, 48, 7, 1.0, 3);
  check_against_reference(cpu, 20, 9, 3, 0.8, 4);
  check_against_reference(cpu, 16, 16, 1, 1.5, 5);
  check_against_reference(cpu, 6, 5, 11, 1.5, 6);
}

TEST(gaussian_ssim_matches_dense_reference_cuda) {
  if (!ssim_cuda_available())
    throw mink_test::Skip("CUDA not available");
  const torch::Device cuda(torch::kCUDA, 0);
  check_against_reference(cuda, 37, 53, 11, 1.5, 2);
  check_against_reference(cuda, 6, 5, 11, 1.5, 6);
}

TEST(gaussian_ssim_rejects_bad_arguments) {
  auto [prediction, target] = random_pair(8, 8, 7);
  CHECK_THROWS(ssim(prediction, target, 4, 1.5));
  CHECK_THROWS(ssim(prediction, target.slice(0, 0, 4), 11, 1.5));
  CHECK_THROWS(ssim(prediction.select(2, 0), target.select(2, 0), 11, 1.5));
}
