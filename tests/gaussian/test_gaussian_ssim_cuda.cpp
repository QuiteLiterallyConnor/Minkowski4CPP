// Fused CUDA SSIM (gaussian_ssim_cuda.cu) against the LibTorch filters it replaces and against a
// float64 dense reference: values and gradients for several windows and image sizes (including
// images smaller than the window), non-contiguous inputs, a weighted loss, determinism, and the
// fallbacks (target requiring grad, windows above kSsimMaxWindow).
#include "gaussian/gaussian_cuda.hpp"
#include "gaussian/gaussian_trainer.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>

#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef USE_CUDA

using namespace minkowski::gaussian;
namespace gd = minkowski::gaussian::detail;

namespace {

void require_cuda() {
  if (!torch::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
}

at::Tensor dense_reference(const at::Tensor &prediction, const at::Tensor &target, int64_t window,
                           double sigma) {
  const auto p = prediction.to(at::kDouble), t = target.to(at::kDouble);
  auto c = at::arange(window, p.options()) - static_cast<double>(window / 2);
  auto k = at::exp(-(c * c) / (2.0 * sigma * sigma));
  k = k / k.sum();
  const auto weight = (k.unsqueeze(1) * k.unsqueeze(0)).expand({3, 1, window, window}).contiguous();
  const auto x = p.permute({2, 0, 1}).unsqueeze(0), y = t.permute({2, 0, 1}).unsqueeze(0);
  auto f = [&](const at::Tensor &v) { return at::conv2d(v, weight, {}, 1, window / 2, 1, 3); };
  const auto mx = f(x), my = f(y);
  const auto sxx = f(x * x) - mx * mx, syy = f(y * y) - my * my, sxy = f(x * y) - mx * my;
  constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
  return (((2.0 * mx * my + c1) * (2.0 * sxy + c2)) /
          ((mx * mx + my * my + c1) * (sxx + syy + c2)))
      .mean();
}

std::pair<at::Tensor, at::Tensor> random_pair(int64_t h, int64_t w, uint64_t seed) {
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
  auto target = at::rand({h, w, 3}, gen, at::kFloat);
  auto prediction = (0.7 * target + 0.3 * at::rand({h, w, 3}, gen, at::kFloat)).clamp(0, 1);
  return {prediction, target};
}

struct Run {
  double value;
  at::Tensor grad;
};

// 0.2 * (1 - SSIM), the trainer's weighting, so the incoming gradient is not 1.
Run run(const at::Tensor &prediction, const at::Tensor &target, int64_t window, double sigma,
        bool fused) {
  gd::cuda_fused_ssim() = fused;
  const auto input = prediction.to(at::kCUDA).requires_grad_(true);
  const auto value = ssim(input, target.to(at::kCUDA), window, sigma);
  (0.2 * (1.0 - value)).backward();
  gd::cuda_fused_ssim() = true;
  return {value.item<double>(), input.grad().cpu()};
}

double max_rel(const at::Tensor &a, const at::Tensor &b) {
  const auto a64 = a.to(at::kDouble), b64 = b.to(at::kDouble);
  return (a64 - b64).abs().max().item<double>() / b64.abs().max().item<double>();
}

} // namespace

TEST(gaussian_fused_ssim_matches_libtorch_and_float64) {
  require_cuda();
  using Case = std::tuple<int64_t, int64_t, int64_t, double>; // h, w, window, sigma
  for (const auto &[h, w, window, sigma] :
       std::vector<Case>{{37, 53, 11, 1.5}, {6, 5, 11, 1.5}, {64, 48, 7, 1.0}, {20, 9, 3, 0.8},
                         {16, 16, 1, 1.5}, {40, 33, 15, 2.5}, {128, 96, 11, 1.5}, {17, 300, 11, 1.5}}) {
    const auto [prediction, target] = random_pair(h, w, static_cast<uint64_t>(h * 1000 + w));
    const auto fused = run(prediction, target, window, sigma, true);
    const auto aten = run(prediction, target, window, sigma, false);
    const auto ref_input = prediction.to(at::kDouble).requires_grad_(true);
    const auto ref = dense_reference(ref_input, target, window, sigma);
    (0.2 * (1.0 - ref)).backward();
    const double dv_aten = std::abs(fused.value - aten.value);
    const double dv_ref = std::abs(fused.value - ref.item<double>());
    const double dv_aten_ref = std::abs(aten.value - ref.item<double>());
    const double dg_aten = max_rel(fused.grad, aten.grad);
    const double dg_ref = max_rel(fused.grad, ref_input.grad());
    const double dg_aten_ref = max_rel(aten.grad, ref_input.grad());
    std::cout << "         " << h << "x" << w << " window " << window << ": ssim " << fused.value
              << " |d value| vs LibTorch " << dv_aten << ", vs float64 " << dv_ref
              << " (LibTorch " << dv_aten_ref << "); grad rel vs LibTorch " << dg_aten
              << ", vs float64 " << dg_ref << " (LibTorch " << dg_aten_ref << ")\n";
    CHECK(dv_ref < 2e-6);
    CHECK(dg_ref < 1e-4);
    CHECK(dv_aten < 2e-6);
    CHECK(dg_aten < 1e-4);
  }
}

TEST(gaussian_fused_ssim_non_contiguous_prediction) {
  require_cuda();
  // The renderer's image is a [H, W, 3] slice of a [H, W, 5] buffer.
  const auto [prediction, target] = random_pair(45, 61, 9);
  auto buffer = at::rand({45, 61, 5}, at::kFloat).to(at::kCUDA);
  buffer.slice(2, 0, 3).copy_(prediction.to(at::kCUDA));
  buffer.requires_grad_(true);
  const auto value = ssim(buffer.slice(2, 0, 3), target.to(at::kCUDA));
  value.backward();
  const auto contiguous = run(prediction, target, 11, 1.5, true);
  CHECK_NEAR(value.item<double>(), contiguous.value, 0.0);
  // 0.2 * (1 - S) scales the gradient by -0.2.
  CHECK(max_rel(buffer.grad().slice(2, 0, 3).cpu() * -0.2f, contiguous.grad) < 1e-6);
  CHECK_NEAR(buffer.grad().slice(2, 3, 5).abs().sum().item<double>(), 0.0, 0.0);
}

TEST(gaussian_fused_ssim_is_deterministic) {
  require_cuda();
  const auto [prediction, target] = random_pair(70, 90, 10);
  const auto a = run(prediction, target, 11, 1.5, true);
  const auto b = run(prediction, target, 11, 1.5, true);
  CHECK(a.value == b.value);
  CHECK(at::equal(a.grad, b.grad));
}

TEST(gaussian_fused_ssim_falls_back) {
  require_cuda();
  const auto [prediction, target] = random_pair(30, 26, 11);
  // Target requiring grad: the LibTorch path, which also returns the target's gradient.
  const auto p = prediction.to(at::kCUDA).requires_grad_(true);
  const auto t = target.to(at::kCUDA).requires_grad_(true);
  ssim(p, t).backward();
  CHECK(t.grad().defined());
  CHECK(t.grad().abs().sum().item<double>() > 0.0);
  // A window above kSsimMaxWindow.
  const int64_t big = gd::kSsimMaxWindow + 2;
  const auto fused = run(prediction, target, big, 3.0, true);
  const auto aten = run(prediction, target, big, 3.0, false);
  CHECK_NEAR(fused.value, aten.value, 1e-7);
  CHECK(max_rel(fused.grad, aten.grad) < 1e-6);
}

#endif // USE_CUDA
