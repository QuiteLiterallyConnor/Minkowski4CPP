// Fused CUDA Adam update (gaussian_adam_cuda.cu) against the LibTorch ops it replaces: the kernel
// alone over several steps (including zero, tiny and huge gradients), and whole trainer steps on
// the test scene with the fused update on and off.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_cuda.hpp"
#include "gaussian/gaussian_ply.hpp"
#include "gaussian/gaussian_trainer.hpp"
#include "utils/test_framework.hpp"

#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

using namespace minkowski::gaussian;
namespace gd = minkowski::gaussian::detail;

namespace {

void require_cuda() {
  if (!torch::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
}

double max_rel(const at::Tensor &a, const at::Tensor &b) {
  const auto a64 = a.cpu().to(at::kDouble), b64 = b.cpu().to(at::kDouble);
  const double scale = b64.abs().max().item<double>();
  const double diff = (a64 - b64).abs().max().item<double>();
  return scale > 0 ? diff / scale : diff;
}

} // namespace

TEST(gaussian_fused_adam_matches_libtorch_update) {
  require_cuda();
  torch::manual_seed(3);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const int64_t n = 100003; // not a multiple of the block size
  auto p_ref = torch::randn({n}, f), p = p_ref.clone();
  auto m1_ref = torch::zeros({n}, f), m1 = m1_ref.clone();
  auto m2_ref = torch::zeros({n}, f), m2 = m2_ref.clone();
  const double beta1 = 0.9, beta2 = 0.999, eps = 1e-15, lr = 2.5e-3;
  for (int step = 1; step <= 6; ++step) {
    auto g = torch::randn({n}, f) * 1e-3;
    g.slice(0, 0, 100).zero_();                // zero gradient (first step: m2 = 0, eps only)
    g.slice(0, 100, 200).mul_(1e-20);          // tiny
    g.slice(0, 200, 300).mul_(1e18);           // huge (g^2 stays finite in float32)
    const double bias1 = 1.0 - std::pow(beta1, step), bias2 = 1.0 - std::pow(beta2, step);
    // The trainer's LibTorch update.
    m1_ref.mul_(beta1).add_(g, 1.0 - beta1);
    m2_ref.mul_(beta2).addcmul_(g, g, 1.0 - beta2);
    p_ref.addcdiv_(m1_ref, (m2_ref / bias2).sqrt().add_(eps), -lr / bias1);
    const gd::AdamParams k{static_cast<float>(beta1), static_cast<float>(1.0 - beta1),
                           static_cast<float>(beta2), static_cast<float>(1.0 - beta2),
                           gd::adam_inverse_bias2(bias2), static_cast<float>(eps),
                           static_cast<float>(-lr / bias1)};
    gd::adam_update_cuda(p, g, m1, m2, k);
    // Scale-aware absolute error (not an IEEE ULP distance): cancellation near p=0 must
    // retain the update scale. Permit one float epsilon of |p| + lr per step.
    const auto spacing = (p_ref.abs() + lr) * std::ldexp(1.0, -23);
    const auto ratio = (p - p_ref).abs() / spacing;
    const double ulps = ratio.max().item<double>();
    if (ulps > 1.0) {
      const int64_t w = ratio.argmax().item<int64_t>();
      std::cout.precision(10);
      std::cout << "         worst element " << w << ": p " << p[w].item<float>() << " vs "
                << p_ref[w].item<float>() << ", g " << g[w].item<float>() << ", m1 "
                << m1[w].item<float>() << " vs " << m1_ref[w].item<float>() << ", m2 "
                << m2[w].item<float>() << " vs " << m2_ref[w].item<float>() << "\n";
      std::cout.precision(6);
    }
    const double dm1 = max_rel(m1, m1_ref), dm2 = max_rel(m2, m2_ref);
    const int64_t differ = (p != p_ref).sum().item<int64_t>();
    std::cout << "         step " << step << ": max scaled |dp| " << ulps << ", m1 rel " << dm1
              << ", m2 rel " << dm2 << ", parameters not bit-identical " << differ << " of " << n
              << "\n";
    CHECK(ulps <= step + 1e-9);
    CHECK(dm1 < 1e-6);
    CHECK(dm2 < 1e-6);
    // Elementwise checks prevent the huge-gradient slice from masking errors elsewhere.
    CHECK_ALLCLOSE(m1, m1_ref, 1e-6, 1e-30);
    CHECK_ALLCLOSE(m2, m2_ref, 1e-6, 1e-40);
    CHECK(torch::isfinite(p).all().item<bool>());
  }
}

TEST(gaussian_fused_adam_cancellation_and_bias_rounding) {
  require_cuda();
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  // Ordinary gradients and cancellation with the update exposed the reciprocal rounding bug.
  for (const double beta2 : {0.0, 0.9, 0.999, 0.999999}) {
    auto p = torch::linspace(-0.00251, -0.00249, 257, f), ref = p.clone();
    auto g = torch::full_like(p, -0.0001782548352);
    auto m1 = torch::zeros_like(p), r1 = m1.clone();
    auto m2 = torch::zeros_like(p), r2 = m2.clone();
    for (int step = 1; step <= 16; ++step) {
      const double bias1 = 1.0 - std::pow(0.9, step);
      const double bias2 = 1.0 - std::pow(beta2, step);
      r1.mul_(0.9).add_(g, 0.1);
      r2.mul_(beta2).addcmul_(g, g, 1.0 - beta2);
      ref.addcdiv_(r1, (r2 / bias2).sqrt().add_(1e-15), -0.0025 / bias1);
      const gd::AdamParams k{0.9f, 0.1f, static_cast<float>(beta2),
                             static_cast<float>(1.0 - beta2), gd::adam_inverse_bias2(bias2),
                             1e-15f, static_cast<float>(-0.0025 / bias1)};
      gd::adam_update_cuda(p, g, m1, m2, k);
      CHECK_ALLCLOSE(m1, r1, 1e-6, 1e-30);
      CHECK_ALLCLOSE(m2, r2, 1e-6, 1e-40);
      CHECK_ALLCLOSE(p, ref, step * 1e-7, step * 0.0025 * 1e-7);
    }
  }
}

TEST(gaussian_fused_adam_empty_and_invalid_inputs) {
  require_cuda();
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const gd::AdamParams k{0.9f, 0.1f, 0.999f, 0.001f, gd::adam_inverse_bias2(0.001),
                         1e-15f, -0.025f};
  auto p = torch::empty({0}, f), g = p.clone(), m1 = p.clone(), m2 = p.clone();
  gd::adam_update_cuda(p, g, m1, m2, k);
  CHECK_EQ(p.numel(), 0);
  p = torch::zeros({6}, f); g = p.clone(); m1 = p.clone(); m2 = p.clone();
  const auto bad_type = g.to(torch::kFloat64), bad_device = g.cpu();
  const auto bad_stride = torch::zeros({6, 2}, f).select(1, 0);
  const auto bad_size = g.slice(0, 0, 5);
  CHECK_THROWS(gd::adam_update_cuda(p, bad_type, m1, m2, k));
  CHECK_THROWS(gd::adam_update_cuda(p, bad_device, m1, m2, k));
  CHECK_THROWS(gd::adam_update_cuda(p, bad_stride, m1, m2, k));
  CHECK_THROWS(gd::adam_update_cuda(p, bad_size, m1, m2, k));
  CHECK_EQ(p.abs().sum().item<float>(), 0.0f);
}

TEST(gaussian_fused_adam_current_stream) {
  require_cuda();
  const auto stream = c10::cuda::getStreamFromPool();
  const c10::cuda::CUDAStreamGuard guard(stream);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto p = torch::ones({513}, f), g = torch::full_like(p, 0.5f);
  auto m1 = torch::zeros_like(p), m2 = torch::zeros_like(p);
  const gd::AdamParams k{0, 1, 0, 1, 1, 0, -0.25f};
  gd::adam_update_cuda(p, g, m1, m2, k);
  stream.synchronize();
  CHECK((p == 0.75f).all().item<bool>());
  CHECK((m1 == 0.5f).all().item<bool>());
  CHECK((m2 == 0.25f).all().item<bool>());
}

TEST(gaussian_fused_adam_trainer_steps_match_libtorch) {
  require_cuda();
  const std::string path = std::string(MINK_TEST_DATA_DIR) + "/test.ply";
  auto full = load_gaussian_model_ply(path);
  const auto base = full.select(at::arange(3000, at::kLong)).to(torch::kCUDA);
  const auto center = base.xyz().mean(0).cpu();
  const double cx = center[0].item<double>(), cy = center[1].item<double>(),
               cz = center[2].item<double>();
  const auto camera = Camera::look_at({cx, cy, cz - 6.0}, {cx, cy, cz}, {0, -1, 0}, 64, 48, 1.0);
  const auto target = torch::full({48, 64, 3}, 0.4f, torch::TensorOptions().device(torch::kCUDA));
  auto train = [&](bool fused) {
    gd::cuda_fused_adam() = fused;
    auto model = base.clone();
    GaussianTrainerConfig config;
    config.densify = false;
    config.scene_extent = 5.0;
    GaussianTrainer trainer(model, config);
    std::vector<double> losses;
    for (int i = 0; i < 8; ++i)
      losses.push_back(trainer.step(camera, target).loss);
    gd::cuda_fused_adam() = true;
    return std::make_pair(model.parameters(), losses);
  };
  const auto [fused, fused_loss] = train(true);
  const auto [aten, aten_loss] = train(false);
  const char *names[] = {"xyz", "f_dc", "f_rest", "opacity", "scale", "rotation"};
  for (size_t i = 0; i < fused.size(); ++i) {
    const double d = max_rel(fused[i].detach(), aten[i].detach());
    std::cout << "         " << names[i] << ": relative difference after 8 steps " << d << "\n";
    CHECK(d < 1e-4);
  }
  std::cout << "         loss step 8: fused " << fused_loss.back() << ", LibTorch " << aten_loss.back()
            << "\n";
  CHECK(std::abs(fused_loss.back() - aten_loss.back()) <= 1e-5 * std::abs(aten_loss.back()));
  CHECK(fused_loss.back() < fused_loss.front());
}

#endif // USE_CUDA
