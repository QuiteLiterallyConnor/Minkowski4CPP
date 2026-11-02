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
                           1.0f / static_cast<float>(bias2), static_cast<float>(eps),
                           static_cast<float>(-lr / bias1)};
    gd::adam_update_cuda(p, g, m1, m2, k);
    // The two updates can differ in the last bit (different rounding of the same formula), so
    // p + update can land on a neighbouring float: measure parameter differences in units of the
    // float spacing at |p| + lr (at most one more ulp per step) and the moments relatively.
    const auto spacing = (p_ref.abs() + lr) * std::ldexp(1.0, -23);
    const double ulps = ((p - p_ref).abs() / spacing).max().item<double>();
    const double dm1 = max_rel(m1, m1_ref), dm2 = max_rel(m2, m2_ref);
    const int64_t differ = (p != p_ref).sum().item<int64_t>();
    std::cout << "         step " << step << ": max |dp| " << ulps << " ulp, m1 rel " << dm1
              << ", m2 rel " << dm2 << ", parameters not bit-identical " << differ << " of " << n
              << "\n";
    CHECK(ulps <= step + 1e-9);
    CHECK(dm1 < 1e-6);
    CHECK(dm2 < 1e-6);
    CHECK(torch::isfinite(p).all().item<bool>());
  }
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
