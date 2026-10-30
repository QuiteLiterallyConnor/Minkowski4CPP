#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_model.hpp"
#include "gaussian/gaussian_ply.hpp"
#include "gaussian/gaussian_render.hpp"
#include "gaussian/gaussian_sh.hpp"
#include "gaussian/gaussian_trainer.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif
#include <cmath>
#include <cstdlib>
#include <filesystem>

using namespace minkowski::gaussian;

#ifndef MINK_TEST_DATA_DIR
#define MINK_TEST_DATA_DIR "tests/data"
#endif

namespace {

Camera test_camera(int64_t size = 32) {
  return Camera::look_at({0, 0, -3}, {0, 0, 0}, {0, -1, 0}, size, size, 1.0);
}

// Raw-parameter model with the given positions, color and activated opacity.
GaussianModel make_model(const at::Tensor &xyz, double r, double g, double b,
                         double opacity, double scale, int sh_degree = 0) {
  const int64_t n = xyz.size(0);
  const int64_t k = (sh_degree + 1) * (sh_degree + 1);
  const auto rgb = at::tensor({r, g, b}, at::kFloat).repeat({n, 1});
  auto f_dc = (rgb - 0.5) / kShC0;
  return GaussianModel(
      xyz.clone(), f_dc, at::zeros({n, k - 1, 3}),
      at::full({n, 1}, std::log(opacity / (1.0 - opacity))),
      at::full({n, 3}, std::log(scale)),
      at::tensor({1.0f, 0.0f, 0.0f, 0.0f}).repeat({n, 1}));
}

std::string data_path(const char *name) {
  const char *env = std::getenv("MINK_TEST_DATA_DIR");
  return (std::filesystem::path(env ? env : MINK_TEST_DATA_DIR) / name)
      .string();
}

bool cuda_available() {
#ifdef USE_CUDA
  return at::cuda::is_available();
#else
  return false;
#endif
}

} // namespace

TEST(gaussian_render_single_gaussian_at_center) {
  auto model = make_model(at::zeros({1, 3}), 1.0, 0.0, 0.0, 0.9, 0.3);
  const auto out = render(model, test_camera());
  CHECK(out.image.sizes() == at::IntArrayRef({32, 32, 3}));
  const auto center = out.image[16][16];
  CHECK(center[0].item<float>() > 0.7f);
  CHECK(center[1].item<float>() < 0.05f);
  CHECK(out.alpha[16][16].item<float>() > 0.8f);
  CHECK(out.alpha[0][0].item<float>() < 1e-3f);
  CHECK(out.visible.sum().item<int64_t>() == 1);
  CHECK_NEAR(out.depth[16][16].item<float>() / out.alpha[16][16].item<float>(),
             3.0f, 0.05f);
}

TEST(gaussian_render_empty_scene_returns_background) {
  auto model = make_model(at::zeros({0, 3}), 1, 1, 1, 0.5, 0.1);
  RenderOptions options;
  options.background = at::tensor({0.2f, 0.4f, 0.6f});
  const auto out = render(model, test_camera(16), options);
  CHECK_NEAR(out.image[3][5][0].item<float>(), 0.2f, 1e-6);
  CHECK_NEAR(out.image[3][5][2].item<float>(), 0.6f, 1e-6);
  CHECK_NEAR(out.alpha.sum().item<float>(), 0.0f, 1e-6);
}

TEST(gaussian_render_near_gaussian_occludes_far) {
  auto xyz = at::tensor({0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f}).reshape({2, 3});
  auto front = make_model(xyz.slice(0, 0, 1), 1, 0, 0, 0.99, 0.5);
  auto back = make_model(xyz.slice(0, 1, 2), 0, 0, 1, 0.99, 0.5);
  auto both = GaussianModel(
      at::cat({front.xyz(), back.xyz()}), at::cat({front.f_dc(), back.f_dc()}),
      at::zeros({2, 0, 3}), at::cat({front.opacity_raw(), back.opacity_raw()}),
      at::cat({front.scale_raw(), back.scale_raw()}),
      at::cat({front.rotation_raw(), back.rotation_raw()}));
  const auto pixel = render(both, test_camera()).image[16][16];
  CHECK(pixel[0].item<float>() > 0.8f);
  CHECK(pixel[2].item<float>() < 0.1f);
}

TEST(gaussian_sh_degree_zero_and_degree_one) {
  const auto dirs = at::tensor({0.0f, 0.0f, 1.0f}).reshape({1, 3});
  auto sh = at::zeros({1, 4, 3});
  sh.select(1, 0).fill_(0.25);
  const auto dc = eval_sh_color(0, sh.slice(1, 0, 1), dirs);
  CHECK_NEAR(dc[0][0].item<float>(), 0.5f + kShC0 * 0.25f, 1e-6);
  CHECK_EQ(sh_coefficient_count(3), 16);
  // Degree-1 term along +z is C1 * z * coefficient[2].
  sh.select(1, 2).fill_(1.0);
  const auto c1 = eval_sh_color(1, sh, dirs);
  const float expected = 0.5f + kShC0 * 0.25f + 0.4886025119f * 1.0f;
  CHECK_NEAR(c1[0][1].item<float>(), expected, 1e-5);
}

TEST(gaussian_render_gradients_are_finite_and_nonzero) {
  auto xyz = at::tensor({0.0f, 0.0f, 0.0f, 0.2f, 0.1f, 0.5f}).reshape({2, 3});
  auto model = make_model(xyz, 0.7, 0.3, 0.2, 0.6, 0.25);
  model.enable_gradients();
  auto out = render(model, test_camera());
  out.image.sum().backward();
  for (const auto &p : model.parameters()) {
    if (p.numel() == 0)
      continue;
    CHECK(p.grad().defined());
    CHECK(at::isfinite(p.grad()).all().item<bool>());
  }
  CHECK(model.xyz().grad().abs().sum().item<float>() > 0.0f);
  CHECK(model.f_dc().grad().abs().sum().item<float>() > 0.0f);
  CHECK(model.opacity_raw().grad().abs().sum().item<float>() > 0.0f);
  CHECK(model.scale_raw().grad().abs().sum().item<float>() > 0.0f);
  CHECK(out.means2d.grad().defined());
}

TEST(gaussian_render_color_gradient_matches_finite_difference) {
  auto model = make_model(at::zeros({1, 3}), 0.6, 0.4, 0.2, 0.7, 0.3);
  model.enable_gradients();
  const auto camera = test_camera(16);
  render(model, camera).image.sum().backward();
  const double analytic = model.f_dc().grad()[0][0].item<double>();

  auto sum_with = [&](double delta) {
    torch::NoGradGuard guard;
    auto copy = model.clone();
    copy.f_dc().data_ptr<float>()[0] += static_cast<float>(delta);
    return render(copy, camera).image.sum().item<double>();
  };
  const double h = 1e-2;
  const double numeric = (sum_with(h) - sum_with(-h)) / (2 * h);
  CHECK_NEAR(analytic, numeric, 0.02 * std::abs(numeric) + 1e-3);
}

TEST(gaussian_model_tensor_round_trip) {
  auto xyz = at::randn({8, 3});
  auto model = make_model(xyz, 0.3, 0.5, 0.7, 0.4, 0.2);
  const auto tensor = model.to_gaussian_tensor();
  const auto back = GaussianModel::from_tensor(tensor);
  CHECK_ALLCLOSE(back.xyz(), model.xyz(), 1e-6, 1e-6);
  CHECK_ALLCLOSE(back.opacities(), model.opacities(), 1e-5, 1e-5);
  CHECK_ALLCLOSE(back.scales(), model.scales(), 1e-5, 1e-5);
  CHECK_ALLCLOSE(back.colors(), model.colors(), 1e-5, 1e-5);
}

TEST(gaussian_ply_model_save_load_round_trip) {
  auto model = GaussianModel(
      at::randn({6, 3}), at::randn({6, 3}), at::randn({6, 15, 3}) * 0.1,
      at::randn({6, 1}), at::randn({6, 3}) * 0.2 - 2.0,
      at::randn({6, 4}));
  const auto path =
      (std::filesystem::temp_directory_path() / "mink_gaussian_model.ply")
          .string();
  save_gaussian_ply(model, path);
  const auto loaded = load_gaussian_model_ply(path);
  std::filesystem::remove(path);
  CHECK_EQ(loaded.size(), 6);
  CHECK_EQ(loaded.sh_degree(), 3);
  CHECK_ALLCLOSE(loaded.xyz(), model.xyz(), 1e-6, 1e-6);
  CHECK_ALLCLOSE(loaded.f_dc(), model.f_dc(), 1e-6, 1e-6);
  CHECK_ALLCLOSE(loaded.f_rest(), model.f_rest(), 1e-6, 1e-6);
  CHECK_ALLCLOSE(loaded.opacity_raw(), model.opacity_raw(), 1e-6, 1e-6);
  CHECK_ALLCLOSE(loaded.scale_raw(), model.scale_raw(), 1e-6, 1e-6);
}

TEST(gaussian_ply_models_load_with_sh_degree_three) {
  for (const char *name : {"test.ply", "test_compressed.ply"}) {
    const auto path = data_path(name);
    if (!std::filesystem::exists(path))
      throw mink_test::Skip(std::string(name) + " not found");
    const auto model = load_gaussian_model_ply(path);
    CHECK_EQ(model.sh_degree(), 3);
    CHECK(at::isfinite(model.f_dc()).all().item<bool>());
    CHECK(at::isfinite(model.f_rest()).all().item<bool>());
    const auto colors = model.colors();
    CHECK(colors.min().item<float>() > -0.5f);
    CHECK(colors.max().item<float>() < 3.0f);
  }
}

TEST(gaussian_ply_scene_renders_and_trains) {
  const auto path = data_path("test.ply");
  if (!std::filesystem::exists(path))
    throw mink_test::Skip("test.ply not found");
  auto full = load_gaussian_model_ply(path);
  auto model = full.select(at::arange(2000, at::kLong));
  const auto center = model.xyz().mean(0);
  const auto camera = Camera::look_at(
      {center[0].item<double>(), center[1].item<double>(),
       center[2].item<double>() - 6.0},
      {center[0].item<double>(), center[1].item<double>(),
       center[2].item<double>()},
      {0, -1, 0}, 48, 48, 1.0);
  const auto target = render(model, camera).image.detach().clone();
  CHECK(at::isfinite(target).all().item<bool>());

  GaussianTrainerConfig config;
  config.densify = false;
  config.scene_extent = 5.0;
  GaussianTrainer trainer(model, config);
  const auto result = trainer.step(camera, target);
  CHECK(std::isfinite(result.loss));
  CHECK_EQ(result.iteration, 1);
}

TEST(gaussian_trainer_reduces_loss_on_synthetic_scene) {
  auto truth = make_model(at::tensor({-0.4f, 0.0f, 0.0f, 0.4f, 0.1f, 0.0f})
                              .reshape({2, 3}),
                          0.9, 0.2, 0.1, 0.9, 0.3);
  const auto camera = test_camera(24);
  const auto target = render(truth, camera).image.detach().clone();

  auto model = make_model(at::tensor({-0.3f, 0.1f, 0.0f, 0.3f, 0.0f, 0.0f})
                              .reshape({2, 3}),
                          0.4, 0.4, 0.4, 0.5, 0.25);
  GaussianTrainerConfig config;
  config.densify = false;
  config.scene_extent = 1.0;
  config.lr_position = 5e-3;
  config.lr_dc = 5e-2;
  config.lr_opacity = 0.1;
  GaussianTrainer trainer(model, config);
  const double first = trainer.step(camera, target).loss;
  double last = first;
  for (int i = 0; i < 60; ++i)
    last = trainer.step(camera, target).loss;
  CHECK(last < 0.7 * first);
}

TEST(gaussian_trainer_densify_and_prune) {
  auto model = make_model(at::randn({40, 3}) * 0.3, 0.5, 0.5, 0.5, 0.5, 0.02);
  // Make two Gaussians nearly transparent so they get pruned.
  model.opacity_raw().data_ptr<float>()[0] = -12.0f;
  model.opacity_raw().data_ptr<float>()[1] = -12.0f;
  GaussianTrainerConfig config;
  config.scene_extent = 1.0;
  config.densify_grad_threshold = 0.0;
  GaussianTrainer trainer(model, config);
  const int64_t before = model.size();
  const auto [added, removed] = trainer.densify_and_prune();
  CHECK(added > 0);
  CHECK(removed >= 2);
  CHECK_EQ(model.size(), before + added - removed);
  for (const auto &p : model.parameters())
    CHECK_EQ(p.size(0), model.size());
  CHECK(model.xyz().requires_grad());
}

TEST(gaussian_trainer_reset_opacity_caps_opacity) {
  auto model = make_model(at::randn({10, 3}), 0.5, 0.5, 0.5, 0.9, 0.1);
  GaussianTrainerConfig config;
  GaussianTrainer trainer(model, config);
  trainer.reset_opacity();
  CHECK(model.opacities().max().item<float>() <= 0.0101f);
}

TEST(gaussian_cuda_rasterizer_matches_reference_forward_and_backward) {
  if (!cuda_available())
    throw mink_test::Skip("CUDA not available");
  const torch::Device device(torch::kCUDA, 0);
  at::manual_seed(7);
  const int64_t n = 400;
  auto make = [&]() {
    auto m = GaussianModel(
        at::randn({n, 3}) * 0.6, at::rand({n, 3}) - 0.5, at::zeros({n, 0, 3}),
        at::randn({n, 1}), at::randn({n, 3}) * 0.3 - 2.0, at::randn({n, 4}));
    m = m.to(device);
    m.enable_gradients();
    return m;
  };
  const auto base = make();
  const auto camera = test_camera(50); // not a multiple of the 16px tile
  RenderOptions reference_options;
  reference_options.use_cuda_kernels = false;
  reference_options.background = at::tensor({0.1f, 0.2f, 0.3f}).to(device);
  RenderOptions fast_options = reference_options;
  fast_options.use_cuda_kernels = true;

  const auto weights = at::randn({50, 50, 3}, at::TensorOptions().device(device));
  auto run = [&](const RenderOptions &options) {
    auto model = base.clone();
    model.enable_gradients();
    auto out = render(model, camera, options);
    (out.image * weights).sum().backward();
    return std::make_pair(out, model);
  };
  auto [ref_out, ref_model] = run(reference_options);
  auto [fast_out, fast_model] = run(fast_options);
  CHECK_ALLCLOSE(fast_out.image, ref_out.image, 1e-3, 1e-3);
  CHECK_ALLCLOSE(fast_out.alpha, ref_out.alpha, 1e-3, 1e-3);
  const auto ref_params = ref_model.parameters();
  const auto fast_params = fast_model.parameters();
  for (size_t i = 0; i < ref_params.size(); ++i) {
    if (ref_params[i].numel() == 0)
      continue;
    const double scale = ref_params[i].grad().abs().max().item<double>();
    CHECK_ALLCLOSE(fast_params[i].grad(), ref_params[i].grad(),
                   2e-2, 2e-2 * scale + 1e-4);
  }
}

TEST(gaussian_render_cuda_matches_cpu) {
  if (!cuda_available())
    throw mink_test::Skip("CUDA not available");
  auto model = make_model(at::randn({30, 3}) * 0.4, 0.8, 0.3, 0.1, 0.6, 0.15);
  const auto camera = test_camera(32);
  const auto cpu = render(model, camera).image;
  const auto gpu = render(model.to(torch::Device(torch::kCUDA, 0)), camera)
                       .image.to(torch::kCPU);
  CHECK_ALLCLOSE(gpu, cpu, 1e-3, 1e-3);
}
