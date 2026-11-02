// Fused CPU rasterizer (RenderOptions::use_cpu_kernels) against the LibTorch
// reference path: forward image/depth/alpha, gradients of every input with
// image + depth + alpha in the loss, determinism and thread-count invariance.

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_render.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>

#include <iostream>
#include <string>
#include <vector>

using namespace minkowski::gaussian;

namespace {

struct Inputs {
  at::Tensor positions, rotations, scales, opacities, colors, background;
};

const char *const kNames[] = {"positions", "rotations", "scales",
                              "opacities", "colors",    "background"};

Inputs random_inputs(int64_t n, uint64_t seed, double log_scale,
                     double min_opacity) {
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
  Inputs s;
  s.positions = at::randn({n, 3}, gen, at::kFloat) * 0.7;
  s.rotations = at::randn({n, 4}, gen, at::kFloat);
  s.scales = at::exp(at::randn({n, 3}, gen, at::kFloat) * 0.5 + log_scale);
  s.opacities =
      min_opacity + (0.99 - min_opacity) * at::rand({n, 1}, gen, at::kFloat);
  s.colors = at::rand({n, 3}, gen, at::kFloat);
  s.background = at::tensor({0.2f, 0.1f, 0.3f});
  return s;
}

struct Result {
  RenderOutput out;
  std::vector<at::Tensor> grads;
};

Result run(const Inputs &base, const Camera &camera, bool kernels,
           int64_t tile = 16) {
  std::vector<at::Tensor> live{base.positions.clone(), base.rotations.clone(),
                               base.scales.clone(),    base.opacities.clone(),
                               base.colors.clone(),    base.background.clone()};
  for (auto &t : live)
    t.requires_grad_(true);
  RenderOptions options;
  options.background = live[5];
  options.use_cpu_kernels = kernels;
  options.tile_size = tile;
  Result r;
  r.out = render_gaussians(live[0], live[1], live[2], live[3], live[4], camera,
                           options);
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(31);
  const auto wi = at::randn({camera.height, camera.width, 3}, gen);
  const auto wd = at::randn({camera.height, camera.width}, gen);
  const auto wa = at::randn({camera.height, camera.width}, gen);
  ((r.out.image * wi).sum() + 0.1 * (r.out.depth * wd).sum() +
   (r.out.alpha * wa).sum())
      .backward();
  for (auto &t : live)
    r.grads.push_back(t.grad().defined() ? t.grad() : at::zeros_like(t));
  return r;
}

void expect_close_to_reference(const Inputs &scene, const Camera &camera,
                               const std::string &label) {
  const auto ref = run(scene, camera, false);
  const auto fast = run(scene, camera, true);
  // The LibTorch path stops a whole chunk once every pixel has T < 1e-4,
  // the kernels stop per pixel: outputs differ by at most ~1e-4.
  const double image = (fast.out.image - ref.out.image).abs().max().item<double>();
  const double alpha = (fast.out.alpha - ref.out.alpha).abs().max().item<double>();
  const double depth = (fast.out.depth - ref.out.depth).abs().max().item<double>();
  std::cout << "         " << label << ": max |kernel - libtorch| image "
            << image << ", alpha " << alpha << ", depth " << depth << "\n";
  CHECK(image < 5e-4 && alpha < 5e-4 && depth < 2e-3);
  for (size_t k = 0; k < ref.grads.size(); ++k) {
    const double scale = ref.grads[k].abs().max().item<double>();
    const double err = (fast.grads[k] - ref.grads[k]).abs().max().item<double>();
    std::cout << "         " << label << " grad " << kNames[k] << ": max "
              << scale << ", err " << err << "\n";
    CHECK(scale > 0.0);
    CHECK_ALLCLOSE(fast.grads[k], ref.grads[k], 2e-2, 2e-2 * scale + 1e-4);
  }
}

Camera camera_of(int64_t w, int64_t h) {
  return Camera::look_at({0.3, -0.2, -3.0}, {0, 0, 0}, {0, -1, 0}, w, h, 1.0);
}

} // namespace

TEST(gaussian_cpu_raster_matches_libtorch_sparse) {
  expect_close_to_reference(random_inputs(2000, 41, -4.0, 0.05),
                            camera_of(77, 45), "sparse");
}

TEST(gaussian_cpu_raster_matches_libtorch_saturated) {
  // Large opaque splats: most pixels stop early.
  expect_close_to_reference(random_inputs(300, 43, -2.0, 0.3),
                            camera_of(50, 66), "saturated");
}

TEST(gaussian_cpu_raster_matches_libtorch_other_tile_size) {
  const auto scene = random_inputs(500, 47, -3.0, 0.05);
  const auto camera = camera_of(40, 40);
  const auto ref = run(scene, camera, false, 7);
  const auto fast = run(scene, camera, true, 7);
  CHECK_ALLCLOSE(fast.out.image, ref.out.image, 0.0, 5e-4);
}

TEST(gaussian_cpu_raster_is_deterministic_across_threads) {
  const auto scene = random_inputs(1500, 53, -3.2, 0.05);
  const auto camera = camera_of(64, 48);
  const auto a = run(scene, camera, true);
  const auto b = run(scene, camera, true);
  const int threads = at::get_num_threads();
  at::set_num_threads(1);
  const auto c = run(scene, camera, true);
  at::set_num_threads(threads);
  CHECK(at::equal(a.out.image, b.out.image));
  CHECK(at::equal(a.out.image, c.out.image));
  for (size_t k = 0; k < a.grads.size(); ++k) {
    CHECK(at::equal(a.grads[k], b.grads[k]));
    CHECK(at::equal(a.grads[k], c.grads[k]));
  }
}

TEST(gaussian_cpu_raster_empty_and_culled_scenes) {
  const auto camera = camera_of(20, 20);
  Inputs empty = random_inputs(0, 1, -3.0, 0.1);
  Inputs behind = random_inputs(4, 2, -3.0, 0.5);
  behind.positions.select(1, 2).fill_(-10.0f); // behind the camera
  for (const auto *scene : {&empty, &behind}) {
    const auto r = run(*scene, camera, true);
    const auto bg = scene->background.reshape({1, 1, 3}).expand_as(r.out.image);
    CHECK_ALLCLOSE(r.out.image, bg, 0.0, 1e-7);
    CHECK(r.out.alpha.abs().max().item<double>() == 0.0);
    // d(image . wi)/d(background) = sum of the image weights per channel.
    const auto gen = at::make_generator<at::CPUGeneratorImpl>(31);
    const auto wi = at::randn({20, 20, 3}, gen);
    CHECK_ALLCLOSE(r.grads[5], wi.sum(at::IntArrayRef{0, 1}), 1e-5, 1e-5);
  }
}
