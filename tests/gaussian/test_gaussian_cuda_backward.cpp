// CUDA backward kernel against the LibTorch reference path on scenes where
// most warps see no hit for most Gaussians (the warp-level skip in
// raster_backward_kernel), with partial edge tiles and image + depth + alpha
// in the loss.

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_render.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace minkowski::gaussian;

namespace {

struct Inputs {
  at::Tensor positions, rotations, scales, opacities, colors, background;
  std::vector<at::Tensor *> all() {
    return {&positions, &rotations, &scales, &opacities, &colors, &background};
  }
};

const char *const kNames[] = {"positions", "rotations", "scales",
                              "opacities", "colors",    "background"};

// Returns the gradients of image . wi + depth . wd + alpha . wa.
std::vector<at::Tensor> gradients(const Inputs &base, const Camera &camera,
                                  torch::Device device, bool kernels,
                                  at::Tensor *image) {
  Inputs live;
  live.positions = base.positions.to(device).requires_grad_(true);
  live.rotations = base.rotations.to(device).requires_grad_(true);
  live.scales = base.scales.to(device).requires_grad_(true);
  live.opacities = base.opacities.to(device).requires_grad_(true);
  live.colors = base.colors.to(device).requires_grad_(true);
  live.background = base.background.to(device).requires_grad_(true);
  RenderOptions options;
  options.background = live.background;
  options.use_cuda_kernels = kernels;
  const auto out =
      render_gaussians(live.positions, live.rotations, live.scales,
                       live.opacities, live.colors, camera, options);
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(17);
  const auto wi = at::randn({camera.height, camera.width, 3}, gen).to(device);
  const auto wd = at::randn({camera.height, camera.width}, gen).to(device);
  const auto wa = at::randn({camera.height, camera.width}, gen).to(device);
  ((out.image * wi).sum() + 0.1 * (out.depth * wd).sum() +
   (out.alpha * wa).sum())
      .backward();
  *image = out.image.detach().cpu();
  std::vector<at::Tensor> grads;
  for (auto *t : live.all())
    grads.push_back(t->grad().cpu());
  return grads;
}

void expect_parity(const Inputs &scene, const Camera &camera,
                   const std::string &label) {
  const torch::Device cuda(torch::kCUDA, 0);
  at::Tensor ref_image, fast_image;
  const auto ref = gradients(scene, camera, cuda, false, &ref_image);
  const auto fast = gradients(scene, camera, cuda, true, &fast_image);
  // The LibTorch path stops whole chunks at T < 1e-4, the kernels per pixel.
  CHECK_ALLCLOSE(fast_image, ref_image, 1e-3, 1e-3);
  for (size_t k = 0; k < ref.size(); ++k) {
    const double scale = ref[k].abs().max().item<double>();
    const double err = (fast[k] - ref[k]).abs().max().item<double>();
    std::cout << "         " << label << " " << kNames[k] << ": max |grad| "
              << scale << ", max err " << err << "\n";
    CHECK(scale > 0.0);
    // Same tolerance as gaussian_cuda_rasterizer_matches_reference_* (the
    // kernels use __expf, float atomics and T recovered by division).
    CHECK_ALLCLOSE(fast[k], ref[k], 2e-2, 2e-2 * scale + 1e-4);
  }
}

} // namespace

TEST(gaussian_cuda_backward_sparse_hits_match_reference) {
#ifdef USE_CUDA
  if (!at::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
  // 3000 small splats over a 77 x 45 image (partial tiles on both axes): most
  // (warp, Gaussian) pairs of a tile list have no lane above 1/255.
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(23);
  const int64_t n = 3000;
  Inputs s;
  s.positions = at::randn({n, 3}, gen, at::kFloat) * 0.8;
  s.rotations = at::randn({n, 4}, gen, at::kFloat);
  s.scales = at::exp(at::randn({n, 3}, gen, at::kFloat) * 0.4 - 4.0);
  s.opacities = at::sigmoid(at::randn({n, 1}, gen, at::kFloat));
  s.colors = at::rand({n, 3}, gen, at::kFloat);
  s.background = at::tensor({0.2f, 0.1f, 0.3f});
  const auto camera = Camera::look_at({0.3, -0.2, -3.0}, {0, 0, 0},
                                      {0, -1, 0}, 77, 45, 1.0);
  expect_parity(s, camera, "sparse");
#else
  throw mink_test::Skip("built without CUDA");
#endif
}

TEST(gaussian_cuda_backward_saturated_tiles_match_reference) {
#ifdef USE_CUDA
  if (!at::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
  // Large overlapping splats: every lane is hit, pixels stop early at
  // different Gaussians (per-pixel last_index), mixed with a few tiny ones.
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(29);
  const int64_t n = 400;
  Inputs s;
  s.positions = at::randn({n, 3}, gen, at::kFloat) * 0.5;
  s.rotations = at::randn({n, 4}, gen, at::kFloat);
  s.scales = at::exp(at::randn({n, 3}, gen, at::kFloat) * 0.8 - 2.0);
  s.opacities = 0.2 + 0.75 * at::rand({n, 1}, gen, at::kFloat);
  s.colors = at::rand({n, 3}, gen, at::kFloat);
  s.background = at::tensor({0.0f, 0.5f, 1.0f});
  const auto camera = Camera::look_at({0.0, 0.1, -3.0}, {0, 0, 0},
                                      {0, -1, 0}, 50, 66, 1.0);
  expect_parity(s, camera, "saturated");
#else
  throw mink_test::Skip("built without CUDA");
#endif
}
