// Tile bounds of the rasterizer: every path must match a brute-force
// reference that composites every Gaussian at every pixel (no tiles, no
// culling box), and the bounds must shrink with opacity and follow the
// footprint's aspect ratio.

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_render.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include "gaussian_reference.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace minkowski::gaussian;
using namespace gaussian_reference;

namespace {

bool tile_bounds_cuda_available() {
#ifdef USE_CUDA
  return at::cuda::is_available();
#else
  return false;
#endif
}

// Opaque (o in [0.3, 0.99]), anisotropic, rotated splats; many are centered
// off screen so their footprints reach into the image only partially.
Splats opaque_scene(int64_t n, uint64_t seed, double log_scale = -2.6) {
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
  Splats s;
  s.positions = at::randn({n, 3}, gen, at::kFloat) * 0.9;
  s.rotations = at::randn({n, 4}, gen, at::kFloat);
  s.scales = at::exp(at::randn({n, 3}, gen, at::kFloat) * 0.6 + log_scale);
  s.opacities = 0.3 + 0.69 * at::rand({n, 1}, gen, at::kFloat);
  s.colors = at::rand({n, 3}, gen, at::kFloat);
  s.background = at::tensor({0.05f, 0.1f, 0.15f});
  return s;
}

Camera bounds_camera() {
  // 61 x 47 pixels: neither side is a multiple of the 16-pixel tile.
  return Camera::look_at({0.2, -0.1, -3.0}, {0, 0, 0}, {0, -1, 0}, 61, 47,
                         1.1);
}

// Tolerances: float32 projection and compositing (~1e-5) plus the early stop
// at T < 1e-4, which drops at most 1e-4 x (color + background) or depth.
void expect_matches_reference(const RenderOutput &out, const Reference &ref,
                              const std::string &label) {
  const auto e = reference_error(out, ref);
  std::cout << "         " << label << ": max |render - brute force| image "
            << e.image << ", alpha " << e.alpha << ", depth " << e.depth
            << " (ambiguous pixels " << e.ambiguous_fraction * 100 << "%)\n";
  CHECK(e.ambiguous_fraction < 0.01);
  if (!(e.image < 5e-4 && e.alpha < 5e-4 && e.depth < 2e-3))
    FAIL_AT(label << " differs from the brute-force reference");
}

} // namespace

TEST(gaussian_tile_bounds_match_brute_force_cpu) {
  const auto camera = bounds_camera();
  // Seed 3: larger splats (footprints of several tiles), where a 3-sigma box
  // misses visible alpha in whole tiles.
  for (uint64_t seed : {1u, 2u, 3u}) {
    const auto scene = opaque_scene(seed == 3 ? 60 : 240, seed,
                                    seed == 3 ? -1.6 : -2.6);
    const auto ref = brute_force(scene, camera);
    for (int64_t tile : {16, 7, 64}) {
      torch::NoGradGuard guard;
      expect_matches_reference(render_splats(scene, camera, false, tile), ref,
                               "cpu seed " + std::to_string(seed) + " tile " +
                                   std::to_string(tile));
    }
  }
}

TEST(gaussian_tile_bounds_match_brute_force_cuda) {
  if (!tile_bounds_cuda_available())
    throw mink_test::Skip("CUDA not available");
  const torch::Device device(torch::kCUDA, 0);
  const auto camera = bounds_camera();
  // Seed 3: larger splats (footprints of several tiles), where a 3-sigma box
  // misses visible alpha in whole tiles.
  for (uint64_t seed : {1u, 2u, 3u}) {
    const auto scene = opaque_scene(seed == 3 ? 60 : 240, seed,
                                    seed == 3 ? -1.6 : -2.6);
    const auto ref = brute_force(scene, camera);
    torch::NoGradGuard guard;
    const auto gpu = scene.to(device);
    expect_matches_reference(render_splats(gpu, camera, true), ref,
                             "cuda-kernels seed " + std::to_string(seed));
    expect_matches_reference(render_splats(gpu, camera, false), ref,
                             "cuda-libtorch seed " + std::to_string(seed));
  }
}

TEST(gaussian_tile_bounds_shrink_with_opacity) {
  // Identical geometry: a transparent splat reaches the 1/255 cut-off at a
  // smaller Mahalanobis radius sqrt(2 ln(255 o)) than an opaque one.
  const auto camera = Camera::look_at({0, 0, -3}, {0, 0, 0}, {0, -1, 0}, 128,
                                      128, 1.0);
  Splats s;
  s.positions = at::zeros({2, 3});
  s.rotations = at::tensor({1.0f, 0.0f, 0.0f, 0.0f}).repeat({2, 1});
  s.scales = at::full({2, 3}, 0.2f);
  s.opacities = at::tensor({0.99f, 0.02f}).reshape({2, 1});
  s.colors = at::ones({2, 3});
  s.background = at::zeros({3});
  torch::NoGradGuard guard;
  const auto radii = render_splats(s, camera, false).radii;
  const double sigma = 0.2 * camera.fx / 3.0; // pixels
  const double opaque = std::sqrt(2.0 * std::log(255.0 * 0.99)) * sigma;
  const double faint = std::sqrt(2.0 * std::log(255.0 * 0.02)) * sigma;
  std::cout << "         radii opaque " << radii[0].item<int64_t>()
            << " faint " << radii[1].item<int64_t>() << "\n";
  CHECK(radii[0].item<int64_t>() >= static_cast<int64_t>(opaque));
  CHECK(radii[0].item<int64_t>() <= static_cast<int64_t>(opaque) + 2);
  CHECK(radii[1].item<int64_t>() <= static_cast<int64_t>(faint) + 2);
  CHECK(radii[1].item<int64_t>() < 0.65 * radii[0].item<int64_t>());
}

TEST(gaussian_tile_bounds_follow_anisotropy) {
  // A splat stretched along x covers one row of 16-pixel tiles, not a square.
  const auto camera = Camera::look_at({0, 0, -3}, {0, 0, 0}, {0, -1, 0}, 256,
                                      256, 1.0);
  Splats s;
  // Center row y = 136 px, the middle of tile row 8 ([128, 144)).
  s.positions = at::tensor({0.0f, 0.1025f, 0.0f}).reshape({1, 3});
  s.rotations = at::tensor({1.0f, 0.0f, 0.0f, 0.0f}).reshape({1, 4});
  s.scales = at::tensor({0.4f, 0.004f, 0.004f}).reshape({1, 3});
  s.opacities = at::full({1, 1}, 0.9f);
  s.colors = at::ones({1, 3});
  s.background = at::zeros({3});
  torch::NoGradGuard guard;
  const auto out = render_splats(s, camera, false);
  CHECK(out.tile_pairs.defined());
  const auto pairs = out.tile_pairs.item<int64_t>();
  std::cout << "         needle tile pairs " << pairs << " (radius "
            << out.radii[0].item<int64_t>() << ")\n";
  // Footprint ~2 x 3.33 x 0.4 fx / 3 = 208 px wide, ~4 px tall: one row of
  // 14 tiles. (The square 3-sigma box of the previous bounds: 12 x 12 tiles.)
  CHECK(pairs >= 13 && pairs <= 256 / 16);
  expect_matches_reference(out, brute_force(s, camera), "needle");
}

TEST(gaussian_tile_pairs_counts_rasterized_pairs) {
  const auto camera = bounds_camera();
  const auto scene = opaque_scene(120, 3);
  torch::NoGradGuard guard;
  const auto a = render_splats(scene, camera, false, 16);
  const auto b = render_splats(scene, camera, false, 61);
  CHECK(a.tile_pairs.item<int64_t>() >= a.visible.sum().item<int64_t>());
  // One tile covers the whole image: one pair per visible Gaussian.
  CHECK_EQ(b.tile_pairs.item<int64_t>(), b.visible.sum().item<int64_t>());
  if (tile_bounds_cuda_available()) {
    const auto gpu = render_splats(scene.to(torch::Device(torch::kCUDA, 0)),
                                   camera, true);
    // Same bounds up to float rounding of the projection on another device.
    const auto cpu_pairs = a.tile_pairs.item<int64_t>();
    CHECK(std::abs(gpu.tile_pairs.item<int64_t>() - cpu_pairs) <=
          cpu_pairs / 100 + 1);
  }
}
