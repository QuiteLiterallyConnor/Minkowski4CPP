// Projection robustness: principal-point-aware frustum clamp (crops render
// like the full image), numerically stable 2D covariance determinant for
// needle splats, and a stable depth sort for equal depths.

#include "gaussian/gaussian3d.hpp"
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
#include <iostream>
#include <string>
#include <vector>

using namespace minkowski::gaussian;
using namespace gaussian_reference;

namespace {

struct Path {
  torch::Device device;
  bool kernels;
  const char *name;
};

std::vector<Path> projection_paths() {
  std::vector<Path> paths{{torch::Device(torch::kCPU), false, "cpu"}};
#ifdef USE_CUDA
  if (at::cuda::is_available()) {
    paths.push_back({torch::Device(torch::kCUDA, 0), true, "cuda-kernels"});
    paths.push_back({torch::Device(torch::kCUDA, 0), false, "cuda-libtorch"});
  }
#endif
  return paths;
}

// Semi-transparent (o <= 0.3) splats: their 1/255 footprint lies inside the
// 3-sigma tile bounds, so these tests do not depend on the tile bounds.
Splats splats_at_pixels(const Camera &camera, const at::Tensor &pixels,
                        const at::Tensor &depths, float scale,
                        uint64_t seed) {
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(seed);
  const int64_t n = pixels.size(0);
  // Camera at the origin of a look_at({0,0,-3}) camera: world = camera - 3z.
  const auto x = (pixels.select(1, 0) - camera.cx) / camera.fx * depths;
  const auto y = (pixels.select(1, 1) - camera.cy) / camera.fy * depths;
  Splats s;
  s.positions = at::stack({x, y, depths - 3.0}, 1).to(at::kFloat);
  s.rotations = at::randn({n, 4}, gen, at::kFloat);
  s.scales = scale * at::exp(0.3 * at::randn({n, 3}, gen, at::kFloat));
  s.opacities = 0.1 + 0.2 * at::rand({n, 1}, gen, at::kFloat);
  s.colors = at::rand({n, 3}, gen, at::kFloat);
  s.background = at::tensor({0.1f, 0.2f, 0.3f});
  return s;
}

Camera straight_camera(int64_t width, int64_t height) {
  return Camera::look_at({0, 0, -3}, {0, 0, 0}, {0, -1, 0}, width, height,
                         1.0);
}

} // namespace

TEST(gaussian_projection_crop_matches_full_image) {
  // The left half of a 128 x 96 image rendered as a 64 x 96 camera with the
  // same intrinsics (cx = 64 at its right edge) must give the same pixels.
  // Splat centers lie at x in [-8, 72] px, inside both cameras' clamp limits
  // (W - cx + 0.15 W beyond the principal point), so neither clamps them.
  const auto full = straight_camera(128, 96);
  Camera crop = full;
  crop.width = 64;
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(4);
  const int64_t n = 80;
  const auto pixels = at::stack(
      {-8.0 + 80.0 * at::rand({n}, gen, at::kDouble),
       96.0 * at::rand({n}, gen, at::kDouble)},
      1);
  const auto depths = 2.5 + at::rand({n}, gen, at::kDouble);
  const auto scene = splats_at_pixels(full, pixels, depths, 0.12f, 5);
  for (const auto &path : projection_paths()) {
    torch::NoGradGuard guard;
    const auto s = scene.to(path.device);
    const auto a = render_splats(s, full, path.kernels);
    const auto b = render_splats(s, crop, path.kernels);
    const double image =
        (a.image.slice(1, 0, 64) - b.image).abs().max().item<double>();
    const double alpha =
        (a.alpha.slice(1, 0, 64) - b.alpha).abs().max().item<double>();
    std::cout << "         " << path.name << ": max |crop - full| image "
              << image << ", alpha " << alpha << "\n";
    CHECK(image < 1e-5);
    CHECK(alpha < 1e-5);
  }
}

TEST(gaussian_projection_crop_matches_reference) {
  // Off-center principal point against the float64 reference.
  auto camera = straight_camera(64, 96);
  camera.cx = 64.0;
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(6);
  const int64_t n = 60;
  const auto pixels = at::stack(
      {-20.0 + 100.0 * at::rand({n}, gen, at::kDouble),
       96.0 * at::rand({n}, gen, at::kDouble)},
      1);
  const auto scene = splats_at_pixels(
      camera, pixels, 2.5 + at::rand({n}, gen, at::kDouble), 0.12f, 7);
  const auto ref = brute_force(scene, camera);
  for (const auto &path : projection_paths()) {
    torch::NoGradGuard guard;
    const auto e = reference_error(
        render_splats(scene.to(path.device), camera, path.kernels), ref);
    std::cout << "         " << path.name << ": max |render - reference| image "
              << e.image << ", alpha " << e.alpha << "\n";
    CHECK(e.ambiguous_fraction < 0.01);
    CHECK(e.image < 5e-4);
    CHECK(e.alpha < 5e-4);
  }
}

TEST(gaussian_projection_needle_determinant_is_stable) {
  // A needle ~2000 px long (sigma) and ~0.01 px thin at 45 degrees: the 2D
  // covariance is [[A, A], [A, A]] + 0.3 I with A ~ 2e6, so a * c - b * b
  // loses most digits in float32. The thin direction (set by the 0.3 px
  // low-pass) must match the float64 reference.
  const auto camera = straight_camera(256, 256);
  Splats s;
  s.positions = at::tensor({0.004f, -0.003f, 0.0f}).reshape({1, 3});
  const double half = 0.5 * (3.14159265358979323846 / 4.0);
  s.rotations = at::tensor({static_cast<float>(std::cos(half)), 0.0f, 0.0f,
                            static_cast<float>(std::sin(half))})
                    .reshape({1, 4});
  s.scales = at::tensor({25.6f, 1e-4f, 1e-4f}).reshape({1, 3});
  s.opacities = at::full({1, 1}, 0.3f);
  s.colors = at::tensor({0.9f, 0.6f, 0.3f}).reshape({1, 3});
  s.background = at::zeros({3});
  const auto ref = brute_force(s, camera);
  for (const auto &path : projection_paths()) {
    torch::NoGradGuard guard;
    const auto out = render_splats(s.to(path.device), camera, path.kernels);
    const auto e = reference_error(out, ref);
    std::cout << "         " << path.name << ": needle max |render - reference| "
              << "image " << e.image << ", alpha " << e.alpha << "\n";
    CHECK(out.visible.all().item<bool>());
    // Remaining error: float32 rounding of a, b, c (~0.1 at 2e6) only
    // affects the long axis, where the power is < 0.01 inside the image.
    CHECK(e.alpha < 3e-3);
    CHECK(e.image < 3e-3);
  }
}

TEST(gaussian_projection_equal_depths_keep_index_order) {
  // 600 overlapping splats at exactly the same depth: compositing order must
  // be the index order (stable sort) on every path.
  const auto camera = straight_camera(48, 48);
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(8);
  const int64_t n = 600;
  const auto pixels = 4.0 + 40.0 * at::rand({n, 2}, gen, at::kDouble);
  auto scene = splats_at_pixels(camera, pixels, at::full({n}, 3.0, at::kDouble),
                                0.08f, 9);
  scene.positions.select(1, 2).fill_(0.0f); // bit-identical depths
  scene.opacities.fill_(0.3f);
  const auto ref = brute_force(scene, camera);
  for (const auto &path : projection_paths()) {
    torch::NoGradGuard guard;
    const auto e = reference_error(
        render_splats(scene.to(path.device), camera, path.kernels), ref);
    std::cout << "         " << path.name
              << ": equal depths max |render - reference| image " << e.image
              << "\n";
    CHECK(e.ambiguous_fraction < 0.02);
    CHECK(e.image < 5e-4);
  }
}

TEST(gaussian3d_float_mahalanobis_thin_axes_match_double) {
  // Float Gaussian3D with scale ratios up to 2e4: the Mahalanobis distance of
  // float points must match float64 evaluated at the same (rounded) points.
  for (float thin : {1e-2f, 1e-3f, 1e-4f}) {
    Gaussian3Df gf;
    gf.mean = {0.3f, -1.2f, 4.0f};
    gf.scale = {2.0f, 0.5f, thin};
    gf.orientation = {0.8f, 0.3f, -0.4f, 0.2f};
    Gaussian3Dd gd;
    for (int i = 0; i < 3; ++i) {
      gd.mean[i] = gf.mean[i];
      gd.scale[i] = gf.scale[i];
    }
    for (int i = 0; i < 4; ++i)
      gd.orientation[i] = gf.orientation[i];
    const auto rotation = gd.rotation_matrix();
    for (int axis = 0; axis < 3; ++axis)
      for (double k : {0.5, 1.7, 3.0}) {
        Vector3<float> pf{};
        Vector3<double> pd{};
        for (int r = 0; r < 3; ++r) {
          pf[r] = static_cast<float>(gd.mean[r] +
                                     k * gd.scale[axis] * rotation[r][axis]);
          pd[r] = pf[r];
        }
        const double expected = gd.mahalanobis_squared(pd);
        const double got = gf.mahalanobis_squared(pf);
        // Float rounding of R^T d: ~4e-7 absolute along each axis.
        const double e = 4e-7 / thin;
        CHECK_NEAR(got, expected,
                   1e-5 * expected + 2.0 * std::sqrt(expected) * e + e * e);
      }
  }
}
TEST(gaussian3d_double_mahalanobis_thin_axes_are_exact) {
  // float64 through the explicit inverse lost ~1e-4 relative at thin = 1e-6.
  for (double thin : {1e-4, 1e-5, 1e-6}) { // 1e-6 = kMinimumScale
    Gaussian3Dd g;
    g.mean = {0.3, -1.2, 4.0};
    g.scale = {2.0, 0.5, thin};
    g.orientation = {0.8, 0.3, -0.4, 0.2};
    const auto rotation = g.rotation_matrix();
    for (int axis = 0; axis < 3; ++axis) {
      Vector3<double> p = g.mean;
      for (int r = 0; r < 3; ++r)
        p[r] += 1.7 * g.scale[axis] * rotation[r][axis];
      CHECK_NEAR(g.mahalanobis_squared(p), 2.89, 1e-9);
    }
  }
}
