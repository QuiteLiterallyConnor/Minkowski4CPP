// Central-difference gradient checks for the Gaussian renderer and numerical
// edge cases for the Gaussian value types.
//
// Finite differences are only meaningful where the rendered loss is smooth.
// The rasterizer has deliberate non-smooth points, so the gradcheck scenes are
// built to stay away from all of them (each is asserted as a precondition):
//   * alpha < 1/255 is dropped and alpha is clamped at 0.99: every splat alone
//     covers every pixel with alpha in [2/255, 0.98];
//   * tile culling at the 3-sigma radius: the same condition means every splat
//     reaches every tile;
//   * the Jacobian frustum clamp (|x/z| <= 0.65 W / fx): centers stay inside;
//   * depth-order swaps: depths differ by > 0.25 world units (60x the step);
//   * the CUDA early stop at T < 1e-4: three splats with opacity <= 0.65 keep
//     T >= 0.04;
//   * SH colors are clamped at 0: radiance + 0.5 stays >= 0.05.
//
// Tolerances (documented in docs/testing.md):
//   float32 renderer, loss accumulated in float64, step h per field below,
//   |analytic - numeric| <= rtol * |numeric| + atol * max|numeric of field|
//   with rtol = 1e-2 and atol = 2e-3 (CPU / LibTorch path) or 5e-3 (CUDA kernels,
//   which use __expf and float atomics). SH and the view-direction chain are
//   additionally checked in float64 with rtol = 1e-6.

#include "gaussian/gaussian3d.hpp"
#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_model.hpp"
#include "gaussian/gaussian_render.hpp"
#include "gaussian/gaussian_sh.hpp"
#include "gaussian/gaussian_tensor.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace minkowski::gaussian;

namespace {

constexpr double kPi = 3.14159265358979323846;

bool gradcheck_cuda_available() {
#ifdef USE_CUDA
  return at::cuda::is_available();
#else
  return false;
#endif
}

torch::Device cuda_device() { return torch::Device(torch::kCUDA, 0); }

// Activated renderer inputs (all float32, same device).
struct Scene {
  at::Tensor positions, rotations, scales, opacities, colors, background;

  std::vector<at::Tensor *> fields() {
    return {&positions, &rotations, &scales, &opacities, &colors, &background};
  }
  Scene map(const std::function<at::Tensor(const at::Tensor &)> &f) const {
    Scene s;
    s.positions = f(positions);
    s.rotations = f(rotations);
    s.scales = f(scales);
    s.opacities = f(opacities);
    s.colors = f(colors);
    s.background = f(background);
    return s;
  }
  Scene to(torch::Device device) const {
    return map([&](const at::Tensor &t) { return t.to(device); });
  }
  Scene clone() const {
    return map([](const at::Tensor &t) { return t.detach().clone(); });
  }
  Scene select(int64_t i) const {
    Scene s = clone();
    s.positions = positions.slice(0, i, i + 1).clone();
    s.rotations = rotations.slice(0, i, i + 1).clone();
    s.scales = scales.slice(0, i, i + 1).clone();
    s.opacities = opacities.slice(0, i, i + 1).clone();
    s.colors = colors.slice(0, i, i + 1).clone();
    return s;
  }
};

const char *const kSceneFieldNames[] = {"positions", "rotations", "scales",
                                        "opacities", "colors",    "background"};
// Central-difference step per field (absolute, in the field's units).
const double kSceneSteps[] = {4e-3, 4e-3, 4e-3, 1e-3, 1e-2, 1e-2};

constexpr int64_t kImageSize = 24;

// Oblique camera so the world-to-camera rotation is not the identity.
Camera gradcheck_camera() {
  return Camera::look_at({1.0, -0.8, -3.2}, {0.1, 0.05, 0.0}, {0, -1, 0},
                         kImageSize, kImageSize, 1.0);
}

// Three large, anisotropic, rotated splats with distinct depths (see header).
Scene smooth_scene() {
  Scene s;
  s.positions = at::tensor({-0.25f, 0.15f, 0.0f, 0.3f, -0.2f, 0.6f, 0.05f,
                            0.3f, -0.5f})
                    .reshape({3, 3});
  // Deliberately unnormalized quaternions (w, x, y, z).
  s.rotations = at::tensor({0.9f, 0.2f, -0.3f, 0.1f, 0.6f, -0.4f, 0.5f, 0.3f,
                            0.7f, 0.1f, 0.2f, -0.6f})
                    .reshape({3, 4});
  s.scales = at::tensor({1.6f, 2.4f, 1.9f, 2.2f, 1.7f, 2.0f, 1.8f, 2.6f, 1.5f})
                 .reshape({3, 3});
  s.opacities = at::tensor({0.55f, 0.4f, 0.65f}).reshape({3, 1});
  s.colors = at::tensor({0.9f, 0.2f, 0.1f, 0.1f, 0.7f, 0.3f, 0.2f, 0.3f, 0.8f})
                 .reshape({3, 3});
  s.background = at::tensor({0.1f, 0.25f, 0.4f});
  return s;
}

struct LossWeights {
  at::Tensor image, depth, alpha; // float64
};

LossWeights make_weights(int64_t size, torch::Device device) {
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(1234);
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  LossWeights w;
  w.image = at::randn({size, size, 3}, gen, opts).to(device);
  w.depth = (0.3 * at::randn({size, size}, gen, opts)).to(device);
  w.alpha = at::randn({size, size}, gen, opts).to(device);
  return w;
}

at::Tensor weighted_loss(const RenderOutput &out, const LossWeights &w) {
  return (out.image.to(at::kDouble) * w.image).sum() +
         (out.depth.to(at::kDouble) * w.depth).sum() +
         (out.alpha.to(at::kDouble) * w.alpha).sum();
}

RenderOutput render_scene(const Scene &s, const Camera &camera,
                          bool cuda_kernels) {
  RenderOptions options;
  options.background = s.background;
  options.use_cuda_kernels = cuda_kernels;
  return render_gaussians(s.positions, s.rotations, s.scales, s.opacities,
                          s.colors, camera, options);
}

// Asserts that the scene stays away from every non-smooth point (see header).
void check_smooth_preconditions(const Scene &scene, const Camera &camera,
                                bool cuda_kernels) {
  torch::NoGradGuard guard;
  const auto count = scene.positions.size(0);
  for (int64_t i = 0; i < count; ++i) {
    const auto alpha = render_scene(scene.select(i), camera, cuda_kernels).alpha;
    const double lo = alpha.min().item<double>();
    const double hi = alpha.max().item<double>();
    if (!(lo >= 2.0 / 255.0 && hi <= 0.98))
      FAIL_AT("gradcheck precondition: splat " << i << " alpha range [" << lo
                                               << ", " << hi << "]");
  }
  const auto w2c = camera.world_to_camera.to(scene.positions.device());
  const auto cam = at::matmul(scene.positions, w2c.slice(0, 0, 3)
                                                   .slice(1, 0, 3)
                                                   .transpose(0, 1)) +
                   w2c.slice(0, 0, 3).select(1, 3);
  const auto z = cam.select(1, 2);
  const double limit = 0.65 * static_cast<double>(camera.width) / camera.fx;
  CHECK((cam.select(1, 0) / z).abs().max().item<double>() < 0.9 * limit);
  CHECK((cam.select(1, 1) / z).abs().max().item<double>() < 0.9 * limit);
  const auto sorted = std::get<0>(z.sort());
  CHECK((sorted.slice(0, 1) - sorted.slice(0, 0, -1)).min().item<double>() >
        0.25);
}

struct GradError {
  double worst_ratio = 0.0; // max |a - n| / tolerance (must stay <= 1)
  double max_abs = 0.0;
  double scale = 0.0;
};

// Compares an analytic gradient with central differences of `loss_fn`
// element by element. `perturb(i, value)` must set element i of the field.
GradError compare_field(const std::string &label, const at::Tensor &analytic,
                        const at::Tensor &values, double step, double rtol,
                        double atol,
                        const std::function<double(int64_t, float)> &loss_at) {
  const auto flat = values.detach().to(at::kCPU).to(at::kFloat).contiguous();
  const auto a =
      analytic.defined()
          ? analytic.detach().to(at::kCPU).to(at::kDouble).reshape({-1})
          : at::zeros({flat.numel()}, at::kDouble);
  const float *x = flat.data_ptr<float>();
  std::vector<double> numeric(static_cast<size_t>(flat.numel()));
  double scale = 0.0;
  for (int64_t i = 0; i < flat.numel(); ++i) {
    const float xp = static_cast<float>(x[i] + step);
    const float xm = static_cast<float>(x[i] - step);
    const double lp = loss_at(i, xp);
    const double lm = loss_at(i, xm);
    numeric[i] = (lp - lm) / (static_cast<double>(xp) - static_cast<double>(xm));
    scale = std::max(scale, std::abs(numeric[i]));
  }
  GradError err;
  err.scale = scale;
  for (int64_t i = 0; i < flat.numel(); ++i) {
    const double ai = a[i].item<double>();
    const double diff = std::abs(ai - numeric[i]);
    const double tol = rtol * std::abs(numeric[i]) + atol * scale + 1e-9;
    err.max_abs = std::max(err.max_abs, diff);
    err.worst_ratio = std::max(err.worst_ratio, diff / tol);
    if (!(diff <= tol))
      FAIL_AT(label << "[" << i << "]: analytic " << ai << " vs numeric "
                    << numeric[i] << " (tol " << tol << ")");
  }
  return err;
}

void report(const std::string &label, const GradError &e) {
  std::cout << "         " << std::left << std::setw(34) << label
            << " max|grad| " << std::setprecision(4) << e.scale << "  max err "
            << e.max_abs << "  err/tol " << e.worst_ratio << "\n";
}

// Full gradcheck of render_gaussians w.r.t. every activated input.
void gradcheck_render(torch::Device device, bool cuda_kernels, double atol,
                      const char *label) {
  const auto camera = gradcheck_camera();
  const auto base = smooth_scene().to(device);
  check_smooth_preconditions(base, camera, cuda_kernels);
  const auto weights = make_weights(kImageSize, device);

  Scene live = base.clone();
  for (auto *t : live.fields())
    t->set_requires_grad(true);
  weighted_loss(render_scene(live, camera, cuda_kernels), weights).backward();

  Scene probe = base.clone();
  auto probe_fields = probe.fields();
  auto live_fields = live.fields();
  for (size_t f = 0; f < probe_fields.size(); ++f) {
    at::Tensor &field = *probe_fields[f];
    const auto original = field.clone();
    auto loss_at = [&](int64_t i, float v) {
      torch::NoGradGuard guard;
      field.view({-1})[i].fill_(v);
      const double loss =
          weighted_loss(render_scene(probe, camera, cuda_kernels), weights)
              .item<double>();
      field.copy_(original);
      return loss;
    };
    const auto err = compare_field(
        std::string(label) + " " + kSceneFieldNames[f], live_fields[f]->grad(),
        original, kSceneSteps[f], 1e-2, atol, loss_at);
    report(std::string(label) + " " + kSceneFieldNames[f], err);
    CHECK(err.scale > 0.0);
  }
}

// Raw-parameter model with SH degree 3 built from the smooth scene.
GaussianModel smooth_sh_model() {
  const auto s = smooth_scene();
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(99);
  // DC colors in [0.26, 0.74] and small SH bands keep radiance + 0.5 well
  // above the clamp at 0 (asserted in gradcheck_model).
  const auto f_rest = 0.03 * at::randn({3, 15, 3}, gen, at::kFloat);
  const auto rgb = 0.2 + 0.6 * s.colors;
  return GaussianModel(s.positions.clone(), (rgb - 0.5) / kShC0, f_rest,
                       at::log(s.opacities / (1.0 - s.opacities)),
                       at::log(s.scales), s.rotations.clone());
}

void gradcheck_model(torch::Device device, bool cuda_kernels, double atol,
                     const char *label) {
  const auto camera = gradcheck_camera();
  const auto base = smooth_sh_model().to(device);
  RenderOptions options;
  options.background = at::tensor({0.1f, 0.25f, 0.4f}).to(device);
  options.use_cuda_kernels = cuda_kernels;
  const auto weights = make_weights(kImageSize, device);
  {
    // SH clamp precondition: radiance + 0.5 >= 0.05 for every channel.
    torch::NoGradGuard guard;
    const auto dirs = base.xyz() - camera.position().to(device);
    const auto unit = dirs / dirs.norm(2, {1}, true);
    const auto radiance =
        (sh_basis(3, unit).unsqueeze(-1) * base.sh_coefficients()).sum(1) + 0.5;
    CHECK(radiance.min().item<double>() >= 0.05);
  }

  auto live = base.clone();
  live.enable_gradients();
  weighted_loss(render(live, camera, options), weights).backward();
  const auto analytic = live.parameters();

  const char *names[] = {"xyz",       "f_dc",      "f_rest",
                         "opacity_raw", "scale_raw", "rotation_raw"};
  const double steps[] = {4e-3, 1e-2, 1e-2, 5e-3, 2e-3, 4e-3};
  auto probe = base.clone();
  auto params = probe.parameters();
  for (size_t f = 0; f < params.size(); ++f) {
    at::Tensor field = params[f];
    const auto original = field.clone();
    auto loss_at = [&](int64_t i, float v) {
      torch::NoGradGuard guard;
      field.view({-1})[i].fill_(v);
      const double loss =
          weighted_loss(render(probe, camera, options), weights).item<double>();
      field.copy_(original);
      return loss;
    };
    const auto err =
        compare_field(std::string(label) + " model." + names[f],
                      analytic[f].grad(), original, steps[f], 1e-2, atol, loss_at);
    report(std::string(label) + " model." + names[f], err);
    CHECK(err.scale > 0.0);
  }
}

// Checks finite outputs and finite gradients for every input; returns the
// rendered output.
RenderOutput render_and_check_finite(const Scene &scene, const Camera &camera,
                                     bool cuda_kernels,
                                     Scene *grads = nullptr) {
  Scene live = scene.clone();
  for (auto *t : live.fields())
    t->set_requires_grad(true);
  auto out = render_scene(live, camera, cuda_kernels);
  CHECK(at::isfinite(out.image).all().item<bool>());
  CHECK(at::isfinite(out.depth).all().item<bool>());
  CHECK(at::isfinite(out.alpha).all().item<bool>());
  CHECK(out.alpha.min().item<double>() >= 0.0);
  CHECK(out.alpha.max().item<double>() <= 1.0);
  const auto weights = make_weights(camera.height, scene.positions.device());
  weighted_loss(out, weights).backward();
  for (auto *t : live.fields()) {
    if (t->numel() == 0)
      continue;
    CHECK(t->grad().defined());
    CHECK(at::isfinite(t->grad()).all().item<bool>());
  }
  if (grads)
    *grads = live.map([](const at::Tensor &t) {
      return t.grad().defined() ? t.grad().detach().clone()
                                : at::zeros_like(t.detach());
    });
  return out;
}

// Paths available on this build: CPU, LibTorch-on-CUDA, CUDA kernels.
struct RenderPath {
  torch::Device device;
  bool cuda_kernels;
  const char *name;
};

std::vector<RenderPath> render_paths() {
  std::vector<RenderPath> paths{{torch::Device(torch::kCPU), false, "cpu"}};
  if (gradcheck_cuda_available()) {
    paths.push_back({cuda_device(), false, "cuda-libtorch"});
    paths.push_back({cuda_device(), true, "cuda-kernels"});
  }
  return paths;
}

Scene make_scene(const std::vector<float> &positions, float scale,
                 float opacity) {
  const int64_t n = static_cast<int64_t>(positions.size() / 3);
  Scene s;
  s.positions = at::tensor(positions).reshape({n, 3});
  s.rotations = at::tensor({1.0f, 0.0f, 0.0f, 0.0f}).repeat({n, 1});
  s.scales = at::full({n, 3}, scale);
  s.opacities = at::full({n, 1}, opacity);
  s.colors = at::rand({n, 3}, at::make_generator<at::CPUGeneratorImpl>(5));
  s.background = at::tensor({0.2f, 0.4f, 0.6f});
  return s;
}

// Camera at the origin looking down +z.
Camera origin_camera(int64_t size = 32) {
  return Camera::look_at({0, 0, 0}, {0, 0, 1}, {0, -1, 0}, size, size, 1.0);
}

// Image equals the background, alpha/depth are zero, and the background
// gradient is the per-channel sum of the image weights.
void check_background_only(const Scene &scene, const Camera &camera,
                           const RenderPath &path) {
  Scene grads;
  const auto s = scene.to(path.device);
  const auto out = render_and_check_finite(s, camera, path.cuda_kernels, &grads);
  const auto bg = s.background.reshape({1, 1, 3}).expand_as(out.image);
  CHECK_ALLCLOSE(out.image, bg, 0.0, 1e-6);
  CHECK(out.alpha.abs().max().item<double>() == 0.0);
  CHECK(out.depth.abs().max().item<double>() == 0.0);
  if (s.positions.size(0) > 0) {
    CHECK(out.visible.sum().item<int64_t>() == 0);
    CHECK(out.radii.abs().sum().item<int64_t>() == 0);
    for (auto *g : {&grads.positions, &grads.rotations, &grads.scales,
                    &grads.opacities, &grads.colors})
      CHECK(g->abs().max().item<double>() == 0.0);
  }
  const auto weights = make_weights(camera.height, path.device);
  const auto expected = weights.image.sum(at::IntArrayRef{0, 1}).to(at::kFloat);
  CHECK_ALLCLOSE(grads.background, expected, 1e-5, 1e-4);
}

} // namespace

// --- Central-difference gradient checks -----------------------------------

TEST(gaussian_gradcheck_render_cpu_all_inputs) {
  gradcheck_render(torch::Device(torch::kCPU), false, 2e-3, "cpu");
}

TEST(gaussian_gradcheck_render_cpu_tile_size_one_tile) {
  // Same check with a single tile covering the image (no tile seams).
  const auto camera = gradcheck_camera();
  const auto base = smooth_scene();
  const auto weights = make_weights(kImageSize, torch::Device(torch::kCPU));
  auto run = [&](int64_t tile) {
    Scene live = base.clone();
    for (auto *t : live.fields())
      t->set_requires_grad(true);
    RenderOptions options;
    options.background = live.background;
    options.tile_size = tile;
    const auto out = render_gaussians(live.positions, live.rotations,
                                      live.scales, live.opacities, live.colors,
                                      camera, options);
    weighted_loss(out, weights).backward();
    return std::make_pair(out, live);
  };
  auto [tiled_out, tiled] = run(16);
  auto [single_out, single] = run(kImageSize);
  CHECK_ALLCLOSE(tiled_out.image, single_out.image, 1e-6, 1e-6);
  auto a = tiled.fields();
  auto b = single.fields();
  for (size_t f = 0; f < a.size(); ++f)
    CHECK_ALLCLOSE(a[f]->grad(), b[f]->grad(), 1e-5, 1e-5);
}

TEST(gaussian_gradcheck_render_cuda_all_inputs) {
  if (!gradcheck_cuda_available())
    throw mink_test::Skip("CUDA not available");
  gradcheck_render(cuda_device(), true, 5e-3, "cuda-kernels");
  gradcheck_render(cuda_device(), false, 2e-3, "cuda-libtorch");
}

TEST(gaussian_gradcheck_model_sh3_cpu) {
  gradcheck_model(torch::Device(torch::kCPU), false, 2e-3, "cpu");
}

TEST(gaussian_gradcheck_model_sh3_cuda) {
  if (!gradcheck_cuda_available())
    throw mink_test::Skip("CUDA not available");
  gradcheck_model(cuda_device(), true, 5e-3, "cuda-kernels");
}

TEST(gaussian_gradcheck_sh_double_precision) {
  // eval_sh_color and the render() view-direction chain in float64.
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(3);
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  const int64_t n = 5;
  const auto camera_center = at::tensor({0.3, -0.2, -2.5}, opts);
  for (int degree = 0; degree <= 3; ++degree) {
    const int64_t k = sh_coefficient_count(degree);
    const auto xyz0 = at::randn({n, 3}, gen, opts);
    const auto sh0 = 0.2 * at::randn({n, k, 3}, gen, opts);
    const auto weights = at::randn({n, 3}, gen, opts);
    auto color_loss = [&](const at::Tensor &xyz, const at::Tensor &sh) {
      auto d = xyz - camera_center;
      d = d / d.norm(2, {1}, true).clamp_min(1e-12);
      return (eval_sh_color(degree, sh, d) * weights).sum();
    };
    {
      // Clamp precondition.
      auto d = xyz0 - camera_center;
      d = d / d.norm(2, {1}, true);
      const auto radiance =
          (sh_basis(degree, d).unsqueeze(-1) * sh0).sum(1) + 0.5;
      CHECK(radiance.min().item<double>() > 0.05);
    }
    auto xyz = xyz0.clone().set_requires_grad(true);
    auto sh = sh0.clone().set_requires_grad(true);
    color_loss(xyz, sh).backward();
    const double h = 1e-6;
    for (int which = 0; which < 2; ++which) {
      const auto &base = which == 0 ? xyz0 : sh0;
      // Degree 0 does not depend on the view direction: no xyz gradient.
      const auto grad = which == 0 ? xyz.grad() : sh.grad();
      const auto analytic =
          grad.defined() ? grad.reshape({-1}) : at::zeros({base.numel()}, opts);
      for (int64_t i = 0; i < base.numel(); ++i) {
        auto p = base.clone(), m = base.clone();
        p.view({-1})[i] += h;
        m.view({-1})[i] -= h;
        const double lp = which == 0 ? color_loss(p, sh0).item<double>()
                                     : color_loss(xyz0, p).item<double>();
        const double lm = which == 0 ? color_loss(m, sh0).item<double>()
                                     : color_loss(xyz0, m).item<double>();
        const double numeric = (lp - lm) / (2 * h);
        CHECK_NEAR(analytic[i].item<double>(), numeric,
                   1e-6 * std::abs(numeric) + 1e-8);
      }
    }
  }
}

TEST(gaussian_gradcheck_sh_basis_is_orthonormal) {
  // Fibonacci-sphere quadrature of the real SH basis up to degree 3:
  // integral of Y_i * Y_j over the sphere must be delta_ij.
  const int64_t points = 40000;
  const auto opts = at::TensorOptions().dtype(at::kDouble);
  const auto i = at::arange(points, opts) + 0.5;
  const auto z = 1.0 - 2.0 * i / static_cast<double>(points);
  const auto r = at::sqrt(1.0 - z * z);
  const auto phi = i * (kPi * (3.0 - std::sqrt(5.0)));
  const auto dirs = at::stack({r * at::cos(phi), r * at::sin(phi), z}, 1);
  const auto basis = sh_basis(3, dirs); // [P, 16]
  const auto gram =
      at::matmul(basis.transpose(0, 1), basis) * (4.0 * kPi / points);
  CHECK_ALLCLOSE(gram, at::eye(16, opts), 0.0, 1e-3);
}

// --- CPU vs CUDA parity with image, depth and alpha in the loss -----------

TEST(gaussian_gradcheck_cuda_parity_image_depth_alpha) {
  if (!gradcheck_cuda_available())
    throw mink_test::Skip("CUDA not available");
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(11);
  const int64_t n = 300;
  Scene base;
  base.positions = at::randn({n, 3}, gen, at::kFloat) * 0.6;
  base.rotations = at::randn({n, 4}, gen, at::kFloat);
  base.scales = at::exp(at::randn({n, 3}, gen, at::kFloat) * 0.3 - 2.3);
  base.opacities = at::sigmoid(at::randn({n, 1}, gen, at::kFloat));
  base.colors = at::rand({n, 3}, gen, at::kFloat);
  base.background = at::tensor({0.1f, 0.2f, 0.3f});
  const auto camera = Camera::look_at({0.4, -0.3, -3.0}, {0, 0, 0},
                                      {0, -1, 0}, 50, 40, 1.0);
  const auto weights = make_weights(50, cuda_device());
  const LossWeights w{weights.image.slice(0, 0, 40),
                      weights.depth.slice(0, 0, 40),
                      weights.alpha.slice(0, 0, 40)};
  auto run = [&](torch::Device device, bool kernels) {
    // clone(): base.to(kCPU) aliases base, and the CPU run must not turn
    // the shared inputs into autograd leaves for the CUDA run.
    Scene live = base.to(device).clone();
    for (auto *t : live.fields())
      t->set_requires_grad(true);
    const auto out = render_scene(live, camera, kernels);
    LossWeights wd{w.image.to(device), w.depth.to(device), w.alpha.to(device)};
    weighted_loss(out, wd).backward();
    return std::make_pair(out, live);
  };
  auto [ref_out, ref] = run(torch::Device(torch::kCPU), false);
  auto [fast_out, fast] = run(cuda_device(), true);
  CHECK_ALLCLOSE(fast_out.image.cpu(), ref_out.image, 1e-3, 1e-3);
  CHECK_ALLCLOSE(fast_out.depth.cpu(), ref_out.depth, 1e-3, 3e-3);
  CHECK_ALLCLOSE(fast_out.alpha.cpu(), ref_out.alpha, 1e-3, 1e-3);
  auto a = ref.fields();
  auto b = fast.fields();
  for (size_t f = 0; f < a.size(); ++f) {
    const auto ga = a[f]->grad();
    const auto gb = b[f]->grad().cpu();
    const double scale = ga.abs().max().item<double>();
    const double err = (ga - gb).abs().max().item<double>();
    report(std::string("parity ") + kSceneFieldNames[f],
           GradError{err / (2e-2 * scale + 1e-4), err, scale});
    CHECK_ALLCLOSE(gb, ga, 2e-2, 2e-2 * scale + 1e-4);
  }
}

// --- Edge cases ------------------------------------------------------------

TEST(gaussian_gradcheck_empty_scene_all_paths) {
  const auto camera = origin_camera(20);
  for (const auto &path : render_paths())
    check_background_only(make_scene({}, 0.1f, 0.5f), camera, path);
}

TEST(gaussian_gradcheck_all_culled_all_paths) {
  const auto camera = origin_camera(20);
  // Behind the camera, before the near plane, far off screen, too transparent.
  const auto behind = make_scene({0, 0, -2, 0.1f, 0, -0.5f}, 0.2f, 0.8f);
  const auto near = make_scene({0, 0, 0.005f}, 0.001f, 0.8f);
  const auto off = make_scene({50, 0, 2, 0, -60, 3, 1e6f, 1e6f, 1e6f}, 0.1f, 0.8f);
  const auto faint = make_scene({0, 0, 2, 0.1f, 0.1f, 3}, 0.3f, 0.003f);
  for (const auto &path : render_paths())
    for (const auto *scene : {&behind, &near, &off, &faint})
      check_background_only(*scene, camera, path);
}

TEST(gaussian_gradcheck_equal_depths_are_consistent) {
  // Overlapping splats with exactly equal depth: the result must not depend on
  // the run or on the device.
  const auto camera = origin_camera(32);
  std::vector<float> xyz;
  for (int i = 0; i < 64; ++i) {
    xyz.push_back(0.02f * static_cast<float>(i % 8) - 0.07f);
    xyz.push_back(0.02f * static_cast<float>(i / 8) - 0.07f);
    xyz.push_back(2.0f);
  }
  const auto scene = make_scene(xyz, 0.15f, 0.7f);
  const auto first = render_and_check_finite(scene, camera, false);
  const auto second = render_and_check_finite(scene, camera, false);
  CHECK(at::equal(first.image, second.image));
  for (const auto &path : render_paths()) {
    const auto out = render_and_check_finite(scene.to(path.device), camera,
                                             path.cuda_kernels);
    CHECK_ALLCLOSE(out.image.cpu(), first.image, 1e-3, 1e-3);
  }
}

TEST(gaussian_gradcheck_near_plane_crossing_splats) {
  const auto camera = origin_camera(32);
  // Centers just past the near plane (0.01) with extents crossing it, and a
  // large splat straddling the plane.
  const auto close = make_scene({0, 0, 0.011f, 0.002f, -0.001f, 0.02f}, 0.05f, 0.6f);
  const auto straddle = make_scene({0.1f, 0.05f, 0.3f}, 0.5f, 0.6f);
  for (const auto &path : render_paths())
    for (const auto *scene : {&close, &straddle}) {
      const auto out =
          render_and_check_finite(scene->to(path.device), camera, path.cuda_kernels);
      CHECK(out.alpha.max().item<double>() > 0.1);
    }
}

TEST(gaussian_gradcheck_camera_inside_huge_splat) {
  const auto camera = origin_camera(32);
  // Center slightly in front of the camera: covers the whole image.
  const auto front = make_scene({0.01f, -0.02f, 0.05f}, 50.0f, 0.5f);
  // Center behind the camera: culled like in 3DGS (documented behaviour).
  const auto back = make_scene({0.0f, 0.0f, -0.05f}, 50.0f, 0.5f);
  for (const auto &path : render_paths()) {
    const auto out =
        render_and_check_finite(front.to(path.device), camera, path.cuda_kernels);
    CHECK(out.alpha.min().item<double>() > 0.45);
    check_background_only(back, camera, path);
  }
}

TEST(gaussian_gradcheck_extreme_scales_and_coordinates) {
  const auto camera = origin_camera(32);
  std::vector<Scene> scenes;
  scenes.push_back(make_scene({0.05f, 0.0f, 2.0f}, 1e-7f, 0.9f)); // sub-pixel
  scenes.push_back(make_scene({0.05f, 0.0f, 2.0f}, 1e4f, 0.5f));  // huge
  scenes.push_back(make_scene({0.05f, 0.0f, 2.0f}, 1e12f, 0.5f)); // overflowing
  {
    auto flat = make_scene({0.0f, 0.1f, 2.0f, -0.1f, 0.0f, 3.0f}, 0.2f, 0.7f);
    flat.scales.select(1, 2).fill_(0.0f); // degenerate disc
    flat.scales[1].fill_(0.0f);           // fully degenerate point
    scenes.push_back(flat);
    auto needle = make_scene({0.0f, 0.0f, 2.0f}, 1e-6f, 0.8f);
    needle.scales[0][0] = 3.0f; // 3e6 aspect ratio
    needle.rotations = at::tensor({0.92f, 0.1f, 0.3f, 0.2f}).reshape({1, 4});
    scenes.push_back(needle);
  }
  scenes.push_back(make_scene({0.0f, 0.0f, 1e6f}, 1.0f, 0.9f));     // far
  scenes.push_back(make_scene({1e20f, 1e20f, 1e20f}, 1.0f, 0.9f)); // extreme
  scenes.push_back(make_scene({-1e30f, 0.0f, 1e30f}, 1e29f, 0.9f));
  for (const auto &path : render_paths())
    for (size_t i = 0; i < scenes.size(); ++i) {
      try {
        render_and_check_finite(scenes[i].to(path.device), camera,
                                path.cuda_kernels);
      } catch (const mink_test::Failure &e) {
        FAIL_AT("scene " << i << " on " << path.name << ": " << e.what());
      }
    }
}

TEST(gaussian_gradcheck_extreme_scenes_match_across_paths) {
  if (!gradcheck_cuda_available())
    throw mink_test::Skip("CUDA not available");
  const auto camera = origin_camera(32);
  const auto huge = make_scene({0.05f, 0.0f, 2.0f}, 1e4f, 0.5f);
  auto needle = make_scene({0.0f, 0.0f, 2.0f}, 1e-3f, 0.8f);
  needle.scales[0][0] = 0.6f;
  needle.rotations = at::tensor({0.92f, 0.1f, 0.3f, 0.2f}).reshape({1, 4});
  for (const Scene *scene : std::vector<const Scene *>{&huge, &needle}) {
    const auto ref = render_and_check_finite(*scene, camera, false);
    const auto out =
        render_and_check_finite(scene->to(cuda_device()), camera, true);
    CHECK_ALLCLOSE(out.image.cpu(), ref.image, 1e-3, 1e-3);
    CHECK_ALLCLOSE(out.alpha.cpu(), ref.alpha, 1e-3, 1e-3);
  }
}

// --- Gaussian3D / GaussianTensor near-degenerate covariance ---------------

TEST(gaussian_gradcheck_gaussian3d_near_degenerate_inverse) {
  // Thin rotated Gaussians: the inverse covariance must reproduce the exact
  // Mahalanobis distance k^2 along every principal axis.
  for (double thin : {1e-3, 1e-6}) {
    Gaussian3Dd g;
    g.mean = {0.3, -1.2, 4.0};
    g.scale = {2.0, 0.5, thin};
    g.orientation = {0.8, 0.3, -0.4, 0.2};
    const auto rotation = g.rotation_matrix();
    const auto cov = g.covariance();
    const auto inv = g.inverse_covariance();
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) {
        CHECK_NEAR(cov[r][c], cov[c][r], 1e-15);
        CHECK_NEAR(inv[r][c], inv[c][r], 1e-15 * std::abs(inv[r][c]) + 1e-12);
      }
    // The Mahalanobis distance is checked at thin = 1e-3 only: through the
    // explicit inverse covariance (entries ~1/thin^2) even float64 loses
    // ~1e-4 relative at thin <= 1e-5 (2.89038 vs 2.89). Fixed and tested in
    // fix/gaussian-projection-stability.
    for (int axis = 0; axis < 3 && thin >= 1e-3; ++axis) {
      const double k = 1.7;
      Vector3<double> p = g.mean;
      for (int r = 0; r < 3; ++r)
        p[r] += k * g.scale[axis] * rotation[r][axis];
      CHECK_NEAR(g.mahalanobis_squared(p), k * k, 1e-6 * k * k);
    }
    CHECK_NEAR(g.determinant(), std::pow(2.0 * 0.5 * thin, 2),
               1e-12 * std::pow(2.0 * 0.5 * thin, 2));
    // Not checked here: Gaussian3D<float>::mahalanobis_squared via the
    // explicit inverse covariance loses ~5% at scale ratios of 2e3 (scales
    // {2, 0.5, 1e-3}: 2.75 instead of 2.89 for k^2 = 2.89). Fixed and tested
    // in fix/gaussian-projection-stability.
  }
}

TEST(gaussian_gradcheck_gaussian3d_overlap_near_degenerate_float) {
  // overlap_integral of thin Gaussians in float must match double.
  for (double thin : {1e-2, 1e-3}) {
    Gaussian3Dd a, b;
    a.scale = {1.0, 0.8, thin};
    a.orientation = {0.9, 0.2, -0.1, 0.3};
    b.scale = {0.7, 1.1, thin};
    b.orientation = {0.9, 0.2, -0.1, 0.32};
    b.mean = {0.1, -0.05, 0.0};
    const double expected = a.overlap_integral(b);
    Gaussian3Df af, bf;
    for (int i = 0; i < 3; ++i) {
      af.scale[i] = static_cast<float>(a.scale[i]);
      bf.scale[i] = static_cast<float>(b.scale[i]);
      af.mean[i] = static_cast<float>(a.mean[i]);
      bf.mean[i] = static_cast<float>(b.mean[i]);
    }
    for (int i = 0; i < 4; ++i) {
      af.orientation[i] = static_cast<float>(a.orientation[i]);
      bf.orientation[i] = static_cast<float>(b.orientation[i]);
    }
    const double got = af.overlap_integral(bf);
    std::cout << "         overlap thin=" << thin << " double " << expected
              << " float " << got << "\n";
    CHECK_NEAR(got, expected, 2e-2 * expected);
  }
}

TEST(gaussian_gradcheck_tensor_covariance_near_degenerate) {
  const auto gen = at::make_generator<at::CPUGeneratorImpl>(21);
  const int64_t n = 64;
  auto rotations = at::randn({n, 4}, gen, at::kDouble);
  auto scales = at::exp(at::rand({n, 3}, gen, at::kDouble) * 2.0 - 1.0);
  scales.select(1, 2).fill_(1e-4); // near-degenerate axis
  scales.slice(0, 0, 8).select(1, 1).fill_(1e-4);
  const auto zeros3 = at::zeros({n, 3}, at::kDouble);
  const GaussianTensor ref(zeros3, rotations, scales, at::ones({n, 1}, at::kDouble),
                           zeros3);
  const auto cov64 = ref.covariances();
  // Eigenvalues are the squared scales.
  const auto eig = std::get<0>(at::linalg_eigh(cov64));
  const auto expected = std::get<0>((scales * scales).sort(1));
  CHECK_ALLCLOSE(eig, expected, 1e-9, 1e-14);
  CHECK_ALLCLOSE(cov64, cov64.transpose(1, 2), 0.0, 1e-15);

  const GaussianTensor f32(zeros3.to(at::kFloat), rotations.to(at::kFloat),
                           scales.to(at::kFloat),
                           at::ones({n, 1}, at::kFloat), zeros3.to(at::kFloat));
  const auto cov32 = f32.covariances();
  CHECK_ALLCLOSE(cov32.to(at::kDouble), cov64,
                 0.0, 1e-6 * cov64.abs().amax().item<double>());
  if (gradcheck_cuda_available()) {
    const auto gpu = f32.to(cuda_device()).covariances().cpu();
    CHECK_ALLCLOSE(gpu, cov32, 1e-5, 1e-6);
    const auto gpu64 = ref.to(cuda_device()).covariances().cpu();
    CHECK_ALLCLOSE(gpu64, cov64, 1e-12, 1e-15);
  }
}
