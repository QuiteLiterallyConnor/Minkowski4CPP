// Fused CUDA projection (gaussian_project_cuda.cu) against the LibTorch projection it replaces:
// whole renders (image, depth, alpha and every input gradient, plus means2d.grad for
// densification) with the fused path on and off, and the projection columns and gradients
// against a float64 LibTorch reference of the same math, including clamped / degenerate cases.
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <utility>
#include <vector>

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_cuda.hpp"
#include "gaussian/gaussian_render.hpp"
#include "utils/test_framework.hpp"

#ifdef USE_CUDA // the fused projection and its hook exist only in CUDA builds

using namespace minkowski::gaussian;
namespace gd = minkowski::gaussian::detail;

namespace {

void require_cuda() {
  if (!torch::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
}

struct Scene {
  at::Tensor positions, rotations, scales, opacities, colors;
};

// N Gaussians around the origin; some behind the camera, some far outside the image (so the
// Jacobian clamp is active), some needle-thin, some with a near-zero quaternion.
Scene make_scene(int64_t n, uint64_t seed, const at::Device &dev) {
  torch::manual_seed(seed);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32);
  Scene s;
  s.positions = torch::randn({n, 3}, f) * 1.2;
  s.positions.slice(0, 0, n / 20).select(1, 2).fill_(-3.5);           // behind the camera
  s.positions.slice(0, n / 20, n / 10).select(1, 0).mul_(8.0);        // far off to the sides
  s.rotations = torch::randn({n, 4}, f);
  s.rotations.slice(0, n / 10, n / 10 + 3).mul_(1e-7);                // near-degenerate quaternions
  s.scales = torch::rand({n, 3}, f) * 0.15 + 0.005;
  s.scales.slice(0, n / 5, n / 4).select(1, 1).fill_(1e-4);           // needles
  s.opacities = torch::rand({n, 1}, f) * 0.9 + 0.05;
  s.colors = torch::rand({n, 3}, f);
  for (auto *t : {&s.positions, &s.rotations, &s.scales, &s.opacities, &s.colors})
    *t = t->to(dev).requires_grad_(true);
  return s;
}

Camera make_camera(int64_t w, int64_t h, double cx_shift = 0.0) {
  auto cam = Camera::look_at({0.3, -0.2, -3.0}, {0, 0, 0}, {0, -1, 0}, w, h, 1.0);
  cam.cx += cx_shift; // off-centre principal point: asymmetric frustum clamp
  return cam;
}

struct Result {
  at::Tensor image, depth, alpha;
  std::vector<at::Tensor> grads; // positions, rotations, scales, opacities, colors, means2d
  at::Tensor visible, radii;
};

Result run(const Scene &base, const Camera &cam, bool fused) {
  gd::cuda_fused_projection() = fused;
  Scene s;
  const std::initializer_list<std::pair<at::Tensor *, const at::Tensor *>> fields{
      {&s.positions, &base.positions}, {&s.rotations, &base.rotations},
      {&s.scales, &base.scales}, {&s.opacities, &base.opacities}, {&s.colors, &base.colors}};
  for (auto [dst, src] : fields)
    *dst = src->detach().clone().requires_grad_(true);
  auto out = render_gaussians(s.positions, s.rotations, s.scales, s.opacities, s.colors, cam);
  torch::manual_seed(7);
  const auto p_img = torch::randn_like(out.image), p_d = torch::randn_like(out.depth),
             p_a = torch::randn_like(out.alpha);
  ((out.image * p_img).sum() + (out.depth * p_d).sum() + (out.alpha * p_a).sum()).backward();
  gd::cuda_fused_projection() = true;
  Result r{out.image.detach(), out.depth.detach(), out.alpha.detach(), {}, out.visible,
           out.radii};
  for (const auto *t : {&s.positions, &s.rotations, &s.scales, &s.opacities, &s.colors})
    r.grads.push_back(t->grad());
  r.grads.push_back(out.means2d.grad());
  return r;
}

// max |a - b| <= rtol * max |b| + atol
void check_close(const at::Tensor &a, const at::Tensor &b, double rtol, double atol,
                 const char *what) {
  CHECK(a.defined() && b.defined());
  const double diff = (a.cpu().to(torch::kFloat64) - b.cpu().to(torch::kFloat64)).abs().max().item<double>();
  const double scale = b.cpu().to(torch::kFloat64).abs().max().item<double>();
  std::cout << "         " << what << ": max abs diff " << diff << " (relative "
            << (scale > 0 ? diff / scale : 0.0) << ")\n";
  if (!(diff <= rtol * scale + atol))
    FAIL_AT(what << ": max abs diff " << diff << " vs scale " << scale);
}

// The LibTorch projection of render_gaussians, in float64, returning the 10 kernel columns.
at::Tensor reference_projection(const at::Tensor &p, const at::Tensor &q, const at::Tensor &s,
                                const at::Tensor &w2c, const gd::ProjectParams &k) {
  const auto R = w2c.slice(1, 0, 3), t = w2c.select(1, 3);
  const auto cam = at::matmul(p, R.transpose(0, 1)) + t;
  const auto z = cam.select(1, 2);
  const auto sz = z.clamp_min(static_cast<double>(k.near_plane));
  const auto tx = cam.select(1, 0) / sz, ty = cam.select(1, 1) / sz;
  const auto x2 = k.fx * tx + k.cx, y2 = k.fy * ty + k.cy;
  const auto txc = tx.clamp(k.tx_lo, k.tx_hi), tyc = ty.clamp(k.ty_lo, k.ty_hi);
  const auto zeros = at::zeros_like(sz);
  const auto J = at::stack({at::stack({k.fx / sz, zeros, -k.fx * txc / sz}, 1),
                            at::stack({zeros, k.fy / sz, -k.fy * tyc / sz}, 1)},
                           1);
  const auto n = q / q.norm(2, {1}, true).clamp_min(1e-12);
  const auto w = n.select(1, 0), x = n.select(1, 1), y = n.select(1, 2), zz = n.select(1, 3);
  const auto Rq = at::stack(
      {at::stack({1 - 2 * (y * y + zz * zz), 2 * (x * y - w * zz), 2 * (x * zz + w * y)}, 1),
       at::stack({2 * (x * y + w * zz), 1 - 2 * (x * x + zz * zz), 2 * (y * zz - w * x)}, 1),
       at::stack({2 * (x * zz - w * y), 2 * (y * zz + w * x), 1 - 2 * (x * x + y * y)}, 1)},
      1);
  const auto P = at::matmul(at::matmul(J, R), Rq * s.unsqueeze(1));
  const auto m1 = P.select(1, 0), m2 = P.select(1, 1);
  const auto cov_a = (m1 * m1).sum(1), b = (m1 * m2).sum(1), cov_c = (m2 * m2).sum(1);
  const auto a = cov_a + 0.3, c = cov_c + 0.3;
  const auto det = at::linalg_cross(m1, m2, 1).square().sum(1) + 0.3 * (cov_a + cov_c) + 0.09;
  const auto inv = 1.0 / det.clamp_min(1e-12);
  return at::stack({x2, y2, c * inv, -b * inv, a * inv, z, a, b, c, det}, 1);
}

} // namespace

TEST(gaussian_fused_projection_matches_libtorch_render) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  for (const double shift : {0.0, 9.0}) {
    const auto scene = make_scene(3000, 11, dev);
    const auto cam = make_camera(96, 64, shift);
    const auto fused = run(scene, cam, true);
    const auto aten = run(scene, cam, false);
    // Culling and radii come from float32 values that differ in the last bits between the two
    // paths, so a splat right at a boundary may flip; allow a handful.
    const int64_t vis_flips = (fused.visible.cpu() != aten.visible.cpu()).sum().item<int64_t>();
    const int64_t radius_flips = (fused.radii.cpu() != aten.radii.cpu()).sum().item<int64_t>();
    std::cout << "         culling flips: visible " << vis_flips << ", radii " << radius_flips
              << " of " << fused.visible.numel() << "\n";
    CHECK(vis_flips <= 3);
    CHECK(radius_flips <= 3);
    check_close(fused.image, aten.image, 1e-4, 1e-5, "image");
    check_close(fused.depth, aten.depth, 1e-4, 1e-5, "depth");
    check_close(fused.alpha, aten.alpha, 1e-4, 1e-5, "alpha");
    const char *names[] = {"grad positions", "grad rotations", "grad scales", "grad opacities",
                           "grad colors", "means2d.grad"};
    for (size_t i = 0; i < fused.grads.size(); ++i)
      check_close(fused.grads[i], aten.grads[i], 2e-3, 1e-6, names[i]);
  }
}

TEST(gaussian_fused_projection_matches_float64_reference) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  const auto scene = make_scene(4000, 23, dev);
  const auto cam = make_camera(128, 96, -11.0);
  const double w = 128, h = 96;
  const gd::ProjectParams k{static_cast<float>(cam.fx), static_cast<float>(cam.fy),
                                static_cast<float>(cam.cx), static_cast<float>(cam.cy),
                                static_cast<float>(cam.near_plane),
                                static_cast<float>(-(cam.cx + 0.15 * w) / cam.fx),
                                static_cast<float>((w - cam.cx + 0.15 * w) / cam.fx),
                                static_cast<float>(-(cam.cy + 0.15 * h) / cam.fy),
                                static_cast<float>((h - cam.cy + 0.15 * h) / cam.fy)};
  const auto p = scene.positions.detach().contiguous(), q = scene.rotations.detach().contiguous(),
             s = scene.scales.detach().contiguous();
  const auto w2c = cam.world_to_camera.to(dev).slice(0, 0, 3).contiguous();
  auto out = at::empty({p.size(0), gd::kProjectColumns}, p.options());
  gd::project_forward_cuda(p, q, s, w2c, k, out);

  // float64 reference with autograd; parameters cast to float first, as the kernel sees them.
  auto p64 = p.to(at::kDouble).requires_grad_(true), q64 = q.to(at::kDouble).requires_grad_(true),
       s64 = s.to(at::kDouble).requires_grad_(true);
  const auto ref = reference_projection(p64, q64, s64, w2c.to(at::kDouble), k);
  check_close(out.slice(1, 0, 6), ref.detach().slice(1, 0, 6), 1e-4, 1e-6, "columns 0-5");
  check_close(out.slice(1, 6, 10), ref.detach().slice(1, 6, 10), 1e-4, 1e-6, "columns 6-9");

  torch::manual_seed(3);
  auto g = torch::randn({p.size(0), gd::kProjectColumns}, p.options());
  g.slice(1, 6, 10).zero_(); // only the first 6 columns are differentiable outputs
  // Scale the conic gradients like the rasterizer's (conics are ~1e-2..1e2).
  (ref * g.to(at::kDouble)).sum().backward();
  auto gp = at::empty_like(p), gq = at::empty_like(q), gs = at::empty_like(s);
  gd::project_backward_cuda(p, q, s, w2c, k, g, gp, gq, gs);
  // Exclude the near-degenerate quaternions from the rotation check: their gradient scales
  // with 1/|q| ~ 1e7 and float32 cannot match it closely; they are checked to be finite.
  const int64_t n = p.size(0);
  auto keep = torch::ones({n}, torch::kBool);
  keep.slice(0, n / 10, n / 10 + 3).fill_(false);
  const auto kd = keep.to(dev);
  check_close(gp, p64.grad(), 2e-3, 1e-6, "grad positions");
  check_close(gq.index({kd}), q64.grad().index({kd}), 2e-3, 1e-6, "grad rotations");
  check_close(gs, s64.grad(), 2e-3, 1e-6, "grad scales");
  CHECK(torch::isfinite(gq).all().item<bool>());
}

TEST(gaussian_fused_projection_falls_back_when_the_camera_needs_grad) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  const auto scene = make_scene(500, 5, dev);
  auto cam = make_camera(48, 48);
  cam.world_to_camera = cam.world_to_camera.to(dev).requires_grad_(true);
  auto out = render_gaussians(scene.positions, scene.rotations, scene.scales, scene.opacities,
                              scene.colors, cam);
  out.image.sum().backward();
  CHECK(cam.world_to_camera.grad().defined()); // the LibTorch path carries the pose gradient
  CHECK(scene.positions.grad().defined());
}

TEST(gaussian_fused_projection_empty_scene) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  auto pos = torch::zeros({0, 3}, f).requires_grad_(true);
  auto out = render_gaussians(pos, torch::zeros({0, 4}, f), torch::zeros({0, 3}, f),
                              torch::zeros({0, 1}, f), torch::zeros({0, 3}, f),
                              make_camera(16, 16));
  CHECK_EQ(out.image.size(0), int64_t(16));
  CHECK_NEAR(out.alpha.abs().sum().item<double>(), 0.0, 0.0);
}

#endif // USE_CUDA
