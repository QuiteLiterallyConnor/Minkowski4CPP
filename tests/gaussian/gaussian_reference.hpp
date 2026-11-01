// Brute-force float64 reference for the Gaussian rasterizer (tests only).
//
// Same projection (EWA, 0.3 px low-pass, frustum-clamped Jacobian) and alpha
// rules (alpha = min(0.99, o * exp(power)), dropped below 1/255) as
// render_gaussians, but every Gaussian in front of the near plane is
// composited at every pixel in depth order (ties in index order), without
// tiles, culling boxes or early stopping. The frustum clamp uses the
// principal-point-aware limits (identical to the symmetric 1.3 x half-FOV
// clamp when cx = W/2 and cy = H/2).
#ifndef MINK_TESTS_GAUSSIAN_REFERENCE_HPP
#define MINK_TESTS_GAUSSIAN_REFERENCE_HPP

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_render.hpp"

#include <ATen/ATen.h>

#include <tuple>

namespace gaussian_reference {

using minkowski::gaussian::Camera;
using minkowski::gaussian::RenderOptions;
using minkowski::gaussian::RenderOutput;

struct Splats {
  at::Tensor positions, rotations, scales, opacities, colors, background;
  Splats to(torch::Device device) const {
    return {positions.to(device), rotations.to(device), scales.to(device),
            opacities.to(device), colors.to(device),    background.to(device)};
  }
};

inline RenderOutput render_splats(const Splats &s, const Camera &camera,
                                  bool cuda_kernels, int64_t tile_size = 16) {
  RenderOptions options;
  options.background = s.background;
  options.use_cuda_kernels = cuda_kernels;
  options.tile_size = tile_size;
  return minkowski::gaussian::render_gaussians(
      s.positions, s.rotations, s.scales, s.opacities, s.colors, camera,
      options);
}

struct Reference {
  at::Tensor image, depth, alpha; // float64 [H, W, 3], [H, W], [H, W]
  // [H, W] bool: some alpha lies within float rounding of the 1/255 cut-off,
  // where float32 renderers may legitimately keep or drop the splat.
  at::Tensor ambiguous;
};

inline Reference brute_force(const Splats &splats, const Camera &camera) {
  const auto d = at::TensorOptions().dtype(at::kDouble);
  auto f64 = [](const at::Tensor &t) {
    return t.detach().to(at::kCPU).to(at::kDouble);
  };
  const auto w2c = f64(camera.world_to_camera);
  const auto R = w2c.slice(0, 0, 3).slice(1, 0, 3);
  const auto t = w2c.slice(0, 0, 3).select(1, 3);
  const auto cam = at::matmul(f64(splats.positions), R.transpose(0, 1)) + t;
  const auto keep =
      (cam.select(1, 2) > camera.near_plane).nonzero().squeeze(1);
  const auto p = cam.index_select(0, keep);
  const auto z = p.select(1, 2);
  const auto tx = p.select(1, 0) / z, ty = p.select(1, 1) / z;
  const auto mx = camera.fx * tx + camera.cx;
  const auto my = camera.fy * ty + camera.cy;
  const double W = static_cast<double>(camera.width);
  const double H = static_cast<double>(camera.height);
  const auto txc = tx.clamp(-(camera.cx + 0.15 * W) / camera.fx,
                            (W - camera.cx + 0.15 * W) / camera.fx);
  const auto tyc = ty.clamp(-(camera.cy + 0.15 * H) / camera.fy,
                            (H - camera.cy + 0.15 * H) / camera.fy);
  const auto zero = at::zeros_like(z);
  const auto J = at::stack(
      {at::stack({camera.fx / z, zero, -camera.fx * txc / z}, 1),
       at::stack({zero, camera.fy / z, -camera.fy * tyc / z}, 1)},
      1);
  auto q = f64(splats.rotations).index_select(0, keep);
  q = q / q.norm(2, {1}, true);
  const auto w = q.select(1, 0), x = q.select(1, 1), y = q.select(1, 2),
             k = q.select(1, 3);
  const auto Rq = at::stack(
      {at::stack({1 - 2 * (y * y + k * k), 2 * (x * y - w * k),
                  2 * (x * k + w * y)},
                 1),
       at::stack({2 * (x * y + w * k), 1 - 2 * (x * x + k * k),
                  2 * (y * k - w * x)},
                 1),
       at::stack({2 * (x * k - w * y), 2 * (y * k + w * x),
                  1 - 2 * (x * x + y * y)},
                 1)},
      1);
  const auto s = f64(splats.scales).index_select(0, keep);
  const auto M = at::matmul(at::matmul(J, R), Rq * s.unsqueeze(1));
  const auto cov = at::matmul(M, M.transpose(1, 2));
  const auto a = cov.select(1, 0).select(1, 0) + 0.3;
  const auto b = cov.select(1, 0).select(1, 1);
  const auto c = cov.select(1, 1).select(1, 1) + 0.3;
  const auto det = a * c - b * b;
  const auto ca = c / det, cb = -b / det, cc = a / det;
  const auto o = f64(splats.opacities).reshape({-1}).index_select(0, keep);
  const auto rgb = f64(splats.colors).index_select(0, keep);
  const auto order = std::get<1>(z.sort(/*stable=*/true, 0, false));

  const int64_t h = camera.height, wd = camera.width;
  const auto grid = at::meshgrid(
      {at::arange(h, d) + 0.5, at::arange(wd, d) + 0.5}, "ij");
  const auto py = grid[0].reshape({1, -1}), px = grid[1].reshape({1, -1});
  auto col = [&](const at::Tensor &v) {
    return v.index_select(0, order).unsqueeze(1);
  };
  const auto dx = px - col(mx), dy = py - col(my);
  const auto power =
      -0.5 * (col(ca) * dx * dx + col(cc) * dy * dy) - col(cb) * dx * dy;
  auto alpha = (col(o) * at::exp(power)).clamp_max(0.99);
  const double cut = 1.0 / 255.0;
  Reference ref;
  ref.ambiguous = ((alpha - cut).abs() < 1e-4 * cut).any(0).reshape({h, wd});
  alpha = at::where(alpha >= cut, alpha, at::zeros_like(alpha));
  const auto transmit = at::cumprod(1.0 - alpha, 0);
  const auto before =
      at::cat({at::ones({1, h * wd}, d), transmit.slice(0, 0, -1)}, 0);
  const auto weight = (before * alpha).transpose(0, 1); // [P, N]
  const auto final_t =
      transmit.size(0) > 0 ? transmit.select(0, -1) : at::ones({h * wd}, d);
  ref.image = (at::matmul(weight, rgb.index_select(0, order)) +
               final_t.unsqueeze(1) * f64(splats.background).reshape({1, 3}))
                  .reshape({h, wd, 3});
  ref.depth = at::matmul(weight, z.index_select(0, order)).reshape({h, wd});
  ref.alpha = (1.0 - final_t).reshape({h, wd});
  return ref;
}

struct ReferenceError {
  double image = 0.0, alpha = 0.0, depth = 0.0, ambiguous_fraction = 0.0;
};

// Max |render - reference| over the non-ambiguous pixels.
inline ReferenceError reference_error(const RenderOutput &out,
                                      const Reference &ref) {
  const auto valid = (~ref.ambiguous).to(at::kDouble);
  auto err = [&](const at::Tensor &a, const at::Tensor &b) {
    return ((a.detach().to(at::kCPU).to(at::kDouble) - b).abs() * valid)
        .max()
        .item<double>();
  };
  ReferenceError e;
  e.ambiguous_fraction = ref.ambiguous.to(at::kDouble).mean().item<double>();
  const auto image_err =
      (out.image.detach().to(at::kCPU).to(at::kDouble) - ref.image)
          .abs()
          .amax(2);
  e.image = (image_err * valid).max().item<double>();
  e.alpha = err(out.alpha, ref.alpha);
  e.depth = err(out.depth, ref.depth);
  return e;
}

} // namespace gaussian_reference

#endif
