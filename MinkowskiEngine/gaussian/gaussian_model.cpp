#include "gaussian/gaussian_model.hpp"

#include "gaussian/gaussian_sh.hpp"

#include <algorithm>
#include <cmath>

namespace minkowski {
namespace gaussian {
namespace {

constexpr double kProbabilityEpsilon = 1e-6;

at::Tensor logit_of(const at::Tensor &p) {
  const auto clamped = p.clamp(kProbabilityEpsilon, 1.0 - kProbabilityEpsilon);
  return at::log(clamped / (1.0 - clamped));
}

} // namespace

GaussianModel::GaussianModel(at::Tensor xyz, at::Tensor f_dc, at::Tensor f_rest,
                             at::Tensor opacity_raw, at::Tensor scale_raw,
                             at::Tensor rotation_raw)
    : xyz_(std::move(xyz)), f_dc_(std::move(f_dc)), f_rest_(std::move(f_rest)),
      opacity_raw_(std::move(opacity_raw)), scale_raw_(std::move(scale_raw)),
      rotation_raw_(std::move(rotation_raw)) {
  validate();
}

void GaussianModel::validate() const {
  TORCH_CHECK(xyz_.defined() && xyz_.dim() == 2 && xyz_.size(1) == 3,
              "xyz must have shape [N, 3]");
  const auto n = xyz_.size(0);
  TORCH_CHECK(f_dc_.dim() == 2 && f_dc_.size(0) == n && f_dc_.size(1) == 3,
              "f_dc must have shape [N, 3]");
  TORCH_CHECK(f_rest_.dim() == 3 && f_rest_.size(0) == n && f_rest_.size(2) == 3,
              "f_rest must have shape [N, K-1, 3]");
  const auto rest = f_rest_.size(1);
  TORCH_CHECK(rest == 0 || rest == 3 || rest == 8 || rest == 15,
              "f_rest must hold SH degree 0, 1, 2 or 3");
  TORCH_CHECK(opacity_raw_.dim() == 2 && opacity_raw_.size(0) == n &&
                  opacity_raw_.size(1) == 1,
              "opacity_raw must have shape [N, 1]");
  TORCH_CHECK(scale_raw_.dim() == 2 && scale_raw_.size(0) == n &&
                  scale_raw_.size(1) == 3,
              "scale_raw must have shape [N, 3]");
  TORCH_CHECK(rotation_raw_.dim() == 2 && rotation_raw_.size(0) == n &&
                  rotation_raw_.size(1) == 4,
              "rotation_raw must have shape [N, 4]");
  for (const auto *t : {&xyz_, &f_dc_, &f_rest_, &opacity_raw_, &scale_raw_,
                        &rotation_raw_}) {
    TORCH_CHECK(t->scalar_type() == at::kFloat,
                "Gaussian model parameters must be float32");
    TORCH_CHECK(t->device() == xyz_.device(),
                "Gaussian model parameters must share a device");
  }
}

int GaussianModel::sh_degree() const {
  switch (f_rest_.size(1)) {
  case 0: return 0;
  case 3: return 1;
  case 8: return 2;
  default: return 3;
  }
}

at::Tensor GaussianModel::rotations() const {
  return rotation_raw_ / rotation_raw_.norm(2, {1}, true).clamp_min(1e-12);
}

at::Tensor GaussianModel::scales() const { return at::exp(scale_raw_); }

at::Tensor GaussianModel::opacities() const {
  return at::sigmoid(opacity_raw_);
}

at::Tensor GaussianModel::colors() const { return 0.5 + kShC0 * f_dc_; }

at::Tensor GaussianModel::sh_coefficients() const {
  return at::cat({f_dc_.unsqueeze(1), f_rest_}, 1);
}

GaussianTensor GaussianModel::to_gaussian_tensor() const {
  return GaussianTensor(xyz_, rotations(), scales(), opacities(), colors());
}

GaussianModel GaussianModel::from_tensor(const GaussianTensor &gaussians) {
  const auto options = at::TensorOptions().dtype(at::kFloat).device(
      gaussians.device());
  auto f = [&](const at::Tensor &t) { return t.detach().to(options).clone(); };
  const auto n = gaussians.size();
  return GaussianModel(
      f(gaussians.positions()),
      f((gaussians.colors() - 0.5) / kShC0),
      at::zeros({n, 0, 3}, options),
      logit_of(f(gaussians.opacities())),
      at::log(f(gaussians.scales()).clamp_min(1e-7)),
      f(gaussians.rotations()));
}

GaussianModel GaussianModel::from_points(const at::Tensor &positions,
                                         const at::Tensor &colors,
                                         int sh_degree,
                                         double initial_opacity) {
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
              "positions must have shape [N, 3]");
  TORCH_CHECK(colors.sizes() == positions.sizes(),
              "colors must have shape [N, 3]");
  TORCH_CHECK(initial_opacity > 0.0 && initial_opacity < 1.0,
              "initial_opacity must be in (0, 1)");
  const auto options = at::TensorOptions().dtype(at::kFloat).device(
      positions.device());
  const auto xyz = positions.detach().to(options).clone();
  const auto n = xyz.size(0);
  const auto rest = sh_coefficient_count(sh_degree) - 1;

  // Mean squared distance to the 3 nearest neighbors, computed in chunks.
  at::Tensor mean_dist2 = at::ones({n}, options);
  if (n > 1) {
    const int64_t k = std::min<int64_t>(4, n);
    constexpr int64_t chunk = 512;
    for (int64_t begin = 0; begin < n; begin += chunk) {
      const auto rows = xyz.slice(0, begin, std::min(begin + chunk, n));
      auto d2 = at::cdist(rows, xyz).square();
      auto nearest = std::get<0>(d2.topk(k, 1, false, true));
      mean_dist2.slice(0, begin, begin + rows.size(0))
          .copy_(nearest.slice(1, 1, k).mean(1));
    }
  }
  mean_dist2 = mean_dist2.clamp_min(1e-7);
  const auto scale_raw =
      (0.5 * at::log(mean_dist2)).unsqueeze(1).repeat({1, 3}).contiguous();

  auto rotation = at::zeros({n, 4}, options);
  rotation.select(1, 0).fill_(1.0);
  return GaussianModel(
      xyz, ((colors.detach().to(options) - 0.5) / kShC0).contiguous(),
      at::zeros({n, rest, 3}, options),
      at::full({n, 1}, std::log(initial_opacity / (1.0 - initial_opacity)),
               options),
      scale_raw, rotation);
}

std::vector<at::Tensor> GaussianModel::parameters() const {
  return {xyz_, f_dc_, f_rest_, opacity_raw_, scale_raw_, rotation_raw_};
}

void GaussianModel::set_parameters(const std::vector<at::Tensor> &parameters) {
  TORCH_CHECK(parameters.size() == 6, "expected 6 Gaussian parameter tensors");
  GaussianModel replacement(parameters[0], parameters[1], parameters[2],
                            parameters[3], parameters[4], parameters[5]);
  *this = std::move(replacement);
}

std::vector<at::Tensor> GaussianModel::enable_gradients() {
  for (auto *t : {&xyz_, &f_dc_, &f_rest_, &opacity_raw_, &scale_raw_,
                  &rotation_raw_})
    *t = t->detach().set_requires_grad(true);
  return parameters();
}

GaussianModel GaussianModel::to(torch::Device device) const {
  auto move = [&](const at::Tensor &t) { return t.detach().to(device); };
  return GaussianModel(move(xyz_), move(f_dc_), move(f_rest_),
                       move(opacity_raw_), move(scale_raw_),
                       move(rotation_raw_));
}

GaussianModel GaussianModel::clone() const {
  auto copy = [](const at::Tensor &t) { return t.detach().clone(); };
  return GaussianModel(copy(xyz_), copy(f_dc_), copy(f_rest_),
                       copy(opacity_raw_), copy(scale_raw_),
                       copy(rotation_raw_));
}

GaussianModel GaussianModel::select(const at::Tensor &indices) const {
  TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == at::kLong,
              "indices must be a one-dimensional int64 tensor");
  auto pick = [&](const at::Tensor &t) {
    return t.detach().index_select(0, indices);
  };
  return GaussianModel(pick(xyz_), pick(f_dc_), pick(f_rest_),
                       pick(opacity_raw_), pick(scale_raw_),
                       pick(rotation_raw_));
}

} // namespace gaussian
} // namespace minkowski
