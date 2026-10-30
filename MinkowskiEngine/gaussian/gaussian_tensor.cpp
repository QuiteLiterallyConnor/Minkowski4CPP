#include "gaussian_tensor.hpp"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <utility>

#ifdef USE_CUDA
#include "gaussian_cuda.hpp"
#endif

namespace minkowski {
namespace gaussian {
namespace {

void check_matrix(const at::Tensor &tensor, int64_t rows, int64_t columns,
                  const char *name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(1) == columns, name,
              " must have shape [N, ", columns, "]");
  TORCH_CHECK(tensor.size(0) == rows, name, " row count must match positions");
}

at::Tensor normalize_quaternions(const at::Tensor &quaternions) {
  const auto norms = quaternions.square().sum(1, true).sqrt();
  const auto valid =
      at::isfinite(quaternions).all() & (norms > 0).all() & at::isfinite(norms).all();
  TORCH_CHECK(valid.item<bool>(),
              "quaternion rotations must have finite nonzero norm");
  return quaternions / norms;
}

at::Tensor quaternion_rotation_matrices(const at::Tensor &quaternions) {
  const auto q = normalize_quaternions(quaternions);
  const auto w = q.select(1, 0);
  const auto x = q.select(1, 1);
  const auto y = q.select(1, 2);
  const auto z = q.select(1, 3);
  const auto two = at::full_like(w, 2);
  return at::stack(
             {1 - two * (y * y + z * z), two * (x * y - z * w),
              two * (x * z + y * w), two * (x * y + z * w),
              1 - two * (x * x + z * z), two * (y * z - x * w),
              two * (x * z - y * w), two * (y * z + x * w),
              1 - two * (x * x + y * y)},
             1)
      .reshape({quaternions.size(0), 3, 3});
}

at::Tensor quaternion_product(const at::Tensor &a, const at::Tensor &b) {
  const auto aw = a.select(1, 0), ax = a.select(1, 1);
  const auto ay = a.select(1, 2), az = a.select(1, 3);
  const auto bw = b.select(1, 0), bx = b.select(1, 1);
  const auto by = b.select(1, 2), bz = b.select(1, 3);
  return at::stack({aw * bw - ax * bx - ay * by - az * bz,
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw},
                   1);
}

at::Tensor transform_vector(const at::Tensor &vector, const at::Tensor &rotation,
                            const at::Tensor &translation) {
  auto matrices = quaternion_rotation_matrices(rotation);
  return at::bmm(matrices, vector.unsqueeze(2)).squeeze(2) + translation;
}

} // namespace

GaussianTensor::GaussianTensor(at::Tensor positions, at::Tensor rotations,
                               at::Tensor scales, at::Tensor opacities,
                               at::Tensor colors, at::Tensor features,
                               at::Tensor velocities)
    : positions_(std::move(positions)), rotations_(std::move(rotations)),
      scales_(std::move(scales)), opacities_(std::move(opacities)),
      colors_(std::move(colors)), features_(std::move(features)),
      velocities_(std::move(velocities)) {
  TORCH_CHECK(positions_.defined() && positions_.dim() == 2 &&
                  positions_.size(1) == 3,
              "positions must have shape [N, 3]");
  if (!features_.defined())
    features_ = positions_.new_empty({positions_.size(0), 0});
  validate();
}

GaussianTensor GaussianTensor::empty(int64_t count,
                                     const at::TensorOptions &options,
                                     int64_t feature_channels,
                                     bool with_velocities) {
  TORCH_CHECK(count >= 0, "Gaussian count must be nonnegative");
  TORCH_CHECK(feature_channels >= 0,
              "feature channel count must be nonnegative");
  auto positions = at::empty({count, 3}, options);
  auto rotations = at::empty({count, 4}, options);
  auto scales = at::empty({count, 3}, options);
  auto opacities = at::empty({count, 1}, options);
  auto colors = at::empty({count, 3}, options);
  auto features = at::empty({count, feature_channels}, options);
  auto velocities =
      with_velocities ? at::empty({count, 3}, options) : at::Tensor();
  return GaussianTensor(std::move(positions), std::move(rotations),
                        std::move(scales), std::move(opacities),
                        std::move(colors), std::move(features),
                        std::move(velocities));
}

void GaussianTensor::validate() const {
  TORCH_CHECK(positions_.defined() && positions_.dim() == 2 &&
                  positions_.size(1) == 3,
              "positions must have shape [N, 3]");
  const auto count = positions_.size(0);
  check_matrix(rotations_, count, 4, "rotations");
  check_matrix(scales_, count, 3, "scales");
  check_matrix(opacities_, count, 1, "opacities");
  check_matrix(colors_, count, 3, "colors");
  check_matrix(features_, count, features_.dim() == 2 ? features_.size(1) : -1,
               "features");
  if (velocities_.defined())
    check_matrix(velocities_, count, 3, "velocities");

  TORCH_CHECK(positions_.is_floating_point() &&
                  (positions_.scalar_type() == at::kFloat ||
                   positions_.scalar_type() == at::kDouble),
              "Gaussian tensors support float32 and float64");
  const auto device = positions_.device();
  const auto type = positions_.scalar_type();
  for (const auto *tensor :
       {&rotations_, &scales_, &opacities_, &colors_, &features_}) {
    TORCH_CHECK(tensor->device() == device, "all Gaussian fields must share a device");
    TORCH_CHECK(tensor->scalar_type() == type,
                "all Gaussian fields must share a dtype");
    TORCH_CHECK(tensor->is_floating_point(),
                "Gaussian fields must use a floating-point dtype");
  }
  if (velocities_.defined()) {
    TORCH_CHECK(velocities_.device() == device,
                "all Gaussian fields must share a device");
    TORCH_CHECK(velocities_.scalar_type() == type,
                "all Gaussian fields must share a dtype");
  }
}

at::Tensor GaussianTensor::covariances() const {
#ifdef USE_CUDA
  if (positions_.is_cuda() && !rotations_.requires_grad() &&
      !scales_.requires_grad()) {
    const auto rotations = normalize_quaternions(rotations_.contiguous());
    const auto scales = scales_.contiguous();
    auto output = at::empty({size(), 3, 3}, positions_.options());
    detail::covariance_cuda(rotations, scales, output);
    return output;
  }
#endif
  const auto rotation = quaternion_rotation_matrices(rotations_);
  const auto variances = scales_.square().unsqueeze(1);
  return at::bmm(at::bmm(rotation, at::diag_embed(variances.squeeze(1))),
                 rotation.transpose(1, 2));
}

GaussianTensor GaussianTensor::select(const at::Tensor &indices) const {
  TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == at::kLong,
              "indices must be a one-dimensional int64 tensor");
  TORCH_CHECK(indices.device() == device(),
              "indices and Gaussian tensors must be on the same device");
  return GaussianTensor(positions_.index_select(0, indices),
                        rotations_.index_select(0, indices),
                        scales_.index_select(0, indices),
                        opacities_.index_select(0, indices),
                        colors_.index_select(0, indices),
                        features_.index_select(0, indices),
                        velocities_.defined()
                            ? velocities_.index_select(0, indices)
                            : at::Tensor());
}

GaussianTensor GaussianTensor::remove(const at::Tensor &indices) const {
  TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == at::kLong,
              "indices must be a one-dimensional int64 tensor");
  TORCH_CHECK(indices.device() == device(),
              "indices and Gaussian tensors must be on the same device");
  auto keep = at::ones({size()}, positions_.options().dtype(at::kBool));
  keep.index_fill_(0, indices, false);
  return select(at::nonzero(keep).reshape({-1}));
}

GaussianTensor GaussianTensor::filter_opacity(double minimum_opacity) const {
  TORCH_CHECK(std::isfinite(minimum_opacity) && minimum_opacity >= 0.0 &&
                  minimum_opacity <= 1.0,
              "minimum_opacity must be finite and in [0, 1]");
  auto indices = at::nonzero(opacities_.select(1, 0) >= minimum_opacity)
                     .reshape({-1});
  return select(indices);
}

GaussianTensor GaussianTensor::filter_bounds(const at::Tensor &lower,
                                             const at::Tensor &upper) const {
  TORCH_CHECK(lower.sizes() == at::IntArrayRef({3}) &&
                  upper.sizes() == at::IntArrayRef({3}),
              "lower and upper bounds must have shape [3]");
  TORCH_CHECK(lower.device() == device() && upper.device() == device(),
              "bounds and Gaussian tensors must be on the same device");
  TORCH_CHECK(lower.scalar_type() == dtype() && upper.scalar_type() == dtype(),
              "bounds and Gaussian tensors must have the same dtype");
  TORCH_CHECK(at::all(lower < upper).item<bool>(),
              "each lower bound must be smaller than its upper bound");
  auto inside = ((positions_ >= lower) & (positions_ < upper)).all(1);
  return select(at::nonzero(inside).reshape({-1}));
}

GaussianTensor GaussianTensor::append(const GaussianTensor &other) const {
  return concatenate({*this, other});
}

GaussianTensor
GaussianTensor::concatenate(const std::vector<GaussianTensor> &tensors) {
  TORCH_CHECK(!tensors.empty(), "cannot concatenate an empty Gaussian tensor list");
  const auto &first = tensors.front();
  std::vector<at::Tensor> positions, rotations, scales, opacities, colors,
      features, velocities;
  positions.reserve(tensors.size());
  rotations.reserve(tensors.size());
  scales.reserve(tensors.size());
  opacities.reserve(tensors.size());
  colors.reserve(tensors.size());
  features.reserve(tensors.size());
  const bool with_velocities = first.has_velocities();
  for (const auto &tensor : tensors) {
    TORCH_CHECK(tensor.device() == first.device() &&
                    tensor.dtype() == first.dtype() &&
                    tensor.feature_channels() == first.feature_channels() &&
                    tensor.has_velocities() == with_velocities,
                "concatenated Gaussian tensors must have matching device, dtype, "
                "feature channels, and velocity presence");
    positions.push_back(tensor.positions_);
    rotations.push_back(tensor.rotations_);
    scales.push_back(tensor.scales_);
    opacities.push_back(tensor.opacities_);
    colors.push_back(tensor.colors_);
    features.push_back(tensor.features_);
    if (with_velocities)
      velocities.push_back(tensor.velocities_);
  }
  return GaussianTensor(at::cat(positions, 0), at::cat(rotations, 0),
                        at::cat(scales, 0), at::cat(opacities, 0),
                        at::cat(colors, 0), at::cat(features, 0),
                        with_velocities ? at::cat(velocities, 0) : at::Tensor());
}

GaussianTensor GaussianTensor::clone() const {
  return GaussianTensor(positions_.clone(), rotations_.clone(), scales_.clone(),
                        opacities_.clone(), colors_.clone(), features_.clone(),
                        velocities_.defined() ? velocities_.clone() : at::Tensor());
}

GaussianTensor GaussianTensor::to(torch::Device target_device) const {
  return GaussianTensor(positions_.to(target_device), rotations_.to(target_device),
                        scales_.to(target_device), opacities_.to(target_device),
                        colors_.to(target_device), features_.to(target_device),
                        velocities_.defined() ? velocities_.to(target_device)
                                              : at::Tensor());
}

std::vector<at::Tensor> GaussianTensor::enable_gradients() {
  positions_ = positions_.detach().set_requires_grad(true);
  rotations_ = rotations_.detach().set_requires_grad(true);
  scales_ = scales_.detach().set_requires_grad(true);
  opacities_ = opacities_.detach().set_requires_grad(true);
  colors_ = colors_.detach().set_requires_grad(true);
  std::vector<at::Tensor> parameters{
      positions_, rotations_, scales_, opacities_, colors_};
  if (features_.numel() > 0) {
    features_ = features_.detach().set_requires_grad(true);
    parameters.push_back(features_);
  }
  if (velocities_.defined()) {
    velocities_ = velocities_.detach().set_requires_grad(true);
    parameters.push_back(velocities_);
  }
  return parameters;
}

void GaussianTensor::project_constraints() {
  at::NoGradGuard guard;
  scales_.clamp_(1e-6);
  opacities_.clamp_(0.0, 1.0);
}

GaussianTensor GaussianTensor::transform(const at::Tensor &rotation,
                                         const at::Tensor &translation) const {
  TORCH_CHECK(rotation.dim() == 1 || rotation.dim() == 2,
              "rotation must have shape [4] or [N, 4]");
  TORCH_CHECK(translation.dim() == 1 || translation.dim() == 2,
              "translation must have shape [3] or [N, 3]");
  TORCH_CHECK(rotation.size(rotation.dim() - 1) == 4 &&
                  translation.size(translation.dim() - 1) == 3,
              "rotation and translation must have final dimensions 4 and 3; got ",
              rotation.sizes(), " and ", translation.sizes());
  TORCH_CHECK(rotation.device() == device() && translation.device() == device(),
              "transform and Gaussian tensors must be on the same device");
  TORCH_CHECK(rotation.scalar_type() == dtype() &&
                  translation.scalar_type() == dtype(),
              "transform and Gaussian tensors must have the same dtype");

  auto rotations = rotation.dim() == 1 ? rotation.unsqueeze(0).expand({size(), 4})
                                       : rotation;
  auto translations =
      translation.dim() == 1 ? translation.unsqueeze(0).expand({size(), 3})
                             : translation;
  TORCH_CHECK(rotations.size(0) == size() && translations.size(0) == size(),
              "batched transform row counts must equal Gaussian count");
#ifdef USE_CUDA
  if (positions_.is_cuda() && !positions_.requires_grad() &&
      !rotations_.requires_grad() && !rotation.requires_grad() &&
      !translation.requires_grad() &&
      (!velocities_.defined() || !velocities_.requires_grad())) {
    auto normalized_transforms =
        normalize_quaternions(rotations.contiguous()).contiguous();
    auto normalized_gaussians =
        normalize_quaternions(rotations_.contiguous()).contiguous();
    auto input_positions = positions_.contiguous();
    auto input_translations = translations.contiguous();
    auto output_positions = at::empty_like(input_positions);
    auto output_rotations = at::empty_like(normalized_gaussians);
    auto input_velocities =
        velocities_.defined() ? velocities_.contiguous() : at::Tensor();
    auto output_velocities =
        velocities_.defined() ? at::empty_like(input_velocities) : at::Tensor();
    detail::transform_cuda(
        input_positions, normalized_gaussians, normalized_transforms,
        input_translations, input_velocities, output_positions, output_rotations,
        output_velocities);
    return GaussianTensor(std::move(output_positions), std::move(output_rotations),
                          scales_, opacities_, colors_, features_,
                          std::move(output_velocities));
  }
#endif
  auto rotated_positions = transform_vector(positions_, rotations, translations);
  auto normalized_rotation = normalize_quaternions(rotations);
  auto normalized_gaussians = normalize_quaternions(rotations_);
  auto transformed_rotations =
      quaternion_product(normalized_rotation, normalized_gaussians);
  auto transformed_velocities =
      velocities_.defined()
          ? at::bmm(quaternion_rotation_matrices(rotations),
                    velocities_.unsqueeze(2))
                .squeeze(2)
          : at::Tensor();
  return GaussianTensor(std::move(rotated_positions),
                        std::move(transformed_rotations), scales_, opacities_,
                        colors_, features_, std::move(transformed_velocities));
}

} // namespace gaussian
} // namespace minkowski
