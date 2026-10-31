#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_MODEL_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_MODEL_HPP

#include "gaussian/gaussian_tensor.hpp"

#include <torch/torch.h>

#include <vector>

namespace minkowski {
namespace gaussian {

// Trainable Gaussian-splat parameters in the "raw" space used by 3DGS:
//   xyz [N,3], f_dc [N,3], f_rest [N,K-1,3] (K = (sh_degree+1)^2),
//   opacity_raw [N,1] (logit), scale_raw [N,3] (log), rotation_raw [N,4]
//   (unnormalized quaternion, w x y z).
// Activations: opacity = sigmoid, scale = exp, rotation = normalize,
// color = 0.5 + C0 * f_dc (view-independent) or full SH evaluation in render().
class GaussianModel {
public:
  GaussianModel(at::Tensor xyz, at::Tensor f_dc, at::Tensor f_rest,
                at::Tensor opacity_raw, at::Tensor scale_raw,
                at::Tensor rotation_raw);

  // Inverts the activations of an activated tensor (SH degree 0).
  static GaussianModel from_tensor(const GaussianTensor &gaussians);

  // Initializes a model from a colored point cloud: scales from the mean
  // distance to the nearest neighbors, identity rotations, constant opacity.
  static GaussianModel from_points(const at::Tensor &positions,
                                   const at::Tensor &colors, int sh_degree = 3,
                                   double initial_opacity = 0.1);

  int64_t size() const { return xyz_.size(0); }
  int sh_degree() const;
  torch::Device device() const { return xyz_.device(); }

  const at::Tensor &xyz() const { return xyz_; }
  const at::Tensor &f_dc() const { return f_dc_; }
  const at::Tensor &f_rest() const { return f_rest_; }
  const at::Tensor &opacity_raw() const { return opacity_raw_; }
  const at::Tensor &scale_raw() const { return scale_raw_; }
  const at::Tensor &rotation_raw() const { return rotation_raw_; }

  // Activated views (differentiable).
  at::Tensor positions() const { return xyz_; }
  at::Tensor rotations() const;
  at::Tensor scales() const;
  at::Tensor opacities() const;
  at::Tensor colors() const;
  // [N, K, 3] with the DC term first.
  at::Tensor sh_coefficients() const;

  GaussianTensor to_gaussian_tensor() const;

  // Fixed order: xyz, f_dc, f_rest, opacity_raw, scale_raw, rotation_raw.
  std::vector<at::Tensor> parameters() const;
  void set_parameters(const std::vector<at::Tensor> &parameters);
  // Makes every parameter a leaf that requires gradients.
  std::vector<at::Tensor> enable_gradients();

  GaussianModel to(torch::Device device) const;
  GaussianModel clone() const;
  GaussianModel select(const at::Tensor &indices) const;

private:
  void validate() const;

  at::Tensor xyz_, f_dc_, f_rest_, opacity_raw_, scale_raw_, rotation_raw_;
};

} // namespace gaussian
} // namespace minkowski

#endif
