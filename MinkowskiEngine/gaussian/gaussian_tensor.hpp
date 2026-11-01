#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_TENSOR_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_TENSOR_HPP

#include <torch/torch.h>

#include <vector>

namespace minkowski {
namespace gaussian {

class GaussianTensor {
public:
  // Fields are stored as contiguous [N, C] LibTorch tensors.
  GaussianTensor(at::Tensor positions, at::Tensor rotations, at::Tensor scales,
                 at::Tensor opacities, at::Tensor colors,
                 at::Tensor features = {}, at::Tensor velocities = {});

  static GaussianTensor empty(int64_t count, const at::TensorOptions &options,
                              int64_t feature_channels = 0,
                              bool with_velocities = false);

  int64_t size() const { return positions_.size(0); }
  int64_t feature_channels() const { return features_.size(1); }
  bool has_velocities() const { return velocities_.defined(); }
  torch::Device device() const { return positions_.device(); }
  c10::ScalarType dtype() const { return positions_.scalar_type(); }

  const at::Tensor &positions() const { return positions_; }
  const at::Tensor &rotations() const { return rotations_; }
  const at::Tensor &scales() const { return scales_; }
  const at::Tensor &opacities() const { return opacities_; }
  const at::Tensor &colors() const { return colors_; }
  const at::Tensor &features() const { return features_; }
  const at::Tensor &velocities() const { return velocities_; }

  at::Tensor covariances() const;
  GaussianTensor select(const at::Tensor &indices) const;
  GaussianTensor remove(const at::Tensor &indices) const;
  GaussianTensor filter_opacity(double minimum_opacity) const;
  GaussianTensor filter_bounds(const at::Tensor &lower,
                               const at::Tensor &upper) const;
  GaussianTensor append(const GaussianTensor &other) const;
  static GaussianTensor concatenate(const std::vector<GaussianTensor> &tensors);
  GaussianTensor clone() const;
  GaussianTensor to(torch::Device device) const;
  std::vector<at::Tensor> enable_gradients();
  void project_constraints();

  // Quaternions use (w, x, y, z); transform inputs may be shared [4]/[3] or
  // per-Gaussian [N, 4]/[N, 3]. Autograd-enabled inputs use LibTorch operators;
  // inference inputs on CUDA use dedicated kernels.
  GaussianTensor transform(const at::Tensor &rotation,
                           const at::Tensor &translation) const;

private:
  at::Tensor positions_;
  at::Tensor rotations_;
  at::Tensor scales_;
  at::Tensor opacities_;
  at::Tensor colors_;
  at::Tensor features_;
  at::Tensor velocities_;

  void validate() const;
};

} // namespace gaussian
} // namespace minkowski

#endif
