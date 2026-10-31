#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_OPTIMIZER_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_OPTIMIZER_HPP

#include "gaussian_tensor.hpp"

#include <torch/optim/adam.h>

namespace minkowski {
namespace gaussian {

class GaussianOptimizer {
public:
  explicit GaussianOptimizer(GaussianTensor &gaussians,
                             double learning_rate = 1e-2);

  static at::Tensor rgb_reconstruction_loss(const at::Tensor &prediction,
                                           const at::Tensor &target);
  static at::Tensor depth_reconstruction_loss(const at::Tensor &prediction,
                                             const at::Tensor &target);
  static at::Tensor opacity_regularization(const GaussianTensor &gaussians);
  static at::Tensor scale_regularization(const GaussianTensor &gaussians);
  static at::Tensor objective(const GaussianTensor &gaussians,
                              const at::Tensor &predicted_rgb,
                              const at::Tensor &target_rgb,
                              double opacity_weight = 0.0,
                              double scale_weight = 0.0);

  // Backpropagates a scalar objective through LibTorch autograd and updates
  // the Gaussian fields owned by the tensor passed to the constructor. That
  // tensor must outlive this optimizer.
  void step(const at::Tensor &loss);
  void zero_grad();

private:
  GaussianTensor *gaussians_;
  std::vector<at::Tensor> parameters_;
  torch::optim::Adam optimizer_;
};

} // namespace gaussian
} // namespace minkowski

#endif
