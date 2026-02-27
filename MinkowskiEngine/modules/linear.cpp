#include "linear.hpp"
namespace minkowski {
MinkowskiLinearImpl::MinkowskiLinearImpl(int in_features, int out_features,
                                         bool bias) {
  linear_ = register_module(
      "linear",
      torch::nn::Linear(
          torch::nn::LinearOptions(in_features, out_features).bias(bias)));
}
void MinkowskiLinearImpl::reset() {
  linear_->reset_parameters();
}
SparseTensor
MinkowskiLinearImpl::forward(const SparseTensor &input) {
  auto output = linear_->forward(input.F());
  return SparseTensor(output, input.coordinate_map_key(),
                      input.coordinate_manager());
}
} 
