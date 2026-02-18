#include "nonlinearity.hpp"
namespace minkowski {
MinkowskiSinusoidalImpl::MinkowskiSinusoidalImpl(int in_channels,
                                                  int out_channels)
    : in_channels_(in_channels), out_channels_(out_channels) {
  kernel_ = register_parameter("kernel",
                                torch::rand({in_channels, out_channels}));
  bias_ = register_parameter("bias", torch::rand({1, out_channels}));
  coef_ = register_parameter("coef", torch::rand({1, out_channels}));
}
void MinkowskiSinusoidalImpl::reset() {
  torch::NoGradGuard no_grad;
  kernel_.uniform_(0, 1);
  bias_.uniform_(0, 1);
  coef_.uniform_(0, 1);
}
SparseTensor
MinkowskiSinusoidalImpl::forward(const SparseTensor &input) {
  auto out_F =
      torch::sin(input.F().mm(kernel_) + bias_) * coef_;
  return SparseTensor(out_F, input.coordinate_map_key(),
                      input.coordinate_manager());
}
} 
