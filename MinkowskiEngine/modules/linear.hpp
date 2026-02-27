#ifndef MINK_CPP_MODULES_LINEAR_HPP
#define MINK_CPP_MODULES_LINEAR_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include <torch/torch.h>
namespace minkowski {
class MinkowskiLinearImpl
    : public torch::nn::Cloneable<MinkowskiLinearImpl> {
public:
  MinkowskiLinearImpl(int in_features, int out_features, bool bias = true);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  torch::nn::Linear linear_{nullptr};
};
TORCH_MODULE(MinkowskiLinear);
} 
#endif 
