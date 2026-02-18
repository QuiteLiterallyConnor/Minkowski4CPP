#ifndef MINK_CPP_MODULES_PRUNING_HPP
#define MINK_CPP_MODULES_PRUNING_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "../coordinate_manager.hpp"
#include "../autograd/pruning_fn.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>
namespace minkowski {
class MinkowskiPruningImpl
    : public torch::nn::Cloneable<MinkowskiPruningImpl> {
public:
  MinkowskiPruningImpl() = default;
  void reset() override {}
  SparseTensor forward(const SparseTensor &input, const at::Tensor &mask);
};
TORCH_MODULE(MinkowskiPruning);
} 
#endif 
