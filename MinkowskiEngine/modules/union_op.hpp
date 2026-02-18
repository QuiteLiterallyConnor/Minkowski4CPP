#ifndef MINK_CPP_MODULES_UNION_OP_HPP
#define MINK_CPP_MODULES_UNION_OP_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "../coordinate_manager.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>
namespace minkowski {
class MinkowskiUnionImpl
    : public torch::nn::Cloneable<MinkowskiUnionImpl> {
public:
  MinkowskiUnionImpl() = default;
  void reset() override {}
  SparseTensor forward(const std::vector<SparseTensor> &inputs);
};
TORCH_MODULE(MinkowskiUnion);
} 
#endif 
