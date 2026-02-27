#ifndef MINK_CPP_MODULES_INTERPOLATION_HPP
#define MINK_CPP_MODULES_INTERPOLATION_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "../coordinate_manager.hpp"
#include "../autograd/interpolation_fn.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>
#include <tuple>
namespace minkowski {
struct InterpolationResult {
  at::Tensor output;
  at::Tensor in_map;
  at::Tensor out_map;
  at::Tensor weights;
};
class MinkowskiInterpolationImpl
    : public torch::nn::Cloneable<MinkowskiInterpolationImpl> {
public:
  MinkowskiInterpolationImpl(bool return_kernel_map = false,
                              bool return_weights = false);
  void reset() override {}
  at::Tensor forward(const SparseTensor &input, const at::Tensor &tfield);
  InterpolationResult forward_full(const SparseTensor &input,
                                    const at::Tensor &tfield);
  bool return_kernel_map_;
  bool return_weights_;
};
TORCH_MODULE(MinkowskiInterpolation);
} 
#endif 
