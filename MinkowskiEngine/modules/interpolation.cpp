#include "interpolation.hpp"
namespace minkowski {
MinkowskiInterpolationImpl::MinkowskiInterpolationImpl(
    bool return_kernel_map, bool return_weights)
    : return_kernel_map_(return_kernel_map),
      return_weights_(return_weights) {}
at::Tensor
MinkowskiInterpolationImpl::forward(const SparseTensor &input,
                                     const at::Tensor &tfield) {
  auto result = forward_full(input, tfield);
  return result.output;
}
InterpolationResult
MinkowskiInterpolationImpl::forward_full(const SparseTensor &input,
                                          const at::Tensor &tfield) {
  auto manager = input.coordinate_manager();
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto results = InterpolationFunction::apply(input.F(), tfield, in_key,
                                               manager);
  InterpolationResult result;
  result.output = results[0];
  result.in_map = results[1];
  result.out_map = results[2];
  result.weights = results[3];
  return result;
}
} 
