#include "pruning.hpp"
namespace minkowski {
SparseTensor
MinkowskiPruningImpl::forward(const SparseTensor &input,
                              const at::Tensor &mask) {
  auto manager = input.coordinate_manager();
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto out_key = std::make_shared<CoordinateMapKey>(
      input.coordinate_map_key().get_coordinate_size());
  auto output = PruningFunction::apply(input.F(), mask, in_key, out_key,
                                        manager);
  return SparseTensor(output, *out_key, manager);
}
} 
