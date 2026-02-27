
#include "broadcast.hpp"

namespace minkowski {

MinkowskiBroadcastBaseImpl::MinkowskiBroadcastBaseImpl(
    BroadcastMode::Type operation_type)
    : operation_type_(operation_type) {}

SparseTensor
MinkowskiBroadcastBaseImpl::forward(const SparseTensor &input,
                                    const SparseTensor &input_glob) {
  auto manager = input.coordinate_manager();
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto glob_key =
      std::make_shared<CoordinateMapKey>(input_glob.coordinate_map_key());

  auto output = BroadcastFunction::apply(input.F(), input_glob.F(),
                                          operation_type_, in_key, glob_key,
                                          manager);

  return SparseTensor(output, input.coordinate_map_key(), manager);
}

MinkowskiBroadcastAdditionImpl::MinkowskiBroadcastAdditionImpl() {
  base_ = std::make_shared<MinkowskiBroadcastBaseImpl>(
      BroadcastMode::ELEMENTWISE_ADDITON);
  register_module("base", base_);
}

SparseTensor MinkowskiBroadcastAdditionImpl::forward(
    const SparseTensor &input, const SparseTensor &input_glob) {
  return base_->forward(input, input_glob);
}

MinkowskiBroadcastMultiplicationImpl::
    MinkowskiBroadcastMultiplicationImpl() {
  base_ = std::make_shared<MinkowskiBroadcastBaseImpl>(
      BroadcastMode::ELEMENTWISE_MULTIPLICATION);
  register_module("base", base_);
}

SparseTensor MinkowskiBroadcastMultiplicationImpl::forward(
    const SparseTensor &input, const SparseTensor &input_glob) {
  return base_->forward(input, input_glob);
}

SparseTensor
MinkowskiBroadcastImpl::forward(const SparseTensor &input,
                                const SparseTensor &input_glob) {
  auto manager = input.coordinate_manager();
  int64_t n_rows = static_cast<int64_t>(manager->size(input.coordinate_map_key()));
  int64_t n_cols = input_glob.F().size(1);

  auto broadcast_feat =
      torch::zeros({n_rows, n_cols}, input.F().options());

  auto origin_maps = manager->union_map(
      {input.coordinate_map_key()},
      const_cast<CoordinateMapKey &>(
          *std::make_shared<CoordinateMapKey>(
              input.coordinate_map_key())));

  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto glob_key =
      std::make_shared<CoordinateMapKey>(input_glob.coordinate_map_key());

  auto zero_input = torch::zeros_like(input.F().narrow(1, 0, n_cols));
  auto output = BroadcastFunction::apply(
      zero_input, input_glob.F(), BroadcastMode::ELEMENTWISE_ADDITON,
      in_key, glob_key, manager);

  return SparseTensor(output, input.coordinate_map_key(), manager);
}

SparseTensor MinkowskiBroadcastConcatenationImpl::forward(
    const SparseTensor &input, const SparseTensor &input_glob) {
  auto manager = input.coordinate_manager();
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto glob_key =
      std::make_shared<CoordinateMapKey>(input_glob.coordinate_map_key());

  int64_t n_rows = static_cast<int64_t>(manager->size(input.coordinate_map_key()));
  int64_t glob_cols = input_glob.F().size(1);
  auto zero_input = torch::zeros({n_rows, glob_cols}, input.F().options());

  auto broadcast_feat = BroadcastFunction::apply(
      zero_input, input_glob.F(), BroadcastMode::ELEMENTWISE_ADDITON,
      in_key, glob_key, manager);

  auto cat_feat = torch::cat({input.F(), broadcast_feat}, 1);
  return SparseTensor(cat_feat, input.coordinate_map_key(), manager);
}

} 
