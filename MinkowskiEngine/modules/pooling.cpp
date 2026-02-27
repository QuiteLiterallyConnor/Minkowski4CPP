#include "pooling.hpp"
namespace minkowski {
MinkowskiPoolingBaseImpl::MinkowskiPoolingBaseImpl(int kernel_size, int stride,
                                                   int dilation,
                                                   PoolingMode::Type pooling_mode,
                                                   int dimension)
    : kernel_generator_(
          {kernel_size},
          {stride},
          {dilation},
          false,
          RegionType::HYPER_CUBE,
          {},
          false,
          {},
          dimension),
      pooling_mode_(pooling_mode), dimension_(dimension) {}
SparseTensor
MinkowskiPoolingBaseImpl::forward(const SparseTensor &input) {
  TORCH_CHECK(input.dimension() == dimension_,
              "Input dimension mismatch: expected ", dimension_, " got ",
              input.dimension());
  auto manager = input.coordinate_manager();
  auto kparams =
      kernel_generator_.get_kernel(input.tensor_stride(), false);
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto out_key = std::make_shared<CoordinateMapKey>(
      input.coordinate_map_key().get_coordinate_size());
  at::Tensor offsets = kparams.region_offsets;
  if (!offsets.defined()) {
    offsets = torch::empty({0}, torch::kInt);
  }
  auto outfeat = LocalPoolingFunction::apply(
      input.F(), pooling_mode_, kparams.kernel_size, kparams.kernel_stride,
      kparams.kernel_dilation, kparams.region_type, offsets, in_key, out_key,
      manager);
  return SparseTensor(outfeat, *out_key, manager);
}
MinkowskiAvgPoolingImpl::MinkowskiAvgPoolingImpl(int kernel_size, int stride,
                                                 int dilation, int dimension) {
  base_ = std::make_shared<MinkowskiPoolingBaseImpl>(
      kernel_size, stride, dilation, PoolingMode::LOCAL_AVG_POOLING, dimension);
  register_module("base", base_);
}
SparseTensor
MinkowskiAvgPoolingImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
MinkowskiSumPoolingImpl::MinkowskiSumPoolingImpl(int kernel_size, int stride,
                                                 int dilation, int dimension) {
  base_ = std::make_shared<MinkowskiPoolingBaseImpl>(
      kernel_size, stride, dilation, PoolingMode::LOCAL_SUM_POOLING, dimension);
  register_module("base", base_);
}
SparseTensor
MinkowskiSumPoolingImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
MinkowskiMaxPoolingImpl::MinkowskiMaxPoolingImpl(int kernel_size, int stride,
                                                 int dilation, int dimension) {
  base_ = std::make_shared<MinkowskiPoolingBaseImpl>(
      kernel_size, stride, dilation, PoolingMode::LOCAL_MAX_POOLING, dimension);
  register_module("base", base_);
}
SparseTensor
MinkowskiMaxPoolingImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
MinkowskiPoolingTransposeImpl::MinkowskiPoolingTransposeImpl(
    int kernel_size, int stride, int dilation, bool expand_coordinates,
    int dimension)
    : kernel_generator_(
          {kernel_size},
          {stride},
          {dilation},
          true,
          RegionType::HYPER_CUBE,
          {},
          expand_coordinates,
          {},
          dimension),
      pooling_mode_(PoolingMode::LOCAL_SUM_POOLING), dimension_(dimension) {}
SparseTensor
MinkowskiPoolingTransposeImpl::forward(const SparseTensor &input) {
  TORCH_CHECK(input.dimension() == dimension_,
              "Input dimension mismatch: expected ", dimension_, " got ",
              input.dimension());
  auto manager = input.coordinate_manager();
  auto kparams =
      kernel_generator_.get_kernel(input.tensor_stride(), true);
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto out_key = std::make_shared<CoordinateMapKey>(
      input.coordinate_map_key().get_coordinate_size());
  at::Tensor offsets = kparams.region_offsets;
  if (!offsets.defined()) {
    offsets = torch::empty({0}, torch::kInt);
  }
  auto outfeat = LocalPoolingTransposeFunction::apply(
      input.F(), pooling_mode_, kparams.kernel_size, kparams.kernel_stride,
      kparams.kernel_dilation, kparams.region_type, offsets,
      kernel_generator_.expand_coordinates(), in_key, out_key, manager);
  return SparseTensor(outfeat, *out_key, manager);
}
MinkowskiGlobalPoolingImpl::MinkowskiGlobalPoolingImpl(
    PoolingMode::Type mode)
    : pooling_mode_(mode) {}
SparseTensor
MinkowskiGlobalPoolingImpl::forward(const SparseTensor &input) {
  auto manager = input.coordinate_manager();
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto out_key = std::make_shared<CoordinateMapKey>(
      input.coordinate_map_key().get_coordinate_size());
  auto outfeat = GlobalPoolingFunction::apply(input.F(), pooling_mode_,
                                               in_key, out_key, manager);
  return SparseTensor(outfeat, *out_key, manager);
}
MinkowskiGlobalSumPoolingImpl::MinkowskiGlobalSumPoolingImpl(
    PoolingMode::Type mode) {
  base_ = std::make_shared<MinkowskiGlobalPoolingImpl>(mode);
  register_module("base", base_);
}
SparseTensor
MinkowskiGlobalSumPoolingImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
MinkowskiGlobalAvgPoolingImpl::MinkowskiGlobalAvgPoolingImpl(
    PoolingMode::Type mode) {
  base_ = std::make_shared<MinkowskiGlobalPoolingImpl>(mode);
  register_module("base", base_);
}
SparseTensor
MinkowskiGlobalAvgPoolingImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
MinkowskiGlobalMaxPoolingImpl::MinkowskiGlobalMaxPoolingImpl(
    PoolingMode::Type mode) {
  base_ = std::make_shared<MinkowskiGlobalPoolingImpl>(mode);
  register_module("base", base_);
}
SparseTensor
MinkowskiGlobalMaxPoolingImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
} 
