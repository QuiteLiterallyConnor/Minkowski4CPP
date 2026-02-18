#include "convolution.hpp"
#include <cmath>
namespace minkowski {
MinkowskiConvolutionBaseImpl::MinkowskiConvolutionBaseImpl(
    int in_channels, int out_channels, int kernel_size, int stride,
    int dilation, bool bias, bool is_transpose, bool expand_coordinates,
    ConvolutionMode::Type convolution_mode, int dimension,
    RegionType::Type region_type)
    : in_channels_(in_channels), out_channels_(out_channels),
      is_transpose_(is_transpose), use_mm_(false), dimension_(dimension),
      convolution_mode_(convolution_mode),
      kernel_generator_(
          {kernel_size},
          {stride},
          {dilation},
          is_transpose,
          region_type,
          {},
          expand_coordinates,
          {},
          dimension) {
  if (kernel_generator_.kernel_volume() == 1 &&
      kernel_generator_.requires_strided_coordinates()) {
    use_mm_ = true;
    kernel_ = register_parameter(
        "kernel",
        torch::empty({in_channels_, out_channels_}, torch::kFloat));
  } else {
    kernel_ = register_parameter(
        "kernel",
        torch::empty({kernel_generator_.kernel_volume(), in_channels_,
                       out_channels_},
                      torch::kFloat));
  }
  if (bias) {
    bias_ = register_parameter("bias",
                                torch::empty({1, out_channels_}, torch::kFloat));
  }
  reset_parameters(is_transpose);
}
void MinkowskiConvolutionBaseImpl::reset() {
  reset_parameters(is_transpose_);
}
void MinkowskiConvolutionBaseImpl::reset_parameters(bool is_transpose) {
  torch::NoGradGuard no_grad;
  int n = (is_transpose ? out_channels_ : in_channels_) *
          kernel_generator_.kernel_volume();
  double stdv = 1.0 / std::sqrt(static_cast<double>(n));
  kernel_.uniform_(-stdv, stdv);
  if (bias_.defined()) {
    bias_.uniform_(-stdv, stdv);
  }
}
SparseTensor MinkowskiConvolutionBaseImpl::forward(const SparseTensor &input) {
  TORCH_CHECK(input.dimension() == dimension_,
              "Input dimension mismatch: expected ", dimension_, " got ",
              input.dimension());
  auto manager = input.coordinate_manager();
  at::Tensor outfeat;
  if (use_mm_) {
    outfeat = input.F().mm(kernel_);
    if (bias_.defined()) {
      outfeat += bias_;
    }
    return SparseTensor(outfeat, input.coordinate_map_key(), manager);
  }
  auto kparams =
      kernel_generator_.get_kernel(input.tensor_stride(), is_transpose_);
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto out_key = std::make_shared<CoordinateMapKey>(
      input.coordinate_map_key().get_coordinate_size());
  at::Tensor offsets = kparams.region_offsets;
  if (!offsets.defined()) {
    offsets = torch::empty({0}, torch::kInt);
  }
  if (is_transpose_) {
    outfeat = ConvolutionTransposeFunction::apply(
        input.F(), kernel_, kparams.kernel_size, kparams.kernel_stride,
        kparams.kernel_dilation, kparams.region_type, offsets,
        kernel_generator_.expand_coordinates(), convolution_mode_, in_key,
        out_key, manager);
  } else {
    outfeat = ConvolutionFunction::apply(
        input.F(), kernel_, kparams.kernel_size, kparams.kernel_stride,
        kparams.kernel_dilation, kparams.region_type, offsets,
        kernel_generator_.expand_coordinates(), convolution_mode_, in_key,
        out_key, manager);
  }
  if (bias_.defined()) {
    outfeat += bias_;
  }
  return SparseTensor(outfeat, *out_key, manager);
}
MinkowskiConvolutionImpl::MinkowskiConvolutionImpl(
    int in_channels, int out_channels, int kernel_size, int stride,
    int dilation, bool bias, bool expand_coordinates,
    ConvolutionMode::Type convolution_mode, int dimension) {
  base_ = std::make_shared<MinkowskiConvolutionBaseImpl>(
      in_channels, out_channels, kernel_size, stride, dilation, bias,
      false, expand_coordinates, convolution_mode, dimension);
  register_module("base", base_);
}
void MinkowskiConvolutionImpl::reset() { base_->reset(); }
SparseTensor
MinkowskiConvolutionImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
MinkowskiConvolutionTransposeImpl::MinkowskiConvolutionTransposeImpl(
    int in_channels, int out_channels, int kernel_size, int stride,
    int dilation, bool bias, bool expand_coordinates,
    ConvolutionMode::Type convolution_mode, int dimension) {
  base_ = std::make_shared<MinkowskiConvolutionBaseImpl>(
      in_channels, out_channels, kernel_size, stride, dilation, bias,
      true, expand_coordinates, convolution_mode, dimension);
  register_module("base", base_);
}
void MinkowskiConvolutionTransposeImpl::reset() { base_->reset(); }
SparseTensor
MinkowskiConvolutionTransposeImpl::forward(const SparseTensor &input) {
  return base_->forward(input);
}
MinkowskiGenerativeConvolutionTransposeImpl::
    MinkowskiGenerativeConvolutionTransposeImpl(
        int in_channels, int out_channels, int kernel_size, int stride,
        int dilation, bool bias, ConvolutionMode::Type convolution_mode,
        int dimension) {
  base_ = std::make_shared<MinkowskiConvolutionBaseImpl>(
      in_channels, out_channels, kernel_size, stride, dilation, bias,
      true, true, convolution_mode,
      dimension);
  register_module("base", base_);
}
void MinkowskiGenerativeConvolutionTransposeImpl::reset() { base_->reset(); }
SparseTensor MinkowskiGenerativeConvolutionTransposeImpl::forward(
    const SparseTensor &input) {
  return base_->forward(input);
}
} 
