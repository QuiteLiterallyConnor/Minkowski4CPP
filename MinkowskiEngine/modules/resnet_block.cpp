#include "resnet_block.hpp"
namespace minkowski {
BasicBlockImpl::BasicBlockImpl(int inplanes, int planes, int stride,
                               int dilation,
                               torch::nn::AnyModule downsample,
                               double bn_momentum, int dimension)
    : inplanes_(inplanes), planes_(planes), stride_(stride),
      dilation_(dilation), bn_momentum_(bn_momentum), dimension_(dimension),
      downsample_(std::move(downsample)) {
  TORCH_CHECK(dimension > 0, "Dimension must be > 0, got ", dimension);
  has_downsample_ = !downsample_.is_empty();
  reset();
}
void BasicBlockImpl::reset() {
  conv1_ = register_module(
      "conv1",
      MinkowskiConvolution(inplanes_, planes_, 3, stride_,
                           dilation_, false,
                           false,
                           ConvolutionMode::DEFAULT, dimension_));
  norm1_ = register_module(
      "norm1", MinkowskiBatchNorm(planes_, 1e-5, bn_momentum_));
  conv2_ = register_module(
      "conv2",
      MinkowskiConvolution(planes_, planes_, 3, 1,
                           dilation_, false,
                           false,
                           ConvolutionMode::DEFAULT, dimension_));
  norm2_ = register_module(
      "norm2", MinkowskiBatchNorm(planes_, 1e-5, bn_momentum_));
  relu_ = register_module(
      "relu", MinkowskiReLU(torch::nn::ReLUOptions().inplace(true)));
}
SparseTensor BasicBlockImpl::forward(const SparseTensor &x) {
  SparseTensor residual = x;
  auto out = conv1_->forward(x);
  out = norm1_->forward(out);
  out = relu_->forward(out);
  out = conv2_->forward(out);
  out = norm2_->forward(out);
  if (has_downsample_) {
    residual = downsample_.forward<SparseTensor>(x);
  }
  out += residual;
  out = relu_->forward(out);
  return out;
}
BottleneckImpl::BottleneckImpl(int inplanes, int planes, int stride,
                               int dilation,
                               torch::nn::AnyModule downsample,
                               double bn_momentum, int dimension)
    : inplanes_(inplanes), planes_(planes), stride_(stride),
      dilation_(dilation), bn_momentum_(bn_momentum), dimension_(dimension),
      downsample_(std::move(downsample)) {
  TORCH_CHECK(dimension > 0, "Dimension must be > 0, got ", dimension);
  has_downsample_ = !downsample_.is_empty();
  reset();
}
void BottleneckImpl::reset() {
  conv1_ = register_module(
      "conv1",
      MinkowskiConvolution(inplanes_, planes_, 1, 1,
                           1, false,
                           false,
                           ConvolutionMode::DEFAULT, dimension_));
  norm1_ = register_module(
      "norm1", MinkowskiBatchNorm(planes_, 1e-5, bn_momentum_));
  conv2_ = register_module(
      "conv2",
      MinkowskiConvolution(planes_, planes_, 3, stride_,
                           dilation_, false,
                           false,
                           ConvolutionMode::DEFAULT, dimension_));
  norm2_ = register_module(
      "norm2", MinkowskiBatchNorm(planes_, 1e-5, bn_momentum_));
  conv3_ = register_module(
      "conv3",
      MinkowskiConvolution(planes_, planes_ * expansion, 1,
                           1, 1, false,
                           false,
                           ConvolutionMode::DEFAULT, dimension_));
  norm3_ = register_module(
      "norm3",
      MinkowskiBatchNorm(planes_ * expansion, 1e-5, bn_momentum_));
  relu_ = register_module(
      "relu", MinkowskiReLU(torch::nn::ReLUOptions().inplace(true)));
}
SparseTensor BottleneckImpl::forward(const SparseTensor &x) {
  SparseTensor residual = x;
  auto out = conv1_->forward(x);
  out = norm1_->forward(out);
  out = relu_->forward(out);
  out = conv2_->forward(out);
  out = norm2_->forward(out);
  out = relu_->forward(out);
  out = conv3_->forward(out);
  out = norm3_->forward(out);
  if (has_downsample_) {
    residual = downsample_.forward<SparseTensor>(x);
  }
  out += residual;
  out = relu_->forward(out);
  return out;
}
} 
