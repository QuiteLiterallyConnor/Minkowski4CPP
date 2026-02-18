#include "senet_block.hpp"
namespace minkowski {
SELayerImpl::SELayerImpl(int channel, int reduction, int dimension)
    : channel_(channel), reduction_(reduction), dimension_(dimension) {
  reset();
}
void SELayerImpl::reset() {
  int reduced = channel_ / reduction_;
  TORCH_CHECK(reduced > 0,
              "SELayer: channel / reduction must be > 0. Got channel=",
              channel_, " reduction=", reduction_);
  fc1_ = register_module("fc1", MinkowskiLinear(channel_, reduced, true));
  fc_relu_ = register_module(
      "fc_relu", MinkowskiReLU(torch::nn::ReLUOptions().inplace(true)));
  fc2_ = register_module("fc2", MinkowskiLinear(reduced, channel_, true));
  fc_sigmoid_ = register_module("fc_sigmoid", MinkowskiSigmoid());
  pooling_ = register_module("pooling", MinkowskiGlobalPooling());
  broadcast_mul_ =
      register_module("broadcast_mul", MinkowskiBroadcastMultiplication());
}
SparseTensor SELayerImpl::forward(const SparseTensor &x) {
  auto y = pooling_->forward(x);
  y = fc1_->forward(y);
  y = fc_relu_->forward(y);
  y = fc2_->forward(y);
  y = fc_sigmoid_->forward(y);
  return broadcast_mul_->forward(x, y);
}
SEBasicBlockImpl::SEBasicBlockImpl(int inplanes, int planes, int stride,
                                   int dilation,
                                   torch::nn::AnyModule downsample,
                                   int reduction, double bn_momentum,
                                   int dimension)
    : inplanes_(inplanes), planes_(planes), stride_(stride),
      dilation_(dilation), reduction_(reduction), bn_momentum_(bn_momentum),
      dimension_(dimension), downsample_(std::move(downsample)) {
  TORCH_CHECK(dimension > 0, "Dimension must be > 0, got ", dimension);
  has_downsample_ = !downsample_.is_empty();
  reset();
}
void SEBasicBlockImpl::reset() {
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
  se_ = register_module("se", SELayer(planes_, reduction_, dimension_));
}
SparseTensor SEBasicBlockImpl::forward(const SparseTensor &x) {
  SparseTensor residual = x;
  auto out = conv1_->forward(x);
  out = norm1_->forward(out);
  out = relu_->forward(out);
  out = conv2_->forward(out);
  out = norm2_->forward(out);
  out = se_->forward(out);
  if (has_downsample_) {
    residual = downsample_.forward<SparseTensor>(x);
  }
  out += residual;
  out = relu_->forward(out);
  return out;
}
SEBottleneckImpl::SEBottleneckImpl(int inplanes, int planes, int stride,
                                   int dilation,
                                   torch::nn::AnyModule downsample,
                                   int reduction, double bn_momentum,
                                   int dimension)
    : inplanes_(inplanes), planes_(planes), stride_(stride),
      dilation_(dilation), reduction_(reduction), bn_momentum_(bn_momentum),
      dimension_(dimension), downsample_(std::move(downsample)) {
  TORCH_CHECK(dimension > 0, "Dimension must be > 0, got ", dimension);
  has_downsample_ = !downsample_.is_empty();
  reset();
}
void SEBottleneckImpl::reset() {
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
  se_ = register_module(
      "se", SELayer(planes_ * expansion, reduction_, dimension_));
}
SparseTensor SEBottleneckImpl::forward(const SparseTensor &x) {
  SparseTensor residual = x;
  auto out = conv1_->forward(x);
  out = norm1_->forward(out);
  out = relu_->forward(out);
  out = conv2_->forward(out);
  out = norm2_->forward(out);
  out = relu_->forward(out);
  out = conv3_->forward(out);
  out = norm3_->forward(out);
  out = se_->forward(out);
  if (has_downsample_) {
    residual = downsample_.forward<SparseTensor>(x);
  }
  out += residual;
  out = relu_->forward(out);
  return out;
}
} 
