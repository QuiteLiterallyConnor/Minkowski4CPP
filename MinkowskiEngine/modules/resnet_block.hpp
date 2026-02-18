#ifndef MINK_CPP_MODULES_RESNET_BLOCK_HPP
#define MINK_CPP_MODULES_RESNET_BLOCK_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "convolution.hpp"
#include "normalization.hpp"
#include "nonlinearity.hpp"
#include <torch/torch.h>
#include <memory>
namespace minkowski {
class BasicBlockImpl : public torch::nn::Cloneable<BasicBlockImpl> {
public:
  static constexpr int expansion = 1;
  BasicBlockImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                 torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                 double bn_momentum = 0.1, int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &x);
  MinkowskiConvolution conv1_{nullptr};
  MinkowskiBatchNorm norm1_{nullptr};
  MinkowskiConvolution conv2_{nullptr};
  MinkowskiBatchNorm norm2_{nullptr};
  MinkowskiReLU relu_{nullptr};
  torch::nn::AnyModule downsample_;
  int inplanes_;
  int planes_;
  int stride_;
  int dilation_;
  double bn_momentum_;
  int dimension_;
  bool has_downsample_ = false;
};
TORCH_MODULE(BasicBlock);
class BottleneckImpl : public torch::nn::Cloneable<BottleneckImpl> {
public:
  static constexpr int expansion = 4;
  BottleneckImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                 torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                 double bn_momentum = 0.1, int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &x);
  MinkowskiConvolution conv1_{nullptr};
  MinkowskiBatchNorm norm1_{nullptr};
  MinkowskiConvolution conv2_{nullptr};
  MinkowskiBatchNorm norm2_{nullptr};
  MinkowskiConvolution conv3_{nullptr};
  MinkowskiBatchNorm norm3_{nullptr};
  MinkowskiReLU relu_{nullptr};
  torch::nn::AnyModule downsample_;
  int inplanes_;
  int planes_;
  int stride_;
  int dilation_;
  double bn_momentum_;
  int dimension_;
  bool has_downsample_ = false;
};
TORCH_MODULE(Bottleneck);
} 
#endif 
