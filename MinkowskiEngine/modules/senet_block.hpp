#ifndef MINK_CPP_MODULES_SENET_BLOCK_HPP
#define MINK_CPP_MODULES_SENET_BLOCK_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "resnet_block.hpp"
#include "linear.hpp"
#include "pooling.hpp"
#include "broadcast.hpp"
#include "nonlinearity.hpp"
#include <torch/torch.h>
#include <memory>
namespace minkowski {
class SELayerImpl : public torch::nn::Cloneable<SELayerImpl> {
public:
  SELayerImpl(int channel, int reduction = 16, int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &x);
  MinkowskiLinear fc1_{nullptr};
  MinkowskiReLU fc_relu_{nullptr};
  MinkowskiLinear fc2_{nullptr};
  MinkowskiSigmoid fc_sigmoid_{nullptr};
  MinkowskiGlobalPooling pooling_{nullptr};
  MinkowskiBroadcastMultiplication broadcast_mul_{nullptr};
  int channel_;
  int reduction_;
  int dimension_;
};
TORCH_MODULE(SELayer);
class SEBasicBlockImpl : public torch::nn::Cloneable<SEBasicBlockImpl> {
public:
  static constexpr int expansion = 1;
  SEBasicBlockImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                   torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                   int reduction = 16, double bn_momentum = 0.1,
                   int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &x);
  MinkowskiConvolution conv1_{nullptr};
  MinkowskiBatchNorm norm1_{nullptr};
  MinkowskiConvolution conv2_{nullptr};
  MinkowskiBatchNorm norm2_{nullptr};
  MinkowskiReLU relu_{nullptr};
  SELayer se_{nullptr};
  torch::nn::AnyModule downsample_;
  int inplanes_;
  int planes_;
  int stride_;
  int dilation_;
  int reduction_;
  double bn_momentum_;
  int dimension_;
  bool has_downsample_ = false;
};
TORCH_MODULE(SEBasicBlock);
class SEBottleneckImpl : public torch::nn::Cloneable<SEBottleneckImpl> {
public:
  static constexpr int expansion = 4;
  SEBottleneckImpl(int inplanes, int planes, int stride = 1, int dilation = 1,
                   torch::nn::AnyModule downsample = torch::nn::AnyModule(),
                   int reduction = 16, double bn_momentum = 0.1,
                   int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &x);
  MinkowskiConvolution conv1_{nullptr};
  MinkowskiBatchNorm norm1_{nullptr};
  MinkowskiConvolution conv2_{nullptr};
  MinkowskiBatchNorm norm2_{nullptr};
  MinkowskiConvolution conv3_{nullptr};
  MinkowskiBatchNorm norm3_{nullptr};
  MinkowskiReLU relu_{nullptr};
  SELayer se_{nullptr};
  torch::nn::AnyModule downsample_;
  int inplanes_;
  int planes_;
  int stride_;
  int dilation_;
  int reduction_;
  double bn_momentum_;
  int dimension_;
  bool has_downsample_ = false;
};
TORCH_MODULE(SEBottleneck);
} 
#endif 
