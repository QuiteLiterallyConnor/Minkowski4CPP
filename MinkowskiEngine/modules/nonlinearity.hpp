#ifndef MINK_CPP_MODULES_NONLINEARITY_HPP
#define MINK_CPP_MODULES_NONLINEARITY_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include <torch/torch.h>
namespace minkowski {
template <typename TorchModuleType>
class MinkowskiNonlinearityImpl
    : public torch::nn::Cloneable<MinkowskiNonlinearityImpl<TorchModuleType>> {
public:
  template <typename... Args>
  explicit MinkowskiNonlinearityImpl(Args &&...args)
      : module_(std::forward<Args>(args)...) {
    this->register_module("module", module_);
  }
  MinkowskiNonlinearityImpl() : module_(nullptr) {
    if constexpr (std::is_default_constructible_v<
                      typename TorchModuleType::ContainedType>) {
      module_ = this->register_module(
          "module",
          TorchModuleType(std::make_shared<
                          typename TorchModuleType::ContainedType>()));
    }
  }
  void reset() override {}
  SparseTensor forward(const SparseTensor &input) {
    auto output = module_->forward(input.F());
    return SparseTensor(output, input.coordinate_map_key(),
                        input.coordinate_manager());
  }
private:
  TorchModuleType module_;
};
using MinkowskiReLUImpl = MinkowskiNonlinearityImpl<torch::nn::ReLU>;
using MinkowskiReLU6Impl = MinkowskiNonlinearityImpl<torch::nn::ReLU6>;
using MinkowskiLeakyReLUImpl = MinkowskiNonlinearityImpl<torch::nn::LeakyReLU>;
using MinkowskiELUImpl = MinkowskiNonlinearityImpl<torch::nn::ELU>;
using MinkowskiSELUImpl = MinkowskiNonlinearityImpl<torch::nn::SELU>;
using MinkowskiCELUImpl = MinkowskiNonlinearityImpl<torch::nn::CELU>;
using MinkowskiGELUImpl = MinkowskiNonlinearityImpl<torch::nn::GELU>;
using MinkowskiSiLUImpl = MinkowskiNonlinearityImpl<torch::nn::SiLU>;
using MinkowskiSigmoidImpl = MinkowskiNonlinearityImpl<torch::nn::Sigmoid>;
using MinkowskiTanhImpl = MinkowskiNonlinearityImpl<torch::nn::Tanh>;
using MinkowskiHardshrinkImpl = MinkowskiNonlinearityImpl<torch::nn::Hardshrink>;
using MinkowskiHardtanhImpl = MinkowskiNonlinearityImpl<torch::nn::Hardtanh>;
using MinkowskiLogSigmoidImpl = MinkowskiNonlinearityImpl<torch::nn::LogSigmoid>;
using MinkowskiPReLUImpl = MinkowskiNonlinearityImpl<torch::nn::PReLU>;
using MinkowskiRReLUImpl = MinkowskiNonlinearityImpl<torch::nn::RReLU>;
using MinkowskiSoftplusImpl = MinkowskiNonlinearityImpl<torch::nn::Softplus>;
using MinkowskiSoftshrinkImpl = MinkowskiNonlinearityImpl<torch::nn::Softshrink>;
using MinkowskiSoftsignImpl = MinkowskiNonlinearityImpl<torch::nn::Softsign>;
using MinkowskiTanhshrinkImpl = MinkowskiNonlinearityImpl<torch::nn::Tanhshrink>;
using MinkowskiThresholdImpl = MinkowskiNonlinearityImpl<torch::nn::Threshold>;
using MinkowskiSoftminImpl = MinkowskiNonlinearityImpl<torch::nn::Softmin>;
using MinkowskiSoftmaxImpl = MinkowskiNonlinearityImpl<torch::nn::Softmax>;
using MinkowskiLogSoftmaxImpl = MinkowskiNonlinearityImpl<torch::nn::LogSoftmax>;
using MinkowskiDropoutImpl = MinkowskiNonlinearityImpl<torch::nn::Dropout>;
using MinkowskiAlphaDropoutImpl = MinkowskiNonlinearityImpl<torch::nn::AlphaDropout>;
TORCH_MODULE(MinkowskiReLU);
TORCH_MODULE(MinkowskiReLU6);
TORCH_MODULE(MinkowskiLeakyReLU);
TORCH_MODULE(MinkowskiELU);
TORCH_MODULE(MinkowskiSELU);
TORCH_MODULE(MinkowskiCELU);
TORCH_MODULE(MinkowskiGELU);
TORCH_MODULE(MinkowskiSiLU);
TORCH_MODULE(MinkowskiSigmoid);
TORCH_MODULE(MinkowskiTanh);
TORCH_MODULE(MinkowskiHardshrink);
TORCH_MODULE(MinkowskiHardtanh);
TORCH_MODULE(MinkowskiLogSigmoid);
TORCH_MODULE(MinkowskiPReLU);
TORCH_MODULE(MinkowskiRReLU);
TORCH_MODULE(MinkowskiSoftplus);
TORCH_MODULE(MinkowskiSoftshrink);
TORCH_MODULE(MinkowskiSoftsign);
TORCH_MODULE(MinkowskiTanhshrink);
TORCH_MODULE(MinkowskiThreshold);
TORCH_MODULE(MinkowskiSoftmin);
TORCH_MODULE(MinkowskiSoftmax);
TORCH_MODULE(MinkowskiLogSoftmax);
TORCH_MODULE(MinkowskiDropout);
TORCH_MODULE(MinkowskiAlphaDropout);
class MinkowskiHardsigmoidImpl
    : public torch::nn::Cloneable<MinkowskiHardsigmoidImpl> {
public:
  MinkowskiHardsigmoidImpl() = default;
  void reset() override {}
  SparseTensor forward(const SparseTensor &input) {
    auto output = at::hardsigmoid(input.F());
    return SparseTensor(output, input.coordinate_map_key(),
                        input.coordinate_manager());
  }
};
TORCH_MODULE(MinkowskiHardsigmoid);
class MinkowskiHardswishImpl
    : public torch::nn::Cloneable<MinkowskiHardswishImpl> {
public:
  MinkowskiHardswishImpl() = default;
  void reset() override {}
  SparseTensor forward(const SparseTensor &input) {
    auto output = at::hardswish(input.F());
    return SparseTensor(output, input.coordinate_map_key(),
                        input.coordinate_manager());
  }
};
TORCH_MODULE(MinkowskiHardswish);
class MinkowskiSinusoidalImpl
    : public torch::nn::Cloneable<MinkowskiSinusoidalImpl> {
public:
  MinkowskiSinusoidalImpl(int in_channels, int out_channels);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  int in_channels_;
  int out_channels_;
  torch::Tensor kernel_{nullptr};
  torch::Tensor bias_{nullptr};
  torch::Tensor coef_{nullptr};
};
TORCH_MODULE(MinkowskiSinusoidal);
} 
#endif 
