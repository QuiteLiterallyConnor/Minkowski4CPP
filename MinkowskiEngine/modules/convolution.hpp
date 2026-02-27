#ifndef MINK_CPP_MODULES_CONVOLUTION_HPP
#define MINK_CPP_MODULES_CONVOLUTION_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "../kernel_generator.hpp"
#include "../coordinate_manager.hpp"
#include "../autograd/convolution_fn.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <cmath>
#include <memory>
namespace minkowski {
class MinkowskiConvolutionBaseImpl
    : public torch::nn::Cloneable<MinkowskiConvolutionBaseImpl> {
public:
  MinkowskiConvolutionBaseImpl(
      int in_channels, int out_channels, int kernel_size = 3, int stride = 1,
      int dilation = 1, bool bias = false, bool is_transpose = false,
      bool expand_coordinates = false,
      ConvolutionMode::Type convolution_mode = ConvolutionMode::DEFAULT,
      int dimension = 3,
      RegionType::Type region_type = RegionType::HYPER_CUBE);
  void reset() override;
  void reset_parameters(bool is_transpose = false);
  SparseTensor forward(const SparseTensor &input);
  int in_channels_;
  int out_channels_;
  bool is_transpose_;
  bool use_mm_;
  int dimension_;
  ConvolutionMode::Type convolution_mode_;
  KernelGenerator kernel_generator_;
  torch::Tensor kernel_{nullptr};
  torch::Tensor bias_{nullptr};
};
class MinkowskiConvolutionImpl
    : public torch::nn::Cloneable<MinkowskiConvolutionImpl> {
public:
  MinkowskiConvolutionImpl(int in_channels, int out_channels,
                           int kernel_size = 3, int stride = 1,
                           int dilation = 1, bool bias = false,
                           bool expand_coordinates = false,
                           ConvolutionMode::Type convolution_mode =
                               ConvolutionMode::DEFAULT,
                           int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  std::shared_ptr<MinkowskiConvolutionBaseImpl> base() { return base_; }
private:
  std::shared_ptr<MinkowskiConvolutionBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiConvolution);
class MinkowskiConvolutionTransposeImpl
    : public torch::nn::Cloneable<MinkowskiConvolutionTransposeImpl> {
public:
  MinkowskiConvolutionTransposeImpl(
      int in_channels, int out_channels, int kernel_size = 3, int stride = 1,
      int dilation = 1, bool bias = false, bool expand_coordinates = false,
      ConvolutionMode::Type convolution_mode = ConvolutionMode::DEFAULT,
      int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  std::shared_ptr<MinkowskiConvolutionBaseImpl> base() { return base_; }
private:
  std::shared_ptr<MinkowskiConvolutionBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiConvolutionTranspose);
class MinkowskiGenerativeConvolutionTransposeImpl
    : public torch::nn::Cloneable<MinkowskiGenerativeConvolutionTransposeImpl> {
public:
  MinkowskiGenerativeConvolutionTransposeImpl(
      int in_channels, int out_channels, int kernel_size = 3, int stride = 1,
      int dilation = 1, bool bias = false,
      ConvolutionMode::Type convolution_mode = ConvolutionMode::DEFAULT,
      int dimension = 3);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  std::shared_ptr<MinkowskiConvolutionBaseImpl> base() { return base_; }
private:
  std::shared_ptr<MinkowskiConvolutionBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiGenerativeConvolutionTranspose);
} 
#endif 
