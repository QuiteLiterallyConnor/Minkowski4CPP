#ifndef MINK_CPP_MODULES_POOLING_HPP
#define MINK_CPP_MODULES_POOLING_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "../kernel_generator.hpp"
#include "../coordinate_manager.hpp"
#include "../autograd/pooling_fn.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>
namespace minkowski {
class MinkowskiPoolingBaseImpl
    : public torch::nn::Cloneable<MinkowskiPoolingBaseImpl> {
public:
  MinkowskiPoolingBaseImpl(int kernel_size, int stride = 1, int dilation = 1,
                           PoolingMode::Type pooling_mode =
                               PoolingMode::LOCAL_AVG_POOLING,
                           int dimension = 3);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
  KernelGenerator kernel_generator_;
  PoolingMode::Type pooling_mode_;
  int dimension_;
};
class MinkowskiAvgPoolingImpl
    : public torch::nn::Cloneable<MinkowskiAvgPoolingImpl> {
public:
  MinkowskiAvgPoolingImpl(int kernel_size, int stride = 1, int dilation = 1,
                          int dimension = 3);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
private:
  std::shared_ptr<MinkowskiPoolingBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiAvgPooling);
class MinkowskiSumPoolingImpl
    : public torch::nn::Cloneable<MinkowskiSumPoolingImpl> {
public:
  MinkowskiSumPoolingImpl(int kernel_size, int stride = 1, int dilation = 1,
                          int dimension = 3);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
private:
  std::shared_ptr<MinkowskiPoolingBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiSumPooling);
class MinkowskiMaxPoolingImpl
    : public torch::nn::Cloneable<MinkowskiMaxPoolingImpl> {
public:
  MinkowskiMaxPoolingImpl(int kernel_size, int stride = 1, int dilation = 1,
                          int dimension = 3);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
private:
  std::shared_ptr<MinkowskiPoolingBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiMaxPooling);
class MinkowskiPoolingTransposeImpl
    : public torch::nn::Cloneable<MinkowskiPoolingTransposeImpl> {
public:
  MinkowskiPoolingTransposeImpl(int kernel_size, int stride,
                                int dilation = 1,
                                bool expand_coordinates = false,
                                int dimension = 3);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
private:
  KernelGenerator kernel_generator_;
  PoolingMode::Type pooling_mode_;
  int dimension_;
};
TORCH_MODULE(MinkowskiPoolingTranspose);
class MinkowskiGlobalPoolingImpl
    : public torch::nn::Cloneable<MinkowskiGlobalPoolingImpl> {
public:
  explicit MinkowskiGlobalPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
  PoolingMode::Type pooling_mode_;
};
TORCH_MODULE(MinkowskiGlobalPooling);
class MinkowskiGlobalSumPoolingImpl
    : public torch::nn::Cloneable<MinkowskiGlobalSumPoolingImpl> {
public:
  explicit MinkowskiGlobalSumPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_SUM_POOLING_PYTORCH_INDEX);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
private:
  std::shared_ptr<MinkowskiGlobalPoolingImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiGlobalSumPooling);
class MinkowskiGlobalAvgPoolingImpl
    : public torch::nn::Cloneable<MinkowskiGlobalAvgPoolingImpl> {
public:
  explicit MinkowskiGlobalAvgPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
private:
  std::shared_ptr<MinkowskiGlobalPoolingImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiGlobalAvgPooling);
class MinkowskiGlobalMaxPoolingImpl
    : public torch::nn::Cloneable<MinkowskiGlobalMaxPoolingImpl> {
public:
  explicit MinkowskiGlobalMaxPoolingImpl(
      PoolingMode::Type mode = PoolingMode::GLOBAL_MAX_POOLING_PYTORCH_INDEX);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input);
private:
  std::shared_ptr<MinkowskiGlobalPoolingImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiGlobalMaxPooling);
} 
#endif 
