#ifndef MINK_CPP_MODULES_NORMALIZATION_HPP
#define MINK_CPP_MODULES_NORMALIZATION_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "../coordinate_manager.hpp"
#include "../autograd/instance_norm_fn.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>
namespace minkowski {
class MinkowskiGlobalPoolingImpl;
class MinkowskiBroadcastAdditionImpl;
class MinkowskiBroadcastMultiplicationImpl;
class MinkowskiBatchNormImpl
    : public torch::nn::Cloneable<MinkowskiBatchNormImpl> {
public:
  MinkowskiBatchNormImpl(int num_features, double eps = 1e-5,
                         double momentum = 0.1, bool affine = true,
                         bool track_running_stats = true);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  torch::nn::BatchNorm1d bn_{nullptr};
};
TORCH_MODULE(MinkowskiBatchNorm);
class MinkowskiSyncBatchNormImpl
    : public torch::nn::Cloneable<MinkowskiSyncBatchNormImpl> {
public:
  MinkowskiSyncBatchNormImpl(int num_features, double eps = 1e-5,
                             double momentum = 0.1, bool affine = true,
                             bool track_running_stats = true);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  torch::nn::BatchNorm1d bn_{nullptr};
};
TORCH_MODULE(MinkowskiSyncBatchNorm);
class MinkowskiInstanceNormImpl
    : public torch::nn::Cloneable<MinkowskiInstanceNormImpl> {
public:
  explicit MinkowskiInstanceNormImpl(int num_features);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  int num_features_;
  torch::Tensor weight_{nullptr};
  torch::Tensor bias_{nullptr};
};
TORCH_MODULE(MinkowskiInstanceNorm);
class MinkowskiStableInstanceNormImpl
    : public torch::nn::Cloneable<MinkowskiStableInstanceNormImpl> {
public:
  explicit MinkowskiStableInstanceNormImpl(int num_features);
  void reset() override;
  SparseTensor forward(const SparseTensor &input);
  int num_features_;
  double eps_ = 1e-6;
  torch::Tensor weight_{nullptr};
  torch::Tensor bias_{nullptr};
  std::shared_ptr<MinkowskiGlobalPoolingImpl> mean_in_{nullptr};
  std::shared_ptr<MinkowskiBroadcastAdditionImpl> glob_sum_{nullptr};
  std::shared_ptr<MinkowskiBroadcastAdditionImpl> glob_sum2_{nullptr};
  std::shared_ptr<MinkowskiGlobalPoolingImpl> glob_mean_{nullptr};
  std::shared_ptr<MinkowskiBroadcastMultiplicationImpl> glob_times_{nullptr};
};
TORCH_MODULE(MinkowskiStableInstanceNorm);
} 
#endif 
