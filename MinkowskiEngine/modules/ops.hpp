#ifndef MINK_CPP_MODULES_OPS_HPP
#define MINK_CPP_MODULES_OPS_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include <torch/torch.h>
#include <vector>
namespace minkowski {
SparseTensor cat(const std::vector<SparseTensor> &sparse_tensors);
SparseTensor sum(const std::vector<SparseTensor> &sparse_tensors);
SparseTensor mean(const std::vector<SparseTensor> &sparse_tensors);
SparseTensor var(const std::vector<SparseTensor> &sparse_tensors);
at::Tensor dense_coordinates(at::IntArrayRef shape);
SparseTensor to_sparse(const at::Tensor &dense_tensor);
SparseTensor to_sparse_all(const at::Tensor &dense_tensor,
                            c10::optional<at::Tensor> coordinates = c10::nullopt);
class MinkowskiToSparseTensorImpl
    : public torch::nn::Cloneable<MinkowskiToSparseTensorImpl> {
public:
  MinkowskiToSparseTensorImpl(bool remove_zeros = true,
                               c10::optional<at::Tensor> coordinates = c10::nullopt);
  void reset() override {}
  SparseTensor forward(const at::Tensor &input);
  bool remove_zeros_;
  c10::optional<at::Tensor> coordinates_;
};
TORCH_MODULE(MinkowskiToSparseTensor);
class MinkowskiToDenseTensorImpl
    : public torch::nn::Cloneable<MinkowskiToDenseTensorImpl> {
public:
  explicit MinkowskiToDenseTensorImpl(
      c10::optional<std::vector<int64_t>> shape = c10::nullopt);
  void reset() override {}
  at::Tensor forward(const SparseTensor &input);
  c10::optional<std::vector<int64_t>> shape_;
};
TORCH_MODULE(MinkowskiToDenseTensor);
class MinkowskiToFeatureImpl
    : public torch::nn::Cloneable<MinkowskiToFeatureImpl> {
public:
  MinkowskiToFeatureImpl() = default;
  void reset() override {}
  at::Tensor forward(const SparseTensor &input);
};
TORCH_MODULE(MinkowskiToFeature);
} 
#endif 
