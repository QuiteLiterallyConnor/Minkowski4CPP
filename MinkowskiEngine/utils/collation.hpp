
#ifndef MINK_CPP_UTILS_COLLATION_HPP
#define MINK_CPP_UTILS_COLLATION_HPP

#include <torch/torch.h>
#include <vector>
#include <tuple>

namespace minkowski {

at::Tensor batched_coordinates(
    const std::vector<at::Tensor>& coords,
    c10::ScalarType dtype = torch::kInt32,
    c10::optional<torch::Device> device = c10::nullopt);

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sparse_collate(
    const std::vector<at::Tensor>& coords,
    const std::vector<at::Tensor>& feats,
    const std::vector<at::Tensor>& labels = {},
    c10::ScalarType dtype = torch::kInt32,
    c10::optional<torch::Device> device = c10::nullopt);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_sparse_collate(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& data,
    c10::ScalarType dtype = torch::kInt32,
    c10::optional<torch::Device> device = c10::nullopt);

class SparseCollation {
public:
  explicit SparseCollation(int64_t limit_numpoints = -1,
                           c10::ScalarType dtype = torch::kInt32,
                           c10::optional<torch::Device> device = c10::nullopt);

  std::tuple<at::Tensor, at::Tensor, at::Tensor>
  operator()(const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& list_data) const;

private:
  int64_t limit_numpoints_;
  c10::ScalarType dtype_;
  c10::optional<torch::Device> device_;
};

} 

#endif 
