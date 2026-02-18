#include "ops.hpp"
#include <numeric>
namespace minkowski {
static void validate_same_key(const std::vector<SparseTensor> &tensors) {
  TORCH_CHECK(tensors.size() > 1,
              "Need at least 2 sparse tensors, got ", tensors.size());
  auto manager = tensors[0].coordinate_manager();
  auto key = tensors[0].coordinate_map_key();
  for (size_t i = 1; i < tensors.size(); ++i) {
    TORCH_CHECK(manager.get() == tensors[i].coordinate_manager().get(),
                "All sparse tensors must share the same CoordinateManager");
    TORCH_CHECK(key == tensors[i].coordinate_map_key(),
                "All sparse tensors must share the same CoordinateMapKey. "
                "Use MinkowskiUnion for different sparsity patterns.");
  }
}
SparseTensor cat(const std::vector<SparseTensor> &sparse_tensors) {
  validate_same_key(sparse_tensors);
  std::vector<at::Tensor> feats;
  feats.reserve(sparse_tensors.size());
  for (const auto &s : sparse_tensors) {
    feats.push_back(s.F());
  }
  auto cat_feat = torch::cat(feats, 1);
  return SparseTensor(cat_feat, sparse_tensors[0].coordinate_map_key(),
                      sparse_tensors[0].coordinate_manager());
}
SparseTensor sum(const std::vector<SparseTensor> &sparse_tensors) {
  validate_same_key(sparse_tensors);
  auto result = sparse_tensors[0].F().clone();
  for (size_t i = 1; i < sparse_tensors.size(); ++i) {
    result += sparse_tensors[i].F();
  }
  return SparseTensor(result, sparse_tensors[0].coordinate_map_key(),
                      sparse_tensors[0].coordinate_manager());
}
SparseTensor mean(const std::vector<SparseTensor> &sparse_tensors) {
  validate_same_key(sparse_tensors);
  auto result = sparse_tensors[0].F().clone();
  for (size_t i = 1; i < sparse_tensors.size(); ++i) {
    result += sparse_tensors[i].F();
  }
  result /= static_cast<double>(sparse_tensors.size());
  return SparseTensor(result, sparse_tensors[0].coordinate_map_key(),
                      sparse_tensors[0].coordinate_manager());
}
SparseTensor var(const std::vector<SparseTensor> &sparse_tensors) {
  validate_same_key(sparse_tensors);
  auto m = sparse_tensors[0].F().clone();
  for (size_t i = 1; i < sparse_tensors.size(); ++i) {
    m += sparse_tensors[i].F();
  }
  m /= static_cast<double>(sparse_tensors.size());
  auto v = (sparse_tensors[0].F() - m).pow(2);
  for (size_t i = 1; i < sparse_tensors.size(); ++i) {
    v += (sparse_tensors[i].F() - m).pow(2);
  }
  v /= static_cast<double>(sparse_tensors.size());
  return SparseTensor(v, sparse_tensors[0].coordinate_map_key(),
                      sparse_tensors[0].coordinate_manager());
}
at::Tensor dense_coordinates(at::IntArrayRef shape) {
  int64_t ndim = static_cast<int64_t>(shape.size());
  TORCH_CHECK(ndim > 2,
              "Shape must be BxCxD1xD2x...xDN with at least one spatial dim");
  int64_t B = shape[0];
  int64_t spatial_dim = ndim - 2;
  int64_t total_spatial = 1;
  for (int64_t i = 2; i < ndim; ++i) {
    total_spatial *= shape[i];
  }
  int64_t total = B * total_spatial;
  int64_t coord_dim = 1 + spatial_dim; 
  auto coords = torch::empty({total, coord_dim}, torch::kInt);
  auto coords_a = coords.accessor<int, 2>();
  int64_t idx = 0;
  for (int64_t b = 0; b < B; ++b) {
    std::vector<int64_t> pos(spatial_dim, 0);
    for (int64_t s = 0; s < total_spatial; ++s) {
      coords_a[idx][0] = static_cast<int>(b);
      for (int64_t d = 0; d < spatial_dim; ++d) {
        coords_a[idx][1 + d] = static_cast<int>(pos[d]);
      }
      for (int64_t d = spatial_dim - 1; d >= 0; --d) {
        pos[d]++;
        if (pos[d] < shape[2 + d]) break;
        pos[d] = 0;
      }
      ++idx;
    }
  }
  return coords;
}
SparseTensor to_sparse(const at::Tensor &dense_tensor) {
  TORCH_CHECK(dense_tensor.dim() > 2,
              "Input must have at least 3 dimensions (BxCxD1x...)");
  int64_t ndim = dense_tensor.dim();
  int64_t ch_dim = 1; 
  auto reduced = torch::abs(dense_tensor).sum(ch_dim);
  auto nonzero_indices = torch::where(reduced != 0);
  std::vector<at::Tensor> idx_tensors;
  for (const auto &t : nonzero_indices) {
    idx_tensors.push_back(t);
  }
  auto stacked = torch::stack(idx_tensors, 1).to(torch::kInt);
  int64_t N = stacked.size(0);
  int64_t C = dense_tensor.size(ch_dim);
  auto features = torch::zeros({N, C}, dense_tensor.options());
  for (int64_t i = 0; i < N; ++i) {
    std::vector<at::Tensor> indices;
    for (int64_t d = 0; d < ndim; ++d) {
      if (d == ch_dim) continue;
      int64_t col = d < ch_dim ? d : d - 1;
      indices.push_back(
          stacked.select(0, i).select(0, col).unsqueeze(0).to(torch::kLong));
    }
  }
  std::vector<int64_t> perm;
  perm.push_back(0);
  for (int64_t d = 2; d < ndim; ++d) perm.push_back(d);
  perm.push_back(1);
  auto permuted = dense_tensor.permute(perm).contiguous();
  auto flat = permuted.reshape({-1, C});
  auto flat_reduced = reduced.reshape({-1});
  auto nonzero_mask = flat_reduced != 0;
  auto nonzero_flat_indices = torch::where(nonzero_mask)[0];
  features = flat.index_select(0, nonzero_flat_indices);
  return SparseTensor(features, stacked);
}
SparseTensor
to_sparse_all(const at::Tensor &dense_tensor,
              c10::optional<at::Tensor> coordinates) {
  TORCH_CHECK(dense_tensor.dim() > 2,
              "Input must have at least 3 dimensions (BxCxD1x...)");
  int64_t ndim = dense_tensor.dim();
  int64_t C = dense_tensor.size(1);
  int64_t spatial_dim = ndim - 2;
  at::Tensor coords;
  if (coordinates.has_value()) {
    coords = coordinates.value();
  } else {
    coords = dense_coordinates(dense_tensor.sizes());
  }
  std::vector<int64_t> perm;
  perm.push_back(0);
  for (int64_t d = 2; d < ndim; ++d) perm.push_back(d);
  perm.push_back(1);
  auto permuted = dense_tensor.permute(perm).contiguous();
  auto features = permuted.reshape({-1, C});
  return SparseTensor(features, coords);
}
MinkowskiToSparseTensorImpl::MinkowskiToSparseTensorImpl(
    bool remove_zeros, c10::optional<at::Tensor> coordinates)
    : remove_zeros_(remove_zeros), coordinates_(std::move(coordinates)) {}
SparseTensor
MinkowskiToSparseTensorImpl::forward(const at::Tensor &input) {
  if (remove_zeros_ && !coordinates_.has_value()) {
    return to_sparse(input);
  } else {
    return to_sparse_all(input, coordinates_);
  }
}
MinkowskiToDenseTensorImpl::MinkowskiToDenseTensorImpl(
    c10::optional<std::vector<int64_t>> shape)
    : shape_(std::move(shape)) {}
at::Tensor
MinkowskiToDenseTensorImpl::forward(const SparseTensor &input) {
  return input.dense(shape_);
}
at::Tensor
MinkowskiToFeatureImpl::forward(const SparseTensor &input) {
  return input.F();
}
} 
