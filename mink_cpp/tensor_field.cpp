#include "tensor_field.hpp"
#include <stdexcept>

namespace minkowski {

TensorField::TensorField(
    at::Tensor features,
    at::Tensor coordinates,
    std::shared_ptr<CoordinateManager> manager,
    SparseTensorQuantizationMode::Type quantization_mode)
    : features_(features),
      field_key_(coordinates.size(1)),
      D_(0) {
  
  // Validate inputs
  TORCH_CHECK(features.dim() == 2, "Features must be a 2D tensor");
  TORCH_CHECK(coordinates.dim() == 2, "Coordinates must be a 2D tensor");
  TORCH_CHECK(features.size(0) == coordinates.size(0),
              "Features and coordinates must have same number of rows");
  TORCH_CHECK(coordinates.scalar_type() == at::kFloat || coordinates.scalar_type() == at::kDouble,
              "TensorField coordinates must be float or double");
  
  D_ = coordinates.size(1) - 1;  // Last column is batch index
  
  // Create manager if needed
  if (!manager) {
    auto backend = coordinates.is_cuda() ? CoordinateMapBackend::CUDA : CoordinateMapBackend::CPU;
    manager = std::make_shared<CoordinateManager>(D_, backend);
  }
  manager_ = manager;
  
  // Insert field coordinates
  stride_type tensor_stride(D_, 1);  // Default stride is 1
  field_key_ = manager_->insert_field(coordinates, tensor_stride, "");
}

SparseTensor TensorField::sparse(
    stride_type tensor_stride,
    c10::optional<CoordinateMapKey> coordinate_map_key,
    SparseTensorQuantizationMode::Type quantization_mode) const {
  
  if (tensor_stride.empty()) {
    tensor_stride = stride_type(D_, 1);
  }
  
  // Convert field to sparse coordinates
  auto [sparse_key, maps] = manager_->field_to_sparse_insert_and_map(
      field_key_, tensor_stride, "");
  auto& [unique_index, inverse_mapping] = maps;
  
  // Apply quantization
  at::Tensor quantized_features;
  uint32_t size = manager_->size(sparse_key);
  
  switch (quantization_mode) {
    case SparseTensorQuantizationMode::RANDOM_SUBSAMPLE:
      quantized_features = features_.index({unique_index});
      break;
      
    case SparseTensorQuantizationMode::UNWEIGHTED_SUM: {
      quantized_features = torch::zeros({static_cast<int64_t>(size), features_.size(1)},
                                       features_.options());
      quantized_features.index_put_({inverse_mapping}, features_, true);
      break;
    }
      
    case SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE: {
      at::Tensor sum = torch::zeros({static_cast<int64_t>(size), features_.size(1)},
                                   features_.options());
      sum.index_put_({inverse_mapping}, features_, true);
      
      at::Tensor count = torch::zeros({static_cast<int64_t>(size)},
                                     torch::TensorOptions().dtype(features_.dtype()).device(features_.device()));
      at::Tensor ones = torch::ones({features_.size(0)}, count.options());
      count.index_put_({inverse_mapping}, ones, true);
      
      quantized_features = sum / count.unsqueeze(1);
      break;
    }
      
    case SparseTensorQuantizationMode::MAX_POOL: {
      // Use public API for max pooling
      quantized_features = torch::scatter_reduce(
          torch::zeros({static_cast<int64_t>(size), features_.size(1)}, features_.options()),
          0,
          inverse_mapping.unsqueeze(1).expand({-1, features_.size(1)}),
          features_,
          "amax",
          false);
      break;
    }
      
    default:
      quantized_features = features_;
  }
  
  // Cache inverse mapping
  inverse_mappings_[sparse_key] = inverse_mapping;
  
  // Return SparseTensor
  return SparseTensor(quantized_features, sparse_key, manager_);
}

SparseTensor TensorField::splat() const {
  // Splat creates sparse tensor with linear interpolation to 2^D neighbors
  // Generate splat coordinates by finding floor and adding all 2^D corners
  
  at::Tensor field_coords = C();
  
  // Generate 2^D offsets
  std::vector<std::vector<int>> offsets;
  int num_offsets = 1 << D_;  // 2^D
  for (int i = 0; i < num_offsets; ++i) {
    std::vector<int> offset;
    offset.push_back(0);  // Batch dimension offset is always 0
    for (int d = 0; d < D_; ++d) {
      offset.push_back((i >> d) & 1);
    }
    offsets.push_back(offset);
  }
  
  // Floor coordinates and add offsets
  at::Tensor floored = torch::floor(field_coords).to(torch::kInt32);
  
  // Create expanded coordinates with all 2^D neighbors
  std::vector<at::Tensor> expanded_coords;
  for (const auto& offset : offsets) {
    at::Tensor offset_tensor = torch::tensor(offset, floored.options()).unsqueeze(0);
    expanded_coords.push_back(floored + offset_tensor);
  }
  at::Tensor all_coords = torch::cat(expanded_coords, 0);  // [N*2^D, D+1]
  
  // Compute interpolation weights for each neighbor
  at::Tensor field_coords_frac = field_coords - torch::floor(field_coords);
  std::vector<at::Tensor> weights;
  for (const auto& offset : offsets) {
    at::Tensor weight = torch::ones({field_coords.size(0)}, field_coords.options());
    for (int d = 1; d <= D_; ++d) {  // Skip batch dimension
      if (offset[d] == 1) {
        weight = weight * field_coords_frac.index({at::indexing::Slice(), d});
      } else {
        weight = weight * (1.0 - field_coords_frac.index({at::indexing::Slice(), d}));
      }
    }
    weights.push_back(weight);
  }
  at::Tensor all_weights = torch::cat(weights, 0);  // [N*2^D]
  
  // Replicate features for each neighbor and weight them
  at::Tensor expanded_features = features_.repeat({num_offsets, 1});
  expanded_features = expanded_features * all_weights.unsqueeze(1);
  
  // Create SparseTensor with quantization
  return SparseTensor(expanded_features, all_coords, manager_,
                     SparseTensorQuantizationMode::UNWEIGHTED_SUM);
}

at::Tensor TensorField::inverse_mapping(const CoordinateMapKey& sparse_key) const {
  // Check cache
  auto it = inverse_mappings_.find(sparse_key);
  if (it != inverse_mappings_.end()) {
    return it->second;
  }
  
  // Compute inverse mapping via field_to_sparse conversion
  stride_type sparse_stride = sparse_key.get_tensor_stride();
  auto [_, maps] = manager_->field_to_sparse_insert_and_map(field_key_, sparse_stride, "");
  auto& [unique_index, inverse_mapping] = maps;
  
  // Cache and return
  inverse_mappings_[sparse_key] = inverse_mapping;
  return inverse_mapping;
}

at::Tensor TensorField::C() const {
  if (!coordinates_.has_value()) {
    coordinates_ = manager_->get_coordinates(field_key_);
  }
  return coordinates_.value();
}

} // namespace minkowski
