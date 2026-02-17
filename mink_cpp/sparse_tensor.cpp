#include "sparse_tensor.hpp"
#include <stdexcept>

namespace minkowski {

SparseTensor::SparseTensor(
    at::Tensor features,
    at::Tensor coordinates,
    std::shared_ptr<CoordinateManager> manager,
    SparseTensorQuantizationMode::Type quantization_mode,
    GPUMemoryAllocatorBackend::Type alloc_type,
    MinkowskiAlgorithm::Mode algo)
    : features_(features),
      key_(coordinates.size(1)),
      D_(0) {
  
  // Validate inputs
  TORCH_CHECK(features.dim() == 2, "Features must be a 2D tensor");
  TORCH_CHECK(coordinates.dim() == 2, "Coordinates must be a 2D tensor");
  TORCH_CHECK(features.size(0) == coordinates.size(0), 
              "Features and coordinates must have same number of rows");
  
  D_ = coordinates.size(1) - 1;  // Last column is batch index
  
  // Create manager if needed
  if (!manager) {
    auto backend = coordinates.is_cuda() ? CoordinateMapBackend::CUDA : CoordinateMapBackend::CPU;
    manager = std::make_shared<CoordinateManager>(D_, backend, alloc_type, algo);
  }
  manager_ = manager;
  
  // Insert coordinates and get mapping
  stride_type tensor_stride(D_, 1);  // Default stride is 1
  auto [new_key, maps] = manager_->insert_and_map(coordinates, tensor_stride, "");
  auto& [unique_index, inverse_mapping] = maps;
  
  key_ = new_key;
  
  // Apply quantization
  features_ = apply_quantization(features, unique_index, inverse_mapping,
                                  manager_->size(key_), quantization_mode);
}

SparseTensor::SparseTensor(
    at::Tensor features,
    CoordinateMapKey coordinate_map_key,
    std::shared_ptr<CoordinateManager> manager)
    : features_(features),
      key_(coordinate_map_key),
      manager_(manager),
      D_(coordinate_map_key.get_coordinate_size() - 1) {
  
  TORCH_CHECK(manager, "Manager must not be null");
  TORCH_CHECK(features.dim() == 2, "Features must be 2D");
}

at::Tensor SparseTensor::C() const {
  if (!coordinates_.has_value()) {
    coordinates_ = manager_->get_coordinates(key_);
  }
  return coordinates_.value();
}

stride_type SparseTensor::tensor_stride() const {
  return key_.get_tensor_stride();
}

std::vector<at::Tensor> SparseTensor::decomposed_coordinates() const {
  at::Tensor coords = C();
  at::Tensor batch_indices = coords.index({at::indexing::Slice(), 0});
  int64_t max_batch = batch_indices.max().item<int64_t>();
  
  std::vector<at::Tensor> result;
  for (int64_t b = 0; b <= max_batch; ++b) {
    auto mask = batch_indices == b;
    auto batch_coords = coords.index({mask});
    result.push_back(batch_coords);
  }
  return result;
}

std::vector<at::Tensor> SparseTensor::decomposed_features() const {
  at::Tensor coords = C();
  at::Tensor batch_indices = coords.index({at::indexing::Slice(), 0});
  int64_t max_batch = batch_indices.max().item<int64_t>();
  
  std::vector<at::Tensor> result;
  for (int64_t b = 0; b <= max_batch; ++b) {
    auto mask = batch_indices == b;
    auto batch_feats = features_.index({mask});
    result.push_back(batch_feats);
  }
  return result;
}

std::vector<std::pair<at::Tensor, at::Tensor>> 
SparseTensor::decomposed_coordinates_and_features() const {
  at::Tensor coords = C();
  at::Tensor batch_indices = coords.index({at::indexing::Slice(), 0});
  int64_t max_batch = batch_indices.max().item<int64_t>();
  
  std::vector<std::pair<at::Tensor, at::Tensor>> result;
  for (int64_t b = 0; b <= max_batch; ++b) {
    auto mask = batch_indices == b;
    auto batch_coords = coords.index({mask});
    auto batch_feats = features_.index({mask});
    result.emplace_back(batch_coords, batch_feats);
  }
  return result;
}

at::Tensor SparseTensor::dense(
    c10::optional<std::vector<int64_t>> shape,
    c10::optional<at::Tensor> min_coordinate) const {
  
  // TODO: Full implementation - this is a complex operation
  // For now, provide basic stub
  throw std::runtime_error("SparseTensor::dense not fully implemented yet");
}

at::Tensor SparseTensor::sparse() const {
  // Convert to PyTorch sparse COO format
  at::Tensor coords = C();
  // Transpose coordinates for  torch.sparse format (COO expects [ndim, nnz])
  at::Tensor indices = coords.t().to(torch::kLong);
  
  // Create sparse tensor
  std::vector<int64_t> size_vec;
  for (int d = 0; d <= D_; ++d) {
    int64_t max_val = coords.index({at::indexing::Slice(), d}).max().item<int64_t>();
    size_vec.push_back(max_val + 1);
  }
  size_vec.push_back(features_.size(1));  // Feature dimension
  
  return torch::sparse_coo_tensor(indices, features_, size_vec);
}

SparseTensor SparseTensor::operator+(const SparseTensor& other) const {
  TORCH_CHECK(manager_ == other.manager_, "Cannot add tensors from different managers");
  
  if (key_ == other.key_) {
    // Same key: element-wise addition
    return SparseTensor(features_ + other.features_, key_, manager_);
  } else {
    // Different keys: need union - complex operation
    throw std::runtime_error("Cross-key addition requires union - not yet implemented");
  }
}

SparseTensor SparseTensor::operator-(const SparseTensor& other) const {
  TORCH_CHECK(manager_ == other.manager_, "Cannot subtract tensors from different managers");
  
  if (key_ == other.key_) {
    return SparseTensor(features_ - other.features_, key_, manager_);
  } else {
    throw std::runtime_error("Cross-key subtraction requires union - not yet implemented");
  }
}

SparseTensor SparseTensor::operator*(const SparseTensor& other) const {
  TORCH_CHECK(manager_ == other.manager_, "Cannot multiply tensors from different managers");
  
  if (key_ == other.key_) {
    return SparseTensor(features_ * other.features_, key_, manager_);
  } else {
    throw std::runtime_error("Cross-key multiplication requires union - not yet implemented");
  }
}

SparseTensor SparseTensor::operator/(const SparseTensor& other) const {
  TORCH_CHECK(manager_ == other.manager_, "Cannot divide tensors from different managers");
  
  if (key_ == other.key_) {
    return SparseTensor(features_ / other.features_, key_, manager_);
  } else {
    throw std::runtime_error("Cross-key division requires union - not yet implemented");
  }
}

SparseTensor& SparseTensor::operator+=(const SparseTensor& other) {
  TORCH_CHECK(manager_ == other.manager_, "Cannot add tensors from different managers");
  TORCH_CHECK(key_ == other.key_, "In-place addition requires same key");
  features_ += other.features_;
  return *this;
}

SparseTensor SparseTensor::operator+(at::Scalar s) const {
  return SparseTensor(features_ + s, key_, manager_);
}

SparseTensor SparseTensor::operator*(at::Scalar s) const {
  return SparseTensor(features_ * s, key_, manager_);
}

SparseTensor SparseTensor::to(torch::Device device) const {
  // TODO: Need to handle coordinate manager device transfer
  throw std::runtime_error("Device transfer not fully implemented yet");
}

SparseTensor SparseTensor::to(c10::ScalarType dtype) const {
  return SparseTensor(features_.to(dtype), key_, manager_);
}

SparseTensor SparseTensor::float_() const {
  return to(at::kFloat);
}

SparseTensor SparseTensor::double_() const {
  return to(at::kDouble);
}

SparseTensor SparseTensor::detach() const {
  return SparseTensor(features_.detach(), key_, manager_);
}

at::Tensor SparseTensor::apply_quantization(
    const at::Tensor& features,
    const at::Tensor& unique_index,
    const at::Tensor& inverse_mapping,
    uint32_t size,
    SparseTensorQuantizationMode::Type mode) const {
  
  switch (mode) {
    case SparseTensorQuantizationMode::RANDOM_SUBSAMPLE:
      // Simply take first occurrence of each unique coordinate
      return features.index({unique_index});
      
    case SparseTensorQuantizationMode::NO_QUANTIZATION:
      // No duplicates expected
      return features;
      
    case SparseTensorQuantizationMode::UNWEIGHTED_SUM: {
      // Sum features for duplicate coordinates
      at::Tensor result = torch::zeros({static_cast<int64_t>(size), features.size(1)}, 
                                       features.options());
      result.index_put_({inverse_mapping}, features, true);  // accumulate=true
      return result;
    }
      
    case SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE: {
      // Average features for duplicate coordinates
      at::Tensor sum = torch::zeros({static_cast<int64_t>(size), features.size(1)}, 
                                    features.options());
      sum.index_put_({inverse_mapping}, features, true);  // accumulate
      
      at::Tensor count = torch::zeros({static_cast<int64_t>(size)}, 
                                      torch::TensorOptions().dtype(features.dtype()).device(features.device()));
      at::Tensor ones = torch::ones({features.size(0)}, count.options());
      count.index_put_({inverse_mapping}, ones, true);  // Count occurrences
      
      return sum / count.unsqueeze(1);
    }
      
    case SparseTensorQuantizationMode::MAX_POOL: {
      // Max pooling for duplicate coordinates using public API
      at::Tensor result = torch::scatter_reduce(
          torch::zeros({static_cast<int64_t>(size), features.size(1)}, features.options()),
          0,
          inverse_mapping.unsqueeze(1).expand({-1, features.size(1)}),
          features,
          "amax",
          false);
      return result;
    }
      
    default:
      throw std::runtime_error("Unknown quantization mode");
  }
}

} // namespace minkowski
