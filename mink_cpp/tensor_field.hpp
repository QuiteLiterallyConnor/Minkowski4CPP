#ifndef MINK_CPP_TENSOR_FIELD_HPP
#define MINK_CPP_TENSOR_FIELD_HPP

#include "types.hpp"
#include "coordinate_manager.hpp"
#include "../mink/coordinate_map_key.hpp"
#include "sparse_tensor.hpp"
#include <torch/torch.h>
#include <memory>
#include <unordered_map>

namespace minkowski {

class TensorField {
public:
  // Construct from float coordinates + features
  TensorField(
    at::Tensor features,
    at::Tensor coordinates,  // float coordinates
    std::shared_ptr<CoordinateManager> manager = nullptr,
    SparseTensorQuantizationMode::Type quantization_mode = SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE);

  // Convert to SparseTensor
  SparseTensor sparse(
    stride_type tensor_stride = {1},
    c10::optional<CoordinateMapKey> coordinate_map_key = c10::nullopt,
    SparseTensorQuantizationMode::Type quantization_mode = SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE) const;

  // Linear interpolation to 2^D neighbors
  SparseTensor splat() const;

  // Get inverse mapping for a sparse key
  at::Tensor inverse_mapping(const CoordinateMapKey& sparse_key) const;

  // Accessors
  const at::Tensor& F() const { return features_; }
  at::Tensor C() const;  // Lazy-fetched from manager
  const CoordinateMapKey& coordinate_field_map_key() const { return field_key_; }
  std::shared_ptr<CoordinateManager> coordinate_manager() const { return manager_; }
  int dimension() const { return D_; }
  torch::Device device() const { return features_.device(); }
  c10::ScalarType dtype() const { return features_.scalar_type(); }

private:
  at::Tensor features_;
  mutable c10::optional<at::Tensor> coordinates_;  // lazy-cached
  CoordinateMapKey field_key_;
  std::shared_ptr<CoordinateManager> manager_;
  int D_;
  
  // Cache inverse mappings
  struct CoordinateMapKeyHash {
    std::size_t operator()(const CoordinateMapKey& key) const {
      // Hash the string component of the key
      return std::hash<std::string>{}(key.get_key().second);
    }
  };
  mutable std::unordered_map<CoordinateMapKey, at::Tensor, CoordinateMapKeyHash> inverse_mappings_;
};

} // namespace minkowski

#endif // MINK_CPP_TENSOR_FIELD_HPP
