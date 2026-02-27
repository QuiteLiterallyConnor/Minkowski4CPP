#ifndef MINK_CPP_TENSOR_FIELD_HPP
#define MINK_CPP_TENSOR_FIELD_HPP

#include "types.hpp"
#include "coordinate_manager.hpp"
#include "src/coordinate_map_key.hpp"
#include "sparse_tensor.hpp"
#include <torch/torch.h>
#include <memory>
#include <unordered_map>

namespace minkowski {

class TensorField {
public:

  TensorField(
    at::Tensor features,
    at::Tensor coordinates,  
    std::shared_ptr<CoordinateManager> manager = nullptr,
    SparseTensorQuantizationMode::Type quantization_mode = SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE);

  SparseTensor sparse(
    stride_type tensor_stride = {},
    c10::optional<CoordinateMapKey> coordinate_map_key = c10::nullopt,
    SparseTensorQuantizationMode::Type quantization_mode = SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE) const;

  SparseTensor splat() const;

  at::Tensor inverse_mapping(const CoordinateMapKey& sparse_key) const;

  const at::Tensor& F() const { return features_; }
  at::Tensor C() const;  
  const CoordinateMapKey& coordinate_field_map_key() const { return field_key_; }
  std::shared_ptr<CoordinateManager> coordinate_manager() const { return manager_; }
  int dimension() const { return D_; }
  torch::Device device() const { return features_.device(); }
  c10::ScalarType dtype() const { return features_.scalar_type(); }

private:
  at::Tensor features_;
  mutable c10::optional<at::Tensor> coordinates_;  
  CoordinateMapKey field_key_;
  std::shared_ptr<CoordinateManager> manager_;
  int D_;

  struct CoordinateMapKeyHash {
    std::size_t operator()(const CoordinateMapKey& key) const {

      return std::hash<std::string>{}(key.get_key().second);
    }
  };
  mutable std::unordered_map<CoordinateMapKey, at::Tensor, CoordinateMapKeyHash> inverse_mappings_;
};

} 

#endif 
