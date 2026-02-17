#ifndef MINK_CPP_SPARSE_TENSOR_HPP
#define MINK_CPP_SPARSE_TENSOR_HPP

#include "types.hpp"
#include "coordinate_manager.hpp"
#include "../mink/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace minkowski {

class SparseTensor {
public:
  // Construct from coordinates + features (inserts into manager)
  SparseTensor(
    at::Tensor features,
    at::Tensor coordinates,
    std::shared_ptr<CoordinateManager> manager = nullptr,
    SparseTensorQuantizationMode::Type quantization_mode = SparseTensorQuantizationMode::RANDOM_SUBSAMPLE,
    GPUMemoryAllocatorBackend::Type alloc_type = GPUMemoryAllocatorBackend::PYTORCH,
    MinkowskiAlgorithm::Mode algo = MinkowskiAlgorithm::DEFAULT);

  // Construct from existing key + features (wrap existing coordinate map)
  SparseTensor(
    at::Tensor features,
    CoordinateMapKey coordinate_map_key,
    std::shared_ptr<CoordinateManager> manager);

  // Accessors
  const at::Tensor& F() const { return features_; }
  at::Tensor C() const;  // Lazy-fetched from manager
  const CoordinateMapKey& coordinate_map_key() const { return key_; }
  std::shared_ptr<CoordinateManager> coordinate_manager() const { return manager_; }
  stride_type tensor_stride() const;
  int dimension() const { return D_; }
  torch::Device device() const { return features_.device(); }
  c10::ScalarType dtype() const { return features_.scalar_type(); }
  bool requires_grad() const { return features_.requires_grad(); }

  // Batch decomposition
  std::vector<at::Tensor> decomposed_coordinates() const;
  std::vector<at::Tensor> decomposed_features() const;
  std::vector<std::pair<at::Tensor, at::Tensor>> decomposed_coordinates_and_features() const;

  // Conversions
  at::Tensor dense(
    c10::optional<std::vector<int64_t>> shape = c10::nullopt,
    c10::optional<at::Tensor> min_coordinate = c10::nullopt) const;
  at::Tensor sparse() const;  // torch.sparse COO

  // Operator overloads
  SparseTensor operator+(const SparseTensor& other) const;
  SparseTensor operator-(const SparseTensor& other) const;
  SparseTensor operator*(const SparseTensor& other) const;
  SparseTensor operator/(const SparseTensor& other) const;
  SparseTensor& operator+=(const SparseTensor& other);

  // Scalar overloads
  SparseTensor operator+(at::Scalar s) const;
  SparseTensor operator*(at::Scalar s) const;

  // Type casting
  SparseTensor to(torch::Device device) const;
  SparseTensor to(c10::ScalarType dtype) const;
  SparseTensor float_() const;
  SparseTensor double_() const;
  SparseTensor detach() const;

private:
  at::Tensor features_;
  mutable c10::optional<at::Tensor> coordinates_;  // lazy-cached
  CoordinateMapKey key_;
  std::shared_ptr<CoordinateManager> manager_;
  int D_;

  // Helper for quantization during construction
  at::Tensor apply_quantization(
    const at::Tensor& features,
    const at::Tensor& unique_index,
    const at::Tensor& inverse_mapping,
    uint32_t size,
    SparseTensorQuantizationMode::Type mode) const;
};

} // namespace minkowski

#endif // MINK_CPP_SPARSE_TENSOR_HPP
