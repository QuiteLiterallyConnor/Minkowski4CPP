#ifndef MINK_CPP_COORDINATE_MANAGER_HPP
#define MINK_CPP_COORDINATE_MANAGER_HPP

#include "types.hpp"
#include "src/coordinate_map_key.hpp"
#include "src/coordinate_map_manager.hpp"
#include <torch/torch.h>
#include <memory>
#include <variant>
#include <vector>

namespace minkowski {

class CoordinateManager {
public:

  CoordinateManager(
    int D,
    CoordinateMapBackend::Type map_type = CoordinateMapBackend::CUDA,
    GPUMemoryAllocatorBackend::Type alloc_type = GPUMemoryAllocatorBackend::PYTORCH,
    MinkowskiAlgorithm::Mode algo = MinkowskiAlgorithm::DEFAULT);

  ~CoordinateManager();

  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
  insert_and_map(
    const at::Tensor& coordinates,
    stride_type tensor_stride,
    std::string string_id = "");

  CoordinateMapKey stride(
    const CoordinateMapKey& in_key,
    const stride_type& kernel_stride,
    std::string string_id = "");

  std::pair<CoordinateMapKey, bool> origin();

  at::Tensor get_coordinates(const CoordinateMapKey& key) const;
  uint32_t size(const CoordinateMapKey& key) const;

  template<typename ManagerType>
  auto const& kernel_map(
    const CoordinateMapKey& in_key,
    const CoordinateMapKey& out_key,
    const stride_type& kernel_size,
    const stride_type& kernel_stride,
    const stride_type& kernel_dilation,
    RegionType::Type region_type,
    const at::Tensor& offsets,
    bool is_transpose,
    bool is_pool) const;

  template<typename ManagerType>
  auto const& origin_map(const CoordinateMapKey& key) const;

  std::vector<at::Tensor> union_map(
    const std::vector<CoordinateMapKey>& in_keys,
    CoordinateMapKey& out_key);

  std::vector<at::Tensor> interpolation_map_weight(
    const at::Tensor& tfield,
    const CoordinateMapKey& key);

  CoordinateMapKey insert_field(
    const at::Tensor& coordinates,
    stride_type tensor_stride,
    std::string string_id = "");

  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
  field_to_sparse_insert_and_map(
    const CoordinateMapKey& field_key,
    stride_type sparse_stride,
    std::string string_id = "");

  template<typename ManagerType>
  ManagerType* manager_ptr();

#ifdef CPU_ONLY
  auto& get_manager_variant() { return manager_; }
  const auto& get_manager_variant() const { return manager_; }
#else
  auto& get_manager_variant() { return manager_; }
  const auto& get_manager_variant() const { return manager_; }
#endif

  int D() const { return D_; }
  CoordinateMapBackend::Type backend() const { return map_type_; }

private:
  int D_;
  CoordinateMapBackend::Type map_type_;
  GPUMemoryAllocatorBackend::Type alloc_type_;
  MinkowskiAlgorithm::Mode algo_;

#ifdef CPU_ONLY
  std::variant<
    std::unique_ptr<cpu_manager_type<int32_t>>
  > manager_;
#else
  std::variant<
    std::unique_ptr<cpu_manager_type<int32_t>>,
    std::unique_ptr<gpu_default_manager_type<int32_t>>,
    std::unique_ptr<gpu_c10_manager_type<int32_t>>
  > manager_;
#endif
};

} 

#endif 
