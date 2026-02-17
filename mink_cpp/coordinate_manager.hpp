#ifndef MINK_CPP_COORDINATE_MANAGER_HPP
#define MINK_CPP_COORDINATE_MANAGER_HPP

#include "types.hpp"
#include "../mink/coordinate_map_key.hpp"
#include "../mink/coordinate_map_manager.hpp"
#include <torch/torch.h>
#include <memory>
#include <variant>
#include <vector>

namespace minkowski {

// Note: Manager type aliases (cpu_manager_type, gpu_default_manager_type, gpu_c10_manager_type)
// are already defined in mink/coordinate_map_manager.hpp and available here

class CoordinateManager {
public:
  // Constructor
  CoordinateManager(
    int D,
    CoordinateMapBackend::Type map_type = CoordinateMapBackend::CUDA,
    GPUMemoryAllocatorBackend::Type alloc_type = GPUMemoryAllocatorBackend::PYTORCH,
    MinkowskiAlgorithm::Mode algo = MinkowskiAlgorithm::DEFAULT);

  // Destructor - defined in .cpp to avoid incomplete type issues with variant
  ~CoordinateManager();

  // Core coordinate operations
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

  // Kernel maps
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

  // Origin map
  template<typename ManagerType>
  auto const& origin_map(const CoordinateMapKey& key) const;

  // Union operations
  std::vector<at::Tensor> union_map(
    const std::vector<CoordinateMapKey>& in_keys,
    CoordinateMapKey& out_key);

  // Interpolation
  std::vector<at::Tensor> interpolation_map_weight(
    const at::Tensor& tfield,
    const CoordinateMapKey& key);

  // Field operations
  CoordinateMapKey insert_field(
    const at::Tensor& coordinates,
    stride_type tensor_stride,
    std::string string_id = "");

  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
  field_to_sparse_insert_and_map(
    const CoordinateMapKey& field_key,
    stride_type sparse_stride,
    std::string string_id = "");

  // Access internal manager (for passing to kernel functions)
  template<typename ManagerType>
  ManagerType* manager_ptr();

  int D() const { return D_; }
  CoordinateMapBackend::Type backend() const { return map_type_; }

private:
  int D_;
  CoordinateMapBackend::Type map_type_;
  GPUMemoryAllocatorBackend::Type alloc_type_;
  MinkowskiAlgorithm::Mode algo_;

  // Type-erased internal manager
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

} // namespace minkowski

#endif // MINK_CPP_COORDINATE_MANAGER_HPP
