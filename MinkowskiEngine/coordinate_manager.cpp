#ifdef CPU_ONLY

#include "coordinate_manager.hpp"
#include "src/types.hpp"
#include <stdexcept>

namespace minkowski {

CoordinateManager::CoordinateManager(
    int D,
    CoordinateMapBackend::Type map_type,
    GPUMemoryAllocatorBackend::Type alloc_type,
    MinkowskiAlgorithm::Mode algo)
    : D_(D), map_type_(map_type), alloc_type_(alloc_type), algo_(algo) {

  TORCH_CHECK(D > 0, "Dimension D must be positive");

#ifdef CPU_ONLY
  if (map_type != CoordinateMapBackend::CPU) {
    throw std::runtime_error("GPU backends not available in CPU_ONLY build");
  }
  manager_ = std::make_unique<cpu_manager_type<int32_t>>(algo, -1);
#else
  if (map_type == CoordinateMapBackend::CPU) {
    manager_ = std::make_unique<cpu_manager_type<int32_t>>(algo, -1);
  } else if (alloc_type == GPUMemoryAllocatorBackend::PYTORCH) {
    manager_ = std::make_unique<gpu_c10_manager_type<int32_t>>(algo, -1);
  } else {
    manager_ = std::make_unique<gpu_default_manager_type<int32_t>>(algo, -1);
  }
#endif
}

std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
CoordinateManager::insert_and_map(
    const at::Tensor& coordinates,
    stride_type tensor_stride,
    std::string string_id) {

  return std::visit([&](auto& manager_ptr) {
    return manager_ptr->insert_and_map_cpp(coordinates, tensor_stride, string_id);
  }, manager_);
}

CoordinateMapKey CoordinateManager::stride(
    const CoordinateMapKey& in_key,
    const stride_type& kernel_stride,
    std::string string_id) {

  return std::visit([&](auto& manager_ptr) {
    auto key_pair = manager_ptr->stride(in_key.get_key(), kernel_stride, string_id);
    return CoordinateMapKey(D_ + 1, key_pair.first);
  }, manager_);
}

std::pair<CoordinateMapKey, bool> CoordinateManager::origin() {
  return std::visit([&](auto& manager_ptr) {
    auto [key_pair, is_new] = manager_ptr->origin();
    return std::make_pair(CoordinateMapKey(D_ + 1, key_pair), is_new);
  }, manager_);
}

at::Tensor CoordinateManager::get_coordinates(const CoordinateMapKey& key) const {
  return std::visit([&](const auto& manager_ptr) -> at::Tensor {
    return manager_ptr->get_coordinates(&key);
  }, manager_);
}

uint32_t CoordinateManager::size(const CoordinateMapKey& key) const {
  return std::visit([&](const auto& manager_ptr) {
    return manager_ptr->size(&key);
  }, manager_);
}

std::vector<at::Tensor> CoordinateManager::union_map(
    const std::vector<CoordinateMapKey>& in_keys,
    CoordinateMapKey& out_key) {

  return std::visit([&](auto& manager_ptr) {

    std::vector<coordinate_map_key_type> internal_keys;
    internal_keys.reserve(in_keys.size());
    for (const auto& key : in_keys) {
      internal_keys.push_back(key.get_key());
    }

    auto result = manager_ptr->union_map(internal_keys);
    auto out_internal_key = result.first;
    auto maps = result.second;

    out_key = CoordinateMapKey(D_ + 1, out_internal_key);

    return maps;
  }, manager_);
}

std::vector<at::Tensor> CoordinateManager::interpolation_map_weight(
    const at::Tensor& tfield,
    const CoordinateMapKey& key) {

  return std::visit([&](auto& manager_ptr) {
    return manager_ptr->interpolation_map_weight(tfield, &key);
  }, manager_);
}

CoordinateMapKey CoordinateManager::insert_field(
    const at::Tensor& coordinates,
    stride_type tensor_stride,
    std::string string_id) {

  return std::visit([&](auto& manager_ptr) {
    return manager_ptr->insert_field_cpp(coordinates, tensor_stride, string_id);
  }, manager_);
}

std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
CoordinateManager::field_to_sparse_insert_and_map(
    const CoordinateMapKey& field_key,
    stride_type sparse_stride,
    std::string string_id) {

  return std::visit([&](auto& manager_ptr) {
    return manager_ptr->field_to_sparse_insert_and_map_cpp(&field_key, sparse_stride, string_id);
  }, manager_);
}

} 

#endif 
