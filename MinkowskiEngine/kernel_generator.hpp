#ifndef MINK_CPP_KERNEL_GENERATOR_HPP
#define MINK_CPP_KERNEL_GENERATOR_HPP

#include "types.hpp"
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace minkowski {

struct StrideHash {
  std::size_t operator()(const stride_type& stride) const {
    std::size_t hash = 0;
    for (const auto& s : stride) {
      hash ^= std::hash<int>{}(s) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
  }
};

class KernelGenerator {
public:
  struct KernelParams {
    stride_type kernel_size;
    stride_type kernel_stride;
    stride_type kernel_dilation;
    RegionType::Type region_type;
    at::Tensor region_offsets;
    bool expand_coordinates;
    int D;
  };

  KernelGenerator(
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef stride = {1},
    torch::IntArrayRef dilation = {1},
    bool is_transpose = false,
    RegionType::Type region_type = RegionType::HYPER_CUBE,
    at::Tensor region_offsets = {},
    bool expand_coordinates = false,
    std::vector<RegionType::Type> axis_types = {},
    int dimension = -1);

  int kernel_volume() const { return kernel_volume_; }
  const stride_type& kernel_size() const { return kernel_size_; }
  const stride_type& kernel_stride() const { return kernel_stride_; }
  const stride_type& kernel_dilation() const { return kernel_dilation_; }
  RegionType::Type region_type() const { return region_type_; }
  const at::Tensor& region_offsets() const { return region_offsets_; }
  bool expand_coordinates() const { return expand_coordinates_; }
  bool requires_strided_coordinates() const;
  int dimension() const { return D_; }

  KernelParams get_kernel(const stride_type& tensor_stride, bool is_transpose) const;

private:
  void initialize();
  std::tuple<RegionType::Type, at::Tensor, int> convert_region_type_impl(
      RegionType::Type region_type,
      const stride_type& tensor_stride,
      const stride_type& kernel_size,
      const stride_type& up_stride,
      const stride_type& dilation,
      const at::Tensor& region_offsets,
      const std::vector<RegionType::Type>& axis_types,
      int dimension,
      bool center = true) const;

  stride_type kernel_size_;
  stride_type kernel_stride_;
  stride_type kernel_dilation_;
  bool is_transpose_;
  RegionType::Type region_type_;
  at::Tensor region_offsets_;
  bool expand_coordinates_;
  std::vector<RegionType::Type> axis_types_;
  int D_;
  int kernel_volume_;
  bool requires_strided_coordinates_;

  mutable std::unordered_map<stride_type, KernelParams, StrideHash> cache_;
};

} 

#endif 
