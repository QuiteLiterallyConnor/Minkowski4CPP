#include "kernel_generator.hpp"
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace minkowski {

static int compute_volume_helper(
    RegionType::Type region_type,
    const stride_type& kernel_size,
    const at::Tensor& region_offsets,
    const std::vector<RegionType::Type>& axis_types,
    int dimension) {

  if (region_type == RegionType::HYPER_CUBE) {

    int volume = 1;
    for (auto k : kernel_size) {
      TORCH_CHECK(k > 0, "kernel_size must be positive");
      volume *= k;
    }
    return volume;

  } else if (region_type == RegionType::HYPER_CROSS) {

    int volume = 1;
    for (auto k : kernel_size) {
      TORCH_CHECK(k > 0, "kernel_size must be positive");
      TORCH_CHECK(k % 2 == 1, "kernel_size must be odd for HYPER_CROSS");
      volume += (k - 1);
    }
    return volume;
  } else if  (region_type == RegionType::CUSTOM) {
    TORCH_CHECK(region_offsets.defined() && region_offsets.numel() > 0,
                "region_offsets must be non-empty for CUSTOM region type");
    TORCH_CHECK(region_offsets.size(1) == dimension,
                "region_offsets must match dimension");
    return region_offsets.size(0);
  }

  throw std::runtime_error("Unknown region type");
}

KernelGenerator::KernelGenerator(
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef stride,
    torch::IntArrayRef dilation,
    bool is_transpose,
    RegionType::Type region_type,
    at::Tensor region_offsets,
    bool expand_coordinates,
    std::vector<RegionType::Type> axis_types,
    int dimension)
    : kernel_size_([&]{ stride_type v; v.reserve(kernel_size.size()); for (auto x : kernel_size) v.push_back(static_cast<uint32_t>(x)); return v; }()),
      kernel_stride_([&]{ stride_type v; v.reserve(stride.size()); for (auto x : stride) v.push_back(static_cast<uint32_t>(x)); return v; }()),
      kernel_dilation_([&]{ stride_type v; v.reserve(dilation.size()); for (auto x : dilation) v.push_back(static_cast<uint32_t>(x)); return v; }()),
      is_transpose_(is_transpose),
      region_type_(region_type),
      region_offsets_(region_offsets),
      expand_coordinates_(expand_coordinates),
      axis_types_(axis_types),
      D_(dimension),
      kernel_volume_(0) {

  initialize();
}

void KernelGenerator::initialize() {

  if (D_ <= 0) {
    D_ = kernel_size_.size();
  }

  if (kernel_size_.size() == 1 && D_ > 1) {
    int val = kernel_size_[0];
    kernel_size_.assign(D_, val);
  }
  if (kernel_stride_.size() == 1 && D_ > 1) {
    int val = kernel_stride_[0];
    kernel_stride_.assign(D_, val);
  }
  if (kernel_dilation_.size() == 1 && D_ > 1) {
    int val = kernel_dilation_[0];
    kernel_dilation_.assign(D_, val);
  }

  TORCH_CHECK(kernel_size_. size() == static_cast<size_t>(D_),
              "kernel_size dimension mismatch");
  TORCH_CHECK(kernel_stride_.size() == static_cast<size_t>(D_),
              "kernel_stride dimension mismatch");
  TORCH_CHECK(kernel_dilation_.size() == static_cast<size_t>(D_),
              "kernel_dilation dimension mismatch");

  if (region_type_ == RegionType::HYPER_CUBE || region_type_ == RegionType::HYPER_CROSS) {
    TORCH_CHECK(axis_types_.empty(), "axis_types must be empty for HYPER_CUBE/HYPER_CROSS");
    if (region_offsets_.defined()) {
      TORCH_CHECK(region_offsets_.numel() == 0, "region_offsets must be empty for HYPER_CUBE/HYPER_CROSS");
    }
  }

  kernel_volume_ = compute_volume_helper(region_type_, kernel_size_, region_offsets_, axis_types_, D_);

  requires_strided_coordinates_ = true;
  for (auto s : kernel_stride_) {
    if (s != 1) {
      requires_strided_coordinates_ = false;
      break;
    }
  }
}

bool KernelGenerator::requires_strided_coordinates() const {
  return requires_strided_coordinates_;
}

KernelGenerator::KernelParams KernelGenerator::get_kernel(
    const stride_type& tensor_stride, bool is_transpose) const {

  auto it = cache_.find(tensor_stride);
  if (it != cache_.end()) {
    return it->second;
  }

  KernelParams params;
  params.kernel_size = kernel_size_;
  params.kernel_stride = kernel_stride_;
  params.kernel_dilation = kernel_dilation_;
  params.expand_coordinates = expand_coordinates_;
  params.D = D_;

  stride_type up_stride(D_, 1);
  if (is_transpose) {
    up_stride = kernel_stride_;
  }

  auto [final_region_type, final_offsets, volume] = convert_region_type_impl(
      region_type_, tensor_stride, kernel_size_, up_stride,
      kernel_dilation_, region_offsets_, axis_types_, D_, true);

  params.region_type = final_region_type;
  params.region_offsets = final_offsets;

  cache_[tensor_stride] = params;

  return params;
}

std::tuple<RegionType::Type, at::Tensor, int>
KernelGenerator::convert_region_type_impl(
    RegionType::Type region_type,
    const stride_type& tensor_stride,
    const stride_type& kernel_size,
    const stride_type& up_stride,
    const stride_type& dilation,
    const at::Tensor& region_offsets,
    const std::vector<RegionType::Type>& axis_types,
    int dimension,
    bool center) const {

  if (region_type == RegionType::HYPER_CUBE) {

    at::Tensor empty_offsets = torch::empty({0, dimension}, torch::kInt32);
    int volume = 1;
    for (auto k : kernel_size) {
      volume *= k;
    }
    return {region_type, empty_offsets, volume};

  } else if (region_type == RegionType::HYPER_CROSS) {

    at::Tensor empty_offsets = torch::empty({0, dimension}, torch::kInt32);
    int volume = 1;
    for (auto k : kernel_size) {
      volume += (k - 1);
    }
    return {region_type, empty_offsets, volume};

  } else if (region_type == RegionType::CUSTOM) {
    TORCH_CHECK(region_offsets.defined() && region_offsets.numel() > 0,
                "region_offsets must be non-empty for CUSTOM");
    TORCH_CHECK(region_offsets.size(1) == dimension,
                "region_offsets dimension mismatch");
    int volume = region_offsets.size(0);
    return {RegionType::CUSTOM, region_offsets, volume};
  }

  throw std::runtime_error("Unknown or unimplemented region type");
}

} 
