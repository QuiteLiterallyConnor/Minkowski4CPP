
#ifndef MINK_CPP_AUTOGRAD_POOLING_FN_HPP
#define MINK_CPP_AUTOGRAD_POOLING_FN_HPP

#include "../types.hpp"
#include "../coordinate_manager.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>

namespace minkowski {

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
LocalPoolingForwardCPU(at::Tensor const &in_feat,
                       default_types::stride_type const &kernel_size,
                       default_types::stride_type const &kernel_stride,
                       default_types::stride_type const &kernel_dilation,
                       RegionType::Type const region_type,
                       at::Tensor const &offset,
                       PoolingMode::Type pooling_mode,
                       CoordinateMapKey *p_in_map_key,
                       CoordinateMapKey *p_out_map_key,
                       cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
LocalPoolingBackwardCPU(at::Tensor const &in_feat,
                        at::Tensor const &grad_out_feat,
                        at::Tensor const &num_nonzero,
                        default_types::stride_type const &kernel_size,
                        default_types::stride_type const &kernel_stride,
                        default_types::stride_type const &kernel_dilation,
                        RegionType::Type const region_type,
                        at::Tensor const &offset,
                        PoolingMode::Type pooling_mode,
                        CoordinateMapKey *p_in_map_key,
                        CoordinateMapKey *p_out_map_key,
                        cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
LocalPoolingTransposeForwardCPU(at::Tensor const &in_feat,
                                default_types::stride_type const &kernel_size,
                                default_types::stride_type const &kernel_stride,
                                default_types::stride_type const &kernel_dilation,
                                RegionType::Type const region_type,
                                at::Tensor const &offset,
                                bool generate_new_coordinates,
                                PoolingMode::Type pooling_mode,
                                CoordinateMapKey *p_in_map_key,
                                CoordinateMapKey *p_out_map_key,
                                cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
LocalPoolingTransposeBackwardCPU(at::Tensor const &in_feat,
                                 at::Tensor const &grad_out_feat,
                                 at::Tensor const &num_nonzero,
                                 default_types::stride_type const &kernel_size,
                                 default_types::stride_type const &kernel_stride,
                                 default_types::stride_type const &kernel_dilation,
                                 RegionType::Type const region_type,
                                 at::Tensor const &offset,
                                 PoolingMode::Type pooling_mode,
                                 CoordinateMapKey *p_in_map_key,
                                 CoordinateMapKey *p_out_map_key,
                                 cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
std::tuple<at::Tensor, at::Tensor>
GlobalPoolingForwardCPU(at::Tensor const &in_feat,
                        PoolingMode::Type const pooling_mode,
                        CoordinateMapKey *p_in_map_key,
                        CoordinateMapKey *p_out_map_key,
                        cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
GlobalPoolingBackwardCPU(at::Tensor const &in_feat,
                         at::Tensor &grad_out_feat,
                         at::Tensor const &num_nonzero,
                         PoolingMode::Type const pooling_mode,
                         CoordinateMapKey *p_in_map_key,
                         CoordinateMapKey *p_out_map_key,
                         cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type, template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor>
LocalPoolingForwardGPU(at::Tensor const &in_feat,
                       default_types::stride_type const &kernel_size,
                       default_types::stride_type const &kernel_stride,
                       default_types::stride_type const &kernel_dilation,
                       RegionType::Type const region_type,
                       at::Tensor const &offset,
                       PoolingMode::Type pooling_mode,
                       CoordinateMapKey *p_in_map_key,
                       CoordinateMapKey *p_out_map_key,
                       gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
LocalPoolingBackwardGPU(at::Tensor const &in_feat,
                        at::Tensor const &grad_out_feat,
                        at::Tensor const &num_nonzero,
                        default_types::stride_type const &kernel_size,
                        default_types::stride_type const &kernel_stride,
                        default_types::stride_type const &kernel_dilation,
                        RegionType::Type const region_type,
                        at::Tensor const &offset,
                        PoolingMode::Type pooling_mode,
                        CoordinateMapKey *p_in_map_key,
                        CoordinateMapKey *p_out_map_key,
                        gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor>
LocalPoolingTransposeForwardGPU(at::Tensor const &in_feat,
                                default_types::stride_type const &kernel_size,
                                default_types::stride_type const &kernel_stride,
                                default_types::stride_type const &kernel_dilation,
                                RegionType::Type const region_type,
                                at::Tensor const &offset,
                                bool generate_new_coordinates,
                                PoolingMode::Type pooling_mode,
                                CoordinateMapKey *p_in_map_key,
                                CoordinateMapKey *p_out_map_key,
                                gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
LocalPoolingTransposeBackwardGPU(at::Tensor const &in_feat,
                                 at::Tensor const &grad_out_feat,
                                 at::Tensor const &num_nonzero,
                                 default_types::stride_type const &kernel_size,
                                 default_types::stride_type const &kernel_stride,
                                 default_types::stride_type const &kernel_dilation,
                                 RegionType::Type const region_type,
                                 at::Tensor const &offset,
                                 PoolingMode::Type pooling_mode,
                                 CoordinateMapKey *p_in_map_key,
                                 CoordinateMapKey *p_out_map_key,
                                 gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
std::tuple<at::Tensor, at::Tensor>
GlobalPoolingForwardGPU(at::Tensor const &in_feat,
                        PoolingMode::Type const pooling_mode,
                        CoordinateMapKey *p_in_map_key,
                        CoordinateMapKey *p_out_map_key,
                        gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
GlobalPoolingBackwardGPU(at::Tensor const &in_feat,
                         at::Tensor &grad_out_feat,
                         at::Tensor const &num_nonzero,
                         PoolingMode::Type const pooling_mode,
                         CoordinateMapKey *p_in_map_key,
                         CoordinateMapKey *p_out_map_key,
                         gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

std::pair<torch::Tensor, torch::Tensor>
max_pool_fw(torch::Tensor const &in_map,
            torch::Tensor const &out_map,
            torch::Tensor const &in_feat,
            int const out_nrows, bool const is_sorted);

torch::Tensor
max_pool_bw(torch::Tensor const &grad_out_feat,
            torch::Tensor const &mask_index,
            int const in_nrows);

class LocalPoolingFunction
    : public torch::autograd::Function<LocalPoolingFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor input_features,
      PoolingMode::Type pooling_mode,
      stride_type kernel_size,
      stride_type kernel_stride,
      stride_type kernel_dilation,
      RegionType::Type region_type,
      at::Tensor region_offsets,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

class LocalPoolingTransposeFunction
    : public torch::autograd::Function<LocalPoolingTransposeFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor input_features,
      PoolingMode::Type pooling_mode,
      stride_type kernel_size,
      stride_type kernel_stride,
      stride_type kernel_dilation,
      RegionType::Type region_type,
      at::Tensor region_offsets,
      bool expand_coordinates,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

class GlobalPoolingFunction
    : public torch::autograd::Function<GlobalPoolingFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor input_features,
      PoolingMode::Type pooling_mode,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

class DirectMaxPoolingFunction
    : public torch::autograd::Function<DirectMaxPoolingFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor in_map,
      at::Tensor out_map,
      at::Tensor in_feat,
      int64_t out_nrows,
      bool is_sorted);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

} 

#endif 
