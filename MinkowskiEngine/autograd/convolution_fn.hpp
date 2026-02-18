#ifndef MINK_CPP_AUTOGRAD_CONVOLUTION_FN_HPP
#define MINK_CPP_AUTOGRAD_CONVOLUTION_FN_HPP

#include "../types.hpp"
#include "../coordinate_manager.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>

namespace minkowski {

template <typename coordinate_type>
at::Tensor
ConvolutionForwardCPU(at::Tensor const &in_feat,
                      at::Tensor const &kernel,
                      default_types::stride_type const &kernel_size,
                      default_types::stride_type const &kernel_stride,
                      default_types::stride_type const &kernel_dilation,
                      RegionType::Type const region_type,
                      at::Tensor const &offset,
                      bool const expand_coordinates,
                      ConvolutionMode::Type const convolution_mode,
                      CoordinateMapKey *p_in_map_key,
                      CoordinateMapKey *p_out_map_key,
                      cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardCPU(at::Tensor const &in_feat,
                       at::Tensor &grad_out_feat,
                       at::Tensor const &kernel,
                       default_types::stride_type const &kernel_size,
                       default_types::stride_type const &kernel_stride,
                       default_types::stride_type const &kernel_dilation,
                       RegionType::Type const region_type,
                       at::Tensor const &offset,
                       ConvolutionMode::Type const convolution_mode,
                       CoordinateMapKey *p_in_map_key,
                       CoordinateMapKey *p_out_map_key,
                       cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
ConvolutionTransposeForwardCPU(at::Tensor const &in_feat,
                               at::Tensor const &kernel,
                               default_types::stride_type const &kernel_size,
                               default_types::stride_type const &kernel_stride,
                               default_types::stride_type const &kernel_dilation,
                               RegionType::Type const region_type,
                               at::Tensor const &offset,
                               bool generate_new_coordinates,
                               ConvolutionMode::Type const convolution_mode,
                               CoordinateMapKey *p_in_map_key,
                               CoordinateMapKey *p_out_map_key,
                               cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
std::pair<at::Tensor, at::Tensor>
ConvolutionTransposeBackwardCPU(at::Tensor const &in_feat,
                                at::Tensor const &grad_out_feat,
                                at::Tensor const &kernel,
                                default_types::stride_type const &kernel_size,
                                default_types::stride_type const &kernel_stride,
                                default_types::stride_type const &kernel_dilation,
                                RegionType::Type const region_type,
                                at::Tensor const &offset,
                                ConvolutionMode::Type const convolution_mode,
                                CoordinateMapKey *p_in_map_key,
                                CoordinateMapKey *p_out_map_key,
                                cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
ConvolutionForwardGPU(at::Tensor const &in_feat,
                      at::Tensor const &kernel,
                      default_types::stride_type const &kernel_size,
                      default_types::stride_type const &kernel_stride,
                      default_types::stride_type const &kernel_dilation,
                      RegionType::Type const region_type,
                      at::Tensor const &offset,
                      bool const expand_coordinates,
                      ConvolutionMode::Type const convolution_mode,
                      CoordinateMapKey *p_in_map_key,
                      CoordinateMapKey *p_out_map_key,
                      gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor>
ConvolutionBackwardGPU(at::Tensor const &in_feat,
                       at::Tensor &grad_out_feat,
                       at::Tensor const &kernel,
                       default_types::stride_type const &kernel_size,
                       default_types::stride_type const &kernel_stride,
                       default_types::stride_type const &kernel_dilation,
                       RegionType::Type const region_type,
                       at::Tensor const &offset,
                       ConvolutionMode::Type const convolution_mode,
                       CoordinateMapKey *p_in_map_key,
                       CoordinateMapKey *p_out_map_key,
                       gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
ConvolutionTransposeForwardGPU(at::Tensor const &in_feat,
                               at::Tensor const &kernel,
                               default_types::stride_type const &kernel_size,
                               default_types::stride_type const &kernel_stride,
                               default_types::stride_type const &kernel_dilation,
                               RegionType::Type const region_type,
                               at::Tensor const &offset,
                               bool generate_new_coordinates,
                               ConvolutionMode::Type const convolution_mode,
                               CoordinateMapKey *p_in_map_key,
                               CoordinateMapKey *p_out_map_key,
                               gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
std::pair<at::Tensor, at::Tensor>
ConvolutionTransposeBackwardGPU(at::Tensor const &in_feat,
                                at::Tensor const &grad_out_feat,
                                at::Tensor const &kernel,
                                default_types::stride_type const &kernel_size,
                                default_types::stride_type const &kernel_stride,
                                default_types::stride_type const &kernel_dilation,
                                RegionType::Type const region_type,
                                at::Tensor const &offset,
                                ConvolutionMode::Type const convolution_mode,
                                CoordinateMapKey *p_in_map_key,
                                CoordinateMapKey *p_out_map_key,
                                gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

class ConvolutionFunction
    : public torch::autograd::Function<ConvolutionFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor input_features,
      at::Tensor kernel,
      stride_type kernel_size,
      stride_type kernel_stride,
      stride_type kernel_dilation,
      RegionType::Type region_type,
      at::Tensor region_offsets,
      bool expand_coordinates,
      ConvolutionMode::Type conv_mode,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

class ConvolutionTransposeFunction
    : public torch::autograd::Function<ConvolutionTransposeFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor input_features,
      at::Tensor kernel,
      stride_type kernel_size,
      stride_type kernel_stride,
      stride_type kernel_dilation,
      RegionType::Type region_type,
      at::Tensor region_offsets,
      bool expand_coordinates,
      ConvolutionMode::Type conv_mode,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

} 

#endif 
