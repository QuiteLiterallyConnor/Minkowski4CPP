
#ifndef MINK_CPP_AUTOGRAD_INTERPOLATION_FN_HPP
#define MINK_CPP_AUTOGRAD_INTERPOLATION_FN_HPP

#include "../types.hpp"
#include "../coordinate_manager.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>

namespace minkowski {

template <typename coordinate_type>
std::vector<at::Tensor>
InterpolationForwardCPU(at::Tensor const &in_feat,
                        at::Tensor const &tfield,
                        CoordinateMapKey *p_in_map_key,
                        cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
InterpolationBackwardCPU(at::Tensor &grad_out_feat,
                         at::Tensor const &in_map,
                         at::Tensor const &out_map,
                         at::Tensor const &weight,
                         CoordinateMapKey *p_in_map_key,
                         cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type, template <typename C> class TemplatedAllocator>
std::vector<at::Tensor>
InterpolationForwardGPU(at::Tensor const &in_feat,
                        at::Tensor const &tfield,
                        CoordinateMapKey *p_in_map_key,
                        gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
InterpolationBackwardGPU(at::Tensor &grad_out_feat,
                         at::Tensor const &in_maps,
                         at::Tensor const &out_maps,
                         at::Tensor const &weights,
                         CoordinateMapKey *p_in_map_key,
                         gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

class InterpolationFunction
    : public torch::autograd::Function<InterpolationFunction> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor input_features,
      at::Tensor tfield,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

} 

#endif 
