
#ifndef MINK_CPP_AUTOGRAD_PRUNING_FN_HPP
#define MINK_CPP_AUTOGRAD_PRUNING_FN_HPP

#include "../types.hpp"
#include "../coordinate_manager.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>

namespace minkowski {

template <typename coordinate_type>
at::Tensor
PruningForwardCPU(at::Tensor const &in_feat,
                  at::Tensor const &keep,
                  CoordinateMapKey *p_in_map_key,
                  CoordinateMapKey *p_out_map_key,
                  cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
PruningBackwardCPU(at::Tensor &grad_out_feat,
                   CoordinateMapKey *p_in_map_key,
                   CoordinateMapKey *p_out_map_key,
                   cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
PruningForwardGPU(at::Tensor const &in_feat,
                  at::Tensor const &keep,
                  CoordinateMapKey *p_in_map_key,
                  CoordinateMapKey *p_out_map_key,
                  gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
PruningBackwardGPU(at::Tensor &grad_out_feat,
                   CoordinateMapKey *p_in_map_key,
                   CoordinateMapKey *p_out_map_key,
                   gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

class PruningFunction
    : public torch::autograd::Function<PruningFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor in_feat,
      at::Tensor mask,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> out_key,
      std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

} 

#endif 
