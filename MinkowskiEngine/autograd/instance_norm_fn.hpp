
#ifndef MINK_CPP_AUTOGRAD_INSTANCE_NORM_FN_HPP
#define MINK_CPP_AUTOGRAD_INSTANCE_NORM_FN_HPP

#include "../types.hpp"
#include "../coordinate_manager.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>

namespace minkowski {

template <typename coordinate_type>
std::tuple<at::Tensor, at::Tensor>
GlobalPoolingForwardCPU(at::Tensor const &in_feat,
                        PoolingMode::Type const pooling_mode,
                        CoordinateMapKey *p_in_map_key,
                        CoordinateMapKey *p_out_map_key,
                        cpu_manager_type<coordinate_type> *p_map_manager);

template <typename coordinate_type>
at::Tensor
BroadcastForwardCPU(at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
                    BroadcastMode::Type const broadcast_mode,
                    CoordinateMapKey *p_in_map_key,
                    CoordinateMapKey *p_glob_map_key,
                    cpu_manager_type<coordinate_type> *p_map_manager);

#ifndef CPU_ONLY
template <typename coordinate_type, template <typename C> class TemplatedAllocator>
std::tuple<at::Tensor, at::Tensor>
GlobalPoolingForwardGPU(at::Tensor const &in_feat,
                        PoolingMode::Type const pooling_mode,
                        CoordinateMapKey *p_in_map_key,
                        CoordinateMapKey *p_out_map_key,
                        gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);

template <typename coordinate_type, template <typename C> class TemplatedAllocator>
at::Tensor
BroadcastForwardGPU(at::Tensor const &in_feat, at::Tensor const &in_feat_glob,
                    BroadcastMode::Type const broadcast_mode,
                    CoordinateMapKey *p_in_map_key,
                    CoordinateMapKey *p_glob_map_key,
                    gpu_manager_type<coordinate_type, TemplatedAllocator> *p_map_manager);
#endif

class InstanceNormFunction
    : public torch::autograd::Function<InstanceNormFunction> {
public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx,
      at::Tensor in_feat,
      std::shared_ptr<CoordinateMapKey> in_key,
      std::shared_ptr<CoordinateMapKey> glob_key,
      std::shared_ptr<CoordinateManager> manager,
      PoolingMode::Type gpooling_mode = PoolingMode::GLOBAL_AVG_POOLING_KERNEL);

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext *ctx,
      torch::autograd::variable_list grad_outputs);
};

} 

#endif 
