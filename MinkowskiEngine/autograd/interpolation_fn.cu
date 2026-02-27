
#include "interpolation_fn.hpp"

namespace minkowski {

static torch::IValue pack_stride(const stride_type &s) {
  std::vector<int64_t> v(s.begin(), s.end());
  return torch::IValue(v);
}
static stride_type unpack_stride(const torch::IValue &iv) {
  auto v = iv.toIntVector();
  return stride_type(v.begin(), v.end());
}

torch::autograd::variable_list InterpolationFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor input_features,
    at::Tensor tfield,
    std::shared_ptr<CoordinateMapKey> in_key,
    std::shared_ptr<CoordinateManager> manager) {

  input_features = input_features.contiguous();

  std::vector<at::Tensor> results;

  if (input_features.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        results = InterpolationForwardGPU(
            input_features, tfield, in_key.get(), mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        results = InterpolationForwardCPU<int32_t>(
            input_features, tfield, in_key.get(), mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  TORCH_CHECK(results.size() == 4, "InterpolationForward should return 4 tensors");
  auto out_feat = results[0];
  auto in_map = results[1];
  auto out_map = results[2];
  auto weights = results[3];

  ctx->save_for_backward({in_map, out_map, weights});
  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return {out_feat, in_map, out_map, weights};
}

torch::autograd::variable_list InterpolationFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto in_map = saved[0];
  auto out_map = saved[1];
  auto weights = saved[2];

  CoordinateMapKey in_key(ctx->saved_data["in_key_coord_size"].toInt(),
    {unpack_stride(ctx->saved_data["in_key_stride"]),
     ctx->saved_data["in_key_str_id"].toStringRef()});

  auto *manager = reinterpret_cast<CoordinateManager *>(
      ctx->saved_data["manager_ptr"].toInt());

  at::Tensor grad_out_feat = grad_outputs[0].contiguous();
  at::Tensor grad_in_feat;

  if (grad_out_feat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        grad_in_feat = InterpolationBackwardGPU(
            grad_out_feat, in_map, out_map, weights,
            &in_key, mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        grad_in_feat = InterpolationBackwardCPU<int32_t>(
            grad_out_feat, in_map, out_map, weights,
            &in_key, mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, at::Tensor(), at::Tensor(), at::Tensor()};
}

} 
