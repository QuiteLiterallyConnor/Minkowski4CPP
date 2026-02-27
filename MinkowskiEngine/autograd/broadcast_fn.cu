#include "broadcast_fn.hpp"

namespace minkowski {

static torch::IValue pack_stride(const stride_type &s) {
  std::vector<int64_t> v(s.begin(), s.end());
  return torch::IValue(v);
}
static stride_type unpack_stride(const torch::IValue &iv) {
  auto v = iv.toIntVector();
  return stride_type(v.begin(), v.end());
}

at::Tensor BroadcastFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor input_features,
    at::Tensor input_features_global,
    BroadcastMode::Type operation_type,
    std::shared_ptr<CoordinateMapKey> in_key,
    std::shared_ptr<CoordinateMapKey> glob_key,
    std::shared_ptr<CoordinateManager> manager) {

  input_features = input_features.contiguous();
  input_features_global = input_features_global.contiguous();

  at::Tensor out_feat;

  if (input_features.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        out_feat = BroadcastForwardGPU(
            input_features, input_features_global, operation_type,
            in_key.get(), glob_key.get(), mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        out_feat = BroadcastForwardCPU<int32_t>(
            input_features, input_features_global, operation_type,
            in_key.get(), glob_key.get(), mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  ctx->save_for_backward({input_features, input_features_global});
  ctx->saved_data["operation_type"] = static_cast<int64_t>(operation_type);
  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["glob_key_stride"] = pack_stride(glob_key->get_key().first);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["glob_key_str_id"] = glob_key->get_key().second;
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["glob_key_coord_size"] = static_cast<int64_t>(glob_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return out_feat;
}

torch::autograd::variable_list BroadcastFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto input_features = saved[0];
  auto input_features_global = saved[1];

  auto operation_type = static_cast<BroadcastMode::Type>(ctx->saved_data["operation_type"].toInt());

  CoordinateMapKey in_key(ctx->saved_data["in_key_coord_size"].toInt(),
    {unpack_stride(ctx->saved_data["in_key_stride"]),
     ctx->saved_data["in_key_str_id"].toStringRef()});
  CoordinateMapKey glob_key(ctx->saved_data["glob_key_coord_size"].toInt(),
    {unpack_stride(ctx->saved_data["glob_key_stride"]),
     ctx->saved_data["glob_key_str_id"].toStringRef()});

  auto *manager = reinterpret_cast<CoordinateManager *>(
      ctx->saved_data["manager_ptr"].toInt());

  at::Tensor grad_out_feat = grad_outputs[0].contiguous();
  at::Tensor grad_in_feat, grad_in_feat_glob;

  if (grad_out_feat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        auto result = BroadcastBackwardGPU(
            input_features, input_features_global, grad_out_feat,
            operation_type, &in_key, &glob_key, mgr_ptr.get());
        grad_in_feat = result.first;
        grad_in_feat_glob = result.second;
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        auto result = BroadcastBackwardCPU<int32_t>(
            input_features, input_features_global, grad_out_feat,
            operation_type, &in_key, &glob_key, mgr_ptr.get());
        grad_in_feat = result.first;
        grad_in_feat_glob = result.second;
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, grad_in_feat_glob, at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor()};
}

} 
