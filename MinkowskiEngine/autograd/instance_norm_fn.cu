
#include "instance_norm_fn.hpp"

namespace minkowski {

static torch::IValue pack_stride(const stride_type &s) {
  std::vector<int64_t> v(s.begin(), s.end());
  return torch::IValue(v);
}
static stride_type unpack_stride(const torch::IValue &iv) {
  auto v = iv.toIntVector();
  return stride_type(v.begin(), v.end());
}

static std::tuple<at::Tensor, at::Tensor>
dispatch_global_pooling_forward(
    at::Tensor const &in_feat,
    PoolingMode::Type pooling_mode,
    CoordinateMapKey *p_in_key,
    CoordinateMapKey *p_glob_key,
    CoordinateManager *manager) {

  std::tuple<at::Tensor, at::Tensor> result;

  if (in_feat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        result = GlobalPoolingForwardGPU(
            in_feat, pooling_mode, p_in_key, p_glob_key, mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        result = GlobalPoolingForwardCPU<int32_t>(
            in_feat, pooling_mode, p_in_key, p_glob_key, mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }
  return result;
}

static at::Tensor dispatch_broadcast_forward(
    at::Tensor const &in_feat,
    at::Tensor const &in_feat_glob,
    BroadcastMode::Type broadcast_mode,
    CoordinateMapKey *p_in_key,
    CoordinateMapKey *p_glob_key,
    CoordinateManager *manager) {

  at::Tensor result;

  if (in_feat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        result = BroadcastForwardGPU(
            in_feat, in_feat_glob, broadcast_mode,
            p_in_key, p_glob_key, mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        result = BroadcastForwardCPU<int32_t>(
            in_feat, in_feat_glob, broadcast_mode,
            p_in_key, p_glob_key, mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }
  return result;
}

at::Tensor InstanceNormFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor in_feat,
    std::shared_ptr<CoordinateMapKey> in_key,
    std::shared_ptr<CoordinateMapKey> glob_key,
    std::shared_ptr<CoordinateManager> manager,
    PoolingMode::Type gpooling_mode) {

  in_feat = in_feat.contiguous();

  auto *mgr = manager.get();

  auto [mean, num_nonzero] = dispatch_global_pooling_forward(
      in_feat, gpooling_mode, in_key.get(), glob_key.get(), mgr);

  auto centered_feat = dispatch_broadcast_forward(
      in_feat, -mean,
      BroadcastMode::ELEMENTWISE_ADDITON,
      in_key.get(), glob_key.get(), mgr);

  auto centered_sq = centered_feat.pow(2);
  auto [variance, num_nonzero2] = dispatch_global_pooling_forward(
      centered_sq, gpooling_mode, in_key.get(), glob_key.get(), mgr);

  constexpr float eps = 1e-8f;
  auto inv_std = 1.0f / (variance + eps).sqrt();

  auto norm_feat = dispatch_broadcast_forward(
      centered_feat, inv_std,
      BroadcastMode::ELEMENTWISE_MULTIPLICATION,
      in_key.get(), glob_key.get(), mgr);

  ctx->save_for_backward({inv_std, norm_feat});

  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["glob_key_stride"] = pack_stride(glob_key->get_key().first);
  ctx->saved_data["glob_key_str_id"] = glob_key->get_key().second;
  ctx->saved_data["glob_key_coord_size"] = static_cast<int64_t>(glob_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(mgr);
  ctx->saved_data["gpooling_mode"] = static_cast<int64_t>(gpooling_mode);

  return norm_feat;
}

torch::autograd::variable_list InstanceNormFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto inv_std = saved[0];
  auto norm_feat = saved[1];

  CoordinateMapKey in_key(ctx->saved_data["in_key_coord_size"].toInt(),
    {unpack_stride(ctx->saved_data["in_key_stride"]),
     ctx->saved_data["in_key_str_id"].toStringRef()});
  CoordinateMapKey glob_key(ctx->saved_data["glob_key_coord_size"].toInt(),
    {unpack_stride(ctx->saved_data["glob_key_stride"]),
     ctx->saved_data["glob_key_str_id"].toStringRef()});

  auto *manager = reinterpret_cast<CoordinateManager *>(
      ctx->saved_data["manager_ptr"].toInt());
  auto gpooling_mode = static_cast<PoolingMode::Type>(
      ctx->saved_data["gpooling_mode"].toInt());

  auto out_grad = grad_outputs[0].contiguous();

  auto [mean_dout, n1] = dispatch_global_pooling_forward(
      out_grad, gpooling_mode, &in_key, &glob_key, manager);

  auto dout_times_out = out_grad * norm_feat;
  auto [mean_dout_feat, n2] = dispatch_global_pooling_forward(
      dout_times_out, gpooling_mode, &in_key, &glob_key, manager);

  auto feat_mean_dout_feat = dispatch_broadcast_forward(
      norm_feat, mean_dout_feat,
      BroadcastMode::ELEMENTWISE_MULTIPLICATION,
      &in_key, &glob_key, manager);

  auto unnorm_din = dispatch_broadcast_forward(
      out_grad - feat_mean_dout_feat, -mean_dout,
      BroadcastMode::ELEMENTWISE_ADDITON,
      &in_key, &glob_key, manager);

  auto norm_din = dispatch_broadcast_forward(
      unnorm_din, inv_std,
      BroadcastMode::ELEMENTWISE_MULTIPLICATION,
      &in_key, &glob_key, manager);

  return {norm_din, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

} 
