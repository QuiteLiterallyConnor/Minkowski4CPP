
#include "pruning_fn.hpp"

namespace minkowski {

static torch::IValue pack_stride(const stride_type &s) {
  std::vector<int64_t> v(s.begin(), s.end());
  return torch::IValue(v);
}
static stride_type unpack_stride(const torch::IValue &iv) {
  auto v = iv.toIntVector();
  return stride_type(v.begin(), v.end());
}

at::Tensor PruningFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor in_feat,
    at::Tensor mask,
    std::shared_ptr<CoordinateMapKey> in_key,
    std::shared_ptr<CoordinateMapKey> out_key,
    std::shared_ptr<CoordinateManager> manager) {

  in_feat = in_feat.contiguous();

  at::Tensor out_feat;

  if (in_feat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        out_feat = PruningForwardGPU(
            in_feat, mask, in_key.get(), out_key.get(), mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        out_feat = PruningForwardCPU<int32_t>(
            in_feat, mask, in_key.get(), out_key.get(), mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["out_key_stride"] = pack_stride(out_key->get_key().first);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["out_key_str_id"] = out_key->get_key().second;
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["out_key_coord_size"] = static_cast<int64_t>(out_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return out_feat;
}

torch::autograd::variable_list PruningFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  CoordinateMapKey in_key(ctx->saved_data["in_key_coord_size"].toInt(),
    {unpack_stride(ctx->saved_data["in_key_stride"]),
     ctx->saved_data["in_key_str_id"].toStringRef()});
  CoordinateMapKey out_key(ctx->saved_data["out_key_coord_size"].toInt(),
    {unpack_stride(ctx->saved_data["out_key_stride"]),
     ctx->saved_data["out_key_str_id"].toStringRef()});

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
        grad_in_feat = PruningBackwardGPU(
            grad_out_feat, &in_key, &out_key, mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        grad_in_feat = PruningBackwardCPU<int32_t>(
            grad_out_feat, &in_key, &out_key, mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

} 
