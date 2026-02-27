
#include "pooling_fn.hpp"

namespace minkowski {

static torch::IValue pack_stride(const stride_type &s) {
  std::vector<int64_t> v(s.begin(), s.end());
  return torch::IValue(v);
}
static stride_type unpack_stride(const torch::IValue &iv) {
  auto v = iv.toIntVector();
  return stride_type(v.begin(), v.end());
}

at::Tensor LocalPoolingFunction::forward(
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
    std::shared_ptr<CoordinateManager> manager) {

  input_features = input_features.contiguous();

  at::Tensor out_feat, num_nonzero;

  if (input_features.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        auto result = LocalPoolingForwardGPU(
            input_features, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, pooling_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
        out_feat = result.first;
        num_nonzero = result.second;
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        auto result = LocalPoolingForwardCPU<int32_t>(
            input_features, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, pooling_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
        out_feat = result.first;
        num_nonzero = result.second;
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  ctx->save_for_backward({input_features, num_nonzero, region_offsets});
  ctx->saved_data["pooling_mode"] = static_cast<int64_t>(pooling_mode);
  ctx->saved_data["kernel_size"] = pack_stride(kernel_size);
  ctx->saved_data["kernel_stride"] = pack_stride(kernel_stride);
  ctx->saved_data["kernel_dilation"] = pack_stride(kernel_dilation);
  ctx->saved_data["region_type"] = static_cast<int64_t>(region_type);
  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["out_key_stride"] = pack_stride(out_key->get_key().first);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["out_key_str_id"] = out_key->get_key().second;
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["out_key_coord_size"] = static_cast<int64_t>(out_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return out_feat;
}

torch::autograd::variable_list LocalPoolingFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto input_features = saved[0];
  auto num_nonzero = saved[1];
  auto region_offsets = saved[2];

  auto pooling_mode = static_cast<PoolingMode::Type>(ctx->saved_data["pooling_mode"].toInt());
  auto kernel_size = unpack_stride(ctx->saved_data["kernel_size"]);
  auto kernel_stride = unpack_stride(ctx->saved_data["kernel_stride"]);
  auto kernel_dilation = unpack_stride(ctx->saved_data["kernel_dilation"]);
  auto region_type = static_cast<RegionType::Type>(ctx->saved_data["region_type"].toInt());

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
        grad_in_feat = LocalPoolingBackwardGPU(
            input_features, grad_out_feat, num_nonzero,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, pooling_mode,
            &in_key, &out_key, mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        grad_in_feat = LocalPoolingBackwardCPU<int32_t>(
            input_features, grad_out_feat, num_nonzero,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, pooling_mode,
            &in_key, &out_key, mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor()};
}

at::Tensor LocalPoolingTransposeFunction::forward(
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
    std::shared_ptr<CoordinateManager> manager) {

  input_features = input_features.contiguous();

  at::Tensor out_feat, num_nonzero;

  if (input_features.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        auto result = LocalPoolingTransposeForwardGPU(
            input_features, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, expand_coordinates, pooling_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
        out_feat = result.first;
        num_nonzero = result.second;
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        auto result = LocalPoolingTransposeForwardCPU<int32_t>(
            input_features, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, expand_coordinates, pooling_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
        out_feat = result.first;
        num_nonzero = result.second;
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  ctx->save_for_backward({input_features, num_nonzero, region_offsets});
  ctx->saved_data["pooling_mode"] = static_cast<int64_t>(pooling_mode);
  ctx->saved_data["kernel_size"] = pack_stride(kernel_size);
  ctx->saved_data["kernel_stride"] = pack_stride(kernel_stride);
  ctx->saved_data["kernel_dilation"] = pack_stride(kernel_dilation);
  ctx->saved_data["region_type"] = static_cast<int64_t>(region_type);
  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["out_key_stride"] = pack_stride(out_key->get_key().first);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["out_key_str_id"] = out_key->get_key().second;
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["out_key_coord_size"] = static_cast<int64_t>(out_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return out_feat;
}

torch::autograd::variable_list LocalPoolingTransposeFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto input_features = saved[0];
  auto num_nonzero = saved[1];
  auto region_offsets = saved[2];

  auto pooling_mode = static_cast<PoolingMode::Type>(ctx->saved_data["pooling_mode"].toInt());
  auto kernel_size = unpack_stride(ctx->saved_data["kernel_size"]);
  auto kernel_stride = unpack_stride(ctx->saved_data["kernel_stride"]);
  auto kernel_dilation = unpack_stride(ctx->saved_data["kernel_dilation"]);
  auto region_type = static_cast<RegionType::Type>(ctx->saved_data["region_type"].toInt());

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
        grad_in_feat = LocalPoolingTransposeBackwardGPU(
            input_features, grad_out_feat, num_nonzero,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, pooling_mode,
            &in_key, &out_key, mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        grad_in_feat = LocalPoolingTransposeBackwardCPU<int32_t>(
            input_features, grad_out_feat, num_nonzero,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, pooling_mode,
            &in_key, &out_key, mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor()};
}

at::Tensor GlobalPoolingFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor input_features,
    PoolingMode::Type pooling_mode,
    std::shared_ptr<CoordinateMapKey> in_key,
    std::shared_ptr<CoordinateMapKey> out_key,
    std::shared_ptr<CoordinateManager> manager) {

  input_features = input_features.contiguous();

  at::Tensor out_feat, num_nonzero;

  if (input_features.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        auto result = GlobalPoolingForwardGPU(
            input_features, pooling_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
        out_feat = std::get<0>(result);
        num_nonzero = std::get<1>(result);
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        auto result = GlobalPoolingForwardCPU<int32_t>(
            input_features, pooling_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
        out_feat = std::get<0>(result);
        num_nonzero = std::get<1>(result);
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  ctx->save_for_backward({input_features, num_nonzero});
  ctx->saved_data["pooling_mode"] = static_cast<int64_t>(pooling_mode);
  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["out_key_stride"] = pack_stride(out_key->get_key().first);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["out_key_str_id"] = out_key->get_key().second;
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["out_key_coord_size"] = static_cast<int64_t>(out_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return out_feat;
}

torch::autograd::variable_list GlobalPoolingFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto input_features = saved[0];
  auto num_nonzero = saved[1];

  auto pooling_mode = static_cast<PoolingMode::Type>(ctx->saved_data["pooling_mode"].toInt());

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
        grad_in_feat = GlobalPoolingBackwardGPU(
            input_features, grad_out_feat, num_nonzero,
            pooling_mode, &in_key, &out_key, mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        grad_in_feat = GlobalPoolingBackwardCPU<int32_t>(
            input_features, grad_out_feat, num_nonzero,
            pooling_mode, &in_key, &out_key, mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

at::Tensor DirectMaxPoolingFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor in_map,
    at::Tensor out_map,
    at::Tensor in_feat,
    int64_t out_nrows,
    bool is_sorted) {

  auto [out_feat, max_mask] = max_pool_fw(in_map, out_map, in_feat,
                                           static_cast<int>(out_nrows), is_sorted);
  ctx->saved_data["in_nrows"] = static_cast<int64_t>(in_feat.size(0));
  ctx->save_for_backward({max_mask});
  return out_feat;
}

torch::autograd::variable_list DirectMaxPoolingFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto max_mask = saved[0];
  int in_nrows = static_cast<int>(ctx->saved_data["in_nrows"].toInt());

  at::Tensor grad_out_feat = grad_outputs[0].contiguous();
  at::Tensor grad_in_feat = max_pool_bw(grad_out_feat, max_mask, in_nrows);

  return {at::Tensor(), at::Tensor(), grad_in_feat, at::Tensor(), at::Tensor()};
}

} 
