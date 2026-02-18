#include "convolution_fn.hpp"
#include <stdexcept>

namespace minkowski {

static torch::IValue pack_stride(const stride_type &s) {
  std::vector<int64_t> v(s.begin(), s.end());
  return torch::IValue(v);
}

static stride_type unpack_stride(const torch::IValue &iv) {
  auto v = iv.toIntVector();
  stride_type s(v.begin(), v.end());
  return s;
}

at::Tensor ConvolutionFunction::forward(
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
    std::shared_ptr<CoordinateManager> manager) {

  input_features = input_features.contiguous();

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
        out_feat = ConvolutionForwardGPU(
            input_features, kernel, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, expand_coordinates, conv_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        out_feat = ConvolutionForwardCPU<int32_t>(
            input_features, kernel, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, expand_coordinates, conv_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  ctx->save_for_backward({input_features, kernel, region_offsets});
  ctx->saved_data["kernel_size"] = pack_stride(kernel_size);
  ctx->saved_data["kernel_stride"] = pack_stride(kernel_stride);
  ctx->saved_data["kernel_dilation"] = pack_stride(kernel_dilation);
  ctx->saved_data["region_type"] = static_cast<int64_t>(region_type);
  ctx->saved_data["conv_mode"] = static_cast<int64_t>(conv_mode);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["out_key_str_id"] = out_key->get_key().second;

  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["out_key_stride"] = pack_stride(out_key->get_key().first);
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["out_key_coord_size"] = static_cast<int64_t>(out_key->get_coordinate_size());

  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return out_feat;
}

torch::autograd::variable_list ConvolutionFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto input_features = saved[0];
  auto kernel = saved[1];
  auto region_offsets = saved[2];

  auto kernel_size = unpack_stride(ctx->saved_data["kernel_size"]);
  auto kernel_stride = unpack_stride(ctx->saved_data["kernel_stride"]);
  auto kernel_dilation = unpack_stride(ctx->saved_data["kernel_dilation"]);
  auto region_type = static_cast<RegionType::Type>(ctx->saved_data["region_type"].toInt());
  auto conv_mode = static_cast<ConvolutionMode::Type>(ctx->saved_data["conv_mode"].toInt());

  auto in_key_stride = unpack_stride(ctx->saved_data["in_key_stride"]);
  auto out_key_stride = unpack_stride(ctx->saved_data["out_key_stride"]);
  auto in_key_str_id = ctx->saved_data["in_key_str_id"].toStringRef();
  auto out_key_str_id = ctx->saved_data["out_key_str_id"].toStringRef();
  auto in_coord_size = ctx->saved_data["in_key_coord_size"].toInt();
  auto out_coord_size = ctx->saved_data["out_key_coord_size"].toInt();

  CoordinateMapKey in_key(in_coord_size, {in_key_stride, in_key_str_id});
  CoordinateMapKey out_key(out_coord_size, {out_key_stride, out_key_str_id});

  auto *manager = reinterpret_cast<CoordinateManager *>(
      ctx->saved_data["manager_ptr"].toInt());

  at::Tensor grad_out_feat = grad_outputs[0].contiguous();

  at::Tensor grad_in_feat, grad_kernel;

  if (grad_out_feat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        auto result = ConvolutionBackwardGPU(
            input_features, grad_out_feat, kernel,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, conv_mode,
            &in_key, &out_key, mgr_ptr.get());
        grad_in_feat = result.first;
        grad_kernel = result.second;
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        auto result = ConvolutionBackwardCPU<int32_t>(
            input_features, grad_out_feat, kernel,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, conv_mode,
            &in_key, &out_key, mgr_ptr.get());
        grad_in_feat = result.first;
        grad_kernel = result.second;
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, grad_kernel, at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

at::Tensor ConvolutionTransposeFunction::forward(
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
    std::shared_ptr<CoordinateManager> manager) {

  input_features = input_features.contiguous();

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
        out_feat = ConvolutionTransposeForwardGPU(
            input_features, kernel, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, expand_coordinates, conv_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        out_feat = ConvolutionTransposeForwardCPU<int32_t>(
            input_features, kernel, kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, expand_coordinates, conv_mode,
            in_key.get(), out_key.get(), mgr_ptr.get());
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  ctx->save_for_backward({input_features, kernel, region_offsets});
  ctx->saved_data["kernel_size"] = pack_stride(kernel_size);
  ctx->saved_data["kernel_stride"] = pack_stride(kernel_stride);
  ctx->saved_data["kernel_dilation"] = pack_stride(kernel_dilation);
  ctx->saved_data["region_type"] = static_cast<int64_t>(region_type);
  ctx->saved_data["conv_mode"] = static_cast<int64_t>(conv_mode);
  ctx->saved_data["in_key_str_id"] = in_key->get_key().second;
  ctx->saved_data["out_key_str_id"] = out_key->get_key().second;
  ctx->saved_data["in_key_stride"] = pack_stride(in_key->get_key().first);
  ctx->saved_data["out_key_stride"] = pack_stride(out_key->get_key().first);
  ctx->saved_data["in_key_coord_size"] = static_cast<int64_t>(in_key->get_coordinate_size());
  ctx->saved_data["out_key_coord_size"] = static_cast<int64_t>(out_key->get_coordinate_size());
  ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get());

  return out_feat;
}

torch::autograd::variable_list ConvolutionTransposeFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto input_features = saved[0];
  auto kernel = saved[1];
  auto region_offsets = saved[2];

  auto kernel_size = unpack_stride(ctx->saved_data["kernel_size"]);
  auto kernel_stride = unpack_stride(ctx->saved_data["kernel_stride"]);
  auto kernel_dilation = unpack_stride(ctx->saved_data["kernel_dilation"]);
  auto region_type = static_cast<RegionType::Type>(ctx->saved_data["region_type"].toInt());
  auto conv_mode = static_cast<ConvolutionMode::Type>(ctx->saved_data["conv_mode"].toInt());

  auto in_key_stride = unpack_stride(ctx->saved_data["in_key_stride"]);
  auto out_key_stride = unpack_stride(ctx->saved_data["out_key_stride"]);
  auto in_key_str_id = ctx->saved_data["in_key_str_id"].toStringRef();
  auto out_key_str_id = ctx->saved_data["out_key_str_id"].toStringRef();
  auto in_coord_size = ctx->saved_data["in_key_coord_size"].toInt();
  auto out_coord_size = ctx->saved_data["out_key_coord_size"].toInt();

  CoordinateMapKey in_key(in_coord_size, {in_key_stride, in_key_str_id});
  CoordinateMapKey out_key(out_coord_size, {out_key_stride, out_key_str_id});

  auto *manager = reinterpret_cast<CoordinateManager *>(
      ctx->saved_data["manager_ptr"].toInt());

  at::Tensor grad_out_feat = grad_outputs[0].contiguous();

  at::Tensor grad_in_feat, grad_kernel;

  if (grad_out_feat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        TORCH_CHECK(false, "Cannot use CPU manager with CUDA tensors");
      } else {
        auto result = ConvolutionTransposeBackwardGPU(
            input_features, grad_out_feat, kernel,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, conv_mode,
            &in_key, &out_key, mgr_ptr.get());
        grad_in_feat = result.first;
        grad_kernel = result.second;
      }
    }, manager->get_manager_variant());
#endif
  } else {
    std::visit([&](auto &mgr_ptr) {
      using MgrType = std::decay_t<decltype(*mgr_ptr)>;
      if constexpr (std::is_same_v<MgrType, cpu_manager_type<int32_t>>) {
        auto result = ConvolutionTransposeBackwardCPU<int32_t>(
            input_features, grad_out_feat, kernel,
            kernel_size, kernel_stride, kernel_dilation,
            region_type, region_offsets, conv_mode,
            &in_key, &out_key, mgr_ptr.get());
        grad_in_feat = result.first;
        grad_kernel = result.second;
      } else {
        TORCH_CHECK(false, "Cannot use GPU manager with CPU tensors");
      }
    }, manager->get_manager_variant());
  }

  return {grad_in_feat, grad_kernel, at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
}

} 
