
#include "quantization.hpp"
#include <cmath>
#include <algorithm>

namespace minkowski {

SparseQuantizeResult sparse_quantize(
    at::Tensor coordinates,
    c10::optional<at::Tensor> features,
    c10::optional<at::Tensor> labels,
    int ignore_label,
    c10::optional<double> quantization_size,
    torch::Device device) {

  TORCH_CHECK(coordinates.dim() == 2,
              "Coordinates must be a 2D matrix. Got shape: ",
              coordinates.sizes());

  bool use_feat = features.has_value();
  bool use_label = labels.has_value();

  if (use_feat) {
    TORCH_CHECK(features->dim() == 2, "Features must be a 2D matrix.");
    TORCH_CHECK(coordinates.size(0) == features->size(0),
                "Coordinate count (", coordinates.size(0),
                ") != feature count (", features->size(0), ")");
  }

  if (use_label) {
    TORCH_CHECK(coordinates.size(0) == labels->size(0),
                "Coordinate count must match label count.");
  }

  int64_t dimension = coordinates.size(1);

  at::Tensor discrete_coordinates;
  if (quantization_size.has_value()) {
    double qs = quantization_size.value();
    if (qs != 1.0) {
      discrete_coordinates = torch::floor(coordinates.to(torch::kFloat) / qs);
    } else {
      discrete_coordinates = torch::floor(coordinates.to(torch::kFloat));
    }
  } else {
    discrete_coordinates = torch::floor(coordinates.to(torch::kFloat));
  }
  discrete_coordinates = discrete_coordinates.to(torch::kInt32);

  SparseQuantizeResult result;

  if (use_label) {

    TORCH_CHECK(!discrete_coordinates.is_cuda(),
                "Quantization with labels requires CPU tensors.");
    TORCH_CHECK(!labels->is_cuda(),
                "Quantization with labels requires CPU tensors.");

    auto label_tensor = labels->to(torch::kInt32);
    auto maps = quantize_label_th(discrete_coordinates, label_tensor, ignore_label);

    auto unique_map = maps[0];
    auto inverse_map = maps[1];
    auto colabels = maps[2];

    result.coordinates = discrete_coordinates.index({unique_map});
    result.unique_index = unique_map;
    result.inverse_mapping = inverse_map;
    result.labels = colabels;

    if (use_feat) {
      result.features = features->index({unique_map});
    }
  } else {

    discrete_coordinates = discrete_coordinates.to(device);

    stride_type tensor_stride(dimension - 1, 1);

    CoordinateMapBackend::Type backend;
    if (device.is_cpu()) {
      backend = CoordinateMapBackend::CPU;
    } else {
      backend = CoordinateMapBackend::CUDA;
    }

    CoordinateManager mgr(static_cast<int>(dimension - 1), backend);
    auto [key, maps] = mgr.insert_and_map(discrete_coordinates, tensor_stride, "");
    auto& [unique_map, inverse_map] = maps;

    result.coordinates = discrete_coordinates.index({unique_map});
    result.unique_index = unique_map;
    result.inverse_mapping = inverse_map;

    if (use_feat) {
      result.features = features->index({unique_map});
    }
  }

  return result;
}

std::pair<at::Tensor, at::Tensor> unique_coordinate_map(
    at::Tensor coordinates,
    stride_type tensor_stride) {

  TORCH_CHECK(coordinates.dim() == 2, "Coordinates must be a 2D matrix.");

  int64_t D = coordinates.size(1) - 1;

  if (tensor_stride.empty()) {
    tensor_stride.assign(D, 1);
  }

  CoordinateMapBackend::Type backend;
  if (coordinates.is_cuda()) {
    backend = CoordinateMapBackend::CUDA;
  } else {
    backend = CoordinateMapBackend::CPU;
  }

  CoordinateManager mgr(static_cast<int>(D), backend);
  auto [key, maps] = mgr.insert_and_map(coordinates, tensor_stride, "");
  return maps;
}

} 
