
#include "serialization.hpp"
#include <fstream>

namespace minkowski {

void save_tensors(
    const std::vector<std::pair<std::string, at::Tensor>> &tensors,
    const std::string &path) {

  torch::serialize::OutputArchive archive;
  for (auto &[name, tensor] : tensors) {
    archive.write(name, tensor);
  }
  archive.save_to(path);
}

std::vector<std::pair<std::string, at::Tensor>>
load_tensors(const std::string &path) {

  torch::serialize::InputArchive archive;
  archive.load_from(path);

  at::Tensor count_tensor;
  archive.read("count", count_tensor);
  int64_t count = count_tensor.item<int64_t>();

  std::vector<std::pair<std::string, at::Tensor>> result;
  result.reserve(count);

  for (int64_t i = 0; i < count; i++) {
    std::string key = "tensor_" + std::to_string(i);
    at::Tensor t;
    archive.read(key, t);

    result.emplace_back(key, std::move(t));
  }
  return result;
}

void save_sparse_tensor(const SparseTensor &stensor, const std::string &path) {
  torch::serialize::OutputArchive archive;

  archive.write("coordinates", stensor.C().contiguous());
  archive.write("features", stensor.F().contiguous());

  archive.write("dimension",
                torch::tensor({static_cast<int64_t>(stensor.dimension())}));

  auto stride = stensor.tensor_stride();
  auto stride_tensor =
      torch::tensor(std::vector<int64_t>(stride.begin(), stride.end()));
  archive.write("tensor_stride", stride_tensor);

  archive.save_to(path);
}

SparseTensorData load_sparse_tensor_data(const std::string &path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);

  SparseTensorData data;
  archive.read("coordinates", data.coordinates);
  archive.read("features", data.features);

  at::Tensor dim_tensor;
  archive.read("dimension", dim_tensor);
  data.dimension = static_cast<int>(dim_tensor.item<int64_t>());

  at::Tensor stride_tensor;
  archive.read("tensor_stride", stride_tensor);
  auto stride_accessor = stride_tensor.accessor<int64_t, 1>();
  data.tensor_stride.resize(stride_accessor.size(0));
  for (int64_t i = 0; i < stride_accessor.size(0); i++) {
    data.tensor_stride[i] = stride_accessor[i];
  }

  return data;
}

SparseTensor reconstruct_sparse_tensor(
    const SparseTensorData &data,
    std::shared_ptr<CoordinateManager> manager) {
  if (!manager) {
    manager = std::make_shared<CoordinateManager>(
        data.dimension, CoordinateMapBackend::CPU);
  }
  return SparseTensor(data.features, data.coordinates, manager);
}

SparseTensor load_sparse_tensor(const std::string &path) {
  auto data = load_sparse_tensor_data(path);
  return reconstruct_sparse_tensor(data);
}

void save_sparse_tensors(const std::vector<SparseTensor> &stensors,
                         const std::string &path) {
  torch::serialize::OutputArchive archive;

  archive.write("count",
                torch::tensor({static_cast<int64_t>(stensors.size())}));

  for (size_t i = 0; i < stensors.size(); i++) {
    std::string prefix = "st_" + std::to_string(i) + "_";
    archive.write(prefix + "coordinates",
                  stensors[i].C().contiguous());
    archive.write(prefix + "features",
                  stensors[i].F().contiguous());
    archive.write(
        prefix + "dimension",
        torch::tensor({static_cast<int64_t>(stensors[i].dimension())}));

    auto stride = stensors[i].tensor_stride();
    archive.write(
        prefix + "tensor_stride",
        torch::tensor(std::vector<int64_t>(stride.begin(), stride.end())));
  }

  archive.save_to(path);
}

std::vector<SparseTensorData>
load_sparse_tensors_data(const std::string &path) {
  torch::serialize::InputArchive archive;
  archive.load_from(path);

  at::Tensor count_tensor;
  archive.read("count", count_tensor);
  int64_t count = count_tensor.item<int64_t>();

  std::vector<SparseTensorData> result;
  result.reserve(count);

  for (int64_t i = 0; i < count; i++) {
    std::string prefix = "st_" + std::to_string(i) + "_";
    SparseTensorData data;

    archive.read(prefix + "coordinates", data.coordinates);
    archive.read(prefix + "features", data.features);

    at::Tensor dim_tensor;
    archive.read(prefix + "dimension", dim_tensor);
    data.dimension = static_cast<int>(dim_tensor.item<int64_t>());

    at::Tensor stride_tensor;
    archive.read(prefix + "tensor_stride", stride_tensor);
    auto acc = stride_tensor.accessor<int64_t, 1>();
    data.tensor_stride.resize(acc.size(0));
    for (int64_t j = 0; j < acc.size(0); j++) {
      data.tensor_stride[j] = acc[j];
    }

    result.push_back(std::move(data));
  }

  return result;
}

} 
