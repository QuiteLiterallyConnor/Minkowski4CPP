
#ifndef MINK_CPP_SERIALIZATION_HPP
#define MINK_CPP_SERIALIZATION_HPP

#include "sparse_tensor.hpp"
#include "coordinate_manager.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

namespace minkowski {

template <typename Module>
void save_model(const Module &module, const std::string &path) {
  torch::save(module, path);
}

template <typename Module>
void load_model(Module &module, const std::string &path) {
  torch::load(module, path);
}

template <typename ModuleImpl>
void save_model(const std::shared_ptr<ModuleImpl> &module,
                const std::string &path) {
  torch::save(module, path);
}

template <typename ModuleImpl>
void load_model(std::shared_ptr<ModuleImpl> &module,
                const std::string &path) {
  torch::load(module, path);
}

void save_tensors(const std::vector<std::pair<std::string, at::Tensor>> &tensors,
                  const std::string &path);

std::vector<std::pair<std::string, at::Tensor>>
load_tensors(const std::string &path);

struct SparseTensorData {
  at::Tensor coordinates; 
  at::Tensor features;    
  std::vector<int64_t> tensor_stride;
  int dimension;

  SparseTensorData() : dimension(0) {}

  SparseTensorData(const at::Tensor &coords, const at::Tensor &feats,
                   const std::vector<int64_t> &stride, int dim)
      : coordinates(coords), features(feats), tensor_stride(stride),
        dimension(dim) {}
};

void save_sparse_tensor(const SparseTensor &stensor, const std::string &path);

SparseTensorData load_sparse_tensor_data(const std::string &path);

SparseTensor reconstruct_sparse_tensor(
    const SparseTensorData &data,
    std::shared_ptr<CoordinateManager> manager = nullptr);

SparseTensor load_sparse_tensor(const std::string &path);

void save_sparse_tensors(const std::vector<SparseTensor> &stensors,
                         const std::string &path);

std::vector<SparseTensorData>
load_sparse_tensors_data(const std::string &path);

} 

#endif 
