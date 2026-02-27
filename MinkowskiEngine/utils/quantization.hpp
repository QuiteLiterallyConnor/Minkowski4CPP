
#ifndef MINK_CPP_UTILS_QUANTIZATION_HPP
#define MINK_CPP_UTILS_QUANTIZATION_HPP

#include "../types.hpp"
#include "../coordinate_manager.hpp"
#include <torch/torch.h>
#include <vector>

namespace minkowski {

std::vector<at::Tensor> quantize_th(at::Tensor& coords);

std::vector<at::Tensor> quantize_label_th(at::Tensor coords, at::Tensor labels,
                                          int invalid_label);

struct SparseQuantizeResult {
  at::Tensor coordinates;                   
  c10::optional<at::Tensor> features;       
  c10::optional<at::Tensor> labels;         
  at::Tensor unique_index;                  
  at::Tensor inverse_mapping;              
};

SparseQuantizeResult sparse_quantize(
    at::Tensor coordinates,
    c10::optional<at::Tensor> features = c10::nullopt,
    c10::optional<at::Tensor> labels = c10::nullopt,
    int ignore_label = -100,
    c10::optional<double> quantization_size = c10::nullopt,
    torch::Device device = torch::kCPU);

std::pair<at::Tensor, at::Tensor> unique_coordinate_map(
    at::Tensor coordinates,
    stride_type tensor_stride = {});

} 

#endif 
