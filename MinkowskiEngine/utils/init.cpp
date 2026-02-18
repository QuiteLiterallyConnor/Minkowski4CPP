
#include "init.hpp"
#include <cmath>
#include <stdexcept>

namespace minkowski {

std::pair<int64_t, int64_t> calculate_fan_in_and_fan_out(const at::Tensor& tensor) {
  int64_t dimensions = tensor.dim();
  TORCH_CHECK(dimensions >= 2,
              "Fan in and fan out cannot be computed for tensor with fewer "
              "than 2 dimensions. Got ", dimensions, " dimensions.");

  int64_t fan_in, fan_out;

  if (dimensions == 2) {

    fan_in = tensor.size(1);
    fan_out = tensor.size(0);
  } else {

    int64_t num_input_fmaps = tensor.size(1);
    int64_t num_output_fmaps = tensor.size(2);
    int64_t receptive_field_size = tensor.size(0);
    fan_in = num_input_fmaps * receptive_field_size;
    fan_out = num_output_fmaps * receptive_field_size;
  }

  return {fan_in, fan_out};
}

static int64_t calculate_correct_fan(const at::Tensor& tensor,
                                     const std::string& mode) {
  TORCH_CHECK(mode == "fan_in" || mode == "fan_out",
              "Mode '", mode, "' not supported. Use 'fan_in' or 'fan_out'.");

  auto [fan_in, fan_out] = calculate_fan_in_and_fan_out(tensor);
  return (mode == "fan_in") ? fan_in : fan_out;
}

at::Tensor& kaiming_normal_(
    at::Tensor& tensor,
    double a,
    const std::string& mode,
    const std::string& nonlinearity) {

  int64_t fan = calculate_correct_fan(tensor, mode);

  double gain;
  if (nonlinearity == "linear" || nonlinearity == "conv1d" ||
      nonlinearity == "conv2d" || nonlinearity == "conv3d" ||
      nonlinearity == "conv_transpose1d" || nonlinearity == "conv_transpose2d" ||
      nonlinearity == "conv_transpose3d" || nonlinearity == "sigmoid") {
    gain = 1.0;
  } else if (nonlinearity == "tanh") {
    gain = 5.0 / 3.0;
  } else if (nonlinearity == "relu") {
    gain = std::sqrt(2.0);
  } else if (nonlinearity == "leaky_relu") {
    gain = std::sqrt(2.0 / (1.0 + a * a));
  } else if (nonlinearity == "selu") {
    gain = 3.0 / 4.0;  
  } else {

    gain = std::sqrt(2.0 / (1.0 + a * a));
  }

  double std = gain / std::sqrt(static_cast<double>(fan));

  {
    torch::NoGradGuard no_grad;
    tensor.normal_(0.0, std);
  }

  return tensor;
}

} 
