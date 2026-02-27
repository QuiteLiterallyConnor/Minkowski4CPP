
#ifndef MINK_CPP_UTILS_INIT_HPP
#define MINK_CPP_UTILS_INIT_HPP

#include <torch/torch.h>
#include <string>

namespace minkowski {

std::pair<int64_t, int64_t> calculate_fan_in_and_fan_out(const at::Tensor& tensor);

at::Tensor& kaiming_normal_(
    at::Tensor& tensor,
    double a = 0.0,
    const std::string& mode = "fan_in",
    const std::string& nonlinearity = "leaky_relu");

} 

#endif 
