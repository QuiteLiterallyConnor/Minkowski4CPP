
#ifndef MINK_CPP_FUNCTIONAL_HPP
#define MINK_CPP_FUNCTIONAL_HPP

#include "types.hpp"
#include "sparse_tensor.hpp"
#include <torch/torch.h>

namespace minkowski {
namespace functional {

inline SparseTensor wrap_tensor(const SparseTensor& input, at::Tensor features) {
  return SparseTensor(std::move(features),
                      input.coordinate_map_key(),
                      input.coordinate_manager());
}

inline SparseTensor relu(const SparseTensor& input, bool inplace = false) {
  return wrap_tensor(input, torch::relu(input.F()));
}

inline SparseTensor leaky_relu(const SparseTensor& input, double negative_slope = 0.01, bool inplace = false) {
  return wrap_tensor(input, torch::nn::functional::leaky_relu(
      input.F(), torch::nn::functional::LeakyReLUFuncOptions().negative_slope(negative_slope).inplace(inplace)));
}

inline SparseTensor elu(const SparseTensor& input, double alpha = 1.0, bool inplace = false) {
  return wrap_tensor(input, torch::elu(input.F(), alpha));
}

inline SparseTensor selu(const SparseTensor& input, bool inplace = false) {
  return wrap_tensor(input, torch::selu(input.F()));
}

inline SparseTensor celu(const SparseTensor& input, double alpha = 1.0, bool inplace = false) {
  return wrap_tensor(input, torch::celu(input.F(), alpha));
}

inline SparseTensor gelu(const SparseTensor& input) {
  return wrap_tensor(input, torch::gelu(input.F()));
}

inline SparseTensor sigmoid(const SparseTensor& input) {
  return wrap_tensor(input, torch::sigmoid(input.F()));
}

inline SparseTensor tanh(const SparseTensor& input) {
  return wrap_tensor(input, torch::tanh(input.F()));
}

inline SparseTensor hardtanh(const SparseTensor& input, double min_val = -1.0, double max_val = 1.0, bool inplace = false) {
  return wrap_tensor(input, torch::hardtanh(input.F(), min_val, max_val));
}

inline SparseTensor hardswish(const SparseTensor& input) {
  return wrap_tensor(input, at::hardswish(input.F()));
}

inline SparseTensor relu6(const SparseTensor& input, bool inplace = false) {
  return wrap_tensor(input, torch::nn::functional::relu6(input.F()));
}

inline SparseTensor hardsigmoid(const SparseTensor& input) {
  return wrap_tensor(input, at::hardsigmoid(input.F()));
}

inline SparseTensor silu(const SparseTensor& input) {
  return wrap_tensor(input, torch::silu(input.F()));
}

inline SparseTensor logsigmoid(const SparseTensor& input) {
  return wrap_tensor(input, torch::nn::functional::logsigmoid(input.F()));
}

inline SparseTensor softplus(const SparseTensor& input, double beta = 1.0, double threshold = 20.0) {
  return wrap_tensor(input, torch::nn::functional::softplus(
      input.F(), torch::nn::functional::SoftplusFuncOptions().beta(beta).threshold(threshold)));
}

inline SparseTensor softsign(const SparseTensor& input) {
  return wrap_tensor(input, torch::nn::functional::softsign(input.F()));
}

inline SparseTensor softshrink(const SparseTensor& input, double lambd = 0.5) {
  return wrap_tensor(input, torch::nn::functional::softshrink(
      input.F(), torch::nn::functional::SoftshrinkFuncOptions().lambda(lambd)));
}

inline SparseTensor hardshrink(const SparseTensor& input, double lambd = 0.5) {
  return wrap_tensor(input, torch::nn::functional::hardshrink(
      input.F(), torch::nn::functional::HardshrinkFuncOptions().lambda(lambd)));
}

inline SparseTensor tanhshrink(const SparseTensor& input) {
  return wrap_tensor(input, torch::nn::functional::tanhshrink(input.F()));
}

inline SparseTensor softmax(const SparseTensor& input, int64_t dim = -1) {
  return wrap_tensor(input, torch::softmax(input.F(), dim));
}

inline SparseTensor softmin(const SparseTensor& input, int64_t dim = -1) {
  return wrap_tensor(input, torch::softmax(-input.F(), dim));
}

inline SparseTensor log_softmax(const SparseTensor& input, int64_t dim = -1) {
  return wrap_tensor(input, torch::log_softmax(input.F(), dim));
}

inline SparseTensor glu(const SparseTensor& input, int64_t dim = -1) {
  return wrap_tensor(input, torch::nn::functional::glu(
      input.F(), torch::nn::functional::GLUFuncOptions().dim(dim)));
}

inline SparseTensor prelu(const SparseTensor& input, const at::Tensor& weight) {
  return wrap_tensor(input, torch::prelu(input.F(), weight));
}

inline SparseTensor rrelu(const SparseTensor& input, double lower = 1.0/8.0, double upper = 1.0/3.0, bool training = false) {
  return wrap_tensor(input, torch::nn::functional::rrelu(
      input.F(), torch::nn::functional::RReLUFuncOptions().lower(lower).upper(upper).training(training)));
}

inline SparseTensor threshold(const SparseTensor& input, double threshold_val, double value, bool inplace = false) {
  return wrap_tensor(input, torch::threshold(input.F(), threshold_val, value));
}

inline SparseTensor batch_norm(
    const SparseTensor& input,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const c10::optional<at::Tensor>& weight = c10::nullopt,
    const c10::optional<at::Tensor>& bias = c10::nullopt,
    bool training = false,
    double momentum = 0.1,
    double eps = 1e-5) {
  auto opts = torch::nn::functional::BatchNormFuncOptions()
      .training(training).momentum(momentum).eps(eps);
  if (weight.has_value()) opts.weight(weight.value());
  if (bias.has_value()) opts.bias(bias.value());
  return wrap_tensor(input, torch::nn::functional::batch_norm(
      input.F(), running_mean, running_var, opts));
}

inline SparseTensor normalize(const SparseTensor& input, double p = 2.0, int64_t dim = 1, double eps = 1e-12) {
  return wrap_tensor(input, torch::nn::functional::normalize(
      input.F(), torch::nn::functional::NormalizeFuncOptions().p(p).dim(dim).eps(eps)));
}

inline SparseTensor linear(const SparseTensor& input, const at::Tensor& weight,
                           const c10::optional<at::Tensor>& bias = c10::nullopt) {
  at::Tensor result;
  if (bias.has_value()) {
    result = at::linear(input.F(), weight, bias.value());
  } else {
    result = at::linear(input.F(), weight);
  }
  return wrap_tensor(input, result);
}

inline SparseTensor dropout(const SparseTensor& input, double p = 0.5, bool training = true, bool inplace = false) {
  return wrap_tensor(input, torch::nn::functional::dropout(
      input.F(), torch::nn::functional::DropoutFuncOptions().p(p).training(training).inplace(inplace)));
}

inline SparseTensor alpha_dropout(const SparseTensor& input, double p = 0.5, bool training = false, bool inplace = false) {
  return wrap_tensor(input, torch::nn::functional::alpha_dropout(
      input.F(), torch::nn::functional::AlphaDropoutFuncOptions().p(p).training(training).inplace(inplace)));
}

inline at::Tensor binary_cross_entropy(const SparseTensor& input, const at::Tensor& target,
                                       const c10::optional<at::Tensor>& weight = c10::nullopt) {
  auto opts = torch::nn::functional::BinaryCrossEntropyFuncOptions();
  if (weight.has_value()) opts.weight(weight.value());
  return torch::nn::functional::binary_cross_entropy(input.F(), target, opts);
}

inline at::Tensor binary_cross_entropy_with_logits(const SparseTensor& input, const at::Tensor& target,
                                                    const c10::optional<at::Tensor>& weight = c10::nullopt) {
  auto opts = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions();
  if (weight.has_value()) opts.weight(weight.value());
  return torch::nn::functional::binary_cross_entropy_with_logits(input.F(), target, opts);
}

inline at::Tensor cross_entropy(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::cross_entropy(input.F(), target);
}

inline at::Tensor mse_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::mse_loss(input.F(), target);
}

inline at::Tensor l1_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::l1_loss(input.F(), target);
}

inline at::Tensor smooth_l1_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::smooth_l1_loss(input.F(), target);
}

inline at::Tensor nll_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::nll_loss(input.F(), target);
}

inline at::Tensor kl_div(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::kl_div(input.F(), target);
}

inline at::Tensor hinge_embedding_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::hinge_embedding_loss(input.F(), target);
}

inline at::Tensor poisson_nll_loss(const SparseTensor& input, const at::Tensor& target,
                                    bool log_input = true, bool full = false, double eps = 1e-8) {
  return torch::nn::functional::poisson_nll_loss(
      input.F(), target,
      torch::nn::functional::PoissonNLLLossFuncOptions().log_input(log_input).full(full).eps(eps));
}

inline at::Tensor multi_margin_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::multi_margin_loss(input.F(), target);
}

inline at::Tensor multilabel_margin_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::multilabel_margin_loss(input.F(), target);
}

inline at::Tensor multilabel_soft_margin_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::multilabel_soft_margin_loss(input.F(), target);
}

inline at::Tensor soft_margin_loss(const SparseTensor& input, const at::Tensor& target) {
  return torch::nn::functional::soft_margin_loss(input.F(), target);
}

} 
} 

#endif 
