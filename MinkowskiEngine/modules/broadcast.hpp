#ifndef MINK_CPP_MODULES_BROADCAST_HPP
#define MINK_CPP_MODULES_BROADCAST_HPP
#include "../types.hpp"
#include "../sparse_tensor.hpp"
#include "../coordinate_manager.hpp"
#include "../autograd/broadcast_fn.hpp"
#include "../src/coordinate_map_key.hpp"
#include <torch/torch.h>
#include <memory>
namespace minkowski {
class MinkowskiBroadcastBaseImpl
    : public torch::nn::Cloneable<MinkowskiBroadcastBaseImpl> {
public:
  explicit MinkowskiBroadcastBaseImpl(BroadcastMode::Type operation_type);
  void reset() override {}
  SparseTensor forward(const SparseTensor &input,
                       const SparseTensor &input_glob);
  BroadcastMode::Type operation_type_;
};
class MinkowskiBroadcastAdditionImpl
    : public torch::nn::Cloneable<MinkowskiBroadcastAdditionImpl> {
public:
  MinkowskiBroadcastAdditionImpl();
  void reset() override {}
  SparseTensor forward(const SparseTensor &input,
                       const SparseTensor &input_glob);
private:
  std::shared_ptr<MinkowskiBroadcastBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiBroadcastAddition);
class MinkowskiBroadcastMultiplicationImpl
    : public torch::nn::Cloneable<MinkowskiBroadcastMultiplicationImpl> {
public:
  MinkowskiBroadcastMultiplicationImpl();
  void reset() override {}
  SparseTensor forward(const SparseTensor &input,
                       const SparseTensor &input_glob);
private:
  std::shared_ptr<MinkowskiBroadcastBaseImpl> base_{nullptr};
};
TORCH_MODULE(MinkowskiBroadcastMultiplication);
class MinkowskiBroadcastImpl
    : public torch::nn::Cloneable<MinkowskiBroadcastImpl> {
public:
  MinkowskiBroadcastImpl() = default;
  void reset() override {}
  SparseTensor forward(const SparseTensor &input,
                       const SparseTensor &input_glob);
};
TORCH_MODULE(MinkowskiBroadcast);
class MinkowskiBroadcastConcatenationImpl
    : public torch::nn::Cloneable<MinkowskiBroadcastConcatenationImpl> {
public:
  MinkowskiBroadcastConcatenationImpl() = default;
  void reset() override {}
  SparseTensor forward(const SparseTensor &input,
                       const SparseTensor &input_glob);
};
TORCH_MODULE(MinkowskiBroadcastConcatenation);
} 
#endif 
