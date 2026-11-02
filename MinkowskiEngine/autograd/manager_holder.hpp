#ifndef MINK_CPP_AUTOGRAD_MANAGER_HOLDER_HPP
#define MINK_CPP_AUTOGRAD_MANAGER_HOLDER_HPP

// Keeps the CoordinateManager alive for as long as an autograd node needs it.
//
// The autograd functions used to store the manager as a raw pointer cast to int64_t in
// ctx->saved_data. If every SparseTensor referring to the manager went out of scope before
// backward() (e.g. `conv->forward(SparseTensor(f, c)).F().sum().backward()`), backward
// dereferenced a destroyed manager. The manager is now stored as an owning reference inside an
// IValue capsule, so the graph keeps it alive until the node itself is released.

#include <memory>
#include <utility>

#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>

#include "../coordinate_manager.hpp"

namespace minkowski {
namespace autograd_detail {

struct ManagerHolder final : torch::CustomClassHolder {
  explicit ManagerHolder(std::shared_ptr<CoordinateManager> m)
      : manager(std::move(m)) {}
  std::shared_ptr<CoordinateManager> manager;
};

// Value for ctx->saved_data that owns a reference to `manager`.
inline c10::IValue pack_manager(std::shared_ptr<CoordinateManager> manager) {
  TORCH_CHECK(manager, "autograd function called without a coordinate manager");
  return c10::IValue::make_capsule(
      c10::make_intrusive<ManagerHolder>(std::move(manager)));
}

// Inverse of pack_manager. The returned reference keeps the manager alive during backward.
inline std::shared_ptr<CoordinateManager> unpack_manager(const c10::IValue &value) {
  TORCH_CHECK(value.isCapsule(), "saved coordinate manager is missing");
  auto holder =
      c10::static_intrusive_pointer_cast<ManagerHolder>(value.toCapsule());
  TORCH_CHECK(holder && holder->manager, "saved coordinate manager is empty");
  return holder->manager;
}

} // namespace autograd_detail
} // namespace minkowski

#endif // MINK_CPP_AUTOGRAD_MANAGER_HOLDER_HPP
