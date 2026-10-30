#include "sphere.hpp"
#include "sphere_geometry.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace minkowski {
namespace adaptive {
namespace {

using sphere_detail::SphereGeometry;

constexpr int kThreads = 256;

// One thread per frontier cell: emits its 8 children, whether each child is still a
// boundary cell, and the change in the volume estimate if this cell is split.
__global__ void expand_kernel(SphereGeometry s, const int64_t *coords, int64_t n, int level,
                              int64_t *child_coords, bool *child_boundary, double *delta) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  const int64_t px = coords[i * 3], py = coords[i * 3 + 1], pz = coords[i * 3 + 2];
  double change = -s.contribution(px, py, pz, level);
  for (int child = 0; child < 8; ++child) {
    const int64_t cx = 2 * px + (child & 1);
    const int64_t cy = 2 * py + ((child >> 1) & 1);
    const int64_t cz = 2 * pz + ((child >> 2) & 1);
    const int64_t o = i * 8 + child;
    child_coords[o * 3] = cx;
    child_coords[o * 3 + 1] = cy;
    child_coords[o * 3 + 2] = cz;
    child_boundary[o] = s.is_boundary(cx, cy, cz, level + 1);
    change += s.contribution(cx, cy, cz, level + 1);
  }
  delta[i] = change;
}

__global__ void occupancy_kernel(SphereGeometry s, const int64_t *coords, const int32_t *levels,
                                 int64_t n, float *features) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  features[i] = s.occupied(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2], levels[i]) ? 1.f
                                                                                           : 0.f;
}

int blocks_for(int64_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

} // namespace

AdaptiveSparseTensor voxelize_sphere_cuda(double radius, double target_accuracy_percent,
                                          std::array<double, 3> center, int64_t max_voxels,
                                          uint32_t max_depth, double *achieved_accuracy_percent) {
  sphere_detail::validate_args(radius, target_accuracy_percent, max_voxels, max_depth);
  if (!torch::cuda::is_available())
    throw std::runtime_error("voxelize_sphere_cuda: CUDA is not available");

  const SphereGeometry sphere{radius};
  const torch::Device device(torch::kCUDA, c10::cuda::current_device());
  const auto i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);
  const auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  const auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
  const auto boolean = torch::TensorOptions().dtype(torch::kBool).device(device);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();

  std::vector<at::Tensor> leaf_coords, leaf_levels;
  const auto add_leaves = [&](const at::Tensor &coords, int level) {
    if (coords.size(0) == 0)
      return;
    leaf_coords.push_back(coords);
    leaf_levels.push_back(torch::full({coords.size(0)}, level, i32));
  };

  // Same refinement as voxelize_sphere: cells of one level are split in order (children
  // of earlier parents first) until the estimate first reaches the target. Per level the
  // incremental estimate is a cumulative sum over the parents' volume deltas, so the
  // first parent whose split reaches the target is found without a per-split loop.
  double volume = sphere.contribution(0, 0, 0, 0);
  int64_t count = 1;
  int level = 0;
  at::Tensor frontier = torch::zeros({0, 3}, i64);
  if (sphere.is_boundary(0, 0, 0, 0))
    frontier = torch::zeros({1, 3}, i64);
  else
    add_leaves(torch::zeros({1, 3}, i64), 0);

  double achieved = sphere.accuracy(volume);
  while (achieved < target_accuracy_percent) {
    const int64_t n = frontier.size(0);
    const int64_t max_splits = n == 0 ? 0 : std::min<int64_t>(n, (max_voxels - count) / 7);
    if (n == 0 || max_splits == 0 || static_cast<uint32_t>(level) >= max_depth)
      throw std::runtime_error("voxelize_sphere: target accuracy not reachable within limits");

    auto child_coords = torch::empty({n * 8, 3}, i64);
    auto child_boundary = torch::empty({n * 8}, boolean);
    auto delta = torch::empty({n}, f64);
    expand_kernel<<<blocks_for(n), kThreads, 0, stream>>>(
        sphere, frontier.data_ptr<int64_t>(), n, level, child_coords.data_ptr<int64_t>(),
        child_boundary.data_ptr<bool>(), delta.data_ptr<double>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const double exact = sphere.exact_volume();
    const auto estimate = delta.cumsum(0) + volume;
    const auto accuracy = 100.0 * (1.0 - (estimate - exact).abs() / exact);
    const auto reached = accuracy.slice(0, 0, max_splits) >= target_accuracy_percent;
    const bool any_reached = max_splits > 0 && reached.any().item<bool>();

    if (!any_reached) {
      if (max_splits < n)
        throw std::runtime_error("voxelize_sphere: target accuracy not reachable within limits");
      // Split every cell of this level and continue with their boundary children.
      add_leaves(child_coords.index({~child_boundary}), level + 1);
      frontier = child_coords.index({child_boundary});
      volume = estimate[n - 1].item<double>();
      achieved = accuracy[n - 1].item<double>();
      count += 7 * n;
      ++level;
      continue;
    }

    // Only the first k parents are split; the remaining ones stay leaves.
    const int64_t k = reached.to(torch::kInt32).argmax().item<int64_t>() + 1;
    const auto kept_coords = child_coords.slice(0, 0, 8 * k);
    const auto kept_boundary = child_boundary.slice(0, 0, 8 * k);
    add_leaves(kept_coords.index({~kept_boundary}), level + 1);
    add_leaves(kept_coords.index({kept_boundary}), level + 1);
    frontier = frontier.slice(0, k, n);
    achieved = accuracy[k - 1].item<double>();
    break;
  }
  add_leaves(frontier, level);
  if (achieved_accuracy_percent)
    *achieved_accuracy_percent = achieved;

  const auto coordinates = torch::cat(leaf_coords, 0).contiguous();
  const auto levels = torch::cat(leaf_levels, 0).contiguous();
  const int64_t total = coordinates.size(0);
  auto features = torch::empty({total, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  occupancy_kernel<<<blocks_for(total), kThreads, 0, stream>>>(
      sphere, coordinates.data_ptr<int64_t>(), levels.data_ptr<int32_t>(), total,
      features.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return AdaptiveSparseTensor::from_voxels(coordinates, levels, features,
                                           sphere_detail::make_config(radius, center, max_depth));
}

} // namespace adaptive
} // namespace minkowski
