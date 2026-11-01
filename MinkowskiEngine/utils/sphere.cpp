#include "sphere.hpp"
#include "sphere_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <vector>

namespace minkowski {
namespace adaptive {

namespace sphere_detail {

void validate_args(double radius, double target_accuracy_percent, int64_t max_voxels,
                   uint32_t max_depth) {
  if (!(radius > 0.0) || !std::isfinite(radius))
    throw std::invalid_argument("voxelize_sphere: radius must be positive and finite");
  if (!(target_accuracy_percent > 0.0 && target_accuracy_percent < 100.0))
    throw std::invalid_argument("voxelize_sphere: target accuracy must be in (0, 100)");
  if (max_voxels < 1 || max_depth > AdaptiveVoxelConfig::kMaxSupportedDepth)
    throw std::invalid_argument("voxelize_sphere: invalid max_voxels or max_depth");
}

AdaptiveVoxelConfig make_config(double radius, const std::array<double, 3> &center,
                                uint32_t max_depth) {
  AdaptiveVoxelConfig config;
  const double side = 2.0 * radius;
  const double min_size = std::ldexp(side, -static_cast<int>(max_depth));
  config.base_voxel_size = {side, side, side};
  config.min_voxel_size = {min_size, min_size, min_size};
  config.origin = {center[0] - radius, center[1] - radius, center[2] - radius};
  config.max_depth = max_depth;
  return config;
}

} // namespace sphere_detail

namespace {

using Cell = std::array<int64_t, 4>; // x, y, z, level (relative to the sphere's bounding cube)
using sphere_detail::SphereGeometry;

bool is_boundary(const SphereGeometry &s, const Cell &c) {
  return s.is_boundary(c[0], c[1], c[2], static_cast<int>(c[3]));
}
bool occupied(const SphereGeometry &s, const Cell &c) {
  return s.occupied(c[0], c[1], c[2], static_cast<int>(c[3]));
}
double contribution(const SphereGeometry &s, const Cell &c) {
  return s.contribution(c[0], c[1], c[2], static_cast<int>(c[3]));
}
double misclassified(const SphereGeometry &s, const Cell &c) {
  return s.misclassified_volume(c[0], c[1], c[2], static_cast<int>(c[3]));
}

Cell child_of(const Cell &parent, int child) {
  return {2 * parent[0] + (child & 1), 2 * parent[1] + ((child >> 1) & 1),
          2 * parent[2] + ((child >> 2) & 1), parent[3] + 1};
}

} // namespace

AdaptiveSparseTensor voxelize_sphere(double radius, double target_accuracy_percent,
                                     std::array<double, 3> center, int64_t max_voxels,
                                     uint32_t max_depth, double *achieved_accuracy_percent,
                                     SphereErrorMetric metric) {
  sphere_detail::validate_args(radius, target_accuracy_percent, max_voxels, max_depth);
  const SphereGeometry sphere{radius};
  const bool geometric = metric == SphereErrorMetric::Geometric;
  const auto accuracy = [&](double vol, double wrong_vol) {
    const double a = sphere.accuracy(vol);
    return geometric ? std::min(a, sphere.geometric_accuracy(wrong_vol)) : a;
  };

  // Refine in FIFO order (all cells of one level before the next) tracking the
  // estimate incrementally; stop at the first leaf count that reaches the target.
  std::vector<Cell> leaves;
  std::deque<Cell> boundary;
  const Cell root{0, 0, 0, 0};
  double volume = contribution(sphere, root);
  double wrong = geometric ? misclassified(sphere, root) : 0.0; // misclassified volume
  int64_t count = 1;
  if (is_boundary(sphere, root))
    boundary.push_back(root);
  else
    leaves.push_back(root);

  double achieved = accuracy(volume, wrong);
  while (achieved < target_accuracy_percent) {
    if (boundary.empty() || count + 7 > max_voxels ||
        static_cast<uint32_t>(boundary.front()[3]) >= max_depth)
      throw std::runtime_error("voxelize_sphere: target accuracy not reachable within limits");
    const Cell parent = boundary.front();
    boundary.pop_front();
    volume -= contribution(sphere, parent);
    if (geometric)
      wrong -= misclassified(sphere, parent);
    count += 7;
    for (int child = 0; child < 8; ++child) {
      const Cell c = child_of(parent, child);
      volume += contribution(sphere, c);
      if (geometric)
        wrong += misclassified(sphere, c);
      if (is_boundary(sphere, c))
        boundary.push_back(c);
      else
        leaves.push_back(c);
    }
    achieved = accuracy(volume, wrong);
  }
  leaves.insert(leaves.end(), boundary.begin(), boundary.end());
  if (achieved_accuracy_percent)
    *achieved_accuracy_percent = achieved;

  const int64_t n = static_cast<int64_t>(leaves.size());
  auto coordinates = torch::zeros({n, 3}, torch::kInt64);
  auto levels = torch::zeros({n}, torch::kInt32);
  auto features = torch::zeros({n, 1}, torch::kFloat32);
  auto coord_a = coordinates.accessor<int64_t, 2>();
  auto level_a = levels.accessor<int32_t, 1>();
  auto feature_a = features.accessor<float, 2>();
  for (int64_t i = 0; i < n; ++i) {
    const Cell &c = leaves[static_cast<size_t>(i)];
    for (int axis = 0; axis < 3; ++axis)
      coord_a[i][axis] = c[axis];
    level_a[i] = static_cast<int32_t>(c[3]);
    feature_a[i][0] = occupied(sphere, c) ? 1.0f : 0.0f;
  }

  return AdaptiveSparseTensor::from_voxels(coordinates, levels, features,
                                           sphere_detail::make_config(radius, center, max_depth));
}

double sphere_volume_estimate(const AdaptiveSparseTensor &sphere) {
  const auto volumes = sphere.volumes().contiguous();
  const auto occupied = sphere.features().to(torch::kCPU).to(torch::kFloat64).reshape({-1});
  return (volumes * (occupied > 0.5).to(torch::kFloat64)).sum().item<double>();
}

#ifdef CPU_ONLY
AdaptiveSparseTensor voxelize_sphere_cuda(double, double, std::array<double, 3>, int64_t,
                                          uint32_t, double *, SphereErrorMetric) {
  throw std::runtime_error("voxelize_sphere_cuda: CUDA not available in CPU_ONLY build");
}
#endif

} // namespace adaptive
} // namespace minkowski
