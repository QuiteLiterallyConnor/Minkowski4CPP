#ifndef MINK_CPP_UTILS_SPHERE_HPP
#define MINK_CPP_UTILS_SPHERE_HPP

#include "../adaptive/adaptive_tensor.hpp"

#include <array>
#include <cstdint>

namespace minkowski {
namespace adaptive {

// How voxelize_sphere[_cuda] measures accuracy.
enum class SphereErrorMetric {
  // accuracy = 100 * (1 - |estimate - exact| / exact) (default). Over- and under-estimating
  // boundary leaves cancel, so refinement can stop at a lucky cancellation with a coarse,
  // visibly wrong shape: radius 100, targets 99.5 and 99.9 both return the same 232 leaves
  // (reported 99.97 %) whose occupied set differs from the ball by ~24 % of its volume.
  SignedVolume,
  // accuracy = min(the SignedVolume accuracy, 100 * (1 - misclassified / exact)), where
  // misclassified estimates the volume of the symmetric difference between the occupied
  // leaves and the ball (per boundary leaf: exact along one axis, 8 x 8 quadrature over the
  // other two). Cannot cancel, so the target bounds the shape error, not only the volume.
  // Needs finer leaves than SignedVolume for the same target.
  Geometric,
};

// Voxelizes a sphere with the fewest leaves that reach `target_accuracy_percent`.
//
// Leaves are cubes refined from one root cube (side 2 * radius) by splitting the
// largest boundary cells first. A leaf is "occupied" when its center lies inside
// the sphere, so the volume estimate is the sum of occupied leaf volumes (see
// sphere_volume_estimate). Accuracy is measured as selected by `metric` (see
// SphereErrorMetric); the default is 100 * (1 - |estimate - exact| / exact).
//
// The returned tensor has one float channel: 1 for occupied leaves, 0 otherwise.
// If achieved_accuracy_percent is non-null it receives the accuracy actually reached.
// Throws std::invalid_argument for bad arguments and std::runtime_error when the
// target cannot be reached within max_voxels leaves / max_depth levels.
AdaptiveSparseTensor voxelize_sphere(double radius, double target_accuracy_percent,
                                     std::array<double, 3> center = {0.0, 0.0, 0.0},
                                     int64_t max_voxels = 10'000'000, uint32_t max_depth = 10,
                                     double *achieved_accuracy_percent = nullptr,
                                     SphereErrorMetric metric = SphereErrorMetric::SignedVolume);

// Same algorithm and result as voxelize_sphere, with the refinement done on the GPU.
// The features of the returned tensor live on the current CUDA device. Throws
// std::runtime_error in CPU_ONLY builds or when CUDA is unavailable.
AdaptiveSparseTensor voxelize_sphere_cuda(double radius, double target_accuracy_percent,
                                          std::array<double, 3> center = {0.0, 0.0, 0.0},
                                          int64_t max_voxels = 10'000'000,
                                          uint32_t max_depth = 10,
                                          double *achieved_accuracy_percent = nullptr,
                                          SphereErrorMetric metric = SphereErrorMetric::SignedVolume);

// Sum of the volumes of occupied leaves of a tensor made by voxelize_sphere[_cuda].
double sphere_volume_estimate(const AdaptiveSparseTensor &sphere);

// Feature-test macro: voxelize_sphere[_cuda] accept a SphereErrorMetric.
#define MINK_SPHERE_HAS_ERROR_METRIC 1

namespace sphere_detail {
void validate_args(double radius, double target_accuracy_percent, int64_t max_voxels,
                   uint32_t max_depth);
AdaptiveVoxelConfig make_config(double radius, const std::array<double, 3> &center,
                                uint32_t max_depth);
} // namespace sphere_detail

} // namespace adaptive
} // namespace minkowski

#endif
