#include "config.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace minkowski {
namespace adaptive {

AxisAlignedRegion AxisAlignedRegion::everything(size_t dimension) {
  AxisAlignedRegion r;
  r.lo.assign(dimension, -std::numeric_limits<double>::infinity());
  r.hi.assign(dimension, std::numeric_limits<double>::infinity());
  return r;
}

void AdaptiveVoxelConfig::validate() const {
  const size_t D = base_voxel_size.size();
  if (D == 0 || D > kMaxDimension)
    throw std::invalid_argument("AdaptiveVoxelConfig: base_voxel_size must have 1.." +
                                std::to_string(kMaxDimension) + " entries");
  for (double s : base_voxel_size)
    if (!(s > 0.0) || !std::isfinite(s))
      throw std::invalid_argument(
          "AdaptiveVoxelConfig: base_voxel_size entries must be positive and finite");
  if (!origin.empty()) {
    if (origin.size() != D)
      throw std::invalid_argument("AdaptiveVoxelConfig: origin dimension mismatch");
    for (double o : origin)
      if (!std::isfinite(o))
        throw std::invalid_argument("AdaptiveVoxelConfig: origin must be finite");
  }
  if (max_depth > kMaxSupportedDepth)
    throw std::invalid_argument("AdaptiveVoxelConfig: max_depth exceeds kMaxSupportedDepth (" +
                                std::to_string(kMaxSupportedDepth) + ")");
  if (!min_voxel_size.empty()) {
    if (min_voxel_size.size() != D)
      throw std::invalid_argument("AdaptiveVoxelConfig: min_voxel_size dimension mismatch");
    const double expected = std::ldexp(1.0, static_cast<int>(max_depth));
    for (size_t d = 0; d < D; ++d) {
      if (!(min_voxel_size[d] > 0.0) || !std::isfinite(min_voxel_size[d]))
        throw std::invalid_argument(
            "AdaptiveVoxelConfig: min_voxel_size entries must be positive and finite");
      const double ratio = base_voxel_size[d] / min_voxel_size[d];
      if (std::fabs(ratio - expected) > 1e-9 * expected)
        throw std::invalid_argument(
            "AdaptiveVoxelConfig: base_voxel_size / min_voxel_size must equal 2^max_depth");
    }
  }
}

} // namespace adaptive
} // namespace minkowski
