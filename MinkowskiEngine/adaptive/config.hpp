#ifndef MINK_CPP_ADAPTIVE_CONFIG_HPP
#define MINK_CPP_ADAPTIVE_CONFIG_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace minkowski {
namespace adaptive {

enum class Reduction { Mean, Sum, Max };
enum class RefinePolicy { Replicate, Split };

// Half-open axis-aligned box [lo, hi).
struct AxisAlignedRegion {
  std::vector<double> lo;
  std::vector<double> hi;

  static AxisAlignedRegion everything(size_t dimension);
};

struct AdaptiveVoxelConfig {
  // Supported spatial dimensions are 1..kMaxDimension.
  static constexpr size_t kMaxDimension = 4;
  // Coordinates are int64 and children use 2*c+bit, so level is capped well
  // below the point where level-0 coordinates of ~2^32 would overflow.
  static constexpr uint32_t kMaxSupportedDepth = 30;

  std::vector<double> base_voxel_size;
  // Optional. When given, base_voxel_size / min_voxel_size must equal 2^max_depth
  // in every dimension.
  std::vector<double> min_voxel_size;
  // Optional (defaults to the zero vector).
  std::vector<double> origin;
  uint32_t max_depth = 0;

  size_t dimension() const { return base_voxel_size.size(); }

  // Throws std::invalid_argument describing the first inconsistency found.
  void validate() const;
};

} // namespace adaptive
} // namespace minkowski

#endif
