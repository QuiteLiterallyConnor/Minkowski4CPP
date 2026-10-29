#ifndef MINK_CPP_ADAPTIVE_VOXEL_GEOMETRY_HPP
#define MINK_CPP_ADAPTIVE_VOXEL_GEOMETRY_HPP

#include "voxel_key.hpp"

#include <array>
#include <cmath>
#include <utility>

namespace minkowski {
namespace adaptive {

// voxel_size(level, d) = base_voxel_size[d] / 2^level. Voxels are half-open boxes
// [lo, hi), so every point belongs to exactly one voxel per level.
template <size_t D> struct AdaptiveVoxelGeometry {
  using Key = AdaptiveVoxelKey<D>;
  using Vec = std::array<double, D>;

  Vec origin{};
  Vec base_voxel_size{};
  uint32_t max_depth = 0;

  Vec voxel_size(uint32_t level) const {
    const double f = std::ldexp(1.0, -static_cast<int>(level));
    Vec s{};
    for (size_t d = 0; d < D; ++d)
      s[d] = base_voxel_size[d] * f;
    return s;
  }

  std::pair<Vec, Vec> bounds(const Key &k) const {
    const Vec s = voxel_size(k.level);
    Vec lo{}, hi{};
    for (size_t d = 0; d < D; ++d) {
      lo[d] = origin[d] + static_cast<double>(k.coordinates[d]) * s[d];
      hi[d] = lo[d] + s[d];
    }
    return {lo, hi};
  }

  Vec center(const Key &k) const {
    auto [lo, hi] = bounds(k);
    Vec c{};
    for (size_t d = 0; d < D; ++d)
      c[d] = lo[d] + 0.5 * (hi[d] - lo[d]);
    return c;
  }

  // The voxel at `level` that owns point p (floor division, so negative
  // coordinates are handled consistently).
  Key locate(const Vec &p, uint32_t level) const {
    const Vec s = voxel_size(level);
    Key k{};
    k.level = level;
    for (size_t d = 0; d < D; ++d)
      k.coordinates[d] =
          static_cast<int64_t>(std::floor((p[d] - origin[d]) / s[d]));
    return k;
  }

  static Key parent(const Key &k) {
    Key p = k;
    p.level = k.level == 0 ? 0 : k.level - 1;
    if (k.level == 0)
      return p;
    for (size_t d = 0; d < D; ++d)
      p.coordinates[d] = k.coordinates[d] >> 1; // arithmetic shift == floor div
    return p;
  }

  static Key child(const Key &k, uint32_t index) {
    Key c{};
    c.level = k.level + 1;
    for (size_t d = 0; d < D; ++d)
      c.coordinates[d] = 2 * k.coordinates[d] + ((index >> d) & 1u);
    return c;
  }

  static std::array<Key, (size_t(1) << D)> children(const Key &k) {
    std::array<Key, (size_t(1) << D)> out{};
    for (uint32_t i = 0; i < out.size(); ++i)
      out[i] = child(k, i);
    return out;
  }

  // True when `a` is a strict ancestor of `b`.
  static bool is_ancestor(const Key &a, const Key &b) {
    if (a.level >= b.level)
      return false;
    const uint32_t diff = b.level - a.level;
    if (diff >= 63)
      return false;
    for (size_t d = 0; d < D; ++d)
      if ((b.coordinates[d] >> diff) != a.coordinates[d])
        return false;
    return true;
  }
};

} // namespace adaptive
} // namespace minkowski

#endif
