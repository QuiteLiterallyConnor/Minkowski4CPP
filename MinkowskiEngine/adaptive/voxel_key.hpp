#ifndef MINK_CPP_ADAPTIVE_VOXEL_KEY_HPP
#define MINK_CPP_ADAPTIVE_VOXEL_KEY_HPP

#include <array>
#include <cstddef>
#include <cstdint>

namespace minkowski {
namespace adaptive {

inline uint64_t mix64(uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Identity of one voxel: integer coordinates on the grid of its refinement level.
// Two keys are equal only if both coordinates and level are equal.
template <size_t D> struct AdaptiveVoxelKey {
  std::array<int64_t, D> coordinates;
  uint32_t level;

  bool operator==(const AdaptiveVoxelKey &) const = default;
};

// The level is mixed in first, so equal coordinates at different levels hash apart.
template <size_t D> struct AdaptiveVoxelKeyHash {
  size_t operator()(const AdaptiveVoxelKey<D> &k) const noexcept {
    uint64_t h = mix64(static_cast<uint64_t>(k.level) + 0x51ed270b0b4a2f1dULL);
    for (size_t d = 0; d < D; ++d)
      h = mix64(h ^ mix64(static_cast<uint64_t>(k.coordinates[d])));
    return static_cast<size_t>(h);
  }
};

} // namespace adaptive
} // namespace minkowski

#endif
