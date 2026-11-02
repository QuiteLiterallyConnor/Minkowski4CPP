#ifndef MINK_CPP_ADAPTIVE_PART_SPLIT_HPP
#define MINK_CPP_ADAPTIVE_PART_SPLIT_HPP

// Internal helpers for immutable topology keys sorted by (level, coordinates).
#include "voxel_key.hpp"

#include <algorithm>
#include <vector>

namespace minkowski {
namespace adaptive {
namespace detail {

template <size_t D>
std::vector<int64_t> part_starts(const std::vector<AdaptiveVoxelKey<D>> &keys, int radius) {
  std::vector<int64_t> occupied;
  uint32_t previous_level = 0;
  int64_t previous_x = 0;
  size_t levels = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto &key = keys[i];
    const int64_t x = key.coordinates[0] >> key.level;
    const bool new_level = i == 0 || key.level != previous_level;
    // Within each level, x is nondecreasing. Other axes and finer x coordinates
    // often repeat the same level-zero cell; keep that cell only once per level.
    if (new_level || x != previous_x) occupied.push_back(x);
    levels += new_level;
    previous_level = key.level;
    previous_x = x;
  }
  if (levels > 1) {
    std::sort(occupied.begin(), occupied.end());
    occupied.erase(std::unique(occupied.begin(), occupied.end()), occupied.end());
  }
  std::vector<int64_t> starts;
  for (size_t i = 0; i < occupied.size(); ++i)
    if (i == 0 || occupied[i] - occupied[i - 1] > radius) starts.push_back(occupied[i]);
  return starts;
}

template <size_t D>
std::vector<std::vector<int64_t>> part_leaves(
    const std::vector<int64_t> &starts, const std::vector<AdaptiveVoxelKey<D>> &keys) {
  std::vector<std::vector<int64_t>> leaves(starts.size());
  size_t part = 0;
  uint32_t previous_level = 0;
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto &key = keys[i];
    const int64_t x = key.coordinates[0] >> key.level;
    if (i == 0 || key.level != previous_level) {
      // A new level can restart at an earlier part or skip many initial parts.
      part = static_cast<size_t>(std::upper_bound(starts.begin(), starts.end(), x) -
                                 starts.begin()) - 1;
      previous_level = key.level;
    } else {
      while (part + 1 < starts.size() && starts[part + 1] <= x) ++part;
    }
    leaves[part].push_back(static_cast<int64_t>(i));
  }
  return leaves;
}

} // namespace detail
} // namespace adaptive
} // namespace minkowski

#endif
