#ifndef MINK_TEST_PART_SPLIT_REFERENCE_HPP
#define MINK_TEST_PART_SPLIT_REFERENCE_HPP

#include "MinkowskiEngine/adaptive/voxel_key.hpp"
#include <algorithm>
#include <vector>

namespace split_reference {
using Key = minkowski::adaptive::AdaptiveVoxelKey<3>;

inline bool less(const Key &a, const Key &b) {
  return a.level == b.level ? a.coordinates < b.coordinates : a.level < b.level;
}

// Original full-leaf sort and per-leaf binary search, retained as an independent
// differential reference and as the benchmark baseline.
inline std::vector<int64_t> starts(const std::vector<Key> &keys, int radius) {
  std::vector<int64_t> xs(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) xs[i] = keys[i].coordinates[0] >> keys[i].level;
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  std::vector<int64_t> result;
  for (size_t i = 0; i < xs.size(); ++i)
    if (i == 0 || xs[i] - xs[i - 1] > radius) result.push_back(xs[i]);
  return result;
}

inline std::vector<std::vector<int64_t>> leaves(const std::vector<int64_t> &first,
                                               const std::vector<Key> &keys) {
  std::vector<std::vector<int64_t>> result(first.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    const auto it = std::upper_bound(first.begin(), first.end(), keys[i].coordinates[0] >> keys[i].level);
    result[static_cast<size_t>(it - first.begin()) - 1].push_back(static_cast<int64_t>(i));
  }
  return result;
}
} // namespace split_reference
#endif
