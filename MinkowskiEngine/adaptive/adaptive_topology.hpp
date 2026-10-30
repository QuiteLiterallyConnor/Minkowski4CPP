#ifndef MINK_CPP_ADAPTIVE_TOPOLOGY_HPP
#define MINK_CPP_ADAPTIVE_TOPOLOGY_HPP

// Internal header. A Topology is the immutable, host-side description of the
// active leaf set. Every topology change creates a new Topology with a fresh id,
// which is what invalidates cached kernel maps.

#include "config.hpp"
#include "voxel_key.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace minkowski {
namespace adaptive {
namespace detail {

constexpr int kMaxDim = static_cast<int>(AdaptiveVoxelConfig::kMaxDimension);
using Key = AdaptiveVoxelKey<kMaxDim>;
using KeyHash = AdaptiveVoxelKeyHash<kMaxDim>;
using DVec = std::array<double, kMaxDim>;

// Order: level first, then coordinates lexicographically. Defines the
// deterministic export order of leaves.
inline bool key_less(const Key &a, const Key &b) {
  if (a.level != b.level)
    return a.level < b.level;
  return a.coordinates < b.coordinates;
}

inline Key parent_key(const Key &k) {
  Key p = k;
  if (k.level == 0)
    return p;
  p.level = k.level - 1;
  for (int d = 0; d < kMaxDim; ++d)
    p.coordinates[d] = k.coordinates[d] >> 1;
  return p;
}

inline Key ancestor_at(Key k, uint32_t level) {
  while (k.level > level)
    k = parent_key(k);
  return k;
}

inline void child_keys(const Key &k, int D, std::vector<Key> &out) {
  const uint32_t n = 1u << D;
  for (uint32_t i = 0; i < n; ++i) {
    Key c{};
    c.level = k.level + 1;
    for (int d = 0; d < D; ++d)
      c.coordinates[d] = 2 * k.coordinates[d] + ((i >> d) & 1u);
    out.push_back(c);
  }
}

inline std::vector<int64_t> sorted_order(const std::vector<Key> &keys) {
  std::vector<int64_t> order(keys.size());
  for (size_t i = 0; i < order.size(); ++i)
    order[i] = static_cast<int64_t>(i);
  std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    return key_less(keys[a], keys[b]);
  });
  return order;
}

// Sorts `keys` by key_less. `order` receives the original position of each sorted key and
// is left empty when the input was already sorted (nothing moved).
void sort_keys(std::vector<Key> &keys, std::vector<int64_t> &order);

#ifdef USE_CUDA
// Device versions (adaptive_topology_cuda.cu) of sort_keys and the ancestor check.
bool cuda_topology_available();
void cuda_sort_keys(std::vector<Key> &keys, std::vector<int64_t> &order);
bool cuda_has_active_ancestor(const std::vector<Key> &keys,
                              const std::vector<int64_t> &level_begin, int64_t first);
#endif

// Calls f(begin, end) over disjoint blocks covering [0, n), on several threads for large n.
// The first exception thrown by f is rethrown after all blocks finish.
void parallel_blocks(int64_t n, const std::function<void(int64_t, int64_t)> &f);

struct Topology {
  uint64_t id = 0;
  int D = 0;
  uint32_t max_depth = 0;
  AdaptiveVoxelConfig config; // origin always filled
  DVec origin{};
  DVec base{};
  std::vector<Key> keys; // sorted by key_less, strictly increasing
  at::Tensor coords;  // [N, D] int64
  at::Tensor levels;  // [N] int32
  at::Tensor volumes; // [N] double
  std::vector<int64_t> counts;
  std::vector<uint32_t> present_levels;

  // Built on first use; most callers never need them.
  const at::Tensor &lo_tensor() const { build_geometry(); return lo_; }
  const at::Tensor &hi_tensor() const { build_geometry(); return hi_; }
  const at::Tensor &sizes_tensor() const { build_geometry(); return sizes_; }
  at::Tensor centers_tensor() const; // [N, D] double, a fresh tensor

  int64_t size() const { return static_cast<int64_t>(keys.size()); }

  DVec size_at(uint32_t level) const {
    const double f = std::ldexp(1.0, -static_cast<int>(level));
    DVec s{};
    for (int d = 0; d < D; ++d)
      s[d] = base[d] * f;
    return s;
  }

  void bounds(const Key &k, DVec &lo_out, DVec &hi_out) const {
    const DVec s = size_at(k.level);
    for (int d = 0; d < D; ++d) {
      lo_out[d] = origin[d] + static_cast<double>(k.coordinates[d]) * s[d];
      hi_out[d] = lo_out[d] + s[d];
    }
  }

  // Returns false for non-finite or out-of-range points.
  bool locate(const DVec &p, uint32_t level, Key &out) const {
    const DVec s = size_at(level);
    out = Key{};
    out.level = level;
    for (int d = 0; d < D; ++d) {
      const double v = std::floor((p[d] - origin[d]) / s[d]);
      if (!std::isfinite(v) || std::fabs(v) > 4e18)
        return false;
      out.coordinates[d] = static_cast<int64_t>(v);
    }
    return true;
  }

  int64_t find(const Key &k) const {
    const auto &index = index_map();
    auto it = index.find(k);
    return it == index.end() ? -1 : it->second;
  }

  // The active leaf that owns point p, or -1 (leaves never overlap).
  int64_t find_containing(const DVec &p) const {
    Key k;
    for (uint32_t l : present_levels) {
      if (!locate(p, l, k))
        return -1;
      const int64_t j = find(k);
      if (j >= 0)
        return j;
    }
    return -1;
  }

  bool has_overlap() const {
    const auto &index = index_map();
    for (const Key &k : keys)
      for (uint32_t l = 0; l < k.level; ++l)
        if (index.count(ancestor_at(k, l)))
          return true;
    return false;
  }

  // Calls f(leaf_index, overlap_volume) for every leaf overlapping the
  // half-open box [blo, bhi). Cost per present level is the number of grid
  // cells of that level inside the box (finer levels cost 2^(D*level_gap)).
  template <class F>
  void for_each_overlap(const DVec &blo, const DVec &bhi, F &&f) const {
    for (uint32_t l : present_levels) {
      const DVec s = size_at(l);
      std::array<int64_t, kMaxDim> i0{}, i1{};
      bool empty = false;
      for (int d = 0; d < D; ++d) {
        const double a = (blo[d] - origin[d]) / s[d];
        const double b = (bhi[d] - origin[d]) / s[d];
        if (!std::isfinite(a) || !std::isfinite(b) || std::fabs(a) > 4e18 ||
            std::fabs(b) > 4e18) {
          empty = true;
          break;
        }
        i0[d] = static_cast<int64_t>(std::floor(a + 1e-9));
        i1[d] = static_cast<int64_t>(std::ceil(b - 1e-9)) - 1;
        if (i1[d] < i0[d]) {
          empty = true;
          break;
        }
      }
      if (empty)
        continue;
      auto cur = i0;
      while (true) {
        Key k{};
        k.level = l;
        k.coordinates = cur;
        const int64_t j = find(k);
        if (j >= 0) {
          double ov = 1.0;
          for (int d = 0; d < D; ++d) {
            const double clo = origin[d] + static_cast<double>(cur[d]) * s[d];
            const double o = std::min(bhi[d], clo + s[d]) - std::max(blo[d], clo);
            if (o <= 1e-9 * s[d]) {
              ov = 0.0;
              break;
            }
            ov *= o;
          }
          if (ov > 0.0)
            f(j, ov);
        }
        int d = D - 1;
        for (; d >= 0; --d) {
          if (++cur[d] <= i1[d])
            break;
          cur[d] = i0[d];
        }
        if (d < 0)
          break;
      }
    }
  }

  // `sorted_keys` must be strictly increasing under key_less. Throws
  // std::invalid_argument on duplicates, bad levels or overlapping leaves.
  static std::shared_ptr<const Topology> make(const AdaptiveVoxelConfig &cfg,
                                              std::vector<Key> sorted_keys);

private:
  const std::unordered_map<Key, int64_t, KeyHash> &index_map() const;
  void build_geometry() const;

  mutable std::once_flag index_once_;
  mutable std::unordered_map<Key, int64_t, KeyHash> index_;
  mutable std::once_flag geometry_once_;
  mutable at::Tensor lo_, hi_, sizes_; // [N, D] double
};

inline bool same_topology(const Topology &a, const Topology &b) {
  if (a.id == b.id)
    return true;
  if (a.D != b.D || a.keys.size() != b.keys.size() || a.max_depth != b.max_depth)
    return false;
  return a.keys == b.keys && a.origin == b.origin && a.base == b.base;
}

} // namespace detail
} // namespace adaptive
} // namespace minkowski

#endif
