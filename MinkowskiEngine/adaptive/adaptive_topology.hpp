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
// For loops with expensive items: the number of blocks (threads) worth using when the loop
// over n items costs about `work` cheap steps in total, and a runner that calls
// f(block, begin, end) for block b = [n * b / blocks, n * (b + 1) / blocks) on one thread each.
int parallel_block_count(int64_t n, int64_t work);
void run_parallel_blocks(int64_t n, int blocks,
                         const std::function<void(int, int64_t, int64_t)> &f);

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
    if (keys.empty())
      return -1;
    const int64_t v = node_table().lookup(k);
    return v >= 0 ? v : -1;
  }

  // The active leaf that owns point p, or -1 (leaves never overlap). Probes the cells
  // containing p from the coarsest present level down; a cell that is neither a leaf nor an
  // ancestor of one ends the search, since every finer cell containing p lies inside it.
  int64_t find_containing(const DVec &p) const {
    if (keys.empty())
      return -1;
    const NodeTable &nodes = node_table();
    Key k;
    for (uint32_t l : present_levels) {
      if (!locate(p, l, k))
        return -1;
      const int64_t v = nodes.lookup(k);
      if (v != NodeTable::kInternal)
        return v; // leaf index, or kEmpty (-1)
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
  // half-open box [blo, bhi), in increasing leaf index order. Walks down the
  // leaf tree from the coarsest present level and only enters cells that hold
  // a leaf or an ancestor of one, so the cost is about (overlapping leaves) x
  // (level gap) x 2^D lookups instead of every finer grid cell in the box.
  template <class F>
  void for_each_overlap(const DVec &blo, const DVec &bhi, F &&f) const {
    if (keys.empty())
      return;
    const uint32_t lmin = present_levels.front();
    const uint32_t lmax = present_levels.back();
    std::array<OverlapLevel, AdaptiveVoxelConfig::kMaxSupportedDepth + 1> lv;
    for (uint32_t l = lmin; l <= lmax; ++l) {
      OverlapLevel &L = lv[l];
      L.s = size_at(l);
      L.walk = true;
      L.leaf = true;
      for (int d = 0; d < D; ++d) {
        const double a = (blo[d] - origin[d]) / L.s[d];
        const double b = (bhi[d] - origin[d]) / L.s[d];
        if (!std::isfinite(a) || !std::isfinite(b) || std::fabs(a) > 4e18 ||
            std::fabs(b) > 4e18) {
          // Also true for every finer level (|a|, |b| only grow): stop there.
          L.walk = false;
          L.leaf = false;
          break;
        }
        // Leaves are accepted with exactly the tolerance of a grid scan of this level.
        L.i0[d] = static_cast<int64_t>(std::floor(a + 1e-9));
        L.i1[d] = static_cast<int64_t>(std::ceil(b - 1e-9)) - 1;
        if (L.i1[d] < L.i0[d])
          L.leaf = false;
        // Cells are walked without tolerance: [floor(a), ceil(b) - 1] contains the ancestor
        // of every finer cell accepted above, since a and b scale exactly by powers of two.
        L.w0[d] = static_cast<int64_t>(std::floor(a));
        L.w1[d] = static_cast<int64_t>(std::ceil(b)) - 1;
      }
    }
    const OverlapLevel &top = lv[lmin];
    if (!top.walk)
      return;
    for (int d = 0; d < D; ++d)
      if (top.w1[d] < top.w0[d])
        return;
    const NodeTable &nodes = node_table();
    // Per-thread scratch buffer, swapped out so a re-entrant call from f stays safe.
    static thread_local std::vector<std::pair<int64_t, double>> scratch;
    std::vector<std::pair<int64_t, double>> hits;
    hits.swap(scratch);
    hits.clear();
    Key k{};
    k.level = lmin;
    for (int d = 0; d < D; ++d) // coordinates past D stay 0, as in every stored key
      k.coordinates[d] = top.w0[d];
    while (true) {
      overlap_visit(nodes, k, lv, blo, bhi, hits);
      int d = D - 1;
      for (; d >= 0; --d) {
        if (++k.coordinates[d] <= top.w1[d])
          break;
        k.coordinates[d] = top.w0[d];
      }
      if (d < 0)
        break;
    }
    // Keys are sorted (level, coordinates), so index order equals the order of a
    // level-by-level lexicographic grid scan.
    std::sort(hits.begin(), hits.end(),
              [](const auto &x, const auto &y) { return x.first < y.first; });
    for (const auto &h : hits)
      f(h.first, h.second);
    hits.swap(scratch);
  }

  // Reference implementation of for_each_overlap: scans every grid cell of every present
  // level inside the box. Kept for tests and benchmarks.
  template <class F>
  void for_each_overlap_scan(const DVec &blo, const DVec &bhi, F &&f) const {
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

  // Open-addressing table over every leaf (value = leaf index) and every strict
  // ancestor of a leaf down to the coarsest present level (value = kInternal).
  struct NodeTable {
    static constexpr int64_t kEmpty = -1;
    static constexpr int64_t kInternal = -2;
    struct Slot {
      Key key;
      int64_t value;
    };
    std::vector<Slot> slots;
    uint64_t mask = 0;
    int64_t internal_count = 0;

    static uint64_t hash(const Key &k) {
      uint64_t h = k.level;
      for (int d = 0; d < kMaxDim; ++d)
        h = (h ^ static_cast<uint64_t>(k.coordinates[d])) * 0x9e3779b97f4a7c15ULL;
      h ^= h >> 32;
      h *= 0xd6e8feb86659fd93ULL;
      return h ^ (h >> 32);
    }
    // Leaf index, kInternal, or kEmpty.
    int64_t lookup(const Key &k) const {
      for (uint64_t h = hash(k) & mask;; h = (h + 1) & mask) {
        const Slot &s = slots[h];
        if (s.value == kEmpty || s.key == k)
          return s.value;
      }
    }
  };
  // Built on first use and cached for the lifetime of this (immutable) topology.
  const NodeTable &node_table() const;

private:
  // Trivial on purpose: for_each_overlap fills only levels lmin..lmax (and only the first D
  // entries of each range), and reads nothing else.
  struct OverlapLevel {
    bool walk; // the box is representable at this level
    bool leaf; // the leaf acceptance range is non-empty
    DVec s;
    std::array<int64_t, kMaxDim> i0, i1; // leaf acceptance range (with tolerance)
    std::array<int64_t, kMaxDim> w0, w1; // walk range (superset, no tolerance)
  };

  template <size_t NL>
  void overlap_visit(const NodeTable &nodes, const Key &k, const std::array<OverlapLevel, NL> &lv,
                     const DVec &blo, const DVec &bhi,
                     std::vector<std::pair<int64_t, double>> &hits) const {
    const int64_t v = nodes.lookup(k);
    if (v == NodeTable::kEmpty)
      return;
    if (v >= 0) {
      const OverlapLevel &L = lv[k.level];
      if (!L.leaf)
        return;
      for (int d = 0; d < D; ++d)
        if (k.coordinates[d] < L.i0[d] || k.coordinates[d] > L.i1[d])
          return;
      double ov = 1.0;
      for (int d = 0; d < D; ++d) {
        const double clo = origin[d] + static_cast<double>(k.coordinates[d]) * L.s[d];
        const double o = std::min(bhi[d], clo + L.s[d]) - std::max(blo[d], clo);
        if (o <= 1e-9 * L.s[d])
          return;
        ov *= o;
      }
      hits.emplace_back(v, ov);
      return;
    }
    const OverlapLevel &C = lv[k.level + 1];
    if (!C.walk)
      return;
    std::array<int64_t, kMaxDim> c0{}, c1{};
    for (int d = 0; d < D; ++d) {
      c0[d] = std::max(2 * k.coordinates[d], C.w0[d]);
      c1[d] = std::min(2 * k.coordinates[d] + 1, C.w1[d]);
      if (c1[d] < c0[d])
        return;
    }
    Key c{};
    c.level = k.level + 1;
    c.coordinates = c0;
    while (true) {
      overlap_visit(nodes, c, lv, blo, bhi, hits);
      int d = D - 1;
      for (; d >= 0; --d) {
        if (++c.coordinates[d] <= c1[d])
          break;
        c.coordinates[d] = c0[d];
      }
      if (d < 0)
        break;
    }
  }

  const std::unordered_map<Key, int64_t, KeyHash> &index_map() const;
  void build_geometry() const;

  mutable std::once_flag nodes_once_;
  mutable NodeTable nodes_;
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
