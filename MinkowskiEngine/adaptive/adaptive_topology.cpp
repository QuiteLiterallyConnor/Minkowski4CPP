#include "adaptive_topology.hpp"

#include <atomic>
#include <stdexcept>

namespace minkowski {
namespace adaptive {
namespace detail {

std::shared_ptr<const Topology> Topology::make(const AdaptiveVoxelConfig &cfg,
                                               std::vector<Key> sorted_keys) {
  cfg.validate();
  static std::atomic<uint64_t> counter{0};

  auto t = std::make_shared<Topology>();
  t->id = ++counter;
  t->D = static_cast<int>(cfg.dimension());
  t->max_depth = cfg.max_depth;
  t->config = cfg;
  if (t->config.origin.empty())
    t->config.origin.assign(t->D, 0.0);
  for (int d = 0; d < t->D; ++d) {
    t->origin[d] = t->config.origin[d];
    t->base[d] = cfg.base_voxel_size[d];
  }
  t->keys = std::move(sorted_keys);

  const int64_t N = t->size();
  t->index.reserve(static_cast<size_t>(N) * 2);
  for (int64_t i = 0; i < N; ++i) {
    const Key &k = t->keys[i];
    if (k.level > t->max_depth)
      throw std::invalid_argument("adaptive voxel level exceeds max_depth");
    if (i > 0 && !key_less(t->keys[i - 1], k))
      throw std::invalid_argument("duplicate or unsorted adaptive voxel keys");
    t->index.emplace(k, i);
  }
  for (const Key &k : t->keys)
    for (uint32_t l = 0; l < k.level; ++l)
      if (t->index.count(ancestor_at(k, l)))
        throw std::invalid_argument(
            "overlapping active voxels: a leaf and one of its ancestors are both active");

  const int D = t->D;
  t->coords = torch::zeros({N, D}, torch::kInt64);
  t->levels = torch::zeros({N}, torch::kInt32);
  t->lo = torch::zeros({N, D}, torch::kFloat64);
  t->hi = torch::zeros({N, D}, torch::kFloat64);
  t->sizes = torch::zeros({N, D}, torch::kFloat64);
  t->volumes = torch::zeros({N}, torch::kFloat64);
  auto *c = t->coords.data_ptr<int64_t>();
  auto *lv = t->levels.data_ptr<int32_t>();
  auto *lo = t->lo.data_ptr<double>();
  auto *hi = t->hi.data_ptr<double>();
  auto *sz = t->sizes.data_ptr<double>();
  auto *vol = t->volumes.data_ptr<double>();
  t->counts.assign(t->max_depth + 1, 0);
  for (int64_t i = 0; i < N; ++i) {
    const Key &k = t->keys[i];
    DVec l, h;
    t->bounds(k, l, h);
    const DVec s = t->size_at(k.level);
    double v = 1.0;
    for (int d = 0; d < D; ++d) {
      c[i * D + d] = k.coordinates[d];
      lo[i * D + d] = l[d];
      hi[i * D + d] = h[d];
      sz[i * D + d] = s[d];
      v *= s[d];
    }
    lv[i] = static_cast<int32_t>(k.level);
    vol[i] = v;
    ++t->counts[k.level];
  }
  for (uint32_t l = 0; l <= t->max_depth; ++l)
    if (t->counts[l] > 0)
      t->present_levels.push_back(l);
  return t;
}

} // namespace detail
} // namespace adaptive
} // namespace minkowski
