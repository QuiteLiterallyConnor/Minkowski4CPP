#include "adaptive_tensor.hpp"
#include "adaptive_topology.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace minkowski {
namespace adaptive {

using detail::DVec;
using detail::Key;
using detail::Topology;

namespace {

struct Entry {
  Key key;
  int64_t src;
  double scale;
};

struct StepResult {
  std::shared_ptr<const Topology> topology;
  at::Tensor features;
};

at::Tensor index_tensor(const std::vector<int64_t> &v, const torch::Device &dev) {
  return torch::from_blob(const_cast<int64_t *>(v.data()),
                          {static_cast<int64_t>(v.size())}, torch::kInt64)
      .clone()
      .to(dev);
}

// out[dst[m]] (reduce)= scale[m] * f[src[m]], built from differentiable torch ops so
// gradients reach f. Rows of `out` without entries stay zero.
at::Tensor remap_features(const at::Tensor &f, const std::vector<int64_t> &src,
                          const std::vector<int64_t> &dst, const std::vector<double> *scale,
                          int64_t nnew, Reduction reduction) {
  TORCH_CHECK(f.dim() == 2, "adaptive features must be 2-D [N, C]");
  const auto dev = f.device();
  const int64_t M = static_cast<int64_t>(src.size());
  const int64_t C = f.size(1);
  auto src_t = index_tensor(src, dev);
  auto dst_t = index_tensor(dst, dev);
  auto g = f.index_select(0, src_t);
  if (scale) {
    auto sc = torch::from_blob(const_cast<double *>(scale->data()), {M}, torch::kFloat64)
                  .clone()
                  .to(f.scalar_type())
                  .to(dev)
                  .unsqueeze(1);
    g = g * sc;
  }
  auto out = torch::zeros({nnew, C}, f.options());
  if (M == 0) // still record the (empty) dependency on f so the output stays in the graph
    return out.index_add_(0, dst_t, g);
  switch (reduction) {
  case Reduction::Sum:
    return out.index_add_(0, dst_t, g);
  case Reduction::Mean: {
    std::vector<double> cnt(static_cast<size_t>(nnew), 0.0);
    for (int64_t d : dst)
      cnt[static_cast<size_t>(d)] += 1.0;
    for (double &c : cnt)
      c = std::max(c, 1.0);
    auto cnt_t = torch::from_blob(cnt.data(), {nnew}, torch::kFloat64)
                     .clone()
                     .to(f.scalar_type())
                     .to(dev)
                     .unsqueeze(1);
    return out.index_add_(0, dst_t, g) / cnt_t;
  }
  case Reduction::Max:
    return torch::scatter_reduce(out, 0, dst_t.unsqueeze(1).expand({M, C}), g, "amax", false);
  }
  return out;
}

// Sorts entries by key, merges equal keys into one new leaf and builds features.
StepResult finish(const AdaptiveVoxelConfig &cfg, std::vector<Entry> entries,
                  const at::Tensor &features, Reduction reduction, bool use_scale) {
  std::stable_sort(entries.begin(), entries.end(),
                   [](const Entry &a, const Entry &b) { return detail::key_less(a.key, b.key); });
  std::vector<Key> keys;
  std::vector<int64_t> src, dst;
  std::vector<double> scale;
  src.reserve(entries.size());
  dst.reserve(entries.size());
  for (const Entry &e : entries) {
    if (keys.empty() || !(keys.back() == e.key))
      keys.push_back(e.key);
    src.push_back(e.src);
    dst.push_back(static_cast<int64_t>(keys.size()) - 1);
    scale.push_back(e.scale);
  }
  const int64_t nnew = static_cast<int64_t>(keys.size());
  auto feats = remap_features(features, src, dst, use_scale ? &scale : nullptr, nnew, reduction);
  return {Topology::make(cfg, std::move(keys)), feats};
}

bool intersects(const Topology &t, const Key &k, const AxisAlignedRegion &r) {
  DVec lo, hi;
  t.bounds(k, lo, hi);
  for (int d = 0; d < t.D; ++d)
    if (!(lo[d] < r.hi[d] && hi[d] > r.lo[d]))
      return false;
  return true;
}

bool inside(const Topology &t, const Key &k, const AxisAlignedRegion &r) {
  DVec lo, hi;
  t.bounds(k, lo, hi);
  for (int d = 0; d < t.D; ++d)
    if (!(lo[d] >= r.lo[d] && hi[d] <= r.hi[d]))
      return false;
  return true;
}

void check_region(const Topology &t, const AxisAlignedRegion &r) {
  if (r.lo.size() != static_cast<size_t>(t.D) || r.hi.size() != static_cast<size_t>(t.D))
    throw std::invalid_argument("region dimension does not match the tensor dimension");
}

// Replaces every selected leaf by its 2^D children.
StepResult refine_step(const Topology &t, const at::Tensor &f, const std::vector<char> &sel,
                       RefinePolicy policy) {
  std::vector<Entry> entries;
  std::vector<Key> kids;
  const double split = policy == RefinePolicy::Split ? 1.0 / double(1u << t.D) : 1.0;
  for (int64_t i = 0; i < t.size(); ++i) {
    if (!sel[static_cast<size_t>(i)]) {
      entries.push_back({t.keys[i], i, 1.0});
      continue;
    }
    kids.clear();
    detail::child_keys(t.keys[i], t.D, kids);
    for (const Key &c : kids)
      entries.push_back({c, i, split});
  }
  return finish(t.config, std::move(entries), f, Reduction::Sum, policy == RefinePolicy::Split);
}

// Merges leaves at `level` into their parents when the parent box is inside region.
StepResult coarsen_step(const Topology &t, const at::Tensor &f, const AxisAlignedRegion &region,
                        uint32_t level, Reduction reduction, bool &changed) {
  std::vector<Entry> entries;
  changed = false;
  for (int64_t i = 0; i < t.size(); ++i) {
    const Key &k = t.keys[i];
    if (k.level == level && level >= 1) {
      const Key p = detail::parent_key(k);
      if (inside(t, p, region)) {
        entries.push_back({p, i, 1.0});
        changed = true;
        continue;
      }
    }
    entries.push_back({k, i, 1.0});
  }
  if (!changed)
    return {};
  return finish(t.config, std::move(entries), f, reduction, false);
}

// Leaf index per point (or -1).
std::vector<int64_t> assign_points(const Topology &t, const at::Tensor &points) {
  const int64_t P = points.size(0);
  const double *p = points.data_ptr<double>();
  std::vector<int64_t> out(static_cast<size_t>(P), -1);
  for (int64_t i = 0; i < P; ++i) {
    DVec q{};
    for (int d = 0; d < t.D; ++d)
      q[d] = p[i * t.D + d];
    out[static_cast<size_t>(i)] = t.find_containing(q);
  }
  return out;
}

void check_features(const at::Tensor &f) {
  TORCH_CHECK(f.dim() == 2, "adaptive features must be 2-D [N, C]");
}

} // namespace

AdaptiveSparseTensor AdaptiveSparseTensor::from_topology(
    std::shared_ptr<const Topology> topology, at::Tensor features,
    std::shared_ptr<const SourcePoints> source) {
  check_features(features);
  if (features.size(0) != topology->size())
    throw std::invalid_argument("feature rows (" + std::to_string(features.size(0)) +
                                ") do not match the number of voxels (" +
                                std::to_string(topology->size()) + ")");
  AdaptiveSparseTensor t;
  t.topology_ = std::move(topology);
  t.features_ = std::move(features);
  t.source_ = std::move(source);
  return t;
}

AdaptiveSparseTensor AdaptiveSparseTensor::from_voxels(const at::Tensor &coordinates,
                                                       const at::Tensor &levels,
                                                       const at::Tensor &features,
                                                       const AdaptiveVoxelConfig &config) {
  config.validate();
  check_features(features);
  const int D = static_cast<int>(config.dimension());
  auto coords = coordinates.to(torch::kInt64).cpu().contiguous();
  auto lv = levels.to(torch::kInt32).cpu().contiguous();
  if (coords.dim() != 2 || coords.size(1) != D)
    throw std::invalid_argument("coordinates must have shape [N, D]");
  const int64_t N = coords.size(0);
  if (lv.numel() != N || features.size(0) != N)
    throw std::invalid_argument("coordinates, levels and features must have the same length");

  std::vector<Key> keys(static_cast<size_t>(N));
  const auto *c = coords.data_ptr<int64_t>();
  const auto *l = lv.data_ptr<int32_t>();
  detail::parallel_blocks(N, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      if (l[i] < 0)
        throw std::invalid_argument("negative voxel level");
      Key &key = keys[i];
      key = Key{};
      key.level = static_cast<uint32_t>(l[i]);
      for (int d = 0; d < D; ++d)
        key.coordinates[d] = c[i * D + d];
    }
  });
  std::vector<int64_t> order;
  detail::sort_keys(keys, order);
  auto feats = order.empty() ? features
                             : features.index_select(0, index_tensor(order, features.device()));
  return from_topology(Topology::make(config, std::move(keys)), feats);
}

AdaptiveSparseTensor AdaptiveSparseTensor::from_points(const at::Tensor &points,
                                                       const at::Tensor &features,
                                                       const AdaptiveVoxelConfig &config) {
  config.validate();
  check_features(features);
  const int D = static_cast<int>(config.dimension());
  auto pts = points.to(torch::kFloat64).cpu().contiguous();
  if (pts.dim() != 2 || pts.size(1) != D)
    throw std::invalid_argument("points must have shape [P, D]");
  const int64_t P = pts.size(0);
  if (features.size(0) != P)
    throw std::invalid_argument("points and features must have the same length");

  AdaptiveVoxelConfig cfg = config;
  if (cfg.origin.empty())
    cfg.origin.assign(D, 0.0);
  const double *p = pts.data_ptr<double>();
  std::unordered_map<Key, int64_t, detail::KeyHash> group;
  std::vector<Key> gkeys;
  std::vector<int64_t> point_group(static_cast<size_t>(P));
  for (int64_t i = 0; i < P; ++i) {
    Key k{};
    for (int d = 0; d < D; ++d) {
      const double v = std::floor((p[i * D + d] - cfg.origin[d]) / cfg.base_voxel_size[d]);
      if (!std::isfinite(v) || std::fabs(v) > 4e18)
        throw std::invalid_argument("point coordinate is not finite or out of range");
      k.coordinates[d] = static_cast<int64_t>(v);
    }
    auto it = group.emplace(k, static_cast<int64_t>(gkeys.size()));
    if (it.second)
      gkeys.push_back(k);
    point_group[i] = it.first->second;
  }
  auto order = detail::sorted_order(gkeys);
  std::vector<int64_t> pos(gkeys.size());
  std::vector<Key> sorted(gkeys.size());
  for (size_t i = 0; i < order.size(); ++i) {
    sorted[i] = gkeys[order[i]];
    pos[order[i]] = static_cast<int64_t>(i);
  }
  std::vector<int64_t> src(static_cast<size_t>(P)), dst(static_cast<size_t>(P));
  for (int64_t i = 0; i < P; ++i) {
    src[i] = i;
    dst[i] = pos[point_group[i]];
  }
  auto feats = remap_features(features, src, dst, nullptr,
                              static_cast<int64_t>(sorted.size()), Reduction::Mean);
  auto source = std::make_shared<SourcePoints>();
  source->points = pts;
  // Not detached: revoxelize_from_points() rebuilds leaf features from these, and gradients
  // must reach the point features through it (refinement criteria read them under no_grad).
  source->features = features;
  return from_topology(Topology::make(cfg, std::move(sorted)), feats, source);
}

int64_t AdaptiveSparseTensor::size() const { return topology_->size(); }
int AdaptiveSparseTensor::dimension() const { return topology_->D; }
int64_t AdaptiveSparseTensor::channels() const { return features_.size(1); }
const AdaptiveVoxelConfig &AdaptiveSparseTensor::config() const { return topology_->config; }
at::Tensor AdaptiveSparseTensor::coordinates() const { return topology_->coords; }
at::Tensor AdaptiveSparseTensor::levels() const { return topology_->levels; }
at::Tensor AdaptiveSparseTensor::voxel_sizes() const { return topology_->sizes_tensor(); }
at::Tensor AdaptiveSparseTensor::volumes() const { return topology_->volumes; }
at::Tensor AdaptiveSparseTensor::centers() const {
  return topology_->centers_tensor();
}
std::pair<at::Tensor, at::Tensor> AdaptiveSparseTensor::bounds() const {
  return {topology_->lo_tensor(), topology_->hi_tensor()};
}
std::vector<int64_t> AdaptiveSparseTensor::counts_by_level() const { return topology_->counts; }
uint64_t AdaptiveSparseTensor::topology_version() const { return topology_->id; }
bool AdaptiveSparseTensor::has_overlapping_leaves() const { return topology_->has_overlap(); }

void AdaptiveSparseTensor::set_features(at::Tensor features) {
  check_features(features);
  if (features.size(0) != size())
    throw std::invalid_argument("feature rows do not match the number of voxels");
  features_ = std::move(features);
}

int64_t AdaptiveSparseTensor::index_of(std::array<int64_t, 3> coordinates, uint32_t level) const {
  return index_of_key(std::vector<int64_t>(coordinates.begin(), coordinates.end()), level);
}

int64_t AdaptiveSparseTensor::index_of_key(const std::vector<int64_t> &coordinates,
                                           uint32_t level) const {
  if (coordinates.size() != static_cast<size_t>(topology_->D))
    throw std::invalid_argument("coordinate dimension does not match the tensor dimension");
  Key k{};
  k.level = level;
  for (size_t d = 0; d < coordinates.size(); ++d)
    k.coordinates[d] = coordinates[d];
  return topology_->find(k);
}

void AdaptiveSparseTensor::refine(const AxisAlignedRegion &region, uint32_t target_level,
                                  RefinePolicy policy) {
  const Topology &t0 = *topology_;
  check_region(t0, region);
  if (target_level > t0.max_depth)
    throw std::invalid_argument("refine target_level exceeds max_depth");
  while (true) {
    const Topology &t = *topology_;
    std::vector<char> sel(static_cast<size_t>(t.size()), 0);
    bool any = false;
    for (int64_t i = 0; i < t.size(); ++i)
      if (t.keys[i].level < target_level && intersects(t, t.keys[i], region)) {
        sel[i] = 1;
        any = true;
      }
    if (!any)
      break;
    auto r = refine_step(t, features_, sel, policy);
    topology_ = std::move(r.topology);
    features_ = std::move(r.features);
  }
}

void AdaptiveSparseTensor::coarsen(const AxisAlignedRegion &region, uint32_t target_level,
                                   Reduction reduction) {
  check_region(*topology_, region);
  for (int64_t level = static_cast<int64_t>(topology_->max_depth);
       level > static_cast<int64_t>(target_level); --level) {
    bool changed = false;
    auto r = coarsen_step(*topology_, features_, region, static_cast<uint32_t>(level), reduction,
                          changed);
    if (changed) {
      topology_ = std::move(r.topology);
      features_ = std::move(r.features);
    }
  }
}

void AdaptiveSparseTensor::refine_if(
    const std::function<bool(int64_t, const std::array<double, 4> &, const std::array<double, 4> &,
                             uint32_t)> &criterion) {
  while (true) {
    const Topology &t = *topology_;
    std::vector<char> sel(static_cast<size_t>(t.size()), 0);
    bool any = false;
    for (int64_t i = 0; i < t.size(); ++i) {
      if (t.keys[i].level >= t.max_depth)
        continue;
      DVec lo, hi;
      t.bounds(t.keys[i], lo, hi);
      if (criterion(i, lo, hi, t.keys[i].level)) {
        sel[i] = 1;
        any = true;
      }
    }
    if (!any)
      break;
    auto r = refine_step(t, features_, sel, RefinePolicy::Replicate);
    topology_ = std::move(r.topology);
    features_ = std::move(r.features);
  }
}

void AdaptiveSparseTensor::refine_by_density(int64_t threshold) {
  if (!source_)
    throw std::invalid_argument("refine_by_density requires source points (use from_points)");
  while (true) {
    const Topology &t = *topology_;
    std::vector<int64_t> count(static_cast<size_t>(t.size()), 0);
    for (int64_t j : assign_points(t, source_->points))
      if (j >= 0)
        ++count[j];
    std::vector<char> sel(static_cast<size_t>(t.size()), 0);
    bool any = false;
    for (int64_t i = 0; i < t.size(); ++i)
      if (count[i] > threshold && t.keys[i].level < t.max_depth) {
        sel[i] = 1;
        any = true;
      }
    if (!any)
      break;
    auto r = refine_step(t, features_, sel, RefinePolicy::Replicate);
    topology_ = std::move(r.topology);
    features_ = std::move(r.features);
  }
}

void AdaptiveSparseTensor::refine_by_variance(double threshold) {
  if (!source_)
    throw std::invalid_argument("refine_by_variance requires source points (use from_points)");
  while (true) {
    const Topology &t = *topology_;
    const auto leaf = assign_points(t, source_->points);
    std::vector<int64_t> pidx, lidx;
    for (size_t p = 0; p < leaf.size(); ++p)
      if (leaf[p] >= 0) {
        pidx.push_back(static_cast<int64_t>(p));
        lidx.push_back(leaf[p]);
      }
    std::vector<char> sel(static_cast<size_t>(t.size()), 0);
    bool any = false;
    if (!pidx.empty()) {
      torch::NoGradGuard no_grad; // the variance only drives a topology decision
      auto x = source_->features.index_select(0, index_tensor(pidx, source_->features.device()))
                   .to(torch::kFloat64)
                   .cpu();
      auto li = index_tensor(lidx, torch::kCPU);
      const int64_t C = x.size(1);
      auto cnt = torch::zeros({t.size()}, torch::kFloat64).index_add_(0, li,
                                                                      torch::ones({(int64_t)lidx.size()}, torch::kFloat64));
      auto s1 = torch::zeros({t.size(), C}, torch::kFloat64).index_add_(0, li, x);
      auto s2 = torch::zeros({t.size(), C}, torch::kFloat64).index_add_(0, li, x * x);
      auto n = cnt.clamp_min(1.0).unsqueeze(1);
      auto var = (s2 / n - (s1 / n) * (s1 / n)).clamp_min(0.0).mean(1);
      const double *v = var.data_ptr<double>();
      for (int64_t i = 0; i < t.size(); ++i)
        if (v[i] > threshold && t.keys[i].level < t.max_depth) {
          sel[i] = 1;
          any = true;
        }
    }
    if (!any)
      break;
    auto r = refine_step(t, features_, sel, RefinePolicy::Replicate);
    topology_ = std::move(r.topology);
    features_ = std::move(r.features);
  }
}

void AdaptiveSparseTensor::revoxelize_from_points() {
  if (!source_)
    throw std::invalid_argument("revoxelize_from_points requires source points");
  const Topology &t = *topology_;
  const auto leaf = assign_points(t, source_->points);
  std::vector<int64_t> src, dst;
  for (size_t p = 0; p < leaf.size(); ++p)
    if (leaf[p] >= 0) {
      src.push_back(static_cast<int64_t>(p));
      dst.push_back(leaf[p]);
    }
  features_ = remap_features(source_->features, src, dst, nullptr, t.size(), Reduction::Mean);
}

AdaptiveSparseTensor AdaptiveSparseTensor::with_features(at::Tensor features) const {
  return from_topology(topology_, std::move(features), source_);
}

AdaptiveSparseTensor AdaptiveSparseTensor::to(torch::Device device) const {
  return from_topology(topology_, features_.to(device), source_);
}

AdaptiveSparseTensor AdaptiveSparseTensor::resample_uniform(uint32_t level,
                                                            Reduction reduction) const {
  if (level > topology_->max_depth)
    throw std::invalid_argument("resample level exceeds max_depth");
  AdaptiveSparseTensor out = *this;
  out.source_ = nullptr;
  const auto all = AxisAlignedRegion::everything(static_cast<size_t>(topology_->D));
  out.refine(all, level, RefinePolicy::Replicate);
  out.coarsen(all, level, reduction);
  return out;
}

AdaptiveSparseTensor AdaptiveSparseTensor::resample_to(const AdaptiveSparseTensor &reference,
                                                       Reduction reduction) const {
  const Topology &s = *topology_;
  const Topology &r = *reference.topology_;
  if (s.D != r.D)
    throw std::invalid_argument("resample_to: dimension mismatch");
  std::vector<int64_t> src, dst;
  for (int64_t j = 0; j < r.size(); ++j) {
    const Key &rk = r.keys[j];
    // Same or coarser source leaf covering this reference leaf.
    for (uint32_t l = 0; l <= rk.level; ++l) {
      const int64_t i = s.find(detail::ancestor_at(rk, l));
      if (i >= 0) {
        src.push_back(i);
        dst.push_back(j);
        break;
      }
    }
  }
  for (int64_t i = 0; i < s.size(); ++i) {
    const Key &sk = s.keys[i];
    // Strictly finer source leaf inside a reference leaf.
    for (uint32_t l = 0; l < sk.level; ++l) {
      const int64_t j = r.find(detail::ancestor_at(sk, l));
      if (j >= 0) {
        src.push_back(i);
        dst.push_back(j);
        break;
      }
    }
  }
  auto feats = remap_features(features_, src, dst, nullptr, r.size(), reduction);
  return from_topology(reference.topology_, feats);
}

AdaptiveSparseTensor AdaptiveSparseTensor::operator+(const AdaptiveSparseTensor &other) const {
  if (!detail::same_topology(*topology_, *other.topology_))
    throw std::invalid_argument(
        "adaptive tensors have different topologies; align with resample_to() first");
  return from_topology(topology_, features_ + other.features_, source_);
}

AdaptiveSparseTensor concatenate(const AdaptiveSparseTensor &a, const AdaptiveSparseTensor &b) {
  if (!detail::same_topology(*a.topology(), *b.topology()))
    throw std::invalid_argument(
        "adaptive tensors have different topologies; align with resample_to() first");
  return AdaptiveSparseTensor::from_topology(a.topology(),
                                             torch::cat({a.features(), b.features()}, 1));
}

} // namespace adaptive
} // namespace minkowski
