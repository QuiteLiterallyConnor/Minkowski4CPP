#include "kernel_map.hpp"
#include "adaptive_topology.hpp"

#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>

namespace minkowski {
namespace adaptive {

using detail::DVec;
using detail::Key;
using detail::Topology;

int64_t kernel_index(std::array<int, 3> offset, int kernel_size) {
  const int r = (kernel_size - 1) / 2;
  return (static_cast<int64_t>(offset[0] + r) * kernel_size + (offset[1] + r)) * kernel_size +
         (offset[2] + r);
}

namespace {

int64_t num_kernel_slots(int kernel_size, int D) {
  int64_t n = 1;
  for (int d = 0; d < D; ++d)
    n *= kernel_size;
  return n;
}

// Output leaves of a strided convolution: every input leaf mapped `shift` levels
// coarser (clamped at 0). If a coarsened key ends up under another output key,
// only the coarser one is kept so outputs never overlap.
std::shared_ptr<const Topology> derive_output_topology(const std::shared_ptr<const Topology> &in,
                                                       int shift) {
  if (shift == 0)
    return in;
  if (shift > 0)
    throw std::invalid_argument("output_level_shift > 0 (refining outputs) is not supported");
  std::set<Key, bool (*)(const Key &, const Key &)> keys(detail::key_less);
  for (const Key &k : in->keys) {
    const int64_t l = std::max<int64_t>(static_cast<int64_t>(k.level) + shift, 0);
    keys.insert(detail::ancestor_at(k, static_cast<uint32_t>(l)));
  }
  std::vector<Key> kept;
  for (const Key &k : keys) {
    bool covered = false;
    for (uint32_t l = 0; l < k.level && !covered; ++l)
      covered = keys.count(detail::ancestor_at(k, l)) > 0;
    if (!covered)
      kept.push_back(k);
  }
  return Topology::make(in->config, std::move(kept));
}

AdaptiveKernelMap build_pairs(const Topology &out, const Topology &in,
                              const AdaptiveKernelConfig &kc) {
  const int K = kc.kernel_size;
  if (K < 1 || K % 2 == 0)
    throw std::invalid_argument("kernel_size must be a positive odd number");
  if (out.D != in.D)
    throw std::invalid_argument("dimension mismatch between input and output topology");
  const int D = out.D;
  const int r = (K - 1) / 2;
  const int64_t nk = num_kernel_slots(K, D);

  std::vector<std::array<int, detail::kMaxDim>> offsets(static_cast<size_t>(nk));
  for (int64_t idx = 0; idx < nk; ++idx) {
    int64_t rem = idx;
    for (int d = D - 1; d >= 0; --d) { // first dimension slowest
      offsets[idx][d] = static_cast<int>(rem % K) - r;
      rem /= K;
    }
  }

  std::vector<std::vector<int64_t>> in_k(static_cast<size_t>(nk)), out_k(static_cast<size_t>(nk));
  std::vector<std::vector<double>> w_k(static_cast<size_t>(nk));
  std::vector<std::pair<int64_t, double>> avg; // CenterAveraged scratch

  for (int64_t o = 0; o < out.size(); ++o) {
    const Key &ok = out.keys[o];
    DVec lo, hi;
    out.bounds(ok, lo, hi);
    const DVec s = out.size_at(ok.level);
    double Vo = 1.0;
    DVec c{};
    for (int d = 0; d < D; ++d) {
      Vo *= s[d];
      c[d] = lo[d] + 0.5 * s[d];
    }
    for (int64_t k = 0; k < nk; ++k) {
      DVec t{};
      for (int d = 0; d < D; ++d)
        t[d] = c[d] + offsets[k][d] * s[d];
      if (kc.mode == AdaptiveConvMode::OverlapWeighted) {
        DVec blo{}, bhi{};
        for (int d = 0; d < D; ++d) {
          blo[d] = t[d] - 0.5 * s[d];
          bhi[d] = t[d] + 0.5 * s[d];
        }
        in.for_each_overlap(blo, bhi, [&](int64_t j, double ov) {
          in_k[k].push_back(j);
          out_k[k].push_back(o);
          w_k[k].push_back(ov / Vo);
        });
      } else {
        const int64_t j = in.find_containing(t);
        if (kc.mode == AdaptiveConvMode::CenterAveraged &&
            (j < 0 || in.keys[j].level > ok.level)) {
          // The tap cell is split into finer leaves (or empty at this level): average them.
          DVec blo{}, bhi{};
          for (int d = 0; d < D; ++d) {
            blo[d] = t[d] - 0.5 * s[d];
            bhi[d] = t[d] + 0.5 * s[d];
          }
          avg.clear();
          double total = 0.0;
          in.for_each_overlap(blo, bhi, [&](int64_t i, double ov) {
            avg.emplace_back(i, ov);
            total += ov;
          });
          for (const auto &h : avg) {
            in_k[k].push_back(h.first);
            out_k[k].push_back(o);
            w_k[k].push_back(h.second / total);
          }
          continue;
        }
        if (j < 0)
          continue;
        in_k[k].push_back(j);
        out_k[k].push_back(o);
        w_k[k].push_back(kc.mode == AdaptiveConvMode::VolumeWeighted
                             ? in.volumes.data_ptr<double>()[j] / Vo
                             : 1.0);
      }
    }
  }

  AdaptiveKernelMap m;
  m.num_kernels = nk;
  m.kernel_offsets.assign(static_cast<size_t>(nk) + 1, 0);
  std::vector<int64_t> in_all, out_all, k_all;
  std::vector<double> w_all;
  for (int64_t k = 0; k < nk; ++k) {
    in_all.insert(in_all.end(), in_k[k].begin(), in_k[k].end());
    out_all.insert(out_all.end(), out_k[k].begin(), out_k[k].end());
    w_all.insert(w_all.end(), w_k[k].begin(), w_k[k].end());
    k_all.insert(k_all.end(), in_k[k].size(), k);
    m.kernel_offsets[k + 1] = static_cast<int64_t>(in_all.size());
  }
  const int64_t P = static_cast<int64_t>(in_all.size());
  auto to_t = [&](std::vector<int64_t> &v) {
    return torch::from_blob(v.data(), {P}, torch::kInt64).clone();
  };
  m.in_index = to_t(in_all);
  m.out_index = to_t(out_all);
  m.kernel_index = to_t(k_all);
  m.weight = torch::from_blob(w_all.data(), {P}, torch::kFloat64).clone();
  return m;
}

using CacheKey = std::tuple<uint64_t, int, int, int>;

} // namespace

struct KernelMapCache::Impl {
  mutable std::mutex mutex;
  std::map<CacheKey, KernelMapEntry> entries;
  std::deque<CacheKey> order;
  size_t capacity = 64;
  size_t hits = 0;
  size_t misses = 0;
};

KernelMapCache::KernelMapCache() : impl_(new Impl) {}
KernelMapCache::~KernelMapCache() = default;

KernelMapCache &KernelMapCache::instance() {
  static KernelMapCache cache;
  return cache;
}

KernelMapEntry KernelMapCache::get_or_build(const AdaptiveSparseTensor &input,
                                            const AdaptiveKernelConfig &config) {
  const CacheKey key{input.topology()->id, config.kernel_size, static_cast<int>(config.mode),
                     config.output_level_shift};
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto it = impl_->entries.find(key);
    if (it != impl_->entries.end()) {
      ++impl_->hits;
      return it->second;
    }
    ++impl_->misses;
  }
  KernelMapEntry e;
  e.out_topology = derive_output_topology(input.topology(), config.output_level_shift);
  e.map = build_pairs(*e.out_topology, *input.topology(), config);
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->entries.emplace(key, e).second)
      impl_->order.push_back(key);
    while (impl_->entries.size() > impl_->capacity && !impl_->order.empty()) {
      impl_->entries.erase(impl_->order.front());
      impl_->order.pop_front();
    }
  }
  return e;
}

void KernelMapCache::clear() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->entries.clear();
  impl_->order.clear();
  impl_->hits = 0;
  impl_->misses = 0;
}

KernelMapCache::Stats KernelMapCache::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return {impl_->hits, impl_->misses, impl_->entries.size()};
}

void KernelMapCache::set_capacity(size_t max_entries) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->capacity = std::max<size_t>(max_entries, 1);
  while (impl_->entries.size() > impl_->capacity && !impl_->order.empty()) {
    impl_->entries.erase(impl_->order.front());
    impl_->order.pop_front();
  }
}

AdaptiveKernelMap build_adaptive_kernel_map(const AdaptiveSparseTensor &input,
                                            const AdaptiveKernelConfig &config) {
  return KernelMapCache::instance().get_or_build(input, config).map;
}

} // namespace adaptive
} // namespace minkowski
