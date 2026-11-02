#include "kernel_map.hpp"
#include "adaptive_topology.hpp"

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_set>

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
  const int64_t n = in->size();
  std::vector<Key> keys(static_cast<size_t>(n));
  detail::parallel_blocks(n, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const Key &k = in->keys[i];
      const int64_t l = std::max<int64_t>(static_cast<int64_t>(k.level) + shift, 0);
      keys[i] = detail::ancestor_at(k, static_cast<uint32_t>(l));
    }
  });
  // Inputs are sorted by (level, coordinates) and coarsening keeps the order within a level,
  // so a stable sort by level only would do; a full sort keeps this independent of that.
  std::sort(keys.begin(), keys.end(), detail::key_less);
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  // Drop keys with an ancestor in the set (only levels that hold keys can contain one).
  std::vector<uint32_t> levels;
  for (const Key &k : keys)
    if (levels.empty() || levels.back() != k.level)
      levels.push_back(k.level);
  if (levels.size() < 2)
    return Topology::make(in->config, std::move(keys));
  std::unordered_set<Key, detail::KeyHash> present;
  present.reserve(keys.size() * 2);
  for (const Key &k : keys)
    if (k.level != levels.back()) // the finest level is never an ancestor
      present.insert(k);
  const int64_t m = static_cast<int64_t>(keys.size());
  std::vector<char> covered(static_cast<size_t>(m), 0);
  detail::parallel_blocks(m, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i)
      for (uint32_t l : levels) {
        if (l >= keys[i].level)
          break;
        if (present.count(detail::ancestor_at(keys[i], l))) {
          covered[i] = 1;
          break;
        }
      }
  });
  std::vector<Key> kept;
  kept.reserve(keys.size());
  for (int64_t i = 0; i < m; ++i)
    if (!covered[i])
      kept.push_back(keys[i]);
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

  // Outputs are split into contiguous blocks built on separate threads. Each block keeps its
  // own per-slot lists; concatenating them slot by slot in block order reproduces the serial
  // order (slot, then output, then input leaf index) exactly, independent of thread count.
  const int64_t n_out = out.size();
  const bool overlap = kc.mode == AdaptiveConvMode::OverlapWeighted;
  const bool averaged = kc.mode == AdaptiveConvMode::CenterAveraged;
  const int blocks =
      detail::parallel_block_count(n_out, n_out * nk * (overlap || averaged ? 4 : 1));
  struct Lists {
    std::vector<std::vector<int64_t>> in_k, out_k;
    std::vector<std::vector<double>> w_k;
  };
  std::vector<Lists> lists(static_cast<size_t>(blocks));
  if (n_out > 0)
    in.node_table(); // build the shared lookup table once, before the workers start
  const double *in_volumes = in.volumes.data_ptr<double>();
  detail::run_parallel_blocks(n_out, blocks, [&](int b, int64_t o_begin, int64_t o_end) {
    auto &in_k = lists[b].in_k, &out_k = lists[b].out_k;
    auto &w_k = lists[b].w_k;
    in_k.resize(static_cast<size_t>(nk));
    out_k.resize(static_cast<size_t>(nk));
    w_k.resize(static_cast<size_t>(nk));
    std::vector<std::pair<int64_t, double>> avg; // CenterAveraged scratch
    for (int64_t o = o_begin; o < o_end; ++o) {
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
        if (overlap) {
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
          if (averaged && (j < 0 || in.keys[j].level > ok.level)) {
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
          w_k[k].push_back(kc.mode == AdaptiveConvMode::VolumeWeighted ? in_volumes[j] / Vo
                                                                       : 1.0);
        }
      }
    }
  });

  AdaptiveKernelMap m;
  m.num_kernels = nk;
  m.kernel_offsets.assign(static_cast<size_t>(nk) + 1, 0);
  // start[k * blocks + b] = position of block b's slot-k pairs in the final arrays.
  std::vector<int64_t> start(static_cast<size_t>(nk * blocks));
  int64_t P = 0;
  for (int64_t k = 0; k < nk; ++k) {
    for (int b = 0; b < blocks; ++b) {
      start[k * blocks + b] = P;
      P += static_cast<int64_t>(lists[b].in_k[k].size());
    }
    m.kernel_offsets[k + 1] = P;
  }
  m.in_index = torch::empty({P}, torch::kInt64);
  m.out_index = torch::empty({P}, torch::kInt64);
  m.kernel_index = torch::empty({P}, torch::kInt64);
  m.weight = torch::empty({P}, torch::kFloat64);
  int64_t *in_p = m.in_index.data_ptr<int64_t>();
  int64_t *out_p = m.out_index.data_ptr<int64_t>();
  int64_t *k_p = m.kernel_index.data_ptr<int64_t>();
  double *w_p = m.weight.data_ptr<double>();
  const int copy_blocks =
      P >= (int64_t(1) << 18) ? static_cast<int>(std::min<int64_t>(blocks, nk)) : 1;
  detail::run_parallel_blocks(nk, copy_blocks, [&](int, int64_t k_begin, int64_t k_end) {
    for (int64_t k = k_begin; k < k_end; ++k)
      for (int b = 0; b < blocks; ++b) {
        const auto &src_in = lists[b].in_k[k];
        const int64_t at = start[k * blocks + b];
        std::copy(src_in.begin(), src_in.end(), in_p + at);
        std::copy(lists[b].out_k[k].begin(), lists[b].out_k[k].end(), out_p + at);
        std::copy(lists[b].w_k[k].begin(), lists[b].w_k[k].end(), w_p + at);
        std::fill(k_p + at, k_p + at + static_cast<int64_t>(src_in.size()), k);
      }
  });
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
