#include "kernel_map.hpp"
#include "adaptive_topology.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
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

// The leaf containing the centre of grid cell `cell`, or -1: find_containing(centre) in integer
// arithmetic. At a level l <= cell.level the centre lies in the cell's ancestor; at a finer
// level l = cell.level + m in the cell's child whose lower corner is the centre (cells are
// half-open), coordinates c * 2^m + 2^(m-1). No rounding is involved, so the result depends
// only on the keys, never on where the grid sits in space.
int64_t find_containing_center(const Topology &t, const Topology::NodeTable &nodes,
                               const Key &cell) {
  Key k{};
  for (uint32_t l : t.present_levels) {
    k.level = l;
    if (l <= cell.level) {
      const uint32_t m = cell.level - l;
      for (int d = 0; d < t.D; ++d)
        k.coordinates[d] = cell.coordinates[d] >> m; // floor, also for negatives
    } else {
      const uint32_t m = l - cell.level;
      for (int d = 0; d < t.D; ++d) // |c| < 2^53 and m <= max_depth: no overflow
        k.coordinates[d] = cell.coordinates[d] * (int64_t(1) << m) + (int64_t(1) << (m - 1));
    }
    const int64_t v = nodes.lookup(k);
    if (v != Topology::NodeTable::kInternal)
      return v; // leaf index, or kEmpty (-1)
  }
  return -1;
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
  const int blocks = detail::parallel_block_count(n_out, n_out * nk * (overlap ? 4 : 1));
  struct Lists {
    std::vector<std::vector<int64_t>> in_k, out_k;
    std::vector<std::vector<double>> w_k;
  };
  std::vector<Lists> lists(static_cast<size_t>(blocks));
  // Build the shared lookup table once, before the workers start.
  const Topology::NodeTable *in_nodes = n_out > 0 ? &in.node_table() : nullptr;
  const double *in_volumes = in.volumes.data_ptr<double>();
  detail::run_parallel_blocks(n_out, blocks, [&](int b, int64_t o_begin, int64_t o_end) {
    auto &in_k = lists[b].in_k, &out_k = lists[b].out_k;
    auto &w_k = lists[b].w_k;
    in_k.resize(static_cast<size_t>(nk));
    out_k.resize(static_cast<size_t>(nk));
    w_k.resize(static_cast<size_t>(nk));
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
          // The tap point t is the centre of the output-level cell ok + offset; resolve it on
          // the integer grid (same leaf as in.find_containing(t), without rounding).
          Key cell = ok;
          for (int d = 0; d < D; ++d)
            cell.coordinates[d] += offsets[k][d];
          const int64_t j = find_containing_center(in, *in_nodes, cell);
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

// ---------------------------------------------------------------- part reuse

// Kernel map of one part (see KernelMapCache), in the part's own leaf numbering.
struct PartMap {
  std::vector<uint64_t> signature; // D, max_depth, kernel config, base voxel size bits
  std::vector<Key> keys;           // the part's leaves, translated to start at level-0 x = 0
  std::vector<int64_t> out_counts; // output leaves per level
  std::vector<int64_t> offsets;    // num_kernels + 1 prefix offsets
  std::vector<int32_t> in, out;    // pairs, sorted by slot, then out (as in the full map)
  std::vector<double> weight;      // VolumeWeighted only; CenterSampled weights are 1
  uint64_t hash = 0;

  bool weighted() const {
    return signature[3] == static_cast<uint64_t>(AdaptiveConvMode::VolumeWeighted);
  }
  size_t bytes() const {
    return sizeof(PartMap) + signature.size() * sizeof(uint64_t) + keys.size() * sizeof(Key) +
           (out_counts.size() + offsets.size()) * sizeof(int64_t) +
           (in.size() + out.size()) * sizeof(int32_t) + weight.size() * sizeof(double);
  }
};

bool part_reuse_mode(const AdaptiveKernelConfig &kc) {
  return kc.mode == AdaptiveConvMode::CenterSampled || kc.mode == AdaptiveConvMode::VolumeWeighted;
}

std::vector<uint64_t> part_signature(const Topology &t, const AdaptiveKernelConfig &kc) {
  std::vector<uint64_t> s{static_cast<uint64_t>(t.D), t.max_depth,
                          static_cast<uint64_t>(kc.kernel_size), static_cast<uint64_t>(kc.mode),
                          static_cast<uint64_t>(static_cast<int64_t>(kc.output_level_shift))};
  for (int d = 0; d < t.D; ++d) { // VolumeWeighted weights depend on the voxel size bits
    uint64_t bits;
    std::memcpy(&bits, &t.base[d], sizeof(bits));
    s.push_back(bits);
  }
  return s;
}

// The level-0 cell (along the first axis) holding a key.
inline int64_t level0_x(const Key &k) { return k.coordinates[0] >> k.level; }

// Leaves of a topology grouped into parts: runs of occupied level-0 x cells separated by at
// least r empty ones. A tap lands at most r level-0 cells from the level-0 cell of its output
// leaf (outputs are never coarser than level 0), so no tap crosses between parts.
struct Split {
  std::vector<int64_t> first_x;             // first level-0 x cell of each part, ascending
  std::vector<std::vector<int64_t>> leaves; // leaf indices of each part, ascending

  void assign(const Topology &t) {
    leaves.assign(first_x.size(), {});
    for (int64_t i = 0; i < t.size(); ++i) {
      const auto it = std::upper_bound(first_x.begin(), first_x.end(), level0_x(t.keys[i]));
      leaves[static_cast<size_t>(it - first_x.begin()) - 1].push_back(i);
    }
  }
};

// first_x stays empty when the topology does not split into at least two parts.
Split split_parts(const Topology &t, int r) {
  Split s;
  std::vector<int64_t> xs(static_cast<size_t>(t.size()));
  for (int64_t i = 0; i < t.size(); ++i)
    xs[i] = level0_x(t.keys[i]);
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
  for (size_t j = 0; j < xs.size(); ++j)
    if (j == 0 || xs[j] - xs[j - 1] > r)
      s.first_x.push_back(xs[j]);
  if (s.first_x.size() < 2)
    s.first_x.clear();
  return s;
}

uint64_t part_hash(const std::vector<uint64_t> &signature, const std::vector<Key> &keys) {
  uint64_t h = mix64(keys.size());
  for (uint64_t v : signature)
    h = mix64(h ^ v);
  const detail::KeyHash kh;
  for (const Key &k : keys)
    h = mix64(h ^ static_cast<uint64_t>(kh(k)));
  return h;
}

// Converts a map built on a part's own topology. False if an index does not fit in int32.
bool to_part_map(const AdaptiveKernelMap &m, const Topology &out, PartMap &pm) {
  if (out.size() > INT32_MAX || static_cast<int64_t>(pm.keys.size()) > INT32_MAX)
    return false;
  const int64_t P = m.in_index.numel();
  pm.out_counts = out.counts;
  pm.offsets = m.kernel_offsets;
  pm.in.resize(static_cast<size_t>(P));
  pm.out.resize(static_cast<size_t>(P));
  const int64_t *ip = m.in_index.data_ptr<int64_t>();
  const int64_t *op = m.out_index.data_ptr<int64_t>();
  for (int64_t q = 0; q < P; ++q) {
    pm.in[q] = static_cast<int32_t>(ip[q]);
    pm.out[q] = static_cast<int32_t>(op[q]);
  }
  if (pm.weighted()) {
    const double *wp = m.weight.data_ptr<double>();
    pm.weight.assign(wp, wp + P);
  }
  return true;
}

// Cuts the full map of a split topology into per-part maps (pm[p] already holds signature and
// keys). False if a pair links two parts or an index does not fit in int32.
bool extract_parts(const AdaptiveKernelMap &m, const Topology &in, const Topology &out,
                   const Split &in_split, const Split &out_split, std::vector<PartMap> &pm) {
  const int64_t np = static_cast<int64_t>(pm.size());
  if (in.size() > INT32_MAX || out.size() > INT32_MAX)
    return false;
  // Part of every input leaf and its index inside the part.
  std::vector<int32_t> in_part(static_cast<size_t>(in.size())), in_local(in_part.size());
  // Per part and level: the part's outputs at that level are one run of the whole topology's
  // outputs (level first, then parts along x); level_begin indexes out_split.leaves[p].
  std::vector<std::vector<int64_t>> level_begin(static_cast<size_t>(np));
  const size_t nl = out.counts.size();
  std::atomic<bool> ok{true};
  const int64_t *ip = m.in_index.data_ptr<int64_t>();
  const int64_t *op = m.out_index.data_ptr<int64_t>();
  const double *wp = m.weight.data_ptr<double>();
  const bool weighted = pm[0].weighted();
  const int blocks = detail::parallel_block_count(np, m.in_index.numel() + in.size());
  detail::run_parallel_blocks(np, blocks, [&](int, int64_t p_begin, int64_t p_end) {
    for (int64_t p = p_begin; p < p_end; ++p) {
      const auto &leaves = in_split.leaves[p];
      for (size_t i = 0; i < leaves.size(); ++i) {
        in_part[leaves[i]] = static_cast<int32_t>(p);
        in_local[leaves[i]] = static_cast<int32_t>(i);
      }
      const auto &outs = out_split.leaves[p];
      PartMap &a = pm[p];
      a.out_counts.assign(nl, 0);
      for (int64_t o : outs)
        ++a.out_counts[out.keys[o].level];
      auto &lb = level_begin[p];
      lb.assign(nl + 1, 0);
      for (size_t l = 0; l < nl; ++l) {
        lb[l + 1] = lb[l] + a.out_counts[l];
        if (a.out_counts[l] > 0 && outs[lb[l + 1] - 1] - outs[lb[l]] + 1 != a.out_counts[l])
          ok = false; // not one run: the split assumption does not hold
      }
    }
  });
  if (!ok)
    return false;
  detail::run_parallel_blocks(np, blocks, [&](int, int64_t p_begin, int64_t p_end) {
    for (int64_t p = p_begin; p < p_end && ok; ++p) {
      PartMap &a = pm[p];
      const auto &outs = out_split.leaves[p];
      const auto &lb = level_begin[p];
      a.offsets.assign(1, 0);
      for (int64_t k = 0; k < m.num_kernels; ++k) {
        const int64_t *kb = op + m.kernel_offsets[k], *ke = op + m.kernel_offsets[k + 1];
        for (size_t l = 0; l < nl; ++l) {
          if (lb[l + 1] == lb[l])
            continue;
          const int64_t first = outs[lb[l]], last = outs[lb[l + 1] - 1];
          const int64_t q0 = std::lower_bound(kb, ke, first) - op;
          const int64_t q1 = std::upper_bound(op + q0, ke, last) - op;
          for (int64_t q = q0; q < q1; ++q) {
            if (in_part[ip[q]] != p) {
              ok = false; // a tap crossed between parts
              return;
            }
            a.in.push_back(in_local[ip[q]]);
            a.out.push_back(static_cast<int32_t>(op[q] - first + lb[l]));
            if (weighted)
              a.weight.push_back(wp[q]);
          }
        }
        a.offsets.push_back(static_cast<int64_t>(a.in.size()));
      }
    }
  });
  return ok;
}

// The map of a split topology from its parts' maps. Pairs come out ordered by slot, then by
// output leaf - level first, then part (parts are ordered along the first axis, which key_less
// compares first within a level) - exactly as build_pairs orders them.
AdaptiveKernelMap compose_parts(const std::vector<std::shared_ptr<const PartMap>> &pm,
                                const Split &in_split, const Split &out_split) {
  const size_t np = pm.size();
  const int64_t nk = static_cast<int64_t>(pm[0]->offsets.size()) - 1;
  const size_t nl = pm[0]->out_counts.size();
  const bool weighted = pm[0]->weighted();
  std::vector<std::vector<int64_t>> out_begin(np); // per part: first local output of each level
  for (size_t p = 0; p < np; ++p) {
    out_begin[p].assign(nl + 1, 0);
    for (size_t l = 0; l < nl; ++l)
      out_begin[p][l + 1] = out_begin[p][l] + pm[p]->out_counts[l];
  }
  AdaptiveKernelMap m;
  m.num_kernels = nk;
  m.kernel_offsets.assign(static_cast<size_t>(nk) + 1, 0);
  for (int64_t k = 0; k < nk; ++k) {
    int64_t n = 0;
    for (size_t p = 0; p < np; ++p)
      n += pm[p]->offsets[k + 1] - pm[p]->offsets[k];
    m.kernel_offsets[k + 1] = m.kernel_offsets[k] + n;
  }
  const int64_t P = m.kernel_offsets[nk];
  m.in_index = torch::empty({P}, torch::kInt64);
  m.out_index = torch::empty({P}, torch::kInt64);
  m.kernel_index = torch::empty({P}, torch::kInt64);
  m.weight = torch::empty({P}, torch::kFloat64);
  int64_t *in_p = m.in_index.data_ptr<int64_t>();
  int64_t *out_p = m.out_index.data_ptr<int64_t>();
  int64_t *k_p = m.kernel_index.data_ptr<int64_t>();
  double *w_p = m.weight.data_ptr<double>();
  const int blocks = detail::parallel_block_count(nk, P);
  detail::run_parallel_blocks(nk, blocks, [&](int, int64_t k_begin, int64_t k_end) {
    for (int64_t k = k_begin; k < k_end; ++k) {
      int64_t at = m.kernel_offsets[k];
      for (size_t l = 0; l < nl; ++l)
        for (size_t p = 0; p < np; ++p) {
          const PartMap &a = *pm[p];
          const int32_t *ob = a.out.data() + a.offsets[k];
          const int32_t *oe = a.out.data() + a.offsets[k + 1];
          const int64_t q0 = std::lower_bound(ob, oe, out_begin[p][l]) - a.out.data();
          const int64_t q1 = std::lower_bound(ob, oe, out_begin[p][l + 1]) - a.out.data();
          const int64_t *in_map = in_split.leaves[p].data();
          const int64_t *out_map = out_split.leaves[p].data();
          for (int64_t q = q0; q < q1; ++q, ++at) {
            in_p[at] = in_map[a.in[q]];
            out_p[at] = out_map[a.out[q]];
            w_p[at] = weighted ? a.weight[q] : 1.0;
            k_p[at] = k;
          }
        }
    }
  });
  return m;
}

// Least recently used per-part maps within a byte budget. Guarded by the cache mutex.
struct PartStore {
  using List = std::list<std::shared_ptr<const PartMap>>;
  size_t budget = 0;
  size_t bytes = 0;
  size_t hits = 0;
  size_t misses = 0;
  List lru; // most recently used first
  std::unordered_multimap<uint64_t, List::iterator> index;

  std::shared_ptr<const PartMap> find(const PartMap &probe) {
    const auto range = index.equal_range(probe.hash);
    for (auto it = range.first; it != range.second; ++it) {
      const auto &e = *it->second;
      if (e->signature == probe.signature && e->keys == probe.keys) {
        lru.splice(lru.begin(), lru, it->second);
        return e;
      }
    }
    return nullptr;
  }
  void evict() {
    while (bytes > budget && !lru.empty()) {
      const auto &e = lru.back();
      const auto range = index.equal_range(e->hash);
      for (auto it = range.first; it != range.second; ++it)
        if (it->second->get() == e.get()) {
          index.erase(it);
          break;
        }
      bytes -= e->bytes();
      lru.pop_back();
    }
  }
  void insert(std::shared_ptr<const PartMap> e) {
    if (e->bytes() > budget || find(*e))
      return;
    bytes += e->bytes();
    lru.push_front(std::move(e));
    index.emplace(lru.front()->hash, lru.begin());
    evict();
  }
  void clear() {
    lru.clear();
    index.clear();
    bytes = 0;
  }
};


} // namespace

KernelMapDeviceCache::Arrays KernelMapDeviceCache::get(const AdaptiveKernelMap &map,
                                                       const at::Device &device,
                                                       at::ScalarType dtype, bool with_weight) {
  std::lock_guard<std::mutex> lock(mutex_);
  Slot *slot = nullptr;
  for (auto &s : slots_)
    if (s.device == device)
      slot = &s;
  if (!slot) {
    Slot s;
    s.device = device;
    // One upload per index array; `to` is a no-op for host maps used on the CPU.
    s.in_index = map.in_index.to(device);
    s.out_index = map.out_index.to(device);
    slots_.push_back(std::move(s));
    slot = &slots_.back();
  }
  Arrays a{slot->in_index, slot->out_index, at::Tensor()};
  if (with_weight) {
    for (const auto &w : slot->weights)
      if (w.first == dtype)
        a.weight = w.second;
    if (!a.weight.defined()) {
      // Same conversion order as before caching (move, then cast) so values are unchanged.
      a.weight = map.weight.to(device).to(dtype);
      slot->weights.emplace_back(dtype, a.weight);
    }
  }
  return a;
}

namespace detail {
int64_t &cuda_apply_map_group_bytes() {
  static int64_t bytes = int64_t(32) << 20;
  return bytes;
}
} // namespace detail

struct KernelMapCache::Impl {
  mutable std::mutex mutex;
  std::map<CacheKey, KernelMapEntry> entries;
  std::deque<CacheKey> order;
  size_t capacity = 64;
  size_t hits = 0;
  size_t misses = 0;
  PartStore parts;
};

namespace {

// Builds the entry of a topology that splits into parts, reusing stored part maps; false
// (nothing built) when part reuse does not apply.
bool build_from_parts(const std::shared_ptr<const Topology> &in, const AdaptiveKernelConfig &kc,
                      std::mutex &mutex, PartStore &store, KernelMapEntry &e) {
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (store.budget == 0)
      return false;
  }
  if (!part_reuse_mode(kc) || kc.kernel_size < 1 || kc.kernel_size % 2 == 0 ||
      kc.output_level_shift > 0)
    return false;
  Split in_split = split_parts(*in, (kc.kernel_size - 1) / 2);
  if (in_split.first_x.empty())
    return false;
  in_split.assign(*in);
  const size_t np = in_split.first_x.size();
  const std::vector<uint64_t> signature = part_signature(*in, kc);
  std::vector<PartMap> probe(np);
  for (size_t p = 0; p < np; ++p) {
    PartMap &pm = probe[p];
    pm.signature = signature;
    pm.keys.reserve(in_split.leaves[p].size());
    for (int64_t i : in_split.leaves[p]) {
      Key k = in->keys[i];
      k.coordinates[0] -= in_split.first_x[p] * (int64_t(1) << k.level);
      pm.keys.push_back(k);
    }
    pm.hash = part_hash(signature, pm.keys);
  }
  std::vector<std::shared_ptr<const PartMap>> found(np);
  size_t hits = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    for (size_t p = 0; p < np; ++p)
      if ((found[p] = store.find(probe[p])))
        ++hits;
    store.hits += hits;
    store.misses += np - hits;
  }
  e.out_topology = derive_output_topology(in, kc.output_level_shift);
  Split out_split;
  out_split.first_x = in_split.first_x;
  out_split.assign(*e.out_topology);
  std::vector<std::shared_ptr<const PartMap>> built;
  if (hits == 0) {
    // Nothing to reuse: one full build, cut into the parts' maps for later batches.
    e.map = build_pairs(*e.out_topology, *in, kc);
    if (extract_parts(e.map, *in, *e.out_topology, in_split, out_split, probe))
      for (auto &pm : probe)
        built.push_back(std::make_shared<const PartMap>(std::move(pm)));
  } else {
    for (size_t p = 0; p < np; ++p) {
      if (!found[p]) {
        std::shared_ptr<const Topology> part;
        try {
          part = Topology::make(in->config, probe[p].keys);
        } catch (const std::invalid_argument &) {
          return false; // translated keys out of the representable range: build it whole
        }
        auto part_out = derive_output_topology(part, kc.output_level_shift);
        if (!to_part_map(build_pairs(*part_out, *part, kc), *part_out, probe[p]))
          return false;
        found[p] = std::make_shared<const PartMap>(std::move(probe[p]));
        built.push_back(found[p]);
      }
      // The whole topology's outputs inside this part are the part's own outputs, translated.
      int64_t n_out = 0;
      for (int64_t c : found[p]->out_counts)
        n_out += c;
      if (n_out != static_cast<int64_t>(out_split.leaves[p].size()))
        return false;
    }
    e.map = compose_parts(found, in_split, out_split);
  }
  std::lock_guard<std::mutex> lock(mutex);
  for (auto &pm : built)
    store.insert(std::move(pm));
  return true;
}

} // namespace

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
  if (!build_from_parts(input.topology(), config, impl_->mutex, impl_->parts, e)) {
    e.out_topology = derive_output_topology(input.topology(), config.output_level_shift);
    e.map = build_pairs(*e.out_topology, *input.topology(), config);
  }
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
  impl_->parts.clear();
  impl_->parts.hits = 0;
  impl_->parts.misses = 0;
}

KernelMapCache::Stats KernelMapCache::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  Stats s;
  s.hits = impl_->hits;
  s.misses = impl_->misses;
  s.entries = impl_->entries.size();
  s.part_hits = impl_->parts.hits;
  s.part_misses = impl_->parts.misses;
  s.part_bytes = impl_->parts.bytes;
  return s;
}

void KernelMapCache::set_capacity(size_t max_entries) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->capacity = std::max<size_t>(max_entries, 1);
  while (impl_->entries.size() > impl_->capacity && !impl_->order.empty()) {
    impl_->entries.erase(impl_->order.front());
    impl_->order.pop_front();
  }
}

void KernelMapCache::set_part_reuse_bytes(size_t max_bytes) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->parts.budget = max_bytes;
  impl_->parts.evict();
}

AdaptiveKernelMap build_adaptive_kernel_map(const AdaptiveSparseTensor &input,
                                            const AdaptiveKernelConfig &config) {
  return KernelMapCache::instance().get_or_build(input, config).map;
}

} // namespace adaptive
} // namespace minkowski
