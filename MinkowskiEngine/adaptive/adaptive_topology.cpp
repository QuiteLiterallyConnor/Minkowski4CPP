#include "adaptive_topology.hpp"

#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace minkowski {
namespace adaptive {
namespace detail {

namespace {

// Below this the host path beats a device round trip.
constexpr int64_t kMinDeviceKeys = 1 << 14;

constexpr int64_t kParallelThreshold = 32768;

// Plain std::thread workers on purpose: LibTorch's OpenMP runtime can clash with the
// application's OpenMP runtime on Windows, so no OpenMP / at::parallel_for here.
int thread_count(int64_t n) {
  if (n < kParallelThreshold)
    return 1;
  static const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  return static_cast<int>(std::min<int64_t>(hw, n / 8192));
}

// Runs f(task) for task in [0, tasks), one thread each (the caller runs task 0).
template <class F> void run_tasks(int tasks, F &&f) {
  if (tasks <= 1) {
    f(0);
    return;
  }
  std::exception_ptr error;
  std::mutex error_mutex;
  const auto guarded = [&](int task) {
    try {
      f(task);
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error)
        error = std::current_exception();
    }
  };
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(tasks) - 1);
  for (int task = 1; task < tasks; ++task)
    workers.emplace_back(guarded, task);
  guarded(0);
  for (auto &w : workers)
    w.join();
  if (error)
    std::rethrow_exception(error);
}

// Calls f(begin, end) over [0, n), in parallel blocks for large n.
template <class F> void for_blocks(int64_t n, F &&f) {
  const int tasks = thread_count(n);
  run_tasks(tasks, [&](int t) { f(n * t / tasks, n * (t + 1) / tasks); });
}

struct KeyIdx {
  Key key;
  int64_t idx;
};

inline bool key_idx_less(const KeyIdx &a, const KeyIdx &b) {
  if (key_less(a.key, b.key))
    return true;
  if (key_less(b.key, a.key))
    return false;
  return a.idx < b.idx; // equal keys keep their input order
}

void parallel_sort(std::vector<KeyIdx> &v) {
  const int64_t n = static_cast<int64_t>(v.size());
  int chunks = 1;
  const int threads = thread_count(n);
  while (chunks * 2 <= threads)
    chunks *= 2;
  if (chunks < 2) {
    std::sort(v.begin(), v.end(), key_idx_less);
    return;
  }
  std::vector<int64_t> bound(static_cast<size_t>(chunks) + 1);
  for (int i = 0; i <= chunks; ++i)
    bound[i] = n * i / chunks;
  run_tasks(chunks, [&](int i) {
    std::sort(v.begin() + bound[i], v.begin() + bound[i + 1], key_idx_less);
  });
  for (int width = 1; width < chunks; width *= 2) {
    run_tasks(chunks / (2 * width), [&](int m) {
      const int i = m * 2 * width;
      std::inplace_merge(v.begin() + bound[i], v.begin() + bound[i + width],
                         v.begin() + bound[i + 2 * width], key_idx_less);
    });
  }
}

inline uint64_t fast_hash(const Key &k) { return Topology::NodeTable::hash(k); }

// Open-addressing set of the keys in one level's range of the sorted key array.
struct LevelSet {
  const Key *keys = nullptr;
  std::vector<int64_t> slots;
  uint64_t mask = 0;

  void build(const Key *all, int64_t begin, int64_t end) {
    keys = all;
    size_t cap = 16;
    while (cap < static_cast<size_t>(end - begin) * 2)
      cap <<= 1;
    slots.assign(cap, -1);
    mask = cap - 1;
    for (int64_t i = begin; i < end; ++i) {
      uint64_t h = fast_hash(all[i]) & mask;
      while (slots[h] >= 0)
        h = (h + 1) & mask;
      slots[h] = i;
    }
  }

  bool contains(const Key &k) const {
    uint64_t h = fast_hash(k) & mask;
    for (int64_t s; (s = slots[h]) >= 0; h = (h + 1) & mask)
      if (keys[s] == k)
        return true;
    return false;
  }
};

// True if some leaf has an active strict ancestor. Only levels above the deepest present
// level can own an ancestor, so the (largest) deepest level is never indexed.
bool has_active_ancestor(const std::vector<Key> &keys, const std::vector<uint32_t> &present,
                         const std::vector<int64_t> &level_begin) {
  if (present.size() < 2)
    return false;
#ifdef USE_CUDA
  if (static_cast<int64_t>(keys.size()) >= kMinDeviceKeys && cuda_topology_available())
    return cuda_has_active_ancestor(keys, level_begin, level_begin[present[1]]);
#endif
  const size_t np = present.size();
  std::vector<LevelSet> sets(np - 1);
  for (size_t p = 0; p + 1 < np; ++p)
    sets[p].build(keys.data(), level_begin[present[p]], level_begin[present[p] + 1]);

  const int64_t first = level_begin[present[1]];
  const int64_t total = static_cast<int64_t>(keys.size());
  const int chunks = thread_count(total - first);
  std::atomic<bool> found{false};
  run_tasks(chunks, [&](int c) {
    {
      const int64_t begin = first + (total - first) * c / chunks;
      const int64_t end = first + (total - first) * (c + 1) / chunks;
      // Consecutive keys usually share ancestors; remember the last one checked per level.
      std::vector<Key> last(np - 1);
      std::vector<bool> have(np - 1, false);
      for (int64_t i = begin; i < end && !found.load(std::memory_order_relaxed); ++i) {
        const Key &k = keys[i];
        for (size_t p = 0; p + 1 < np && present[p] < k.level; ++p) {
          Key a{};
          a.level = present[p];
          const uint32_t shift = k.level - a.level;
          for (int d = 0; d < kMaxDim; ++d)
            a.coordinates[d] = shift >= 63 ? (k.coordinates[d] < 0 ? -1 : 0)
                                           : (k.coordinates[d] >> shift);
          if (have[p] && last[p] == a)
            continue;
          last[p] = a;
          have[p] = true;
          if (sets[p].contains(a)) {
            found.store(true, std::memory_order_relaxed);
            break;
          }
        }
      }
    }
  });
  return found.load();
}

} // namespace

void parallel_blocks(int64_t n, const std::function<void(int64_t, int64_t)> &f) {
  for_blocks(n, f);
}

void sort_keys(std::vector<Key> &keys, std::vector<int64_t> &order) {
  order.clear();
  const int64_t n = static_cast<int64_t>(keys.size());
#ifdef USE_CUDA
  if (n >= kMinDeviceKeys && cuda_topology_available()) {
    cuda_sort_keys(keys, order);
    return;
  }
#endif
  std::atomic<bool> sorted{true};
  for_blocks(n, [&](int64_t b, int64_t e) {
    // Each block also compares against the key just before it.
    for (int64_t i = std::max<int64_t>(b, 1); i < e; ++i)
      if (key_less(keys[i], keys[i - 1])) {
        sorted.store(false, std::memory_order_relaxed);
        return;
      }
  });
  if (sorted.load())
    return;
  std::vector<KeyIdx> items(static_cast<size_t>(n));
  for_blocks(n, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i)
      items[i] = KeyIdx{keys[i], i};
  });
  parallel_sort(items);
  order.resize(static_cast<size_t>(n));
  for_blocks(n, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      keys[i] = items[i].key;
      order[i] = items[i].idx;
    }
  });
}

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
  const int D = t->D;
  t->counts.assign(t->max_depth + 1, 0);
  {
    std::mutex counts_mutex;
    const Key *kd = t->keys.data();
    const uint32_t max_level = t->max_depth;
    for_blocks(N, [&](int64_t b, int64_t e) {
      std::vector<int64_t> local(max_level + 1, 0);
      for (int64_t i = b; i < e; ++i) {
        const Key &k = kd[i];
        if (k.level > max_level)
          throw std::invalid_argument("adaptive voxel level exceeds max_depth");
        if (i > 0 && !key_less(kd[i - 1], k))
          throw std::invalid_argument("duplicate or unsorted adaptive voxel keys");
        ++local[k.level];
      }
      std::lock_guard<std::mutex> lock(counts_mutex);
      for (uint32_t l = 0; l <= max_level; ++l)
        t->counts[l] += local[l];
    });
  }
  // level_begin[l] .. level_begin[l + 1] is the range of level l in the sorted keys.
  std::vector<int64_t> level_begin(t->max_depth + 2, 0);
  for (uint32_t l = 0; l <= t->max_depth; ++l) {
    level_begin[l + 1] = level_begin[l] + t->counts[l];
    if (t->counts[l] > 0)
      t->present_levels.push_back(l);
  }
  if (has_active_ancestor(t->keys, t->present_levels, level_begin))
    throw std::invalid_argument(
        "overlapping active voxels: a leaf and one of its ancestors are both active");

  t->coords = torch::empty({N, D}, torch::kInt64);
  t->levels = torch::empty({N}, torch::kInt32);
  t->volumes = torch::empty({N}, torch::kFloat64);
  auto *c = t->coords.data_ptr<int64_t>();
  auto *lv = t->levels.data_ptr<int32_t>();
  auto *vol = t->volumes.data_ptr<double>();
  std::vector<double> volume_at(t->max_depth + 1);
  for (uint32_t l = 0; l <= t->max_depth; ++l) {
    const DVec s = t->size_at(l);
    double v = 1.0;
    for (int d = 0; d < D; ++d)
      v *= s[d];
    volume_at[l] = v;
  }
  const Key *kd = t->keys.data();
  for_blocks(N, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const Key &k = kd[i];
      for (int d = 0; d < D; ++d)
        c[i * D + d] = k.coordinates[d];
      lv[i] = static_cast<int32_t>(k.level);
      vol[i] = volume_at[k.level];
    }
  });
  return t;
}

const std::unordered_map<Key, int64_t, KeyHash> &Topology::index_map() const {
  std::call_once(index_once_, [this] {
    index_.reserve(keys.size() * 2);
    for (size_t i = 0; i < keys.size(); ++i)
      index_.emplace(keys[i], static_cast<int64_t>(i));
  });
  return index_;
}

const Topology::NodeTable &Topology::node_table() const {
  std::call_once(nodes_once_, [this] {
    NodeTable &t = nodes_;
    const int64_t N = size();
    if (N == 0) {
      t.slots.assign(1, NodeTable::Slot{Key{}, NodeTable::kEmpty});
      t.mask = 0;
      return;
    }
    const uint32_t lmin = present_levels.front();
    // Ancestors of sorted keys come in runs, so consecutive leaves mostly stop after one probe.
    // Load factor stays <= 0.7 (full 2^D trees add ~N/(2^D-1) internal nodes to N leaves).
    size_t cap = 16;
    while (cap < static_cast<size_t>(N) * 2)
      cap <<= 1;
    t.slots.assign(cap, NodeTable::Slot{Key{}, NodeTable::kEmpty});
    t.mask = cap - 1;
    int64_t used = 0;
    // Inserts k with value v unless present; returns false if it was already there.
    auto insert = [&](const Key &k, int64_t v) {
      for (uint64_t h = NodeTable::hash(k) & t.mask;; h = (h + 1) & t.mask) {
        NodeTable::Slot &s = t.slots[h];
        if (s.value == NodeTable::kEmpty) {
          s.key = k;
          s.value = v;
          ++used;
          return true;
        }
        if (s.key == k)
          return false;
      }
    };
    auto grow = [&] {
      std::vector<NodeTable::Slot> old;
      old.swap(t.slots);
      const size_t ncap = old.size() * 2;
      t.slots.assign(ncap, NodeTable::Slot{Key{}, NodeTable::kEmpty});
      t.mask = ncap - 1;
      used = 0;
      for (const auto &s : old)
        if (s.value != NodeTable::kEmpty)
          insert(s.key, s.value);
    };
    for (int64_t i = 0; i < N; ++i) {
      // Room for the leaf and all its ancestors at load <= 0.7.
      if (static_cast<size_t>(used + keys[i].level - lmin + 1) * 10 > t.slots.size() * 7)
        grow();
      insert(keys[i], i);
      Key a = keys[i];
      while (a.level > lmin) {
        a = parent_key(a);
        if (!insert(a, NodeTable::kInternal))
          break; // its ancestors are already in the table
        ++t.internal_count;
      }
    }
  });
  return nodes_;
}

void Topology::build_geometry() const {
  std::call_once(geometry_once_, [this] {
    const int64_t N = size();
    lo_ = torch::empty({N, D}, torch::kFloat64);
    hi_ = torch::empty({N, D}, torch::kFloat64);
    sizes_ = torch::empty({N, D}, torch::kFloat64);
    auto *lo = lo_.data_ptr<double>();
    auto *hi = hi_.data_ptr<double>();
    auto *sz = sizes_.data_ptr<double>();
    std::vector<DVec> size_by_level(max_depth + 1);
    for (uint32_t l = 0; l <= max_depth; ++l)
      size_by_level[l] = size_at(l);
    const int Dl = D;
    for_blocks(N, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const Key &k = keys[i];
        const DVec &s = size_by_level[k.level];
        for (int d = 0; d < Dl; ++d) {
          const double l = origin[d] + static_cast<double>(k.coordinates[d]) * s[d];
          lo[i * Dl + d] = l;
          hi[i * Dl + d] = l + s[d];
          sz[i * Dl + d] = s[d];
        }
      }
    });
  });
}

at::Tensor Topology::centers_tensor() const {
  const int64_t N = size();
  auto out = torch::empty({N, D}, torch::kFloat64);
  auto *ct = out.data_ptr<double>();
  std::vector<DVec> size_by_level(max_depth + 1);
  for (uint32_t l = 0; l <= max_depth; ++l)
    size_by_level[l] = size_at(l);
  const int Dl = D;
  for_blocks(N, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      const Key &k = keys[i];
      const DVec &s = size_by_level[k.level];
      for (int d = 0; d < Dl; ++d) {
        const double l = origin[d] + static_cast<double>(k.coordinates[d]) * s[d];
        ct[i * Dl + d] = l + 0.5 * s[d];
      }
    }
  });
  return out;
}

} // namespace detail
} // namespace adaptive
} // namespace minkowski
