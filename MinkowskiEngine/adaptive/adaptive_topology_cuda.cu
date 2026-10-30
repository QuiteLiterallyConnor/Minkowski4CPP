// Device implementations of the heavy topology-build steps (key sort and the
// active-ancestor check). Keys stay host-resident in Topology, so each call uploads
// the key array once and downloads only what the host needs.
#include "adaptive_topology.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

#include <stdexcept>
#include <string>

namespace minkowski {
namespace adaptive {
namespace detail {
namespace {

// Same layout as Key, usable in device code (std::array is not).
struct DKey {
  int64_t c[kMaxDim];
  uint32_t level;
  uint32_t pad;
};
static_assert(sizeof(DKey) == sizeof(Key), "DKey must mirror Key");

struct Item {
  DKey key;
  int64_t idx;
};

__host__ __device__ inline bool dkey_less(const DKey &a, const DKey &b) {
  if (a.level != b.level)
    return a.level < b.level;
  for (int d = 0; d < kMaxDim; ++d)
    if (a.c[d] != b.c[d])
      return a.c[d] < b.c[d];
  return false;
}

struct ItemLess {
  __host__ __device__ bool operator()(const Item &a, const Item &b) const {
    return dkey_less(a.key, b.key);
  }
};

struct MakeItem {
  const DKey *keys;
  __host__ __device__ Item operator()(int64_t i) const { return Item{keys[i], i}; }
};

struct GetKey {
  __host__ __device__ DKey operator()(const Item &it) const { return it.key; }
};
struct GetIdx {
  __host__ __device__ int64_t operator()(const Item &it) const { return it.idx; }
};

struct UnsortedAt {
  const DKey *keys;
  __host__ __device__ bool operator()(int64_t i) const {
    return i > 0 && dkey_less(keys[i], keys[i - 1]);
  }
};

struct Truthy {
  __host__ __device__ bool operator()(bool b) const { return b; }
};

void check(cudaError_t e, const char *what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__global__ void ancestor_kernel(const DKey *keys, int64_t first, int64_t n,
                                const int64_t *level_begin, uint32_t max_level, int *found) {
  const int64_t i = first + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n || *found)
    return;
  const DKey k = keys[i];
  for (uint32_t l = 0; l < k.level && l <= max_level; ++l) {
    const int64_t lo0 = level_begin[l], hi0 = level_begin[l + 1];
    if (lo0 == hi0)
      continue;
    DKey a;
    a.level = l;
    a.pad = 0;
    const uint32_t shift = k.level - l;
    for (int d = 0; d < kMaxDim; ++d)
      a.c[d] = shift >= 63 ? (k.c[d] < 0 ? -1 : 0) : (k.c[d] >> shift);
    int64_t lo = lo0, hi = hi0;
    while (lo < hi) {
      const int64_t mid = lo + (hi - lo) / 2;
      if (dkey_less(keys[mid], a))
        lo = mid + 1;
      else
        hi = mid;
    }
    if (lo < hi0 && !dkey_less(a, keys[lo])) {
      *found = 1;
      return;
    }
  }
}

} // namespace

bool cuda_topology_available() {
  static const bool ok = [] {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }();
  return ok;
}

void cuda_sort_keys(std::vector<Key> &keys, std::vector<int64_t> &order) {
  order.clear();
  const int64_t n = static_cast<int64_t>(keys.size());
  if (n < 2)
    return;
  thrust::device_vector<DKey> d_keys(static_cast<size_t>(n));
  check(cudaMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys.data(), n * sizeof(DKey),
                   cudaMemcpyHostToDevice),
        "key upload");
  const DKey *kp = thrust::raw_pointer_cast(d_keys.data());
  const auto idx = thrust::make_counting_iterator<int64_t>(0);
  if (thrust::none_of(thrust::device, thrust::make_transform_iterator(idx, UnsortedAt{kp}),
                      thrust::make_transform_iterator(idx + n, UnsortedAt{kp}),
                      Truthy{}))
    return;

  thrust::device_vector<Item> items(static_cast<size_t>(n));
  thrust::transform(thrust::device, idx, idx + n, items.begin(), MakeItem{kp});
  thrust::sort(thrust::device, items.begin(), items.end(), ItemLess{});
  thrust::device_vector<int64_t> d_order(static_cast<size_t>(n));
  thrust::transform(thrust::device, items.begin(), items.end(), d_keys.begin(), GetKey{});
  thrust::transform(thrust::device, items.begin(), items.end(), d_order.begin(), GetIdx{});
  order.resize(static_cast<size_t>(n));
  check(cudaMemcpy(keys.data(), thrust::raw_pointer_cast(d_keys.data()), n * sizeof(DKey),
                   cudaMemcpyDeviceToHost),
        "key download");
  check(cudaMemcpy(order.data(), thrust::raw_pointer_cast(d_order.data()), n * sizeof(int64_t),
                   cudaMemcpyDeviceToHost),
        "order download");
}

bool cuda_has_active_ancestor(const std::vector<Key> &keys,
                              const std::vector<int64_t> &level_begin, int64_t first) {
  const int64_t n = static_cast<int64_t>(keys.size());
  if (n - first <= 0)
    return false;
  thrust::device_vector<DKey> d_keys(static_cast<size_t>(n));
  check(cudaMemcpy(thrust::raw_pointer_cast(d_keys.data()), keys.data(), n * sizeof(DKey),
                   cudaMemcpyHostToDevice),
        "key upload");
  thrust::device_vector<int64_t> d_begin(level_begin.begin(), level_begin.end());
  thrust::device_vector<int> d_found(1, 0);
  const int threads = 256;
  const int64_t count = n - first;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  ancestor_kernel<<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_keys.data()), first, n, thrust::raw_pointer_cast(d_begin.data()),
      static_cast<uint32_t>(level_begin.size() - 2), thrust::raw_pointer_cast(d_found.data()));
  check(cudaGetLastError(), "ancestor kernel launch");
  check(cudaDeviceSynchronize(), "ancestor kernel");
  return d_found[0] != 0;
}

} // namespace detail
} // namespace adaptive
} // namespace minkowski
