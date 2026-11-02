#ifndef MINK_CPP_ADAPTIVE_KERNEL_MAP_HPP
#define MINK_CPP_ADAPTIVE_KERNEL_MAP_HPP

#include "adaptive_tensor.hpp"

#include <array>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace minkowski {
namespace adaptive {

enum class AdaptiveConvMode {
  // Sample the leaf containing each tap point; weight 1. Default; reduces to
  // the standard sparse convolution on uniform aligned grids.
  CenterSampled,
  // Same neighbors, weight V_in / V_out.
  VolumeWeighted,
  // Every leaf overlapping the tap box; weight = overlap volume / V_out.
  OverlapWeighted,
};

struct AdaptiveKernelConfig {
  int kernel_size = 3; // odd, per dimension
  AdaptiveConvMode mode = AdaptiveConvMode::CenterSampled;
  // 0 keeps the input leaves as outputs. -n coarsens outputs n levels (never
  // below level 0). Refining outputs is not supported.
  int output_level_shift = 0;
};

// Sparse (out, in, kernel-slot, weight) tuples sorted by kernel slot, then out.
struct AdaptiveKernelMap {
  at::Tensor in_index;     // [P] int64
  at::Tensor out_index;    // [P] int64
  at::Tensor kernel_index; // [P] int64
  at::Tensor weight;       // [P] double
  std::vector<int64_t> kernel_offsets; // num_kernels + 1 prefix offsets into the arrays
  int64_t num_kernels = 0;
};

// Copies of one kernel map's index/weight tensors on the devices and dtypes it was applied
// with, so repeated convolutions upload them once. Shared by every copy of a cache entry and
// freed with it (KernelMapCache::clear() or eviction releases the device memory).
class KernelMapDeviceCache {
public:
  struct Arrays {
    at::Tensor in_index;  // [P] int64 on `device`
    at::Tensor out_index; // [P] int64 on `device`
    at::Tensor weight;    // [P] `dtype` on `device`, undefined if not requested
  };
  // in/out indices on `device`, and the weights as `dtype` if `with_weight`.
  Arrays get(const AdaptiveKernelMap &map, const at::Device &device, at::ScalarType dtype,
             bool with_weight);

private:
  struct Slot {
    at::Device device{at::kCPU};
    at::Tensor in_index, out_index;
    std::vector<std::pair<at::ScalarType, at::Tensor>> weights;
  };
  std::mutex mutex_;
  std::vector<Slot> slots_;
};

struct KernelMapEntry {
  AdaptiveKernelMap map;
  std::shared_ptr<const detail::Topology> out_topology;
  std::shared_ptr<KernelMapDeviceCache> device = std::make_shared<KernelMapDeviceCache>();
};

namespace detail {
// Byte budget of the CUDA apply_map's per-group temporaries (a group's products and their
// concatenation). Consecutive kernel slots are processed together while their pairs fit;
// a slot larger than the budget is its own group. Default 32 MiB; 0 = one group for all
// slots. Process-wide tuning/test hook; set it before convolutions run, not concurrently.
int64_t &cuda_apply_map_group_bytes();
} // namespace detail

// Kernel slot of an offset in {-r..r}^D, r = (kernel_size-1)/2, first dimension slowest.
int64_t kernel_index(std::array<int, 3> offset, int kernel_size);

AdaptiveKernelMap build_adaptive_kernel_map(const AdaptiveSparseTensor &input,
                                            const AdaptiveKernelConfig &config);

// Cache keyed by (input topology id, kernel size, mode, output shift). A topology
// change produces a new id, so stale maps are never returned.
class KernelMapCache {
public:
  struct Stats {
    size_t hits = 0;
    size_t misses = 0;
    size_t entries = 0;
  };

  static KernelMapCache &instance();
  ~KernelMapCache();

  KernelMapEntry get_or_build(const AdaptiveSparseTensor &input,
                              const AdaptiveKernelConfig &config);
  void clear();
  Stats stats() const;
  void set_capacity(size_t max_entries);

private:
  KernelMapCache();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace adaptive
} // namespace minkowski

#endif
