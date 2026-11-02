#ifndef MINK_CPP_ADAPTIVE_KERNEL_MAP_HPP
#define MINK_CPP_ADAPTIVE_KERNEL_MAP_HPP

#include "adaptive_tensor.hpp"

#include <array>
#include <memory>
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
  // Opt-in. Like CenterSampled while the leaf containing the tap point is at the output
  // leaf's level or coarser (weight 1). When it is finer, or no leaf contains the tap point,
  // the tap reads the average of the leaves inside the tap box (the output-sized cell around
  // the tap point), weighted by overlap volume / total overlap volume. Coarse outputs over
  // fine inputs (output_level_shift < 0, or coarse leaves next to refined regions) then read
  // every child instead of the one child whose corner sits on the tap point. Identical to
  // CenterSampled on uniform grids.
  CenterAveraged,
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

struct KernelMapEntry {
  AdaptiveKernelMap map;
  std::shared_ptr<const detail::Topology> out_topology;
};

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
