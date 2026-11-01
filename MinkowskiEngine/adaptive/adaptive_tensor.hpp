#ifndef MINK_CPP_ADAPTIVE_TENSOR_HPP
#define MINK_CPP_ADAPTIVE_TENSOR_HPP

#include "config.hpp"
#include "voxel_geometry.hpp"
#include "voxel_key.hpp"

#include <torch/torch.h>

#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace minkowski {
namespace adaptive {

namespace detail {
struct Topology;
}

// Points (and their features) a tensor was voxelized from. Needed by the
// point-driven refinement criteria and by revoxelize_from_points().
struct SourcePoints {
  at::Tensor points;   // [P, D] double, CPU
  at::Tensor features; // [P, C], kept attached to the autograd graph
};

// Sparse tensor whose active voxels (leaves) can sit at different refinement
// levels. Leaves never overlap. Topology is immutable and shared between
// copies; refine/coarsen swap in a new topology, so copies are independent.
class AdaptiveSparseTensor {
public:
  static AdaptiveSparseTensor from_points(const at::Tensor &points,
                                          const at::Tensor &features,
                                          const AdaptiveVoxelConfig &config);
  static AdaptiveSparseTensor from_voxels(const at::Tensor &coordinates,
                                          const at::Tensor &levels,
                                          const at::Tensor &features,
                                          const AdaptiveVoxelConfig &config);
  static AdaptiveSparseTensor
  from_topology(std::shared_ptr<const detail::Topology> topology,
                at::Tensor features,
                std::shared_ptr<const SourcePoints> source = nullptr);

  int64_t size() const;
  int dimension() const;
  int64_t channels() const;
  const AdaptiveVoxelConfig &config() const;
  torch::Device device() const { return features_.device(); }

  // Leaves in deterministic (level, coordinates) order. Coordinates, levels
  // and geometry tensors are CPU and non-differentiable.
  at::Tensor coordinates() const; // [N, D] int64
  at::Tensor levels() const;      // [N] int32
  at::Tensor voxel_sizes() const; // [N, D] double
  at::Tensor volumes() const;     // [N] double
  at::Tensor centers() const;     // [N, D] double
  std::pair<at::Tensor, at::Tensor> bounds() const; // (lo, hi), half-open
  const at::Tensor &features() const { return features_; }
  void set_features(at::Tensor features);

  std::vector<int64_t> counts_by_level() const; // max_depth + 1 entries
  // Row of the active leaf with exactly this key, or -1.
  int64_t index_of(std::array<int64_t, 3> coordinates, uint32_t level) const;
  int64_t index_of_key(const std::vector<int64_t> &coordinates, uint32_t level) const;
  // Changes whenever the leaf set changes; unchanged by no-op operations.
  uint64_t topology_version() const;
  bool has_overlapping_leaves() const;
  const std::shared_ptr<const detail::Topology> &topology() const { return topology_; }
  bool has_source_points() const { return source_ != nullptr; }

  // Replaces leaves intersecting `region` by all 2^D children until they reach
  // target_level. Already-fine leaves are untouched. Replicate copies the parent
  // feature to each child; Split divides it by 2^D (conserves sums).
  void refine(const AxisAlignedRegion &region, uint32_t target_level,
              RefinePolicy policy = RefinePolicy::Replicate);
  // Merges leaves into ancestors down to target_level, bottom-up, for every
  // parent whose whole box lies inside `region`. Only occupied descendants
  // contribute to the reduction.
  void coarsen(const AxisAlignedRegion &region, uint32_t target_level,
               Reduction reduction = Reduction::Mean);
  // Require source points (from_points). Refine leaves one level at a time, up
  // to max_depth, while the criterion holds; children replicate the parent.
  void refine_by_density(int64_t points_per_leaf_threshold);
  void refine_by_variance(double feature_variance_threshold);
  void refine_if(const std::function<bool(int64_t leaf, const std::array<double, 4> &lo,
                                          const std::array<double, 4> &hi, uint32_t level)>
                     &criterion);
  // Recomputes features of the current leaves from the source points (mean
  // per leaf; leaves containing no point get zeros). Differentiable w.r.t. the
  // point features passed to from_points (detach them first to opt out).
  void revoxelize_from_points();

  AdaptiveSparseTensor with_features(at::Tensor features) const;
  AdaptiveSparseTensor to(torch::Device device) const;
  AdaptiveSparseTensor resample_uniform(uint32_t level,
                                        Reduction reduction = Reduction::Mean) const;
  // Re-expresses this tensor on reference's leaves: finer leaves are reduced,
  // coarser ones replicated, uncovered reference leaves get zeros.
  AdaptiveSparseTensor resample_to(const AdaptiveSparseTensor &reference,
                                   Reduction reduction = Reduction::Mean) const;

  // Element-wise ops require identical topology; use resample_to to align.
  AdaptiveSparseTensor operator+(const AdaptiveSparseTensor &other) const;

private:
  AdaptiveSparseTensor() = default;

  std::shared_ptr<const detail::Topology> topology_;
  at::Tensor features_;
  std::shared_ptr<const SourcePoints> source_;
};

AdaptiveSparseTensor concatenate(const AdaptiveSparseTensor &a,
                                 const AdaptiveSparseTensor &b);

} // namespace adaptive
} // namespace minkowski

#endif
