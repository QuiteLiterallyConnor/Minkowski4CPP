#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_ADAPTIVE_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_ADAPTIVE_HPP

// Bridge between 3D Gaussian splats (GaussianTensor) and adaptive
// mixed-resolution sparse voxels (adaptive::AdaptiveSparseTensor).
//
//   using namespace minkowski;
//   auto cfg = gaussian::adaptive_config_for(scene, /*max_depth=*/8);
//   gaussian::GaussianAdaptiveMap map;                      // optional
//   auto vox = gaussian::gaussians_to_adaptive(scene, cfg, {}, &map);
//   // vox.features(): [L, 4 + F] = [mass, r, g, b, f_0..f_{F-1}]
//   auto init = gaussian::adaptive_to_gaussians(vox);       // one Gaussian per leaf
//   auto leaf = gaussian::find_leaves(vox, scene.positions()); // [N] int64, -1 outside
//
// Level choice. Gaussian i with largest axis scale sigma_i targets the level
//   l_i = clamp(round(log2(max_d base[d] / (cell_per_sigma * sigma_i))),
//               min_level, max_level)
// so its cell edge is about cell_per_sigma * sigma_i (never coarser than the
// level-0 cell, never finer than max_level). Small Gaussians get fine cells,
// large ones coarse cells.
//
// Topology. Every Gaussian deposits into one cell (Center splat: the level-l_i
// cell containing its mean) or into all level-l_i cells within support_sigma
// axis standard deviations (BoxIntegral splat). A deposit cell that contains a
// finer deposit cell is subdivided: its weight is pushed to the 2^3 children in
// proportion to the Gaussian's integral over each child box, recursively, until
// it reaches cells without finer deposits. The result has no overlapping leaves,
// fine leaves where small Gaussians are, coarse leaves elsewhere, and leaves
// only where some Gaussian deposited a nonzero weight. Box integrals use the
// axis-aligned approximation: the product over axes of the 1-D marginal
// integrals with std sqrt(Sigma_dd); cross-axis correlation of rotated Gaussians
// is ignored. BoxIntegral weights are renormalized over the truncated support,
// so each Gaussian's weights sum to 1 in both modes (mass is conserved exactly).
//
// Features (layout [mass, r, g, b, extra features]):
//   mass_j  = sum_i w_ij * m_i,   m_i = opacity_i (* (2 pi)^{3/2} s_x s_y s_z
//                                       when weight_by_volume)
//   color_j = sum_i w_ij * m_i * color_i / mass_j  (mass-weighted mean; or the
//             unnormalized sum when mean_attributes = false), same for features.
// Leaves whose mass is exactly zero have zero attributes. Features are built from
// differentiable torch ops (index_select / index_add), so gradients flow to
// opacities, colors, extra features (and scales when weight_by_volume). The
// topology and the weights w_ij are piecewise constant in positions, rotations
// and scales and are treated as constants (no gradient).
//
// Execution. Levels, keys and weights are computed on the host in double
// precision (positions/scales are copied to the CPU); features are computed
// on the GaussianTensor's device and dtype, and the returned tensor lives there.

#include "gaussian_tensor.hpp"

#include "../adaptive/adaptive_tensor.hpp"
#include "../adaptive/config.hpp"

#include <torch/torch.h>

#include <cstdint>

namespace minkowski {
namespace gaussian {

enum class GaussianSplat {
  Center,     // whole Gaussian into the level-l_i cell containing its mean
  BoxIntegral // spread over level-l_i cells within support_sigma (axis-aligned)
};

struct GaussianToAdaptiveOptions {
  // Target cell edge in units of the largest axis scale (see level choice).
  double cell_per_sigma = 2.0;
  // Clamp range for target levels; max_level < 0 means config.max_depth.
  int min_level = 0;
  int max_level = -1;
  GaussianSplat splat = GaussianSplat::Center;
  // BoxIntegral truncation, in axis standard deviations (> 0).
  double support_sigma = 3.0;
  // BoxIntegral: at most this many cells per axis per Gaussian. A wider support
  // (only possible when the level was clamped at min_level, e.g. a huge
  // background splat) is clipped to a window centered on the mean's cell and
  // renormalized, so mass stays conserved but is more concentrated.
  int64_t max_cells_per_axis = 32;
  // m_i = opacity_i * (2 pi)^{3/2} * prod(scales_i) (integral of the 3DGS density
  // o * exp(-d^2/2)) instead of m_i = opacity_i.
  bool weight_by_volume = false;
  // true: colors/features are mass-weighted means (intensive);
  // false: mass-weighted sums (extensive, sum over leaves conserved).
  bool mean_attributes = true;
  // Append the GaussianTensor's extra features after the colors.
  bool include_features = true;
  // Guard against runaway BoxIntegral splats (total Gaussian-cell entries).
  int64_t max_entries = 50'000'000;
  // Entries whose share of their Gaussian is below min_weight (far tails of
  // subdivided cells, ~1e-40 and smaller) are dropped and the Gaussian's
  // remaining weights renormalized, so mass stays conserved. Such entries only
  // create leaves whose float32 mass is denormal and whose mean attributes are
  // rounding noise. The largest entry of every Gaussian is always kept. 0 keeps
  // every positive entry. Must be in [0, 1).
  double min_weight = 1e-15;
};

// Optional output: how Gaussians were distributed. CPU tensors.
struct GaussianAdaptiveMap {
  at::Tensor gaussian_level; // [N] int32 target level l_i
  at::Tensor gaussian_index; // [E] int64 Gaussian of each entry
  at::Tensor leaf_index;     // [E] int64 leaf row of each entry
  at::Tensor weight;         // [E] double share of the Gaussian in that leaf
};

// Feature channel layout of gaussians_to_adaptive.
constexpr int64_t kAdaptiveMassChannel = 0;
constexpr int64_t kAdaptiveColorChannel = 1; // channels 1..3
constexpr int64_t kAdaptiveFeatureChannel = 4;

// Voxelizes Gaussians into an adaptive sparse tensor. `config` must be 3-D.
// Throws std::invalid_argument for non-finite positions/scales, negative scales,
// zero/non-finite quaternions, a non-3-D config, positions too far from the
// origin to be represented exactly at config.max_depth (|coordinate| >= 2^52
// finest cells), or invalid options, and
// std::length_error when BoxIntegral entries exceed max_entries.
adaptive::AdaptiveSparseTensor
gaussians_to_adaptive(const GaussianTensor &gaussians,
                      const adaptive::AdaptiveVoxelConfig &config,
                      const GaussianToAdaptiveOptions &options = {},
                      GaussianAdaptiveMap *map = nullptr);

enum class AdaptiveOpacity {
  FromMass, // opacity = 1 - exp(-mass), in [0, 1)
  Constant  // opacity = constant_opacity
};

struct AdaptiveToGaussiansOptions {
  // Axis scale = sigma_per_cell * leaf edge (inverse of cell_per_sigma, so a
  // converted leaf maps back to the same level). 1/sqrt(12) moment-matches the box.
  double sigma_per_cell = 0.5;
  // Feature channels; -1 disables. Without a mass channel FromMass falls back
  // to Constant; without a color channel colors are default_color.
  int64_t mass_channel = kAdaptiveMassChannel;
  int64_t color_channel = kAdaptiveColorChannel;
  // Channels from this index on become GaussianTensor features; -1 = none.
  int64_t feature_channel = kAdaptiveFeatureChannel;
  AdaptiveOpacity opacity = AdaptiveOpacity::FromMass;
  double constant_opacity = 0.1;
  double default_color = 0.5;
  // Keep only leaves with mass > min_mass (ignored without a mass channel).
  // Default keeps every leaf with positive mass.
  double min_mass = 0.0;
};

// One axis-aligned Gaussian (identity rotation) per kept leaf, centered in the
// leaf, scale = sigma_per_cell * leaf size, colors/opacity/features read from
// the leaf features through differentiable torch ops (gradients flow back to
// the adaptive features). Output dtype/device follow the features
// (non-floating features are converted to float32).
GaussianTensor adaptive_to_gaussians(const adaptive::AdaptiveSparseTensor &tensor,
                                     const AdaptiveToGaussiansOptions &options = {});

// Spatial-index query: the leaf row containing each point ([P, 3]), or -1.
// Returns a CPU int64 tensor [P]. Half-open leaf boxes, so every point maps
// to at most one leaf.
at::Tensor find_leaves(const adaptive::AdaptiveSparseTensor &tensor,
                       const at::Tensor &points);

// Refine-by-Gaussian-density: returns a copy of `tensor` in which every leaf
// containing more than max_gaussians_per_leaf Gaussian means (half-open leaf
// boxes) is split into its 2^3 children, repeatedly, until no leaf above
// max_level (< 0: config.max_depth) exceeds the threshold. Children replicate
// the parent features (adaptive RefinePolicy::Replicate); re-run
// gaussians_to_adaptive or your own aggregation when per-leaf statistics are
// needed. Leaves are never created where the tensor has none. Throws
// std::invalid_argument for a non-3-D tensor, a negative threshold or a
// max_level above config.max_depth.
adaptive::AdaptiveSparseTensor
refine_by_gaussian_density(const adaptive::AdaptiveSparseTensor &tensor,
                           const GaussianTensor &gaussians, int64_t max_gaussians_per_leaf,
                           int max_level = -1);

// Convenience config: max_depth levels, finest cell = cell_per_sigma times the
// fine_quantile quantile of the Gaussians' largest scale, origin at the minimum
// position corner. Throws std::invalid_argument for an empty input.
adaptive::AdaptiveVoxelConfig adaptive_config_for(const GaussianTensor &gaussians,
                                                  uint32_t max_depth,
                                                  double cell_per_sigma = 2.0,
                                                  double fine_quantile = 0.01);

} // namespace gaussian
} // namespace minkowski

#endif
