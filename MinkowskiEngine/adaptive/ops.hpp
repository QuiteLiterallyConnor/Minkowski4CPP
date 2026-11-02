#ifndef MINK_CPP_ADAPTIVE_OPS_HPP
#define MINK_CPP_ADAPTIVE_OPS_HPP

#include "adaptive_tensor.hpp"
#include "kernel_map.hpp"

namespace minkowski {
namespace adaptive {

enum class InterpMode { Nearest, Multilinear };

// What interpolate() returns for a point that no active leaf contains.
//   Zero:            zeros (default).
//   CoarserAncestor: the volume-weighted mean of the leaves inside the finest grid cell
//                    (any level) that contains the point and at least one leaf, i.e. the
//                    mean of the represented field over that cell's covered part. Fills the
//                    space left by drop_empty refinement with the value of the nearest
//                    common ancestor instead of 0. Points whose level-0 cell holds no leaf
//                    still get zeros. Differentiable w.r.t. the features.
enum class InterpFallback { Zero, CoarserAncestor };
// Feature-test macro: interpolate() accepts an InterpFallback.
#define MINK_ADAPTIVE_HAS_INTERP_FALLBACK 1

// y_o = sum_k sum_i w(o,i,k) * x_i W_k, with weight [K^D, Cin, Cout]. Differentiable
// w.r.t. features and weight. Output leaves are the input leaves unless
// config.output_level_shift < 0.
AdaptiveSparseTensor adaptive_convolution(const AdaptiveSparseTensor &input,
                                          const at::Tensor &weight,
                                          const AdaptiveKernelConfig &config);

// Transpose of a (possibly strided) convolution. `reference` defines the output
// leaves and must be the tensor the forward convolution was applied to; `input`
// must have the topology that forward convolution produces.
AdaptiveSparseTensor adaptive_convolution_transpose(const AdaptiveSparseTensor &input,
                                                    const at::Tensor &weight,
                                                    const AdaptiveKernelConfig &config,
                                                    const AdaptiveSparseTensor &reference);

// [C] tensor. Volume-weighted by default so the result is resolution independent.
at::Tensor adaptive_global_avg_pool(const AdaptiveSparseTensor &input,
                                    bool volume_weighted = true);

// Coarsens every leaf finer than target_level to target_level.
AdaptiveSparseTensor adaptive_pool(const AdaptiveSparseTensor &input, uint32_t target_level,
                                   Reduction reduction = Reduction::Mean);

// points: [M, D]. Points outside every active leaf yield zeros (or, with
// InterpFallback::CoarserAncestor, the mean of their finest occupied ancestor cell, in
// both modes). Nearest takes the containing leaf. Multilinear weights leaves by overlap
// with a box of the containing leaf's size centred on the point (exact multilinear
// interpolation between leaf centres on a uniform grid, partition of unity everywhere).
at::Tensor interpolate(const AdaptiveSparseTensor &input, const at::Tensor &points,
                       InterpMode mode = InterpMode::Nearest,
                       InterpFallback fallback = InterpFallback::Zero);

} // namespace adaptive
} // namespace minkowski

#endif
