#ifndef MINK_CPP_ADAPTIVE_OPS_HPP
#define MINK_CPP_ADAPTIVE_OPS_HPP

#include "adaptive_tensor.hpp"
#include "kernel_map.hpp"

namespace minkowski {
namespace adaptive {

// ContinuousMultilinear (opt-in): normalized tent partition, see interpolate().
enum class InterpMode { Nearest, Multilinear, ContinuousMultilinear };

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

// points: [M, D]. Points outside every active leaf yield zeros. Nearest takes the
// containing leaf. Multilinear weights leaves by overlap with a box of the
// containing leaf's size centred on the point (exact multilinear interpolation
// between leaf centres on a uniform grid, partition of unity everywhere). Multilinear is
// continuous across faces between equal-level leaves but can jump where the containing
// leaf's level changes, because the box size jumps with it.
// ContinuousMultilinear gives every leaf i (centre c_i, edge s_i) the tent
// phi_i(x) = prod_d max(0, 1 - |x_d - c_id| / s_id) and returns
// sum_i phi_i(x) v_i / sum_i phi_i(x) for points inside an active leaf. Every phi_i is
// continuous and the containing leaf has phi >= 2^-D, so the result is continuous everywhere
// inside the union of active leaves (including across level changes), reproduces constants
// exactly, and equals Multilinear wherever every leaf whose tent or box reaches the point has
// the containing leaf's level. Near level changes it is a weighted (Shepard-type) average and
// does not reproduce linear fields exactly. Linear in the features (differentiable).
at::Tensor interpolate(const AdaptiveSparseTensor &input, const at::Tensor &points,
                       InterpMode mode = InterpMode::Nearest);

} // namespace adaptive
} // namespace minkowski

#endif
