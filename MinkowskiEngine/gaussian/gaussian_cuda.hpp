#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_CUDA_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_CUDA_HPP

#include <ATen/ATen.h>

namespace minkowski {
namespace gaussian {
namespace detail {

// The CUDA rasterizer launches one 16x16 thread block per tile.
constexpr int kRasterTileSize = 16;

// attrs [M,10] float32 (depth-sorted), ids [L] int32 (Gaussian rows grouped by
// tile, depth order inside a tile), ranges [tiles,2] int32 (begin,end into ids).
// out [H,W,5], final_t [H,W], last_index [H,W] int32 are written.
void rasterize_forward_cuda(const at::Tensor &attrs, const at::Tensor &ids,
                            const at::Tensor &ranges,
                            const at::Tensor &background, int64_t height,
                            int64_t width, at::Tensor &out,
                            at::Tensor &final_t, at::Tensor &last_index);

// grad_attrs [M,10] must be zero-initialized; gradients are accumulated.
void rasterize_backward_cuda(const at::Tensor &attrs, const at::Tensor &ids,
                             const at::Tensor &ranges,
                             const at::Tensor &background,
                             const at::Tensor &grad_out,
                             const at::Tensor &final_t,
                             const at::Tensor &last_index, int64_t height,
                             int64_t width, at::Tensor &grad_attrs);

void covariance_cuda(const at::Tensor &rotations, const at::Tensor &scales,
                     at::Tensor &output);

void transform_cuda(const at::Tensor &positions, const at::Tensor &gaussian_rotations,
                    const at::Tensor &transform_rotations,
                    const at::Tensor &translations, const at::Tensor &velocities,
                    at::Tensor &output_positions, at::Tensor &output_rotations,
                    at::Tensor &output_velocities);

} // namespace detail
} // namespace gaussian
} // namespace minkowski

#endif
