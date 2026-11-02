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

// Fused projection of render_gaussians (gaussian_project_cuda.cu). Per Gaussian, out [N, 10]:
// x2, y2 (pixel mean), conic a, b, c, camera-space z (differentiable), then a, b, c (2D covariance
// with the low-pass term) and its determinant (for culling). world_to_camera is the [3, 4]
// rotation | translation block, float32, contiguous, on the device.
constexpr int kProjectColumns = 10;
struct ProjectParams {
  float fx, fy, cx, cy, near_plane;
  float tx_lo, tx_hi, ty_lo, ty_hi; // clamped-frustum limits of x/z and y/z for the Jacobian
};
void project_forward_cuda(const at::Tensor &positions, const at::Tensor &rotations,
                          const at::Tensor &scales, const at::Tensor &world_to_camera,
                          const ProjectParams &params, at::Tensor &out);
// Gradients of positions [N,3], rotations [N,4] and scales [N,3] from grad_out [N,10] (only the
// first 6 columns carry gradient); outputs are overwritten.
void project_backward_cuda(const at::Tensor &positions, const at::Tensor &rotations,
                           const at::Tensor &scales, const at::Tensor &world_to_camera,
                           const ProjectParams &params, const at::Tensor &grad_out,
                           at::Tensor &grad_positions, at::Tensor &grad_rotations,
                           at::Tensor &grad_scales);
// Whether render_gaussians uses the fused projection for CUDA inputs (default on; the LibTorch
// path is used when the camera pose requires grad). Process-wide tuning/test hook.
bool &cuda_fused_projection();

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
