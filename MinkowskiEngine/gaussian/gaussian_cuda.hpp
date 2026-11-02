#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_CUDA_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_CUDA_HPP

#include <ATen/ATen.h>
#include <torch/version.h>

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

// Fused Adam update of GaussianTrainer (gaussian_adam_cuda.cu), in place on contiguous float32
// CUDA tensors of one size:
//   m1 = beta1 m1 + (1 - beta1) g,  m2 = beta2 m2 + (1 - beta2) g^2,
//   p += step * m1 / (sqrt(m2 * inv_bias2) + epsilon)  with step = -lr / bias1.
struct AdamParams {
  float beta1, one_minus_beta1, beta2, one_minus_beta2, inv_bias2, epsilon, step;
};
// Match ATen's CUDA division by a CPU scalar. LibTorch 2.10 changed the reciprocal
// calculation from float32 to float64 before casting to the tensor's scalar type.
inline float adam_inverse_bias2(double bias2) {
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 10)
  return static_cast<float>(1.0 / bias2);
#else
  return 1.0f / static_cast<float>(bias2);
#endif
}
void adam_update_cuda(at::Tensor &param, const at::Tensor &grad, at::Tensor &moment1,
                      at::Tensor &moment2, const AdamParams &params);
// Whether GaussianTrainer uses the fused update for CUDA parameters (default on). Process-wide
// tuning/test hook.
bool &cuda_fused_adam();

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
