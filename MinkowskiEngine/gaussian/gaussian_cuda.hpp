#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_CUDA_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_CUDA_HPP

#include <ATen/ATen.h>

namespace minkowski {
namespace gaussian {
namespace detail {

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
