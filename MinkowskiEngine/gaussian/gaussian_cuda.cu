#include "gaussian_cuda.hpp"

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>

namespace minkowski {
namespace gaussian {
namespace detail {
namespace {

template <typename scalar_t>
__global__ void covariance_kernel(const scalar_t *quaternions,
                                  const scalar_t *scales, scalar_t *output,
                                  int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const scalar_t *q = quaternions + index * 4;
  const scalar_t *s = scales + index * 3;
  const scalar_t norm = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] +
                             q[3] * q[3]);
  const scalar_t w = q[0] / norm, x = q[1] / norm;
  const scalar_t y = q[2] / norm, z = q[3] / norm;
  const scalar_t r[9] = {
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
      2 * (x * z + y * w),     2 * (x * y + z * w),
      1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w),     2 * (y * z + x * w),
      1 - 2 * (x * x + y * y)};
  const scalar_t variance[3] = {s[0] * s[0], s[1] * s[1], s[2] * s[2]};
  scalar_t *cov = output + index * 9;
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      cov[row * 3 + column] =
          r[row * 3] * variance[0] * r[column * 3] +
          r[row * 3 + 1] * variance[1] * r[column * 3 + 1] +
          r[row * 3 + 2] * variance[2] * r[column * 3 + 2];
}

template <typename scalar_t>
__device__ void rotate(const scalar_t *q, const scalar_t *v, scalar_t *out) {
  const scalar_t norm =
      sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const scalar_t w = q[0] / norm, x = q[1] / norm;
  const scalar_t y = q[2] / norm, z = q[3] / norm;
  const scalar_t r00 = 1 - 2 * (y * y + z * z);
  const scalar_t r01 = 2 * (x * y - z * w);
  const scalar_t r02 = 2 * (x * z + y * w);
  const scalar_t r10 = 2 * (x * y + z * w);
  const scalar_t r11 = 1 - 2 * (x * x + z * z);
  const scalar_t r12 = 2 * (y * z - x * w);
  const scalar_t r20 = 2 * (x * z - y * w);
  const scalar_t r21 = 2 * (y * z + x * w);
  const scalar_t r22 = 1 - 2 * (x * x + y * y);
  out[0] = r00 * v[0] + r01 * v[1] + r02 * v[2];
  out[1] = r10 * v[0] + r11 * v[1] + r12 * v[2];
  out[2] = r20 * v[0] + r21 * v[1] + r22 * v[2];
}

template <typename scalar_t>
__global__ void transform_kernel(
    const scalar_t *positions, const scalar_t *gaussian_rotations,
    const scalar_t *transform_rotations, const scalar_t *translations,
    const scalar_t *velocities, scalar_t *output_positions,
    scalar_t *output_rotations, scalar_t *output_velocities, int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;

  const scalar_t *q = transform_rotations + index * 4;
  scalar_t rotated[3];
  rotate(q, positions + index * 3, rotated);
  for (int axis = 0; axis < 3; ++axis)
    output_positions[index * 3 + axis] =
        rotated[axis] + translations[index * 3 + axis];

  const scalar_t qnorm =
      sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const scalar_t gnorm =
      sqrt(gaussian_rotations[index * 4] * gaussian_rotations[index * 4] +
           gaussian_rotations[index * 4 + 1] *
               gaussian_rotations[index * 4 + 1] +
           gaussian_rotations[index * 4 + 2] *
               gaussian_rotations[index * 4 + 2] +
           gaussian_rotations[index * 4 + 3] *
               gaussian_rotations[index * 4 + 3]);
  const scalar_t aw = q[0] / qnorm, ax = q[1] / qnorm;
  const scalar_t ay = q[2] / qnorm, az = q[3] / qnorm;
  const scalar_t bw = gaussian_rotations[index * 4] / gnorm;
  const scalar_t bx = gaussian_rotations[index * 4 + 1] / gnorm;
  const scalar_t by = gaussian_rotations[index * 4 + 2] / gnorm;
  const scalar_t bz = gaussian_rotations[index * 4 + 3] / gnorm;
  scalar_t *out_q = output_rotations + index * 4;
  out_q[0] = aw * bw - ax * bx - ay * by - az * bz;
  out_q[1] = aw * bx + ax * bw + ay * bz - az * by;
  out_q[2] = aw * by - ax * bz + ay * bw + az * bx;
  out_q[3] = aw * bz + ax * by - ay * bx + az * bw;

  if (velocities != nullptr)
    rotate(q, velocities + index * 3, output_velocities + index * 3);
}

} // namespace

void covariance_cuda(const at::Tensor &rotations, const at::Tensor &scales,
                     at::Tensor &output) {
  const int64_t count = rotations.size(0);
  if (count == 0)
    return;
  c10::cuda::CUDAGuard guard(rotations.device());
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(rotations.scalar_type(), "gaussian_covariance_cuda", [&] {
    covariance_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
        rotations.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void transform_cuda(const at::Tensor &positions,
                    const at::Tensor &gaussian_rotations,
                    const at::Tensor &transform_rotations,
                    const at::Tensor &translations, const at::Tensor &velocities,
                    at::Tensor &output_positions, at::Tensor &output_rotations,
                    at::Tensor &output_velocities) {
  const int64_t count = positions.size(0);
  if (count == 0)
    return;
  c10::cuda::CUDAGuard guard(positions.device());
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "gaussian_transform_cuda", [&] {
    transform_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
        positions.data_ptr<scalar_t>(), gaussian_rotations.data_ptr<scalar_t>(),
        transform_rotations.data_ptr<scalar_t>(), translations.data_ptr<scalar_t>(),
        velocities.defined() ? velocities.data_ptr<scalar_t>() : nullptr,
        output_positions.data_ptr<scalar_t>(),
        output_rotations.data_ptr<scalar_t>(),
        output_velocities.defined() ? output_velocities.data_ptr<scalar_t>()
                                    : nullptr,
        count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace gaussian
} // namespace minkowski
