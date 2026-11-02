// Packing depth-ordered attributes is a permutation subset, so its adjoint is
// an assignment scatter, not an atomic reduction. The renderer constructs IDs
// from nonzero(visible) followed by argsort: every selected row is unique.
#include "gaussian_cuda.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace minkowski {
namespace gaussian {
namespace detail {
namespace {

__global__ void gather_kernel(int64_t m, const int64_t *ids,
                              const float *means, int64_t ms0, int64_t ms1,
                              const float *proj, const float *opacity,
                              const float *colors, float *out) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m)
    return;
  const int64_t src = ids[i];
  float *a = out + 10 * i;
  a[0] = means[src * ms0];
  a[1] = means[src * ms0 + ms1];
  a[2] = proj[src * 10 + 2];
  a[3] = proj[src * 10 + 3];
  a[4] = proj[src * 10 + 4];
  a[5] = opacity[src];
  a[6] = proj[src * 10 + 5];
  a[7] = colors[src * 3];
  a[8] = colors[src * 3 + 1];
  a[9] = colors[src * 3 + 2];
}

__global__ void scatter_kernel(int64_t m, const int64_t *ids, const float *grad,
                               float *gmeans, float *gproj, float *gopacity,
                               float *gcolors) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= m)
    return;
  const int64_t dst = ids[i];
  const float *a = grad + 10 * i;
  if (gmeans) {
    gmeans[dst * 2] = a[0];
    gmeans[dst * 2 + 1] = a[1];
  }
  if (gproj) {
    gproj[dst * 10 + 2] = a[2];
    gproj[dst * 10 + 3] = a[3];
    gproj[dst * 10 + 4] = a[4];
    gproj[dst * 10 + 5] = a[6];
  }
  if (gopacity) gopacity[dst] = a[5];
  if (gcolors) {
    gcolors[dst * 3] = a[7];
    gcolors[dst * 3 + 1] = a[8];
    gcolors[dst * 3 + 2] = a[9];
  }
}

constexpr int kThreads = 256;
} // namespace

void gather_attributes_forward_cuda(const at::Tensor &means, const at::Tensor &proj,
                                    const at::Tensor &opacity, const at::Tensor &colors,
                                    const at::Tensor &ids, at::Tensor &out) {
  const int64_t m = ids.numel();
  if (!m)
    return;
  const c10::cuda::CUDAGuard guard(proj.device());
  gather_kernel<<<static_cast<unsigned>((m + kThreads - 1) / kThreads), kThreads, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      m, ids.data_ptr<int64_t>(), means.data_ptr<float>(), means.stride(0), means.stride(1),
      proj.data_ptr<float>(), opacity.data_ptr<float>(), colors.data_ptr<float>(),
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_attributes_backward_cuda(const at::Tensor &ids, const at::Tensor &grad,
                                     at::Tensor &gmeans, at::Tensor &gproj,
                                     at::Tensor &gopacity, at::Tensor &gcolors) {
  const int64_t m = ids.numel();
  if (!m)
    return;
  const c10::cuda::CUDAGuard guard(grad.device());
  scatter_kernel<<<static_cast<unsigned>((m + kThreads - 1) / kThreads), kThreads, 0,
                   at::cuda::getCurrentCUDAStream()>>>(
      m, ids.data_ptr<int64_t>(), grad.data_ptr<float>(),
      gmeans.defined() ? gmeans.data_ptr<float>() : nullptr,
      gproj.defined() ? gproj.data_ptr<float>() : nullptr,
      gopacity.defined() ? gopacity.data_ptr<float>() : nullptr,
      gcolors.defined() ? gcolors.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace gaussian
} // namespace minkowski
