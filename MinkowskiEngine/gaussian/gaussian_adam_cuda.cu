// Fused Adam update of GaussianTrainer::adam_step for CUDA parameters. The LibTorch version runs
// eight element-wise kernels per parameter tensor (mul_, add_, mul_, addcmul_, div, sqrt, add_,
// addcdiv_), each reading and writing the whole tensor; this kernel reads p, g, m1, m2 once and
// writes p, m1, m2 once, with the same float32 formulas.
#include "gaussian_cuda.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <initializer_list>

namespace minkowski {
namespace gaussian {
namespace detail {
namespace {

__global__ void adam_kernel(int64_t n, float *__restrict__ p, const float *__restrict__ g,
                            float *__restrict__ m1, float *__restrict__ m2, AdamParams k) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float gi = g[i];
    // m1 = beta1 m1 + (1 - beta1) g;  m2 = beta2 m2 + (1 - beta2) g^2
    // Each mul_ in the LibTorch path stores a rounded float before add_/addcmul_.
    // Preserve those rounding boundaries, and square g before scaling it, as addcmul_ does.
    const float a = fmaf(k.one_minus_beta1, gi, __fmul_rn(m1[i], k.beta1));
    const float b = fmaf(k.one_minus_beta2, __fmul_rn(gi, gi),
                         __fmul_rn(m2[i], k.beta2));
    m1[i] = a;
    m2[i] = b;
    // p += step * m1 / (sqrt(m2 / bias2) + eps), step = -lr / bias1
    p[i] = fmaf(k.step, a / (sqrtf(b * k.inv_bias2) + k.epsilon), p[i]);
  }
}

} // namespace

void adam_update_cuda(at::Tensor &param, const at::Tensor &grad, at::Tensor &moment1,
                      at::Tensor &moment2, const AdamParams &params) {
  const int64_t n = param.numel();
  using List = std::initializer_list<const at::Tensor *>;
  for (const at::Tensor *t : List{&param, &grad, &moment1, &moment2})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous() && t->scalar_type() == at::kFloat &&
                    t->numel() == n && t->device() == param.device(),
                "adam_update_cuda: parameter, gradient and moments must be contiguous float32 CUDA "
                "tensors of the same size on one device");
  if (n == 0)
    return;
  const c10::cuda::CUDAGuard guard(param.device());
  constexpr int threads = 256;
  const int64_t blocks = std::min<int64_t>((n + threads - 1) / threads, 4096);
  adam_kernel<<<static_cast<unsigned>(blocks), threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      n, param.data_ptr<float>(), grad.data_ptr<float>(), moment1.data_ptr<float>(),
      moment2.data_ptr<float>(), params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace gaussian
} // namespace minkowski
