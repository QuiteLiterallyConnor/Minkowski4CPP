// Fused SSIM for the trainer's loss (see ssim() in gaussian_trainer.cpp). The LibTorch version runs
// five depthwise window x window convolutions and ~25 element-wise kernels forward, and about as
// many backward. Here one kernel per 16x16 tile and channel filters the five local moments with the
// separable Gaussian window through shared memory (zero padding, like conv2d) and writes the SSIM
// map plus the three per-pixel partial derivatives the gradient needs; the backward kernel filters
// those partials the same way. No atomics: forward and backward are deterministic.
//
// With S = (2 mx my + c1)(2 sxy + c2) / ((mx^2 + my^2 + c1)(sxx + syy + c2)), mx = G*x,
// sxx = G*(x^2) - mx^2 and sxy = G*(xy) - mx my, the gradient of sum_p S(p) w.r.t. x(q) is
//   (G*A)(q) + 2 x(q) (G*B)(q) + y(q) (G*C)(q)
// with A = dS/dmx (through sxx and sxy too), B = dS/dsxx and C = dS/dsxy at every pixel
// (G is symmetric, so correlation and convolution coincide).
#include "gaussian_cuda.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace minkowski {
namespace gaussian {
namespace detail {
namespace {

constexpr int kT = 16;                          // output tile edge
constexpr int kMaxR = kSsimMaxWindow / 2;       // largest supported radius
constexpr int kMaxS = kT + 2 * kMaxR;           // tile + halo edge
constexpr float kC1 = 0.01f * 0.01f, kC2 = 0.03f * 0.03f;

// Loads plane(r, c) for the tile and its halo; outside the image reads 0 (conv2d zero padding).
template <typename F>
__device__ __forceinline__ void load_halo(float (*dst)[kMaxS], int S, int R, int y0, int x0,
                                          int height, int width, F plane) {
  const int t = threadIdx.y * kT + threadIdx.x;
  for (int i = t; i < S * S; i += kT * kT) {
    const int r = i / S, c = i % S;
    const int gy = y0 - R + r, gx = x0 - R + c;
    dst[r][c] = (gy >= 0 && gy < height && gx >= 0 && gx < width) ? plane(gy, gx) : 0.0f;
  }
}

__global__ void ssim_forward_kernel(const float *__restrict__ x, const float *__restrict__ y,
                                    int height, int width, SsimWindow win,
                                    float *__restrict__ map, float *__restrict__ partials) {
  __shared__ float sx[kMaxS][kMaxS], sy[kMaxS][kMaxS];
  __shared__ float h[5][kMaxS][kT]; // horizontal pass of x, y, xx, yy, xy
  const int ch = blockIdx.z, R = win.radius, S = kT + 2 * R;
  const int y0 = blockIdx.y * kT, x0 = blockIdx.x * kT;
  const auto px = [&](const float *img) {
    return [=](int gy, int gx) { return img[(static_cast<int64_t>(gy) * width + gx) * 3 + ch]; };
  };
  load_halo(sx, S, R, y0, x0, height, width, px(x));
  load_halo(sy, S, R, y0, x0, height, width, px(y));
  __syncthreads();
  const int t = threadIdx.y * kT + threadIdx.x;
  for (int i = t; i < S * kT; i += kT * kT) {
    const int r = i / kT, c = i % kT;
    float a[5] = {0, 0, 0, 0, 0};
    for (int d = 0; d <= 2 * R; ++d) {
      const float w = win.w[d], vx = sx[r][c + d], vy = sy[r][c + d];
      a[0] += w * vx;
      a[1] += w * vy;
      a[2] += w * (vx * vx);
      a[3] += w * (vy * vy);
      a[4] += w * (vx * vy);
    }
#pragma unroll
    for (int k = 0; k < 5; ++k)
      h[k][r][c] = a[k];
  }
  __syncthreads();
  const int gy = y0 + threadIdx.y, gx = x0 + threadIdx.x;
  if (gy >= height || gx >= width)
    return;
  float m[5] = {0, 0, 0, 0, 0};
  for (int d = 0; d <= 2 * R; ++d) {
    const float w = win.w[d];
#pragma unroll
    for (int k = 0; k < 5; ++k)
      m[k] += w * h[k][threadIdx.y + d][threadIdx.x];
  }
  const float mx = m[0], my = m[1];
  const float sxx = m[2] - mx * mx, syy = m[3] - my * my, sxy = m[4] - mx * my;
  const float n1 = 2.0f * (mx * my) + kC1, n2 = 2.0f * sxy + kC2;
  const float d1 = mx * mx + my * my + kC1, d2 = sxx + syy + kC2;
  const float s = (n1 * n2) / (d1 * d2);
  const int64_t plane = static_cast<int64_t>(height) * width;
  const int64_t at = ch * plane + static_cast<int64_t>(gy) * width + gx;
  map[at] = s;
  // dS/dmx with sxx and sxy held fixed, then through sxx = E[xx] - mx^2 and sxy = E[xy] - mx my.
  const float inv = 1.0f / (d1 * d2);
  const float dsxx = -s / d2;
  const float dsxy = 2.0f * n1 * inv;
  const float dmx = 2.0f * my * n2 * inv - s * 2.0f * mx / d1;
  partials[at] = dmx - 2.0f * mx * dsxx - my * dsxy; // A
  partials[3 * plane + at] = dsxx;                   // B
  partials[6 * plane + at] = dsxy;                   // C
}

__global__ void ssim_backward_kernel(const float *__restrict__ x, const float *__restrict__ y,
                                     const float *__restrict__ partials, int height, int width,
                                     SsimWindow win, const float *__restrict__ grad_out,
                                     float inv_count, float *__restrict__ grad_x) {
  __shared__ float sp[3][kMaxS][kMaxS];
  __shared__ float h[3][kMaxS][kT];
  const int ch = blockIdx.z, R = win.radius, S = kT + 2 * R;
  const int y0 = blockIdx.y * kT, x0 = blockIdx.x * kT;
  const int64_t plane = static_cast<int64_t>(height) * width;
  for (int k = 0; k < 3; ++k) {
    const float *src = partials + (3 * k + ch) * plane;
    load_halo(sp[k], S, R, y0, x0, height, width,
              [=](int gy, int gx) { return src[static_cast<int64_t>(gy) * width + gx]; });
  }
  __syncthreads();
  const int t = threadIdx.y * kT + threadIdx.x;
  for (int i = t; i < S * kT; i += kT * kT) {
    const int r = i / kT, c = i % kT;
    float a[3] = {0, 0, 0};
    for (int d = 0; d <= 2 * R; ++d) {
      const float w = win.w[d];
#pragma unroll
      for (int k = 0; k < 3; ++k)
        a[k] += w * sp[k][r][c + d];
    }
#pragma unroll
    for (int k = 0; k < 3; ++k)
      h[k][r][c] = a[k];
  }
  __syncthreads();
  const int gy = y0 + threadIdx.y, gx = x0 + threadIdx.x;
  if (gy >= height || gx >= width)
    return;
  float g[3] = {0, 0, 0};
  for (int d = 0; d <= 2 * R; ++d) {
    const float w = win.w[d];
#pragma unroll
    for (int k = 0; k < 3; ++k)
      g[k] += w * h[k][threadIdx.y + d][threadIdx.x];
  }
  const int64_t i = (static_cast<int64_t>(gy) * width + gx) * 3 + ch;
  const float scale = grad_out[0] * inv_count;
  grad_x[i] = scale * (g[0] + 2.0f * x[i] * g[1] + y[i] * g[2]);
}

dim3 grid_for(int64_t height, int64_t width) {
  return dim3(static_cast<unsigned>((width + kT - 1) / kT),
              static_cast<unsigned>((height + kT - 1) / kT), 3);
}

void check_image(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kFloat && t.dim() == 3 &&
                  t.size(2) == 3,
              "fused SSIM: ", name, " must be a contiguous float32 CUDA tensor [H, W, 3]");
}

} // namespace

void ssim_forward_cuda(const at::Tensor &x, const at::Tensor &y, const SsimWindow &window,
                       at::Tensor &map, at::Tensor &partials) {
  check_image(x, "prediction");
  check_image(y, "target");
  TORCH_CHECK(window.radius >= 0 && window.radius <= kMaxR, "fused SSIM: window too large");
  const int64_t height = x.size(0), width = x.size(1);
  if (height == 0 || width == 0)
    return;
  const c10::cuda::CUDAGuard guard(x.device());
  ssim_forward_kernel<<<grid_for(height, width), dim3(kT, kT), 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), static_cast<int>(height),
      static_cast<int>(width), window, map.data_ptr<float>(), partials.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ssim_backward_cuda(const at::Tensor &x, const at::Tensor &y, const at::Tensor &partials,
                        const SsimWindow &window, const at::Tensor &grad_out, double inv_count,
                        at::Tensor &grad_x) {
  const int64_t height = x.size(0), width = x.size(1);
  if (height == 0 || width == 0)
    return;
  TORCH_CHECK(grad_out.is_cuda() && grad_out.numel() == 1 && grad_out.scalar_type() == at::kFloat,
              "fused SSIM: grad_out must be a float32 CUDA scalar");
  const c10::cuda::CUDAGuard guard(x.device());
  ssim_backward_kernel<<<grid_for(height, width), dim3(kT, kT), 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), partials.data_ptr<float>(),
      static_cast<int>(height), static_cast<int>(width), window,
      grad_out.contiguous().data_ptr<float>(), static_cast<float>(inv_count),
      grad_x.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace gaussian
} // namespace minkowski
