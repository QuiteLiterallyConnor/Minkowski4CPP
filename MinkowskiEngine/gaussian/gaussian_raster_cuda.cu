#include "gaussian_cuda.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>

namespace minkowski {
namespace gaussian {
namespace detail {
namespace {

constexpr int kTile = kRasterTileSize;
constexpr int kBlock = kTile * kTile;
constexpr int kAttrs = 10; // mx my ca cb cc opacity depth r g b
constexpr float kMinAlpha = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.99f;
constexpr float kStopT = 1e-4f;

struct Shared {
  float2 xy[kBlock];
  float4 conic_opacity[kBlock];
  float4 color_depth[kBlock];
};

__device__ __forceinline__ void load_batch(Shared &s, const float *attrs,
                                           const int32_t *ids, int position,
                                           int end) {
  const int t = threadIdx.y * kTile + threadIdx.x;
  const int p = position + t;
  if (p < end) {
    const float *a = attrs + static_cast<int64_t>(ids[p]) * kAttrs;
    s.xy[t] = make_float2(a[0], a[1]);
    s.conic_opacity[t] = make_float4(a[2], a[3], a[4], a[5]);
    s.color_depth[t] = make_float4(a[7], a[8], a[9], a[6]);
  }
}

// Returns alpha for one pixel/Gaussian pair, or -1 if it does not contribute.
__device__ __forceinline__ float eval_alpha(const Shared &s, int j, float px,
                                            float py, float *dx, float *dy,
                                            float *g) {
  const float2 xy = s.xy[j];
  const float4 co = s.conic_opacity[j];
  *dx = px - xy.x;
  *dy = py - xy.y;
  const float power =
      -0.5f * (co.x * *dx * *dx + co.z * *dy * *dy) - co.y * *dx * *dy;
  if (power > 0.0f)
    return -1.0f;
  *g = __expf(power);
  const float alpha = fminf(kMaxAlpha, co.w * *g);
  return alpha < kMinAlpha ? -1.0f : alpha;
}

__global__ void raster_forward_kernel(const float *attrs, const int32_t *ids,
                                      const int32_t *ranges,
                                      const float *background, int height,
                                      int width, int tiles_x, float *out,
                                      float *final_t, int32_t *last_index) {
  __shared__ Shared s;
  const int tile = blockIdx.y * tiles_x + blockIdx.x;
  const int px_i = blockIdx.x * kTile + threadIdx.x;
  const int py_i = blockIdx.y * kTile + threadIdx.y;
  const bool inside = px_i < width && py_i < height;
  const float px = px_i + 0.5f, py = py_i + 0.5f;
  const int begin = ranges[tile * 2], end = ranges[tile * 2 + 1];
  const int count = end - begin;

  float T = 1.0f, r = 0, g = 0, b = 0, d = 0;
  int last = 0;
  bool done = !inside;

  for (int base = 0; base < count; base += kBlock) {
    if (__syncthreads_count(done) == kBlock)
      break;
    load_batch(s, attrs, ids + begin, base, count);
    __syncthreads();
    const int n = min(kBlock, count - base);
    for (int j = 0; j < n && !done; ++j) {
      float dx, dy, gauss;
      const float alpha = eval_alpha(s, j, px, py, &dx, &dy, &gauss);
      if (alpha < 0.0f)
        continue;
      const float next = T * (1.0f - alpha);
      if (next < kStopT) {
        done = true;
        break;
      }
      const float4 c = s.color_depth[j];
      const float w = alpha * T;
      r += c.x * w;
      g += c.y * w;
      b += c.z * w;
      d += c.w * w;
      T = next;
      last = base + j + 1;
    }
    __syncthreads();
  }

  if (inside) {
    const int pixel = py_i * width + px_i;
    float *o = out + static_cast<int64_t>(pixel) * 5;
    o[0] = r + T * background[0];
    o[1] = g + T * background[1];
    o[2] = b + T * background[2];
    o[3] = d;
    o[4] = 1.0f - T;
    final_t[pixel] = T;
    last_index[pixel] = last;
  }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, offset);
  return v;
}

__global__ void raster_backward_kernel(
    const float *attrs, const int32_t *ids, const int32_t *ranges,
    const float *background, int height, int width, int tiles_x,
    const float *grad_out, const float *final_t, const int32_t *last_index,
    float *grad_attrs) {
  __shared__ Shared s;
  __shared__ int block_last;
  const int t = threadIdx.y * kTile + threadIdx.x;
  const int lane = t & 31;
  const int tile = blockIdx.y * tiles_x + blockIdx.x;
  const int px_i = blockIdx.x * kTile + threadIdx.x;
  const int py_i = blockIdx.y * kTile + threadIdx.y;
  const bool inside = px_i < width && py_i < height;
  const float px = px_i + 0.5f, py = py_i + 0.5f;
  const int begin = ranges[tile * 2];

  const int pixel = py_i * width + px_i;
  const int my_last = inside ? last_index[pixel] : 0;
  float T = inside ? final_t[pixel] : 1.0f;
  const float T_final = T;
  float go[5] = {0, 0, 0, 0, 0};
  if (inside)
    for (int k = 0; k < 5; ++k)
      go[k] = grad_out[static_cast<int64_t>(pixel) * 5 + k];
  const float bg_dot = background[0] * go[0] + background[1] * go[1] +
                       background[2] * go[2];

  if (t == 0)
    block_last = 0;
  __syncthreads();
  atomicMax(&block_last, my_last);
  __syncthreads();
  const int total = block_last;

  // accum holds the compositing of everything behind the current Gaussian.
  float acc[4] = {0, 0, 0, 0};
  float last_alpha = 0.0f;
  float last_color[4] = {0, 0, 0, 0};

  for (int base = ((total + kBlock - 1) / kBlock - 1) * kBlock; base >= 0;
       base -= kBlock) {
    __syncthreads();
    load_batch(s, attrs, ids + begin, base, total);
    __syncthreads();
    const int n = min(kBlock, total - base);
    for (int j = n - 1; j >= 0; --j) {
      float v[kAttrs];
#pragma unroll
      for (int k = 0; k < kAttrs; ++k)
        v[k] = 0.0f;

      float dx, dy, gauss;
      float alpha = -1.0f;
      if (inside && base + j < my_last)
        alpha = eval_alpha(s, j, px, py, &dx, &dy, &gauss);
      if (alpha >= 0.0f) {
        T = T / (1.0f - alpha);
        const float4 co = s.conic_opacity[j];
        const float4 cd = s.color_depth[j];
        const float channel[4] = {cd.x, cd.y, cd.z, cd.w};
        const float weight = alpha * T;
        float dalpha = 0.0f;
        for (int k = 0; k < 4; ++k) {
          acc[k] = last_alpha * last_color[k] + (1.0f - last_alpha) * acc[k];
          last_color[k] = channel[k];
          dalpha += (channel[k] - acc[k]) * go[k == 3 ? 3 : k];
        }
        dalpha *= T;
        const float inv = T_final / (1.0f - alpha);
        dalpha += go[4] * inv - bg_dot * inv;
        last_alpha = alpha;

        v[6] = weight * go[3];
        v[7] = weight * go[0];
        v[8] = weight * go[1];
        v[9] = weight * go[2];
        v[5] = gauss * dalpha;
        // Only the unclamped branch passes gradient through min(alpha, 0.99).
        const float unclamped = co.w * gauss;
        if (unclamped <= kMaxAlpha) {
          const float dpower = unclamped * dalpha;
          v[0] = dpower * (co.x * dx + co.y * dy);
          v[1] = dpower * (co.z * dy + co.y * dx);
          v[2] = -0.5f * dx * dx * dpower;
          v[3] = -dx * dy * dpower;
          v[4] = -0.5f * dy * dy * dpower;
        } else {
          v[5] = 0.0f;
        }
      }

      // Whole warp reduces, then one lane issues the atomics.
#pragma unroll
      for (int k = 0; k < kAttrs; ++k)
        v[k] = warp_sum(v[k]);
      if (lane == 0) {
        float *dst = grad_attrs + static_cast<int64_t>(ids[begin + base + j]) *
                                      kAttrs;
#pragma unroll
        for (int k = 0; k < kAttrs; ++k)
          if (v[k] != 0.0f)
            atomicAdd(dst + k, v[k]);
      }
    }
  }
}

} // namespace

void rasterize_forward_cuda(const at::Tensor &attrs, const at::Tensor &ids,
                            const at::Tensor &ranges,
                            const at::Tensor &background, int64_t height,
                            int64_t width, at::Tensor &out,
                            at::Tensor &final_t, at::Tensor &last_index) {
  const c10::cuda::CUDAGuard guard(attrs.device());
  const int tiles_x = static_cast<int>((width + kTile - 1) / kTile);
  const int tiles_y = static_cast<int>((height + kTile - 1) / kTile);
  const dim3 grid(tiles_x, tiles_y), block(kTile, kTile);
  raster_forward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      attrs.data_ptr<float>(), ids.data_ptr<int32_t>(),
      ranges.data_ptr<int32_t>(), background.data_ptr<float>(),
      static_cast<int>(height), static_cast<int>(width), tiles_x,
      out.data_ptr<float>(), final_t.data_ptr<float>(),
      last_index.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rasterize_backward_cuda(const at::Tensor &attrs, const at::Tensor &ids,
                             const at::Tensor &ranges,
                             const at::Tensor &background,
                             const at::Tensor &grad_out,
                             const at::Tensor &final_t,
                             const at::Tensor &last_index, int64_t height,
                             int64_t width, at::Tensor &grad_attrs) {
  const c10::cuda::CUDAGuard guard(attrs.device());
  const int tiles_x = static_cast<int>((width + kTile - 1) / kTile);
  const int tiles_y = static_cast<int>((height + kTile - 1) / kTile);
  const dim3 grid(tiles_x, tiles_y), block(kTile, kTile);
  raster_backward_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      attrs.data_ptr<float>(), ids.data_ptr<int32_t>(),
      ranges.data_ptr<int32_t>(), background.data_ptr<float>(),
      static_cast<int>(height), static_cast<int>(width), tiles_x,
      grad_out.data_ptr<float>(), final_t.data_ptr<float>(),
      last_index.data_ptr<int32_t>(), grad_attrs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace gaussian
} // namespace minkowski
