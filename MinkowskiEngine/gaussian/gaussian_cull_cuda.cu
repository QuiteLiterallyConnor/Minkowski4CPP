// Fused culling, tile bounds and tile keys for the CUDA path of render_gaussians (see
// gaussian_render.cpp). The LibTorch version runs ~30 element-wise kernels for the culling and
// ~15 (plus repeat_interleave's own synchronisation) for the tile lists; these kernels compute the
// same values with the same float32 operations, one thread per Gaussian or per pair.
#include "gaussian_cuda.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace minkowski {
namespace gaussian {
namespace detail {
namespace {

__global__ void cull_kernel(int64_t n, const float *__restrict__ proj,
                            const float *__restrict__ opacity, CullParams k,
                            bool *__restrict__ visible, int64_t *__restrict__ radius,
                            int64_t *__restrict__ bounds, int64_t *__restrict__ counts) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  const float *p = proj + kProjectColumns * i;
  const float mx = p[0], my = p[1], z = p[5], a = p[6], c = p[8], det = p[9];
  const float o = opacity[i];
  // extent = sqrt(max(2 ln(255 max(o, 1/255)), 0) + slack); NaN propagates as in LibTorch.
  const float oc = o < k.min_alpha ? k.min_alpha : o;
  float e2 = 2.0f * logf(oc * 255.0f);
  e2 = e2 < 0.0f ? 0.0f : e2;
  const float extent = sqrtf(e2 + k.extent_slack);
  const float rx = ceilf(extent * sqrtf(a)), ry = ceilf(extent * sqrtf(c));
  const bool vis = (z > k.near_plane) & (det > 1e-12f) & (o >= k.min_alpha) & isfinite(mx) &
                   isfinite(my) & isfinite(rx) & isfinite(ry) & (mx + rx >= 0.0f) &
                   (mx - rx < k.width) & (my + ry >= 0.0f) & (my - ry < k.height);
  visible[i] = vis;
  int64_t *b = bounds + 4 * i;
  if (!vis) {
    radius[i] = 0;
    b[0] = b[1] = b[2] = b[3] = 0;
    counts[i] = 0;
    return;
  }
  radius[i] = static_cast<int64_t>(fmaxf(rx, ry));
  const float tile = static_cast<float>(k.tile);
  const auto cl = [](float v, float hi) { return v < 0.0f ? 0.0f : (v > hi ? hi : v); };
  const float mc = static_cast<float>(k.max_col), mr = static_cast<float>(k.max_row);
  b[0] = static_cast<int64_t>(cl(floorf((mx - rx) / tile), mc));
  b[1] = static_cast<int64_t>(cl(floorf((mx + rx) / tile), mc));
  b[2] = static_cast<int64_t>(cl(floorf((my - ry) / tile), mr));
  b[3] = static_cast<int64_t>(cl(floorf((my + ry) / tile), mr));
  counts[i] = (b[1] - b[0] + 1) * (b[3] - b[2] + 1);
}

// Gaussian g (row of the depth-sorted list) writes (tile * m + g) for every tile it touches,
// starting at its exclusive prefix offset.
__global__ void tile_keys_kernel(int64_t m, const int64_t *__restrict__ bounds,
                                 const int64_t *__restrict__ offsets, int64_t cols,
                                 int64_t *__restrict__ keys) {
  const int64_t g = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (g >= m)
    return;
  const int64_t *b = bounds + 4 * g;
  int64_t at = offsets[g];
  for (int64_t ty = b[2]; ty <= b[3]; ++ty)
    for (int64_t tx = b[0]; tx <= b[1]; ++tx)
      keys[at++] = (ty * cols + tx) * m + g;
}

__global__ void tile_counts_kernel(int64_t m, const int64_t *__restrict__ bounds,
                                   int64_t *__restrict__ counts) {
  const int64_t g = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (g >= m)
    return;
  const int64_t *b = bounds + 4 * g;
  counts[g] = (b[1] - b[0] + 1) * (b[3] - b[2] + 1);
}

// Thread i in [0, total] looks at the sorted tile ids t[i - 1] and t[i] (t[-1] = -1, t[total] =
// tiles): every tile in (t[i - 1], t[i]] starts at pair i, and the tile before it ends there. This
// gives exactly the (end - count, end) ranges of the bincount + cumsum formulation, including
// empty tiles. Thread i < total also writes the Gaussian row of pair i.
__global__ void finish_lists_kernel(int64_t total, const int64_t *__restrict__ keys, int64_t m,
                                    int64_t tiles, int32_t *__restrict__ ids,
                                    int32_t *__restrict__ ranges) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i > total)
    return;
  const int64_t lo = i == 0 ? -1 : keys[i - 1] / m;
  const int64_t hi = i == total ? tiles : keys[i] / m;
  if (i < total)
    ids[i] = static_cast<int32_t>(keys[i] % m);
  for (int64_t t = lo + 1; t <= hi; ++t) {
    if (t < tiles)
      ranges[2 * t] = static_cast<int32_t>(i);
    if (t > 0)
      ranges[2 * (t - 1) + 1] = static_cast<int32_t>(i);
  }
}

constexpr int kThreads = 256;
unsigned blocks_for(int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

} // namespace

void cull_cuda(const at::Tensor &proj, const at::Tensor &opacity, const CullParams &params,
               at::Tensor &visible, at::Tensor &radius, at::Tensor &bounds, at::Tensor &counts) {
  const int64_t n = proj.size(0);
  TORCH_CHECK(proj.is_cuda() && proj.is_contiguous() && proj.scalar_type() == at::kFloat &&
                  proj.size(1) == kProjectColumns,
              "cull_cuda: proj must be a contiguous float32 CUDA tensor [N, 10]");
  TORCH_CHECK(opacity.is_contiguous() && opacity.numel() == n && opacity.scalar_type() == at::kFloat,
              "cull_cuda: opacity must be a contiguous float32 tensor with N elements");
  if (n == 0)
    return;
  const c10::cuda::CUDAGuard guard(proj.device());
  cull_kernel<<<blocks_for(n), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      n, proj.data_ptr<float>(), opacity.data_ptr<float>(), params, visible.data_ptr<bool>(),
      radius.data_ptr<int64_t>(), bounds.data_ptr<int64_t>(), counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tile_keys_cuda(const at::Tensor &bounds, const at::Tensor &offsets, int64_t cols,
                    at::Tensor &keys) {
  const int64_t m = bounds.size(0);
  if (m == 0)
    return;
  const c10::cuda::CUDAGuard guard(bounds.device());
  tile_keys_kernel<<<blocks_for(m), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      m, bounds.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), cols, keys.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void tile_counts_cuda(const at::Tensor &bounds, at::Tensor &counts) {
  const int64_t m = bounds.size(0);
  if (m == 0)
    return;
  const c10::cuda::CUDAGuard guard(bounds.device());
  tile_counts_kernel<<<blocks_for(m), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      m, bounds.data_ptr<int64_t>(), counts.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finish_tile_lists_cuda(const at::Tensor &keys, int64_t m, int64_t tiles, at::Tensor &ids,
                            at::Tensor &ranges) {
  const int64_t total = keys.numel();
  const c10::cuda::CUDAGuard guard(keys.device());
  finish_lists_kernel<<<blocks_for(total + 1), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      total, keys.data_ptr<int64_t>(), m, tiles, ids.data_ptr<int32_t>(),
      ranges.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace gaussian
} // namespace minkowski
