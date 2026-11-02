#include "gaussian/gaussian_render.hpp"

#include "gaussian/gaussian_sh.hpp"
#ifdef USE_CUDA
#include "gaussian_cuda.hpp"
#endif

#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace minkowski {
namespace gaussian {
namespace {

// Per-Gaussian attribute columns after projection and depth sorting.
constexpr int64_t kAttributeCount = 10; // mx my ca cb cc opacity depth r g b
constexpr float kMinAlpha = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.99f;
constexpr float kStopTransmittance = 1e-4f;
constexpr double kLowPass = 0.3;

at::Tensor quaternion_matrices(const at::Tensor &q) {
  const auto n = q / q.norm(2, {1}, true).clamp_min(1e-12);
  const auto w = n.select(1, 0), x = n.select(1, 1), y = n.select(1, 2),
             z = n.select(1, 3);
  const auto row0 = at::stack({1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                               2 * (x * z + w * y)},
                              1);
  const auto row1 = at::stack({2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                               2 * (y * z - w * x)},
                              1);
  const auto row2 = at::stack({2 * (x * z - w * y), 2 * (y * z + w * x),
                               1 - 2 * (x * x + y * y)},
                              1);
  return at::stack({row0, row1, row2}, 1);
}

// Front-to-back compositing of depth-sorted Gaussians over one tile.
// attrs: [K, 10]; px: [P, 2] pixel centers. Returns [P, 5]: rgb (with the
// background blended in), alpha-weighted depth, accumulated alpha.
at::Tensor composite_tile(const at::Tensor &attrs, const at::Tensor &px,
                          const at::Tensor &background, int64_t chunk) {
  const int64_t count = attrs.size(0);
  const int64_t pixels = px.size(0);
  const auto options = attrs.options();
  at::Tensor color = at::zeros({pixels, 3}, options);
  at::Tensor depth = at::zeros({pixels}, options);
  at::Tensor transmittance = at::ones({pixels}, options);
  const auto pixel_x = px.select(1, 0).unsqueeze(0);
  const auto pixel_y = px.select(1, 1).unsqueeze(0);

  for (int64_t begin = 0; begin < count; begin += chunk) {
    const int64_t size = std::min(chunk, count - begin);
    const auto a = attrs.slice(0, begin, begin + size);
    const auto dx = pixel_x - a.select(1, 0).unsqueeze(1);
    const auto dy = pixel_y - a.select(1, 1).unsqueeze(1);
    const auto ca = a.select(1, 2).unsqueeze(1);
    const auto cb = a.select(1, 3).unsqueeze(1);
    const auto cc = a.select(1, 4).unsqueeze(1);
    const auto power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy;
    auto alpha = (a.select(1, 5).unsqueeze(1) * at::exp(power.clamp_max(0.0)))
                     .clamp_max(kMaxAlpha);
    alpha = at::where(alpha >= kMinAlpha, alpha, at::zeros_like(alpha));

    const auto cumulative = at::cumprod(1.0 - alpha, 0);
    const auto before =
        at::cat({at::ones({1, pixels}, options),
                 cumulative.slice(0, 0, size - 1)},
                0) *
        transmittance.unsqueeze(0);
    const auto weight = before * alpha;
    color = color + at::matmul(weight.transpose(0, 1), a.slice(1, 7, 10));
    depth = depth + at::matmul(weight.transpose(0, 1), a.select(1, 6));
    transmittance = transmittance * cumulative.select(0, size - 1);
    if (begin + size < count &&
        transmittance.max().item<float>() < kStopTransmittance)
      break;
  }
  const auto rgb = color + transmittance.unsqueeze(1) * background;
  return at::cat({rgb, depth.unsqueeze(1), (1.0 - transmittance).unsqueeze(1)},
                 1);
}

struct TileGrid {
  int64_t height, width, tile, chunk, rows, cols;
  at::Tensor x0, x1, y0, y1; // per-Gaussian inclusive tile ranges

  TileGrid(const at::Tensor &bounds, const at::Tensor &meta) {
    const auto m = meta.accessor<int64_t, 1>();
    height = m[0];
    width = m[1];
    tile = m[2];
    chunk = m[3];
    rows = (height + tile - 1) / tile;
    cols = (width + tile - 1) / tile;
    x0 = bounds.select(1, 0);
    x1 = bounds.select(1, 1);
    y0 = bounds.select(1, 2);
    y1 = bounds.select(1, 3);
  }

  at::Tensor select(int64_t row, int64_t col) const {
    return ((x0 <= col) & (x1 >= col) & (y0 <= row) & (y1 >= row))
        .nonzero()
        .squeeze(1);
  }
};

at::Tensor tile_pixels(const TileGrid &grid, int64_t row, int64_t col,
                       const at::TensorOptions &options, int64_t *y_begin,
                       int64_t *y_end, int64_t *x_begin, int64_t *x_end) {
  *y_begin = row * grid.tile;
  *y_end = std::min(*y_begin + grid.tile, grid.height);
  *x_begin = col * grid.tile;
  *x_end = std::min(*x_begin + grid.tile, grid.width);
  const auto ys =
      at::arange(*y_begin, *y_end, options.dtype(at::kFloat)) + 0.5;
  const auto xs =
      at::arange(*x_begin, *x_end, options.dtype(at::kFloat)) + 0.5;
  const auto grids = at::meshgrid({ys, xs}, "ij");
  return at::stack({grids[1].reshape({-1}), grids[0].reshape({-1})}, 1);
}

struct RasterFunction : public torch::autograd::Function<RasterFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            const at::Tensor &attrs,
                            const at::Tensor &background,
                            const at::Tensor &bounds, const at::Tensor &meta) {
    ctx->save_for_backward({attrs, background, bounds, meta});
    const TileGrid grid(bounds, meta);
    auto out = at::empty({grid.height, grid.width, 5}, attrs.options());
    for (int64_t row = 0; row < grid.rows; ++row)
      for (int64_t col = 0; col < grid.cols; ++col) {
        int64_t y0, y1, x0, x1;
        const auto px =
            tile_pixels(grid, row, col, attrs.options(), &y0, &y1, &x0, &x1);
        const auto sel = grid.select(row, col);
        const auto tile =
            composite_tile(attrs.index_select(0, sel), px, background,
                           grid.chunk);
        out.slice(0, y0, y1).slice(1, x0, x1).copy_(
            tile.reshape({y1 - y0, x1 - x0, 5}));
      }
    return out;
  }

  // Each tile is recomputed with autograd enabled, so only the inputs are kept
  // alive between forward and backward.
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const auto &attrs = saved[0];
    const auto &background = saved[1];
    const TileGrid grid(saved[2], saved[3]);
    const auto grad_out = grad_outputs[0].contiguous();
    auto grad_attrs = at::zeros_like(attrs);
    auto grad_background = at::zeros_like(background);

    for (int64_t row = 0; row < grid.rows; ++row)
      for (int64_t col = 0; col < grid.cols; ++col) {
        int64_t y0, y1, x0, x1;
        const auto px =
            tile_pixels(grid, row, col, attrs.options(), &y0, &y1, &x0, &x1);
        const auto sel = grid.select(row, col);
        const auto go = grad_out.slice(0, y0, y1)
                            .slice(1, x0, x1)
                            .reshape({(y1 - y0) * (x1 - x0), 5});
        if (sel.numel() == 0) {
          grad_background += go.slice(1, 0, 3).sum(0);
          continue;
        }
        torch::AutoGradMode enable_grad(true);
        const auto a =
            attrs.index_select(0, sel).detach().set_requires_grad(true);
        const auto b = background.detach().set_requires_grad(true);
        const auto tile = composite_tile(a, px, b, grid.chunk);
        const auto grads = torch::autograd::grad({tile}, {a, b}, {go}, false,
                                                 false, true);
        grad_attrs.index_add_(0, sel, grads[0].detach());
        if (grads[1].defined())
          grad_background += grads[1].detach();
      }
    return {grad_attrs, grad_background, at::Tensor(), at::Tensor()};
  }
};

// Groups (tile, Gaussian) pairs by tile; Gaussians are already depth sorted, so
// sorting by tile*M + row keeps depth order inside every tile.
void build_tile_lists(const at::Tensor &bounds, int64_t rows, int64_t cols,
                      at::Tensor &ids, at::Tensor &ranges) {
  const int64_t m = bounds.size(0);
  const int64_t tiles = rows * cols;
  const auto long_options = bounds.options().dtype(at::kLong);
  const auto x0 = bounds.select(1, 0), x1 = bounds.select(1, 1);
  const auto y0 = bounds.select(1, 2), y1 = bounds.select(1, 3);
  const auto widths = x1 - x0 + 1;
  const auto counts = widths * (y1 - y0 + 1);
  const int64_t total = m > 0 ? counts.sum().item<int64_t>() : 0;
  if (total == 0) {
    ids = at::zeros({1}, bounds.options().dtype(at::kInt));
    ranges = at::zeros({tiles, 2}, bounds.options().dtype(at::kInt));
    return;
  }
  const auto gid = at::repeat_interleave(at::arange(m, long_options), counts);
  const auto offsets = at::cumsum(counts, 0) - counts;
  const auto local = at::arange(total, long_options) - offsets.index_select(0, gid);
  const auto w = widths.index_select(0, gid);
  const auto tx = x0.index_select(0, gid) + at::remainder(local, w);
  const auto ty = y0.index_select(0, gid) + at::floor_divide(local, w);
  const auto keys = std::get<0>(((ty * cols + tx) * m + gid).sort());
  const auto tile_id = at::floor_divide(keys, m);
  const auto sorted_gid = at::remainder(keys, m);
  const auto per_tile = at::bincount(tile_id, {}, tiles);
  const auto ends = at::cumsum(per_tile, 0);
  ids = sorted_gid.to(at::kInt).contiguous();
  ranges = at::stack({ends - per_tile, ends}, 1).to(at::kInt).contiguous();
}

// ---------------------------------------------------------------------------
// Fused CPU rasterizer: the CUDA kernels' algorithm with one task per tile
// (at::parallel_for) and a per-pixel loop over the tile's depth-sorted list.
// Per-pixel early stop at T < 1e-4 as on CUDA. The backward writes each
// (tile, Gaussian) pair's gradient into its own row of a [pairs, 10] buffer
// (no races, no atomics) and reduces with one index_add_, so it is
// deterministic.

struct CpuTile {
  int64_t y0, y1, x0, x1, begin, end;
};

CpuTile cpu_tile(const int32_t *ranges, int64_t tile_index, int64_t cols,
                 int64_t tile, int64_t height, int64_t width) {
  const int64_t row = tile_index / cols, col = tile_index % cols;
  CpuTile t;
  t.y0 = row * tile;
  t.y1 = std::min(t.y0 + tile, height);
  t.x0 = col * tile;
  t.x1 = std::min(t.x0 + tile, width);
  t.begin = ranges[2 * tile_index];
  t.end = ranges[2 * tile_index + 1];
  return t;
}

// Alpha of a pixel/Gaussian pair, or -1 if it does not contribute.
inline float cpu_alpha(const float *a, float px, float py, float *dx,
                       float *dy, float *g) {
  *dx = px - a[0];
  *dy = py - a[1];
  const float power =
      -0.5f * (a[2] * *dx * *dx + a[4] * *dy * *dy) - a[3] * *dx * *dy;
  if (power > 0.0f)
    return -1.0f;
  *g = std::exp(power);
  const float alpha = std::min(kMaxAlpha, a[5] * *g);
  return alpha < kMinAlpha ? -1.0f : alpha;
}

// Copies the tile's Gaussians into a contiguous [count, 10] buffer.
void gather_tile(const float *attrs, const int32_t *ids, const CpuTile &t,
                 std::vector<float> &local) {
  const int64_t count = t.end - t.begin;
  local.resize(static_cast<size_t>(count * kAttributeCount));
  for (int64_t k = 0; k < count; ++k)
    std::copy_n(attrs + static_cast<int64_t>(ids[t.begin + k]) * kAttributeCount,
                kAttributeCount, local.data() + k * kAttributeCount);
}

void rasterize_forward_cpu(const at::Tensor &attrs, const at::Tensor &ids,
                           const at::Tensor &ranges,
                           const at::Tensor &background, int64_t height,
                           int64_t width, int64_t tile, at::Tensor &out,
                           at::Tensor &final_t, at::Tensor &last_index) {
  const int64_t rows = (height + tile - 1) / tile;
  const int64_t cols = (width + tile - 1) / tile;
  const float *attr = attrs.data_ptr<float>();
  const int32_t *id = ids.data_ptr<int32_t>();
  const int32_t *range = ranges.data_ptr<int32_t>();
  const float *bg = background.data_ptr<float>();
  float *o = out.data_ptr<float>();
  float *ft = final_t.data_ptr<float>();
  int32_t *li = last_index.data_ptr<int32_t>();
  at::parallel_for(0, rows * cols, 1, [&](int64_t first, int64_t last_tile) {
    std::vector<float> local;
    for (int64_t ti = first; ti < last_tile; ++ti) {
      const auto t = cpu_tile(range, ti, cols, tile, height, width);
      gather_tile(attr, id, t, local);
      const int64_t count = t.end - t.begin;
      for (int64_t y = t.y0; y < t.y1; ++y)
        for (int64_t x = t.x0; x < t.x1; ++x) {
          const float px = static_cast<float>(x) + 0.5f;
          const float py = static_cast<float>(y) + 0.5f;
          float T = 1.0f, c[4] = {0, 0, 0, 0};
          int32_t last = 0;
          for (int64_t k = 0; k < count; ++k) {
            const float *a = local.data() + k * kAttributeCount;
            float dx, dy, g;
            const float alpha = cpu_alpha(a, px, py, &dx, &dy, &g);
            if (alpha < 0.0f)
              continue;
            const float next = T * (1.0f - alpha);
            if (next < kStopTransmittance)
              break;
            const float w = alpha * T;
            c[0] += a[7] * w;
            c[1] += a[8] * w;
            c[2] += a[9] * w;
            c[3] += a[6] * w;
            T = next;
            last = static_cast<int32_t>(k + 1);
          }
          const int64_t pixel = y * width + x;
          float *dst = o + pixel * 5;
          dst[0] = c[0] + T * bg[0];
          dst[1] = c[1] + T * bg[1];
          dst[2] = c[2] + T * bg[2];
          dst[3] = c[3];
          dst[4] = 1.0f - T;
          ft[pixel] = T;
          li[pixel] = last;
        }
    }
  });
}

// grad_pairs [pairs, 10] must be zero-initialized.
void rasterize_backward_cpu(const at::Tensor &attrs, const at::Tensor &ids,
                            const at::Tensor &ranges,
                            const at::Tensor &background,
                            const at::Tensor &grad_out,
                            const at::Tensor &final_t,
                            const at::Tensor &last_index, int64_t height,
                            int64_t width, int64_t tile,
                            at::Tensor &grad_pairs) {
  const int64_t rows = (height + tile - 1) / tile;
  const int64_t cols = (width + tile - 1) / tile;
  const float *attr = attrs.data_ptr<float>();
  const int32_t *id = ids.data_ptr<int32_t>();
  const int32_t *range = ranges.data_ptr<int32_t>();
  const float *bg = background.data_ptr<float>();
  const float *gout = grad_out.data_ptr<float>();
  const float *ft = final_t.data_ptr<float>();
  const int32_t *li = last_index.data_ptr<int32_t>();
  float *gp = grad_pairs.data_ptr<float>();
  at::parallel_for(0, rows * cols, 1, [&](int64_t first, int64_t last_tile) {
    std::vector<float> local;
    for (int64_t ti = first; ti < last_tile; ++ti) {
      const auto t = cpu_tile(range, ti, cols, tile, height, width);
      if (t.end == t.begin)
        continue;
      gather_tile(attr, id, t, local);
      float *tile_grad = gp + t.begin * kAttributeCount;
      for (int64_t y = t.y0; y < t.y1; ++y)
        for (int64_t x = t.x0; x < t.x1; ++x) {
          const int64_t pixel = y * width + x;
          const float px = static_cast<float>(x) + 0.5f;
          const float py = static_cast<float>(y) + 0.5f;
          const float *go = gout + pixel * 5;
          const float T_final = ft[pixel];
          float T = T_final;
          const float bg_dot = bg[0] * go[0] + bg[1] * go[1] + bg[2] * go[2];
          float acc[4] = {0, 0, 0, 0}, last_color[4] = {0, 0, 0, 0};
          float last_alpha = 0.0f;
          for (int64_t k = li[pixel] - 1; k >= 0; --k) {
            const float *a = local.data() + k * kAttributeCount;
            float dx, dy, g;
            const float alpha = cpu_alpha(a, px, py, &dx, &dy, &g);
            if (alpha < 0.0f)
              continue;
            T = T / (1.0f - alpha);
            const float channel[4] = {a[7], a[8], a[9], a[6]};
            const float weight = alpha * T;
            float dalpha = 0.0f;
            for (int c = 0; c < 4; ++c) {
              acc[c] = last_alpha * last_color[c] + (1.0f - last_alpha) * acc[c];
              last_color[c] = channel[c];
              dalpha += (channel[c] - acc[c]) * go[c];
            }
            dalpha *= T;
            const float inv = T_final / (1.0f - alpha);
            dalpha += go[4] * inv - bg_dot * inv;
            last_alpha = alpha;

            float *v = tile_grad + k * kAttributeCount;
            v[6] += weight * go[3];
            v[7] += weight * go[0];
            v[8] += weight * go[1];
            v[9] += weight * go[2];
            // Only the unclamped branch passes gradient through
            // min(alpha, 0.99).
            const float unclamped = a[5] * g;
            if (unclamped <= kMaxAlpha) {
              const float dpower = unclamped * dalpha;
              v[5] += g * dalpha;
              v[0] += dpower * (a[2] * dx + a[3] * dy);
              v[1] += dpower * (a[4] * dy + a[3] * dx);
              v[2] += -0.5f * dx * dx * dpower;
              v[3] += -dx * dy * dpower;
              v[4] += -0.5f * dy * dy * dpower;
            }
          }
        }
    }
  });
}

struct KernelRasterFunction
    : public torch::autograd::Function<KernelRasterFunction> {
  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            const at::Tensor &attrs,
                            const at::Tensor &background,
                            const at::Tensor &bounds, const at::Tensor &meta) {
    const auto m = meta.accessor<int64_t, 1>();
    const int64_t height = m[0], width = m[1], tile = m[2];
    const int64_t rows = (height + tile - 1) / tile;
    const int64_t cols = (width + tile - 1) / tile;
    at::Tensor ids, ranges;
    build_tile_lists(bounds, rows, cols, ids, ranges);
    const auto packed = attrs.contiguous();
    const auto bg = background.contiguous();
    auto out = at::empty({height, width, 5}, attrs.options());
    auto final_t = at::empty({height, width}, attrs.options());
    auto last_index =
        at::empty({height, width}, attrs.options().dtype(at::kInt));
#ifdef USE_CUDA
    if (attrs.is_cuda())
      detail::rasterize_forward_cuda(packed, ids, ranges, bg, height, width,
                                     out, final_t, last_index);
    else
#endif
      rasterize_forward_cpu(packed, ids, ranges, bg, height, width, tile, out,
                            final_t, last_index);
    ctx->save_for_backward(
        {packed, bg, ids, ranges, final_t, last_index, meta});
    return out;
  }

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs) {
    const auto saved = ctx->get_saved_variables();
    const auto m = saved[6].accessor<int64_t, 1>();
    const auto grad_out = grad_outputs[0].contiguous();
    auto grad_attrs = at::zeros_like(saved[0]);
#ifdef USE_CUDA
    if (saved[0].is_cuda())
      detail::rasterize_backward_cuda(saved[0], saved[2], saved[3], saved[1],
                                      grad_out, saved[4], saved[5], m[0], m[1],
                                      grad_attrs);
    else
#endif
    {
      const auto &ids = saved[2];
      const int64_t pairs = saved[3].numel() > 0
                                ? saved[3].select(1, 1).max().item<int64_t>()
                                : 0;
      if (pairs > 0) {
        auto grad_pairs = at::zeros({pairs, kAttributeCount}, saved[0].options());
        rasterize_backward_cpu(saved[0], ids, saved[3], saved[1], grad_out,
                               saved[4], saved[5], m[0], m[1], m[2],
                               grad_pairs);
        grad_attrs.index_add_(0, ids.slice(0, 0, pairs).to(at::kLong),
                              grad_pairs);
      }
    }
    const auto grad_background =
        (saved[4].unsqueeze(-1) * grad_out.slice(2, 0, 3))
            .sum(at::IntArrayRef{0, 1});
    return {grad_attrs, grad_background, at::Tensor(), at::Tensor()};
  }
};

} // namespace

RenderOutput render_gaussians(const at::Tensor &positions,
                              const at::Tensor &rotations,
                              const at::Tensor &scales,
                              const at::Tensor &opacities,
                              const at::Tensor &colors, const Camera &camera,
                              const RenderOptions &options) {
  const auto n = positions.size(0);
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
              "positions must have shape [N, 3]");
  TORCH_CHECK(rotations.sizes() == at::IntArrayRef({n, 4}),
              "rotations must have shape [N, 4]");
  TORCH_CHECK(scales.sizes() == at::IntArrayRef({n, 3}),
              "scales must have shape [N, 3]");
  TORCH_CHECK(opacities.numel() == n, "opacities must have N elements");
  TORCH_CHECK(colors.sizes() == at::IntArrayRef({n, 3}),
              "colors must have shape [N, 3]");
  for (const auto *t : {&positions, &rotations, &scales, &opacities, &colors}) {
    TORCH_CHECK(t->scalar_type() == at::kFloat,
                "render requires float32 Gaussian tensors");
    TORCH_CHECK(t->device() == positions.device(),
                "render inputs must share a device");
  }
  TORCH_CHECK(camera.width > 0 && camera.height > 0 &&
                  camera.world_to_camera.defined(),
              "camera is not initialized");
  TORCH_CHECK(options.tile_size > 0 && options.chunk_size > 0,
              "tile_size and chunk_size must be positive");

  const auto device = positions.device();
#ifdef USE_CUDA
  const bool use_cuda_raster = options.use_cuda_kernels && device.is_cuda();
#else
  const bool use_cuda_raster = false;
#endif
  const bool use_cpu_raster = options.use_cpu_kernels && device.is_cpu();
#ifdef USE_CUDA
  const int64_t tile_size =
      use_cuda_raster ? detail::kRasterTileSize : options.tile_size;
#else
  const int64_t tile_size = options.tile_size;
#endif
  const auto float_options = at::TensorOptions().dtype(at::kFloat).device(device);
  const int64_t width = camera.width, height = camera.height;
  const auto world_to_camera = camera.world_to_camera.to(float_options);
  const auto rotation = world_to_camera.slice(0, 0, 3).slice(1, 0, 3);
  const auto translation = world_to_camera.slice(0, 0, 3).select(1, 3);

  const auto camera_points =
      at::matmul(positions, rotation.transpose(0, 1)) + translation;
  const auto z = camera_points.select(1, 2);
  const auto safe_z = z.clamp_min(static_cast<double>(camera.near_plane));
  const auto tx = camera_points.select(1, 0) / safe_z;
  const auto ty = camera_points.select(1, 1) / safe_z;
  const auto x2 = camera.fx * tx + camera.cx;
  const auto y2 = camera.fy * ty + camera.cy;
  auto means2d = at::stack({x2, y2}, 1);
  if (means2d.requires_grad())
    means2d.retain_grad();

  // Clamp the viewing frustum used by the Jacobian to keep it stable near the
  // image border: 15% of the image size beyond each edge. The limits are
  // measured from the principal point, so they are asymmetric when cx != W/2
  // (cropped or off-center intrinsics); for a centered principal point they
  // equal the symmetric 1.3 x half-FOV clamp of 3DGS.
  const double w = static_cast<double>(width), h = static_cast<double>(height);
  const auto txc = tx.clamp(-(camera.cx + 0.15 * w) / camera.fx,
                            (w - camera.cx + 0.15 * w) / camera.fx);
  const auto tyc = ty.clamp(-(camera.cy + 0.15 * h) / camera.fy,
                            (h - camera.cy + 0.15 * h) / camera.fy);
  const auto zeros = at::zeros_like(safe_z);
  const auto jacobian = at::stack(
      {at::stack({camera.fx / safe_z, zeros, -camera.fx * txc / safe_z}, 1),
       at::stack({zeros, camera.fy / safe_z, -camera.fy * tyc / safe_z}, 1)},
      1); // [N, 2, 3]
  const auto basis =
      quaternion_matrices(rotations) * scales.unsqueeze(1); // R * diag(s)
  const auto projected = at::matmul(at::matmul(jacobian, rotation), basis);
  // cov2d = M M^T for the 2x3 matrix M with rows m1, m2. Its determinant is
  // computed with the Lagrange identity |m1|^2 |m2|^2 - (m1.m2)^2 =
  // |m1 x m2|^2: a sum of squares, instead of a*c - b*b, which cancels
  // catastrophically in float32 for long thin (needle) splats.
  const auto m1 = projected.select(1, 0), m2 = projected.select(1, 1);
  const auto cov_a = (m1 * m1).sum(1);
  const auto b = (m1 * m2).sum(1);
  const auto cov_c = (m2 * m2).sum(1);
  const auto a = cov_a + kLowPass;
  const auto c = cov_c + kLowPass;
  const auto det = at::linalg_cross(m1, m2, 1).square().sum(1) +
                   kLowPass * (cov_a + cov_c) + kLowPass * kLowPass;
  const auto inverse_det = 1.0 / det.clamp_min(1e-12);
  const auto conic_a = c * inverse_det;
  const auto conic_b = -b * inverse_det;
  const auto conic_c = a * inverse_det;

  at::Tensor radii_long, visible, order_ids, bounds;
  {
    torch::NoGradGuard no_grad;
    // Largest eigenvalue; mid^2 - det is evaluated as ((a - c) / 2)^2 + b^2,
    // which cannot cancel.
    const auto mid = 0.5 * (a + c);
    const auto half_diff = 0.5 * (a - c);
    const auto lambda =
        mid + at::sqrt((half_diff * half_diff + b * b).clamp_min(0.1));
    const auto radius = at::ceil(3.0 * at::sqrt(lambda));
    const auto mx = x2.detach(), my = y2.detach();
    visible = (z > camera.near_plane) & (det > 1e-12) &
              (opacities.reshape({-1}) >= kMinAlpha) &
              at::isfinite(mx) & at::isfinite(my) & at::isfinite(radius) &
              (mx + radius >= 0) & (mx - radius < static_cast<double>(width)) &
              (my + radius >= 0) & (my - radius < static_cast<double>(height));
    radii_long = at::where(visible, radius, at::zeros_like(radius)).to(at::kLong);
    const auto ids = visible.nonzero().squeeze(1);
    // Stable: Gaussians at exactly equal depth keep their index order, so the
    // image does not depend on the sort implementation or device.
    const auto order =
        z.index_select(0, ids).argsort(/*stable=*/true, 0, false);
    order_ids = ids.index_select(0, order);

    const auto sx = mx.index_select(0, order_ids);
    const auto sy = my.index_select(0, order_ids);
    const auto sr = radius.index_select(0, order_ids);
    const double tile = static_cast<double>(tile_size);
    const auto max_col = (width + tile_size - 1) / tile_size - 1;
    const auto max_row = (height + tile_size - 1) / tile_size - 1;
    bounds = at::stack(
        {at::floor((sx - sr) / tile).clamp(0, max_col).to(at::kLong),
         at::floor((sx + sr) / tile).clamp(0, max_col).to(at::kLong),
         at::floor((sy - sr) / tile).clamp(0, max_row).to(at::kLong),
         at::floor((sy + sr) / tile).clamp(0, max_row).to(at::kLong)},
        1);
  }

  auto gather = [&](const at::Tensor &t) { return t.index_select(0, order_ids); };
  const auto attrs = at::cat(
      {at::stack({gather(means2d.select(1, 0)), gather(means2d.select(1, 1)),
                  gather(conic_a), gather(conic_b), gather(conic_c),
                  gather(opacities.reshape({-1})), gather(z)},
                 1),
       gather(colors)},
      1);

  at::Tensor background = options.background.defined()
                              ? options.background.to(float_options)
                              : at::zeros({3}, float_options);
  TORCH_CHECK(background.numel() == 3, "background must have 3 elements");
  background = background.reshape({3});
  const auto meta = at::tensor(
      std::vector<int64_t>{height, width, tile_size, options.chunk_size},
      at::kLong);

  const auto out =
      use_cuda_raster || use_cpu_raster
          ? KernelRasterFunction::apply(attrs, background, bounds, meta)
          : RasterFunction::apply(attrs, background, bounds, meta);
  RenderOutput result;
  result.image = out.slice(2, 0, 3);
  result.depth = out.select(2, 3);
  result.alpha = out.select(2, 4);
  result.means2d = means2d;
  result.visible = visible;
  result.radii = radii_long;
  return result;
}

RenderOutput render(const GaussianModel &model, const Camera &camera,
                    const RenderOptions &options) {
  const int degree =
      options.sh_degree < 0 ? model.sh_degree()
                            : std::min(options.sh_degree, model.sh_degree());
  const auto position = camera.position().to(model.device());
  auto directions = model.xyz() - position;
  directions = directions / directions.norm(2, {1}, true).clamp_min(1e-12);
  const auto colors =
      eval_sh_color(degree, model.sh_coefficients(), directions);
  return render_gaussians(model.xyz(), model.rotation_raw(), model.scales(),
                          model.opacities(), colors, camera, options);
}

RenderOutput render(const GaussianTensor &gaussians, const Camera &camera,
                    const RenderOptions &options) {
  return render_gaussians(gaussians.positions(), gaussians.rotations(),
                          gaussians.scales(), gaussians.opacities(),
                          gaussians.colors(), camera, options);
}

} // namespace gaussian
} // namespace minkowski
