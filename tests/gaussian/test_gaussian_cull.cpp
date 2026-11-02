// Fused CUDA culling and tile lists (gaussian_cull_cuda.cu) against the LibTorch code they replace.
// Both read the same float32 projection, so every decision must match exactly: visibility, radii,
// tile bounds, tile pair counts, and the ids / ranges handed to the rasterizer.
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_cuda.hpp"
#include "gaussian/gaussian_render.hpp"
#include "utils/test_framework.hpp"

#ifdef USE_CUDA // the kernels and their hook exist only in CUDA builds

using namespace minkowski::gaussian;
namespace gd = minkowski::gaussian::detail;

namespace {

void require_cuda() {
  if (!torch::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
}

bool same(const at::Tensor &a, const at::Tensor &b) {
  return a.sizes() == b.sizes() && a.cpu().equal(b.cpu());
}

// The LibTorch culling of render_gaussians on a [N,10] projection (columns x2, y2, conic, z, a, b,
// c, det), returning visible, radius (0 when culled) and per-Gaussian bounds.
void reference_cull(const at::Tensor &proj, const at::Tensor &opacity, const gd::CullParams &k,
                    at::Tensor &visible, at::Tensor &radius, at::Tensor &bounds) {
  const auto mx = proj.select(1, 0), my = proj.select(1, 1), z = proj.select(1, 5);
  const auto a = proj.select(1, 6), c = proj.select(1, 8), det = proj.select(1, 9);
  const auto extent = at::sqrt(
      (2.0 * at::log(opacity.clamp_min(1.0f / 255.0f) * 255.0)).clamp_min(0.0) + 1e-2);
  const auto rx = at::ceil(extent * at::sqrt(a)), ry = at::ceil(extent * at::sqrt(c));
  visible = (z > k.near_plane) & (det > 1e-12) & (opacity >= 1.0f / 255.0f) &
            at::isfinite(mx) & at::isfinite(my) & at::isfinite(rx) & at::isfinite(ry) &
            (mx + rx >= 0) & (mx - rx < static_cast<double>(k.width)) & (my + ry >= 0) &
            (my - ry < static_cast<double>(k.height));
  const auto r = at::maximum(rx, ry);
  radius = at::where(visible, r, at::zeros_like(r)).to(at::kLong);
  const double tile = static_cast<double>(k.tile);
  bounds = at::stack({at::floor((mx - rx) / tile).clamp(0, k.max_col).to(at::kLong),
                      at::floor((mx + rx) / tile).clamp(0, k.max_col).to(at::kLong),
                      at::floor((my - ry) / tile).clamp(0, k.max_row).to(at::kLong),
                      at::floor((my + ry) / tile).clamp(0, k.max_row).to(at::kLong)},
                     1);
}

// The LibTorch tile-list construction (repeat_interleave, sort, bincount).
void reference_lists(const at::Tensor &bounds, int64_t rows, int64_t cols, at::Tensor &ids,
                     at::Tensor &ranges) {
  const int64_t m = bounds.size(0), tiles = rows * cols;
  const auto lo = bounds.options().dtype(at::kLong);
  const auto x0 = bounds.select(1, 0), x1 = bounds.select(1, 1);
  const auto y0 = bounds.select(1, 2), y1 = bounds.select(1, 3);
  const auto widths = x1 - x0 + 1;
  const auto counts = widths * (y1 - y0 + 1);
  const int64_t total = counts.sum().item<int64_t>();
  const auto gid = at::repeat_interleave(at::arange(m, lo), counts);
  const auto offsets = at::cumsum(counts, 0) - counts;
  const auto local = at::arange(total, lo) - offsets.index_select(0, gid);
  const auto w = widths.index_select(0, gid);
  const auto tx = x0.index_select(0, gid) + at::remainder(local, w);
  const auto ty = y0.index_select(0, gid) + at::floor_divide(local, w);
  const auto keys = std::get<0>(((ty * cols + tx) * m + gid).sort());
  const auto per_tile = at::bincount(at::floor_divide(keys, m), {}, tiles);
  const auto ends = at::cumsum(per_tile, 0);
  ids = at::remainder(keys, m).to(at::kInt);
  ranges = at::stack({ends - per_tile, ends}, 1).to(at::kInt);
}

void fused_lists(const at::Tensor &bounds, int64_t rows, int64_t cols, at::Tensor &ids,
                 at::Tensor &ranges) {
  const int64_t m = bounds.size(0), tiles = rows * cols;
  auto counts = at::empty({m}, bounds.options());
  gd::tile_counts_cuda(bounds, counts);
  const int64_t total = counts.sum().item<int64_t>();
  auto keys = at::empty({total}, bounds.options());
  gd::tile_keys_cuda(bounds, at::cumsum(counts, 0) - counts, cols, keys);
  keys = std::get<0>(keys.sort());
  ids = at::empty({total}, bounds.options().dtype(at::kInt));
  ranges = at::empty({tiles, 2}, bounds.options().dtype(at::kInt));
  gd::finish_tile_lists_cuda(keys, m, tiles, ids, ranges);
}

struct Scene {
  at::Tensor positions, rotations, scales, opacities, colors;
};

Scene make_scene(int64_t n, uint64_t seed, const at::Device &dev) {
  torch::manual_seed(seed);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32);
  Scene s;
  s.positions = torch::randn({n, 3}, f) * 1.2;
  s.positions.slice(0, 0, n / 20).select(1, 2).fill_(-3.5);    // behind the camera
  s.positions.slice(0, n / 20, n / 10).select(1, 0).mul_(8.0); // far off to the sides
  s.rotations = torch::randn({n, 4}, f);
  s.scales = torch::rand({n, 3}, f) * 0.15 + 0.005;
  s.scales.slice(0, n / 5, n / 4).select(1, 1).fill_(1e-4);    // needles
  s.opacities = torch::rand({n, 1}, f) * 0.9 + 0.05;
  s.opacities.slice(0, n / 4, n / 4 + 20).fill_(1.0f / 255.0f); // exactly at the alpha cut
  s.opacities.slice(0, n / 4 + 20, n / 4 + 40).fill_(0.003f);   // below it
  s.colors = torch::rand({n, 3}, f);
  for (auto *t : {&s.positions, &s.rotations, &s.scales, &s.opacities, &s.colors})
    *t = t->to(dev).requires_grad_(true);
  return s;
}

struct Result {
  RenderOutput out;
  std::vector<at::Tensor> grads;
};

Result run(const Scene &base, const Camera &cam, bool fused) {
  gd::cuda_fused_tiles() = fused;
  Scene s;
  const std::initializer_list<std::pair<at::Tensor *, const at::Tensor *>> fields{
      {&s.positions, &base.positions}, {&s.rotations, &base.rotations},
      {&s.scales, &base.scales}, {&s.opacities, &base.opacities}, {&s.colors, &base.colors}};
  for (auto [dst, src] : fields)
    *dst = src->detach().clone().requires_grad_(true);
  Result r;
  r.out = render_gaussians(s.positions, s.rotations, s.scales, s.opacities, s.colors, cam);
  torch::manual_seed(7);
  const auto p_img = torch::randn_like(r.out.image);
  ((r.out.image * p_img).sum() + r.out.depth.sum() + r.out.alpha.sum()).backward();
  gd::cuda_fused_tiles() = true;
  for (const auto *t : {&s.positions, &s.rotations, &s.scales, &s.opacities, &s.colors})
    r.grads.push_back(t->grad());
  return r;
}

double rel_diff(const at::Tensor &a, const at::Tensor &b) {
  const auto a64 = a.cpu().to(torch::kFloat64), b64 = b.cpu().to(torch::kFloat64);
  const double scale = b64.abs().max().item<double>();
  const double diff = (a64 - b64).abs().max().item<double>();
  return scale > 0 ? diff / scale : diff;
}

} // namespace

TEST(gaussian_fused_cull_matches_libtorch_exactly) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  const int64_t n = 20000;
  torch::manual_seed(5);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32);
  auto proj = torch::empty({n, 10}, f);
  proj.select(1, 0).uniform_(-60, 300); // x2 around a 250 px wide image
  proj.select(1, 1).uniform_(-60, 200); // y2 around a 150 px high image
  proj.slice(1, 2, 5).uniform_(-1, 1);
  proj.select(1, 5).uniform_(-1, 5);    // z, some behind the near plane
  proj.select(1, 6).uniform_(0.3f, 400);
  proj.select(1, 7).uniform_(-5, 5);
  proj.select(1, 8).uniform_(0.3f, 400);
  proj.select(1, 9).uniform_(0.05f, 1e4);
  auto opacity = torch::rand({n}, f);
  // Edge cases: non-finite means / covariance / opacity, tiny determinants, the alpha cut, means
  // exactly on tile boundaries and on the image border.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  proj[0][0] = nan;
  proj[1][1] = inf;
  proj[2][6] = nan;
  proj[3][8] = inf;
  opacity[4] = nan;
  proj[5][9] = 1e-12f;
  proj[6][9] = 0.0f;
  opacity[7] = 1.0f / 255.0f;
  opacity[8] = std::nextafter(1.0f / 255.0f, 0.0f);
  proj[9][0] = 32.0f;
  proj[9][6] = 0.3f;
  proj[10][0] = 250.0f;
  proj[11][1] = -1.0f;
  proj[12][5] = 0.01f; // exactly the near plane
  opacity.slice(0, 13, 200).fill_(1.0f);
  proj = proj.to(dev).contiguous();
  opacity = opacity.to(dev).contiguous();

  const gd::CullParams k{0.01f, 1.0f / 255.0f, 1e-2f, 250.0f, 150.0f, 16, (250 + 15) / 16 - 1,
                         (150 + 15) / 16 - 1};
  auto visible = at::empty({n}, proj.options().dtype(at::kBool));
  auto radius = at::empty({n}, proj.options().dtype(at::kLong));
  auto bounds = at::empty({n, 4}, proj.options().dtype(at::kLong));
  auto counts = at::empty({n}, proj.options().dtype(at::kLong));
  gd::cull_cuda(proj, opacity, k, visible, radius, bounds, counts);

  at::Tensor ref_visible, ref_radius, ref_bounds;
  reference_cull(proj, opacity, k, ref_visible, ref_radius, ref_bounds);
  const int64_t kept = ref_visible.sum().item<int64_t>();
  std::cout << "         visible " << kept << " of " << n << "\n";
  CHECK(kept > n / 10 && kept < n);
  CHECK(same(visible, ref_visible));
  CHECK(same(radius, ref_radius));
  const auto ids = ref_visible.nonzero().squeeze(1);
  CHECK(same(bounds.index_select(0, ids), ref_bounds.index_select(0, ids)));
  const auto vb = ref_bounds.index_select(0, ids);
  const auto ref_counts = (vb.select(1, 1) - vb.select(1, 0) + 1) * (vb.select(1, 3) - vb.select(1, 2) + 1);
  CHECK(same(counts.index_select(0, ids), ref_counts));
  CHECK_EQ(counts.sum().item<int64_t>(), ref_counts.sum().item<int64_t>());
  for (int i : {0, 1, 2, 3, 4, 5, 6, 8, 12})
    CHECK(!visible[i].item<bool>());
}

TEST(gaussian_fused_tile_lists_match_libtorch_exactly) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  const auto lo = torch::TensorOptions().dtype(torch::kLong);
  using Case = std::tuple<int64_t, int64_t, int64_t, int64_t>; // m, rows, cols, max extent
  for (const auto &[m, rows, cols, max_extent] : std::vector<Case>{
           {1, 1, 1, 1},
           {7, 5, 9, 3},
        {3000, 12, 20, 4},
        {500, 64, 64, 2},   // most tiles empty
        {200, 8, 8, 8}}) {  // many Gaussians covering the whole image
    torch::manual_seed(m + rows);
    const auto x0 = torch::randint(0, cols, {m}, lo), y0 = torch::randint(0, rows, {m}, lo);
    const auto x1 = (x0 + torch::randint(0, max_extent, {m}, lo)).clamp_max(cols - 1);
    const auto y1 = (y0 + torch::randint(0, max_extent, {m}, lo)).clamp_max(rows - 1);
    auto bounds = torch::stack({x0, x1, y0, y1}, 1);
    if (m == 500) // leave the first and last tiles empty
      bounds = torch::stack({x0.clamp(1, cols - 2), x1.clamp(1, cols - 2), y0.clamp(1, rows - 2),
                             y1.clamp(1, rows - 2)},
                            1);
    bounds = bounds.to(dev).contiguous();
    at::Tensor ids, ranges, ref_ids, ref_ranges;
    fused_lists(bounds, rows, cols, ids, ranges);
    reference_lists(bounds, rows, cols, ref_ids, ref_ranges);
    CHECK(same(ids, ref_ids));
    CHECK(same(ranges, ref_ranges));
  }
}

TEST(gaussian_fused_cull_tiles_match_libtorch_render) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  using Case = std::tuple<int64_t, int64_t, double>; // width, height, principal point shift
  for (const auto &[w, h, shift] : std::vector<Case>{{96, 64, 0.0}, {131, 77, 9.0}}) {
    const auto scene = make_scene(4000, 13, dev);
    auto cam = Camera::look_at({0.3, -0.2, -3.0}, {0, 0, 0}, {0, -1, 0}, w, h, 1.0);
    cam.cx += shift;
    const auto fused = run(scene, cam, true);
    const auto aten = run(scene, cam, false);
    CHECK(same(fused.out.visible, aten.out.visible));
    CHECK(same(fused.out.radii, aten.out.radii));
    CHECK_EQ(fused.out.tile_pairs.item<int64_t>(), aten.out.tile_pairs.item<int64_t>());
    // Identical tile lists -> the forward rasterizer computes bit-identical images.
    CHECK(same(fused.out.image, aten.out.image));
    CHECK(same(fused.out.depth, aten.out.depth));
    CHECK(same(fused.out.alpha, aten.out.alpha));
    // The backward accumulates with atomics, so allow float reordering only.
    const char *names[] = {"positions", "rotations", "scales", "opacities", "colors"};
    for (size_t i = 0; i < fused.grads.size(); ++i) {
      const double d = rel_diff(fused.grads[i], aten.grads[i]);
      std::cout << "         grad " << names[i] << ": relative diff " << d << "\n";
      CHECK(d <= 1e-5);
    }
  }
}

TEST(gaussian_fused_cull_tiles_empty_and_all_culled) {
  require_cuda();
  const at::Device dev(at::kCUDA);
  const auto cam = Camera::look_at({0, 0, -3.0}, {0, 0, 0}, {0, -1, 0}, 40, 24, 1.0);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
  for (const int64_t n : {0, 5}) {
    auto positions = torch::zeros({n, 3}, f);
    positions.select(1, 2).fill_(-10.0); // all behind the camera
    const auto out = render_gaussians(positions, torch::randn({n, 4}, f), torch::full({n, 3}, 0.1, f),
                                      torch::full({n, 1}, 0.5, f), torch::rand({n, 3}, f), cam);
    CHECK_EQ(out.tile_pairs.item<int64_t>(), int64_t(0));
    CHECK_NEAR(out.alpha.abs().sum().item<double>(), 0.0, 0.0);
  }
}

#endif // USE_CUDA
