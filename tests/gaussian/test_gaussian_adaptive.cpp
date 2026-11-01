// Tests for the Gaussian <-> adaptive sparse voxel bridge (gaussian_adaptive.hpp).
#include "gaussian/gaussian_adaptive.hpp"
#include "adaptive/ops.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <torch/torch.h>

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

using minkowski::adaptive::AdaptiveSparseTensor;
using minkowski::adaptive::AdaptiveVoxelConfig;
using namespace minkowski::gaussian;

namespace {

const auto kD = torch::TensorOptions().dtype(torch::kFloat64);

AdaptiveVoxelConfig make_config(double base, uint32_t depth,
                                std::vector<double> origin = {}) {
  AdaptiveVoxelConfig cfg;
  cfg.base_voxel_size = {base, base, base};
  cfg.max_depth = depth;
  cfg.origin = std::move(origin);
  return cfg;
}

at::Tensor identity_rotations(int64_t n, const torch::TensorOptions &o = kD) {
  auto r = torch::zeros({n, 4}, o);
  r.select(1, 0).fill_(1.0);
  return r;
}

GaussianTensor make_gaussians(const at::Tensor &positions, const at::Tensor &scales,
                              at::Tensor rotations = {}, at::Tensor opacities = {},
                              at::Tensor colors = {}, at::Tensor features = {}) {
  const int64_t n = positions.size(0);
  if (!rotations.defined())
    rotations = identity_rotations(n, positions.options());
  if (!opacities.defined())
    opacities = torch::full({n, 1}, 0.5, positions.options());
  if (!colors.defined())
    colors = torch::rand({n, 3}, positions.options());
  return GaussianTensor(positions, rotations, scales, opacities, colors, features);
}

// Random scene: positions in [0, extent)^3, log-uniform scales, random rotations.
GaussianTensor random_scene(int64_t n, uint64_t seed, double extent = 32.0,
                            double smin = 0.05, double smax = 4.0, int64_t fc = 2) {
  torch::manual_seed(seed);
  auto pos = torch::rand({n, 3}, kD) * extent;
  auto logs = torch::rand({n, 3}, kD) * std::log(smax / smin) + std::log(smin);
  auto rot = torch::randn({n, 4}, kD);
  auto op = torch::rand({n, 1}, kD) * 0.9 + 0.05;
  auto col = torch::rand({n, 3}, kD);
  auto feat = torch::randn({n, fc}, kD);
  return make_gaussians(pos, logs.exp(), rot, op, col, feat);
}

double total(const at::Tensor &t) { return t.sum().item<double>(); }

GaussianToAdaptiveOptions box_options() {
  GaussianToAdaptiveOptions o;
  o.splat = GaussianSplat::BoxIntegral;
  return o;
}

} // namespace

TEST(gaussian_adaptive_mass_conservation) {
  auto g = random_scene(400, 7);
  auto cfg = make_config(16.0, 6);
  const double opacity_sum = total(g.opacities());
  for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
    GaussianToAdaptiveOptions o;
    o.splat = splat;
    GaussianAdaptiveMap map;
    auto t = gaussians_to_adaptive(g, cfg, o, &map);
    CHECK(t.size() > 0);
    CHECK_EQ(t.channels(), 4 + 2);
    CHECK(!t.has_overlapping_leaves());
    CHECK_NEAR(total(t.features().select(1, kAdaptiveMassChannel)), opacity_sum,
               1e-11 * opacity_sum);
    // Each Gaussian's weights sum to one.
    auto per_g = torch::zeros({g.size()}, kD).index_add(0, map.gaussian_index, map.weight);
    CHECK_ALLCLOSE(per_g, torch::ones_like(per_g), 1e-12, 1e-12);
    CHECK(torch::isfinite(t.features()).all().item<bool>());

    // Extensive attributes: sums over leaves reproduce sum_i m_i * color_i.
    o.mean_attributes = false;
    o.weight_by_volume = true;
    auto s = gaussians_to_adaptive(g, cfg, o);
    auto m = g.opacities().select(1, 0) * g.scales().prod(1) * 15.749609945722419;
    CHECK_NEAR(total(s.features().select(1, 0)), total(m), 1e-10 * total(m));
    auto expect = (g.colors() * m.unsqueeze(1)).sum(0);
    CHECK_ALLCLOSE(s.features().narrow(1, 1, 3).sum(0), expect, 1e-10, 1e-10);
    CHECK_EQ(s.size(), t.size()); // weights/topology do not depend on the mass mode
  }
}

TEST(gaussian_adaptive_matches_brute_force_reference) {
  // Independent dense reference: Gaussian i deposits into the window W_i of
  // level-l_i cells (its mean's cell for Center, the cells within support_sigma
  // axis std for BoxIntegral), so leaf j receives
  //   w_ij = prod_d P_d(leaf_j) / prod_d P_d(W_i)   if leaf_j lies inside W_i, else 0,
  // where P_d is the 1-D normal integral with std sqrt(Sigma_dd) (from
  // covariances(), not the bridge's quaternion code).
  auto g = random_scene(60, 41, /*extent=*/12.0, 0.08, 1.5, 1);
  auto cfg = make_config(8.0, 5, {-0.5, 0.25, 0.0});
  auto stdv = g.covariances().diagonal(0, 1, 2).sqrt(); // [N, 3]
  auto mean = g.positions();
  auto phi_seg = [](const at::Tensor &a, const at::Tensor &b) {
    return 0.5 * (torch::erf(b / std::sqrt(2.0)) - torch::erf(a / std::sqrt(2.0)));
  };
  auto origin = torch::tensor({-0.5, 0.25, 0.0}, kD);
  for (auto [splat, min_weight] : {std::pair{GaussianSplat::Center, 0.0},
                                   std::pair{GaussianSplat::BoxIntegral, 0.0},
                                   std::pair{GaussianSplat::Center, 1e-15},
                                   std::pair{GaussianSplat::BoxIntegral, 1e-15}}) {
    GaussianToAdaptiveOptions o;
    o.splat = splat;
    o.min_weight = min_weight; // 0: exact; default 1e-15 drops far-tail entries
    GaussianAdaptiveMap map;
    auto t = gaussians_to_adaptive(g, cfg, o, &map);
    auto [lo, hi] = t.bounds(); // [L, 3]
    const int64_t N = g.size(), L = t.size();
    auto W = torch::zeros({N, L}, kD);
    for (int64_t i = 0; i < N; ++i) {
      const int l = map.gaussian_level[i].item<int>();
      auto cell = torch::full({3}, 8.0 * std::ldexp(1.0, -l), kD);
      auto rel = (mean[i] - origin) / cell;
      at::Tensor wlo, whi;
      if (splat == GaussianSplat::Center) {
        wlo = rel.floor();
        whi = wlo + 1.0;
      } else {
        auto ext = 3.0 * stdv[i] / cell;
        wlo = (rel - ext).floor();
        whi = (rel + ext).floor() + 1.0;
      }
      CHECK((whi - wlo).le(32.0).all().item<bool>()); // no window clipping here
      wlo = origin + wlo * cell;
      whi = origin + whi * cell;
      auto inside = ((lo >= wlo - 1e-12) & (hi <= whi + 1e-12)).all(1); // [L]
      auto pl = phi_seg((lo - mean[i]) / stdv[i], (hi - mean[i]) / stdv[i]).prod(1);
      auto pw = phi_seg((wlo - mean[i]) / stdv[i], (whi - mean[i]) / stdv[i]).prod();
      W[i] = torch::where(inside, pl / pw, torch::zeros_like(pl));
    }
    // Dense weights from the bridge's sparse map.
    auto Wb = torch::zeros({N, L}, kD);
    Wb.index_put_({map.gaussian_index, map.leaf_index}, map.weight, /*accumulate=*/true);
    CHECK_ALLCLOSE(Wb, W, 1e-9, 1e-12);
    auto m = g.opacities().select(1, 0);
    auto mass = (W * m.unsqueeze(1)).sum(0);
    CHECK_ALLCLOSE(t.features().select(1, 0), mass, 1e-9, 1e-12);
    auto colsum = torch::mm(W.t(), g.colors() * m.unsqueeze(1));
    // Compare mass-weighted colours: leaves in far splat tails carry masses
    // ~1e-20, where the normalized mean amplifies last-bit differences.
    CHECK_ALLCLOSE(t.features().narrow(1, 1, 3) * t.features().narrow(1, 0, 1), colsum, 1e-9,
                   1e-12);
    // Every leaf holds weight, and leaves only exist where some window reaches.
    CHECK(W.sum(0).gt(0).all().item<bool>());
  }
}

TEST(gaussian_adaptive_refine_by_density) {
  auto g = random_scene(500, 23, /*extent=*/16.0, 0.05, 0.5, 0);
  // Cluster: 200 more Gaussians in one corner.
  torch::manual_seed(2);
  auto cpos = torch::rand({200, 3}, kD) * 2.0 + 1.0;
  auto cl = make_gaussians(cpos, torch::full({200, 3}, 0.05, kD), {}, {}, {},
                           torch::zeros({200, 0}, kD));
  auto all = g.append(cl);
  auto cfg = make_config(16.0, 6);
  GaussianToAdaptiveOptions o;
  o.max_level = 0; // start from coarse level-0 leaves
  auto coarse = gaussians_to_adaptive(all, cfg, o);
  CHECK(coarse.counts_by_level()[0] == coarse.size());
  const int64_t threshold = 8;
  auto r = refine_by_gaussian_density(coarse, all, threshold, 5);
  CHECK(!r.has_overlapping_leaves());
  CHECK(r.size() > coarse.size());
  CHECK_EQ(r.levels().max().item<int>(), 5);
  CHECK_NEAR(total(r.volumes()), total(coarse.volumes()), 1e-9); // same covered region
  // Brute force: leaves below max_level hold <= threshold means, and every
  // refined leaf's parent held more than threshold.
  auto [lo, hi] = r.bounds();
  auto pos = all.positions();
  auto inside = ((pos.unsqueeze(1) >= lo.unsqueeze(0)) & (pos.unsqueeze(1) < hi.unsqueeze(0))).all(2);
  auto cnt = inside.sum(0);
  auto lv = r.levels();
  CHECK((cnt.le(threshold) | lv.eq(5)).all().item<bool>());
  auto size = r.voxel_sizes();
  auto plo = ((lo - 0.0) / (size * 2.0)).floor() * size * 2.0; // parent box
  auto in_parent = ((pos.unsqueeze(1) >= plo.unsqueeze(0)) &
                    (pos.unsqueeze(1) < (plo + size * 2.0).unsqueeze(0))).all(2).sum(0);
  CHECK((lv.eq(0) | in_parent.gt(threshold)).all().item<bool>());
  // Features of refined leaves replicate the ancestor.
  CHECK_EQ(r.channels(), coarse.channels());
  // Threshold larger than N: nothing changes; invalid arguments throw.
  auto same = refine_by_gaussian_density(coarse, all, all.size(), -1);
  CHECK_EQ(same.size(), coarse.size());
  CHECK_THROWS(refine_by_gaussian_density(coarse, all, -1));
  CHECK_THROWS(refine_by_gaussian_density(coarse, all, 4, 7));
  CHECK_EQ(refine_by_gaussian_density(coarse, GaussianTensor::empty(0, kD), 0).size(),
           coarse.size());
}

TEST(gaussian_adaptive_level_assignment_matches_size) {
  // Separated Gaussians (one per level-0 cell): sigma = base / (2 * 2^l) -> level l.
  auto cfg = make_config(8.0, 5);
  std::vector<double> sig, expect;
  for (int l = 0; l <= 5; ++l) {
    sig.push_back(8.0 / (2.0 * std::ldexp(1.0, l)));
    expect.push_back(l);
  }
  sig.push_back(1e3);   // larger than the coarsest cell -> clamped to 0
  expect.push_back(0);
  sig.push_back(1e-4);  // smaller than the finest cell -> clamped to max_depth
  expect.push_back(5);
  sig.push_back(8.0 / (2.0 * 4.0) * 1.3); // within sqrt(2) of level 2 -> rounds to 2
  expect.push_back(2);
  const int64_t n = static_cast<int64_t>(sig.size());
  auto pos = torch::zeros({n, 3}, kD);
  auto scales = torch::zeros({n, 3}, kD);
  for (int64_t i = 0; i < n; ++i) {
    pos[i][0] = 24.0 * i + 4.1;
    pos[i][1] = 4.3;
    pos[i][2] = 3.9;
    scales[i][0] = sig[i] * 0.1; // anisotropic: the largest axis decides
    scales[i][1] = sig[i];
    scales[i][2] = sig[i] * 0.5;
  }
  auto g = make_gaussians(pos, scales);
  GaussianAdaptiveMap map;
  auto t = gaussians_to_adaptive(g, cfg, {}, &map);
  auto leaf = find_leaves(t, pos);
  auto levels = t.levels();
  for (int64_t i = 0; i < n; ++i) {
    CHECK_EQ(map.gaussian_level[i].item<int>(), static_cast<int>(expect[i]));
    CHECK(leaf[i].item<int64_t>() >= 0);
    CHECK_EQ(levels[leaf[i].item<int64_t>()].item<int>(), static_cast<int>(expect[i]));
  }
  CHECK_EQ(t.size(), n);

  // Level clamps and cell_per_sigma.
  GaussianToAdaptiveOptions o;
  o.min_level = 1;
  o.max_level = 3;
  gaussians_to_adaptive(g, cfg, o, &map);
  CHECK_EQ(map.gaussian_level.min().item<int>(), 1);
  CHECK_EQ(map.gaussian_level.max().item<int>(), 3);
  o = {};
  o.cell_per_sigma = 4.0; // twice coarser cells -> one level less
  gaussians_to_adaptive(g, cfg, o, &map);
  CHECK_EQ(map.gaussian_level[3].item<int>(), 2);
}

TEST(gaussian_adaptive_small_inside_large_refines) {
  // A level-0 Gaussian and a level-4 one in the same level-0 cell: the coarse
  // deposit is pushed down along the path to the fine cell (7 + 7 + 7 + 8 leaves).
  auto cfg = make_config(8.0, 4);
  auto pos = torch::tensor({{4.0, 4.0, 4.0}, {5.1, 5.3, 5.2}}, kD);
  auto scales = torch::tensor({{4.0, 4.0, 4.0}, {0.25, 0.25, 0.25}}, kD);
  auto op = torch::tensor({{0.8}, {0.3}}, kD);
  auto g = make_gaussians(pos, scales, {}, op);
  for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
    GaussianToAdaptiveOptions o;
    o.splat = splat;
    GaussianAdaptiveMap map;
    auto t = gaussians_to_adaptive(g, cfg, o, &map);
    CHECK(!t.has_overlapping_leaves());
    CHECK_EQ(map.gaussian_level[0].item<int>(), 0);
    CHECK_EQ(map.gaussian_level[1].item<int>(), 4);
    CHECK_NEAR(total(t.features().select(1, 0)), 1.1, 1e-12);
    auto counts = t.counts_by_level();
    auto leaf = find_leaves(t, pos);
    CHECK_EQ(t.levels()[leaf[1].item<int64_t>()].item<int>(), 4);
    if (splat == GaussianSplat::Center) {
      CHECK_EQ(counts[0], 0); // the coarse cell was subdivided
      CHECK_EQ(t.size(), 29);
      CHECK_EQ(counts[1], 7);
      CHECK_EQ(counts[4], 8);
    }
    // The small Gaussian stays at its own level; the big one reaches level 4
    // only through the subdivided cell.
    auto small_leaves = map.leaf_index.masked_select(map.gaussian_index == 1);
    CHECK(t.levels().index_select(0, small_leaves).eq(4).all().item<bool>());
    auto big_levels = t.levels().index_select(0, map.leaf_index.masked_select(
                                                     map.gaussian_index == 0));
    CHECK(big_levels.ge(0).all().item<bool>() && big_levels.le(4).all().item<bool>());
    CHECK(big_levels.eq(4).any().item<bool>());
    // 2 deposits, ~30 entries after subdivision: the guard also covers subdivision.
    o.max_entries = 5;
    bool length_error = false;
    try {
      gaussians_to_adaptive(g, cfg, o);
    } catch (const std::length_error &) {
      length_error = true;
    }
    CHECK(length_error);
  }
}

TEST(gaussian_adaptive_round_trip) {
  auto g = random_scene(300, 11);
  auto cfg = make_config(16.0, 6);
  auto t = gaussians_to_adaptive(g, cfg);
  auto back = adaptive_to_gaussians(t);
  const int64_t positive = (t.features().select(1, 0) > 0).sum().item<int64_t>();
  CHECK_EQ(back.size(), positive);
  CHECK_EQ(back.feature_channels(), 2);
  // Each leaf Gaussian is centered in its leaf with sigma = leaf / 2, so it maps
  // back onto exactly the same leaf: the topology is a fixed point.
  GaussianAdaptiveMap map;
  auto t2 = gaussians_to_adaptive(back, cfg, {}, &map);
  CHECK_EQ(t2.size(), t.size());
  CHECK(torch::equal(t2.coordinates(), t.coordinates()));
  CHECK(torch::equal(t2.levels(), t.levels()));
  CHECK_ALLCLOSE(t2.features().narrow(1, 1, 5), t.features().narrow(1, 1, 5), 1e-12, 1e-12);
  CHECK_ALLCLOSE(t2.features().select(1, 0), -torch::expm1(-t.features().select(1, 0)),
                 1e-12, 1e-300);
  // Far-tail leaves have tiny masses; their opacity must not round to zero.
  CHECK(t2.features().select(1, 0).gt(0).all().item<bool>());
  CHECK_ALLCLOSE(back.positions(), t.centers(), 0.0, 0.0);
  CHECK_ALLCLOSE(back.scales(), t.voxel_sizes() * 0.5, 0.0, 0.0);

  // Separated Gaussians survive the round trip exactly (colors, features).
  auto pos = torch::tensor({{1.5, 1.5, 1.5}, {20.5, 3.5, 7.5}, {9.0, 30.0, 2.0}}, kD);
  auto scales = torch::tensor({{0.2, 0.2, 0.2}, {1.0, 0.5, 0.5}, {3.0, 3.0, 3.0}}, kD);
  auto feat = torch::tensor({{1.0}, {2.0}, {3.0}}, kD);
  auto sep = make_gaussians(pos, scales, {}, {}, {}, feat);
  auto ts = gaussians_to_adaptive(sep, cfg);
  CHECK_EQ(ts.size(), 3);
  auto bs = adaptive_to_gaussians(ts);
  auto leaf = find_leaves(ts, pos);
  CHECK_ALLCLOSE(bs.colors().index_select(0, leaf), sep.colors(), 1e-14, 1e-14);
  CHECK_ALLCLOSE(bs.features().index_select(0, leaf), feat, 1e-14, 1e-14);

  // Options: constant opacity, no mass/colour channels.
  AdaptiveToGaussiansOptions ao;
  ao.opacity = AdaptiveOpacity::Constant;
  ao.constant_opacity = 0.25;
  ao.mass_channel = -1;
  ao.color_channel = -1;
  ao.feature_channel = -1;
  auto plain = adaptive_to_gaussians(t, ao);
  CHECK_EQ(plain.size(), t.size());
  CHECK_NEAR(plain.opacities().min().item<double>(), 0.25, 0.0);
  CHECK_NEAR(plain.colors().max().item<double>(), 0.5, 0.0);
  CHECK_EQ(plain.feature_channels(), 0);
  ao = {};
  ao.color_channel = 4; // needs channels 4..6, only 6 exist
  CHECK_THROWS(adaptive_to_gaussians(t, ao));
}

TEST(gaussian_adaptive_gradcheck_double) {
  // Overlapping Gaussians (shared leaves, pushdown) in both splat modes.
  auto g0 = random_scene(12, 3, /*extent=*/6.0, 0.1, 2.0, 2);
  auto cfg = make_config(8.0, 4);
  for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
    for (bool by_volume : {false, true}) {
      GaussianToAdaptiveOptions o;
      o.splat = splat;
      o.weight_by_volume = by_volume;
      auto op = g0.opacities().clone().requires_grad_(true);
      auto col = g0.colors().clone().requires_grad_(true);
      auto fe = g0.features().clone().requires_grad_(true);
      auto build = [&](const at::Tensor &a, const at::Tensor &b, const at::Tensor &c) {
        return gaussians_to_adaptive(
            GaussianTensor(g0.positions(), g0.rotations(), g0.scales(), a, b, c), cfg, o);
      };
      auto t = build(op, col, fe);
      torch::manual_seed(5);
      auto W = torch::randn(t.features().sizes(), kD);
      auto loss = (t.features() * W).sum();
      loss.backward();
      const double eps = 1e-6;
      int checked = 0;
      for (int which = 0; which < 3; ++which) {
        at::Tensor base = which == 0 ? op : which == 1 ? col : fe;
        auto grad = base.grad();
        CHECK(grad.defined());
        auto flat = base.detach().reshape({-1});
        for (int64_t k = 0; k < flat.numel(); ++k) {
          auto eval = [&](double delta) {
            torch::NoGradGuard ng;
            auto p = flat.clone();
            p[k] += delta;
            auto pt = p.reshape(base.sizes());
            auto tt = which == 0   ? build(pt, col.detach(), fe.detach())
                      : which == 1 ? build(op.detach(), pt, fe.detach())
                                   : build(op.detach(), col.detach(), pt);
            return (tt.features() * W).sum().item<double>();
          };
          const double fd = (eval(eps) - eval(-eps)) / (2 * eps);
          const double ad = grad.reshape({-1})[k].item<double>();
          CHECK_NEAR(ad, fd, 1e-6 * std::max(1.0, std::fabs(fd)));
          ++checked;
        }
      }
      CHECK_EQ(checked, 12 * (1 + 3 + 2));
    }
  }

  // Gradients through adaptive_to_gaussians back to the adaptive features.
  auto t = gaussians_to_adaptive(g0, cfg);
  auto f = t.features().detach().clone().requires_grad_(true);
  auto back = adaptive_to_gaussians(t.with_features(f));
  (back.opacities().sum() + back.colors().sum() * 2.0).backward();
  auto keep = f.select(1, 0) > 0;
  auto expect_mass = torch::exp(-f.detach().select(1, 0)).masked_select(keep);
  CHECK_ALLCLOSE(f.grad().select(1, 0).masked_select(keep), expect_mass, 1e-12, 1e-12);
  CHECK_NEAR(f.grad().narrow(1, 1, 3).sum().item<double>(),
             2.0 * 3.0 * keep.sum().item<double>(), 1e-12);
}

TEST(gaussian_adaptive_empty_and_single) {
  auto cfg = make_config(4.0, 3);
  auto empty = GaussianTensor::empty(0, kD, 3);
  for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
    GaussianToAdaptiveOptions o;
    o.splat = splat;
    GaussianAdaptiveMap map;
    auto t = gaussians_to_adaptive(empty, cfg, o, &map);
    CHECK_EQ(t.size(), 0);
    CHECK_EQ(t.channels(), 4 + 3);
    CHECK_EQ(map.gaussian_level.numel(), 0);
    CHECK_EQ(adaptive_to_gaussians(t).size(), 0);
    CHECK_EQ(find_leaves(t, torch::zeros({5, 3}, kD)).eq(-1).sum().item<int64_t>(), 5);
  }
  CHECK_THROWS(adaptive_config_for(empty, 4));

  auto pos = torch::tensor({{1.3, -2.2, 0.7}}, kD);
  auto scales = torch::tensor({{0.3, 0.2, 0.25}}, kD);
  auto op = torch::tensor({{0.6}}, kD);
  auto col = torch::tensor({{0.1, 0.2, 0.3}}, kD);
  auto one = make_gaussians(pos, scales, {}, op, col);
  auto t = gaussians_to_adaptive(one, cfg);
  CHECK_EQ(t.size(), 1);
  CHECK_ALLCLOSE(t.features(), torch::tensor({{0.6, 0.1, 0.2, 0.3}}, kD), 1e-15, 1e-15);
  CHECK_EQ(find_leaves(t, pos)[0].item<int64_t>(), 0);
  auto b = gaussians_to_adaptive(one, cfg, box_options());
  CHECK(b.size() > 1);
  const int level = b.levels()[0].item<int>();
  CHECK(b.levels().eq(level).all().item<bool>()); // one Gaussian -> one level
  CHECK_NEAR(total(b.features().select(1, 0)), 0.6, 1e-14);
  CHECK_ALLCLOSE(b.features().narrow(1, 1, 3), col.expand({b.size(), 3}), 1e-14, 1e-14);
}

TEST(gaussian_adaptive_degenerate_scales) {
  auto cfg = make_config(8.0, 6);
  auto pos = torch::tensor({{1.0, 1.0, 1.0},
                            {1.01, 1.02, 1.03},
                            {30.0, 2.0, 2.0},
                            {50.0, 2.0, 2.0},
                            {70.0, 70.0, 2.0}},
                           kD);
  auto scales = torch::tensor({{0.0, 0.0, 0.0},       // point mass
                               {1e-30, 1e-30, 1e-30}, // denormal-ish
                               {1e30, 1e30, 1e30},    // enormous
                               {1e-6, 1e-6, 10.0},    // needle
                               {5.0, 5.0, 0.0}},      // flat disc
                              kD);
  auto rot = identity_rotations(5);
  rot[3] = torch::tensor({0.9238795, 0.0, 0.3826834, 0.0}, kD); // rotated needle
  auto g = make_gaussians(pos, scales, rot);
  for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
    GaussianToAdaptiveOptions o;
    o.splat = splat;
    GaussianAdaptiveMap map;
    auto t = gaussians_to_adaptive(g, cfg, o, &map);
    CHECK(torch::isfinite(t.features()).all().item<bool>());
    CHECK(!t.has_overlapping_leaves());
    CHECK_NEAR(total(t.features().select(1, 0)), 2.5, 1e-12);
    CHECK_EQ(map.gaussian_level[0].item<int>(), 6);
    CHECK_EQ(map.gaussian_level[1].item<int>(), 6);
    CHECK_EQ(map.gaussian_level[2].item<int>(), 0);
    CHECK_EQ(map.gaussian_level[3].item<int>(), 0); // sigma_max = 10 -> coarse
    if (splat == GaussianSplat::BoxIntegral) {
      // The enormous splat is clipped to 32^3 level-0 cells around its mean
      // (cells holding finer deposits are subdivided, so it has >= 32^3 leaves).
      auto sel = map.gaussian_index == 2;
      CHECK(sel.sum().item<int64_t>() >= 32 * 32 * 32);
      auto [blo, bhi] = t.bounds();
      auto leaves = map.leaf_index.masked_select(sel);
      auto span = (bhi.index_select(0, leaves).amax(0) - blo.index_select(0, leaves).amin(0)) / 8.0;
      CHECK_ALLCLOSE(span, torch::full({3}, 32.0, kD), 0.0, 1e-12);
      CHECK_NEAR(map.weight.masked_select(sel).sum().item<double>(), 1.0, 1e-12);
    }
    // weight_by_volume: zero-volume Gaussians carry zero mass, colors stay finite.
    o.weight_by_volume = true;
    auto v = gaussians_to_adaptive(g, cfg, o);
    CHECK(torch::isfinite(v.features()).all().item<bool>());
  }
  // Invalid inputs.
  auto bad = scales.clone();
  bad[0][0] = -1.0;
  CHECK_THROWS(gaussians_to_adaptive(make_gaussians(pos, bad), cfg));
  bad = scales.clone();
  bad[0][0] = std::numeric_limits<double>::infinity();
  CHECK_THROWS(gaussians_to_adaptive(make_gaussians(pos, bad), cfg));
  auto nan_pos = pos.clone();
  nan_pos[1][2] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(gaussians_to_adaptive(make_gaussians(nan_pos, scales), cfg));
  auto zero_rot = rot.clone();
  zero_rot[0].zero_();
  CHECK_THROWS(gaussians_to_adaptive(make_gaussians(pos, scales, zero_rot), cfg));
}

TEST(gaussian_adaptive_extreme_coordinates) {
  // Far from the origin but representable: 1e9 world units, finest cell 1/64.
  auto cfg = make_config(4.0, 8);
  auto pos = torch::tensor({{1e9 + 0.3, -1e9 + 0.7, 5e8}, {-1e9, 1e9, -5e8}, {0.1, 0.2, 0.3}},
                           kD);
  auto scales = torch::tensor({{0.01, 0.01, 0.01}, {1.0, 1.0, 1.0}, {0.1, 0.1, 0.1}}, kD);
  auto g = make_gaussians(pos, scales);
  for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
    GaussianToAdaptiveOptions o;
    o.splat = splat;
    auto t = gaussians_to_adaptive(g, cfg, o);
    CHECK_NEAR(total(t.features().select(1, 0)), 1.5, 1e-12);
    auto leaf = find_leaves(t, pos);
    CHECK(leaf.ge(0).all().item<bool>());
    auto [lo, hi] = t.bounds();
    auto plo = lo.index_select(0, leaf), phi = hi.index_select(0, leaf);
    CHECK((plo <= pos).all().item<bool>());
    CHECK((pos < phi).all().item<bool>());
  }
  // A non-zero origin shifts the grid.
  auto shifted = make_config(4.0, 8, {1e9, -1e9, 5e8});
  auto ts = gaussians_to_adaptive(g.select(torch::tensor({0}, torch::kInt64)), shifted);
  CHECK(ts.coordinates().abs().max().item<int64_t>() < 1024);
  // Beyond 2^52 finest cells: rejected instead of silently aliasing.
  auto far = torch::tensor({{1e15, 0.0, 0.0}}, kD);
  CHECK_THROWS(gaussians_to_adaptive(make_gaussians(far, torch::full({1, 3}, 0.1, kD)), cfg));
  CHECK(find_leaves(ts, far)[0].item<int64_t>() == -1);
  auto inf = torch::tensor({{std::numeric_limits<double>::infinity(), 0.0, 0.0}}, kD);
  CHECK(find_leaves(ts, inf)[0].item<int64_t>() == -1);
}

TEST(gaussian_adaptive_find_leaves_matches_bounds) {
  auto g = random_scene(200, 21);
  auto cfg = make_config(16.0, 6);
  auto t = gaussians_to_adaptive(g, cfg, box_options());
  torch::manual_seed(9);
  auto q = torch::rand({500, 3}, kD) * 40.0 - 4.0;
  auto got = find_leaves(t, q);
  auto [lo, hi] = t.bounds();
  // Brute force: inside[p, j] = lo_j <= q_p < hi_j.
  auto inside = ((q.unsqueeze(1) >= lo.unsqueeze(0)) & (q.unsqueeze(1) < hi.unsqueeze(0)))
                    .all(2);
  auto hits = inside.sum(1);
  CHECK(hits.le(1).all().item<bool>());
  auto expect = torch::where(hits > 0, inside.to(torch::kFloat64).argmax(1),
                             torch::full_like(hits, -1));
  CHECK(torch::equal(got, expect));
  CHECK(got.ge(0).sum().item<int64_t>() > 0);
  CHECK(got.eq(-1).sum().item<int64_t>() > 0);
}

TEST(gaussian_adaptive_options_and_config) {
  auto g = random_scene(50, 4);
  AdaptiveVoxelConfig two_d;
  two_d.base_voxel_size = {1.0, 1.0};
  two_d.max_depth = 2;
  CHECK_THROWS(gaussians_to_adaptive(g, two_d));
  auto cfg = make_config(16.0, 4);
  GaussianToAdaptiveOptions o;
  o.min_level = 3;
  o.max_level = 2;
  CHECK_THROWS(gaussians_to_adaptive(g, cfg, o));
  o = {};
  o.max_level = 5;
  CHECK_THROWS(gaussians_to_adaptive(g, cfg, o));
  o = {};
  o.cell_per_sigma = 0.0;
  CHECK_THROWS(gaussians_to_adaptive(g, cfg, o));
  o = {};
  o.min_weight = 1.0;
  CHECK_THROWS(gaussians_to_adaptive(g, cfg, o));
  // min_weight only removes negligible entries: same mass, at most as many leaves,
  // and a large threshold still keeps every Gaussian (its largest entry).
  {
    GaussianToAdaptiveOptions exact = box_options(), pruned = box_options();
    exact.min_weight = 0.0;
    pruned.min_weight = 1e-3;
    GaussianAdaptiveMap me, mp;
    auto te = gaussians_to_adaptive(g, cfg, exact, &me);
    auto tp = gaussians_to_adaptive(g, cfg, pruned, &mp);
    CHECK(tp.size() <= te.size());
    // Kept entries were >= 1e-3 (or their Gaussian's only survivor, renormalized to 1).
    CHECK(mp.weight.ge(1e-3).all().item<bool>());
    CHECK_EQ(std::get<0>(at::_unique2(mp.gaussian_index)).numel(), g.size());
    CHECK_NEAR(total(tp.features().select(1, 0)), total(te.features().select(1, 0)),
               1e-12 * total(te.features().select(1, 0)));
    auto per_g = torch::zeros({g.size()}, kD).index_add(0, mp.gaussian_index, mp.weight);
    CHECK_ALLCLOSE(per_g, torch::ones_like(per_g), 1e-12, 1e-12);
  }
  o = box_options();
  o.max_entries = 10;
  bool length_error = false;
  try {
    gaussians_to_adaptive(g, cfg, o);
  } catch (const std::length_error &) {
    length_error = true;
  }
  CHECK(length_error);

  // include_features = false drops the extra channels.
  o = {};
  o.include_features = false;
  CHECK_EQ(gaussians_to_adaptive(g, cfg, o).channels(), 4);

  // adaptive_config_for: valid grid, finest cell from the 1% scale quantile.
  auto auto_cfg = adaptive_config_for(g, 6);
  auto_cfg.validate();
  CHECK_EQ(auto_cfg.max_depth, 6u);
  GaussianAdaptiveMap map;
  auto t = gaussians_to_adaptive(g, auto_cfg, {}, &map);
  CHECK(t.size() > 0);
  CHECK(map.gaussian_level.max().item<int>() == 6);
}

TEST(gaussian_adaptive_feeds_adaptive_ops) {
  // The bridge output is an ordinary AdaptiveSparseTensor: convolution and
  // pooling run on it and gradients reach the Gaussian opacities.
  auto g0 = random_scene(150, 13);
  auto op = g0.opacities().clone().requires_grad_(true);
  GaussianTensor g(g0.positions(), g0.rotations(), g0.scales(), op, g0.colors(),
                   g0.features());
  auto t = gaussians_to_adaptive(g, make_config(16.0, 5));
  torch::manual_seed(1);
  auto w = torch::randn({27, t.channels(), 4}, kD) * 0.1;
  minkowski::adaptive::AdaptiveKernelConfig kc;
  auto y = minkowski::adaptive::adaptive_convolution(t, w, kc);
  minkowski::adaptive::adaptive_global_avg_pool(y).sum().backward();
  CHECK(op.grad().defined());
  CHECK(torch::isfinite(op.grad()).all().item<bool>());
  CHECK(op.grad().abs().sum().item<double>() > 0.0);
}

TEST(gaussian_adaptive_cpu_cuda_agreement) {
  if (!torch::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
  auto g = random_scene(3000, 17, 64.0, 0.05, 6.0, 3);
  auto cfg = make_config(32.0, 7);
  for (auto splat : {GaussianSplat::Center, GaussianSplat::BoxIntegral}) {
    GaussianToAdaptiveOptions o;
    o.splat = splat;
    for (auto dtype : {torch::kFloat64, torch::kFloat32}) {
      auto gc = GaussianTensor(g.positions().to(dtype), g.rotations().to(dtype),
                               g.scales().to(dtype), g.opacities().to(dtype),
                               g.colors().to(dtype), g.features().to(dtype));
      auto gg = gc.to(torch::kCUDA);
      auto tc = gaussians_to_adaptive(gc, cfg, o);
      auto tg = gaussians_to_adaptive(gg, cfg, o);
      CHECK(tg.features().is_cuda());
      CHECK(torch::equal(tc.coordinates(), tg.coordinates()));
      CHECK(torch::equal(tc.levels(), tg.levels()));
      const double tol = dtype == torch::kFloat64 ? 1e-12 : 1e-5;
      // index_add order differs on CUDA (atomics); with min_weight every mass is a
      // normal float, so mean attributes agree to rounding.
      CHECK_ALLCLOSE(tg.features().cpu(), tc.features(), tol, tol);
      auto bc = adaptive_to_gaussians(tc);
      auto bg = adaptive_to_gaussians(tg);
      CHECK_EQ(bc.size(), bg.size());
      CHECK_ALLCLOSE(bg.colors().cpu(), bc.colors(), tol, tol);
      CHECK_ALLCLOSE(bg.opacities().cpu(), bc.opacities(), tol, tol);
      CHECK(torch::equal(find_leaves(tg, gg.positions()), find_leaves(tc, gc.positions())));
    }
  }
}
