// Representation study: 3-D shape classification with identical data, budget
// and evaluation for four input representations.
//
//   uniform            points -> uniform sparse voxels (32^3) -> sparse conv net
//   adaptive           points -> adaptive voxels (levels 0..3, refine_by_density)
//                      -> adaptive conv net
//   gaussian           points -> adaptive Gaussian mixture -> per-Gaussian MLP
//                      (PointNet-style set network over Gaussian parameters)
//   adaptive_gaussian  same Gaussians -> gaussians_to_adaptive (bridge) ->
//                      adaptive conv net
//
// Dataset: 10 procedurally generated shape classes (sphere, cube, cylinder, cone,
// torus, pyramid, capsule, annulus, helix, dumbbell), surface samples with random
// rotation, anisotropic scale, translation, jitter and outliers. Generated from
// --data-seed only, so every method and seed sees the same samples. --seed sets
// the model initialization and the shuffling order.
//
//   ml_representations --method uniform --seed 1 --epochs 12 --output run.json
// --device cuda (default) requires CUDA; cpu is explicit, and auto permits fallback.
//
// Writes a JSON record (per-epoch loss/accuracy/time, final accuracy, macro
// precision/recall/F1, train time, inference latency/throughput, peak CUDA and
// host memory, active voxel counts, conversion cost). Train on CUDA: the CPU
// sparse convolution of upstream main has a GEMM layout bug.
#include "MinkowskiEngine/minkowski.hpp"
#include "adaptive/adaptive_tensor.hpp"
#include "adaptive/ops.hpp"
#include "gaussian/gaussian_adaptive.hpp"
#include "third_party/json.hpp"

#include <torch/torch.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace minkowski;
namespace ad = minkowski::adaptive;
namespace gs = minkowski::gaussian;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int kClasses = 10;
const char *kClassNames[kClasses] = {"sphere",  "cube",    "cylinder", "cone",  "torus",
                                     "pyramid", "capsule", "annulus",  "helix", "dumbbell"};
constexpr double kPi = 3.14159265358979323846;

double now_s() {
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- dataset

using Vec3 = std::array<double, 3>;

struct Sampler {
  std::mt19937_64 rng;
  explicit Sampler(uint64_t seed) : rng(seed) {}
  double u(double a = 0.0, double b = 1.0) {
    return a + (b - a) * std::uniform_real_distribution<double>(0.0, 1.0)(rng);
  }
  double n() { return std::normal_distribution<double>(0.0, 1.0)(rng); }
  Vec3 sphere_dir() {
    while (true) {
      Vec3 v{n(), n(), n()};
      const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      if (r > 1e-9)
        return {v[0] / r, v[1] / r, v[2] / r};
    }
  }
};

// Area-weighted surface sample of shape `cls` in roughly [-1, 1]^3.
Vec3 sample_shape(int cls, Sampler &s) {
  switch (cls) {
  case 0: { // sphere r 0.9
    auto d = s.sphere_dir();
    return {0.9 * d[0], 0.9 * d[1], 0.9 * d[2]};
  }
  case 1: { // cube surface, half edge 0.7
    const int face = static_cast<int>(s.u() * 6) % 6;
    Vec3 p{s.u(-0.7, 0.7), s.u(-0.7, 0.7), s.u(-0.7, 0.7)};
    p[face / 2] = face % 2 ? 0.7 : -0.7;
    return p;
  }
  case 2: { // cylinder r 0.55, half height 0.8, with caps
    const double r = 0.55, h = 0.8;
    const double side = 2 * kPi * r * 2 * h, cap = kPi * r * r;
    const double t = s.u() * (side + 2 * cap);
    if (t < side) {
      const double a = s.u(0, 2 * kPi);
      return {r * std::cos(a), r * std::sin(a), s.u(-h, h)};
    }
    const double rr = r * std::sqrt(s.u()), a = s.u(0, 2 * kPi);
    return {rr * std::cos(a), rr * std::sin(a), t < side + cap ? -h : h};
  }
  case 3: { // cone base r 0.75 at z=-0.8, apex z=0.8
    const double r = 0.75, h = 1.6;
    const double slant = std::sqrt(r * r + h * h);
    const double lat = kPi * r * slant, base = kPi * r * r;
    if (s.u() * (lat + base) < lat) {
      const double t = std::sqrt(s.u()); // area grows linearly with distance from apex
      const double a = s.u(0, 2 * kPi);
      return {t * r * std::cos(a), t * r * std::sin(a), 0.8 - t * h};
    }
    const double rr = r * std::sqrt(s.u()), a = s.u(0, 2 * kPi);
    return {rr * std::cos(a), rr * std::sin(a), -0.8};
  }
  case 4: { // torus R 0.65, r 0.25 (rejection for area weighting)
    const double R = 0.65, r = 0.25;
    while (true) {
      const double a = s.u(0, 2 * kPi), b = s.u(0, 2 * kPi);
      if (s.u() * (R + r) <= R + r * std::cos(b))
        return {(R + r * std::cos(b)) * std::cos(a), (R + r * std::cos(b)) * std::sin(a),
                r * std::sin(b)};
    }
  }
  case 5: { // square pyramid: base half 0.75 at z=-0.7, apex z=0.8
    const double hb = 0.75, z0 = -0.7, z1 = 0.8;
    const double slant = std::sqrt((z1 - z0) * (z1 - z0) + hb * hb);
    const double tri = hb * slant * 2.0; // one side face area = 0.5 * 2hb * slant
    const double base = 4 * hb * hb;
    const double t = s.u() * (base + 4 * tri);
    if (t < base)
      return {s.u(-hb, hb), s.u(-hb, hb), z0};
    const int f = static_cast<int>((t - base) / tri) % 4;
    double a = s.u(), b = s.u();
    if (a + b > 1) {
      a = 1 - a;
      b = 1 - b;
    }
    // triangle (corner0, corner1, apex)
    const double cs[4][2] = {{-hb, -hb}, {hb, -hb}, {hb, hb}, {-hb, hb}};
    const auto &c0 = cs[f];
    const auto &c1 = cs[(f + 1) % 4];
    return {c0[0] + a * (c1[0] - c0[0]) + b * (0 - c0[0]),
            c0[1] + a * (c1[1] - c0[1]) + b * (0 - c0[1]), z0 + b * (z1 - z0)};
  }
  case 6: { // capsule: r 0.4, cylinder half length 0.5
    const double r = 0.4, h = 0.5;
    const double side = 2 * kPi * r * 2 * h, caps = 4 * kPi * r * r;
    if (s.u() * (side + caps) < side) {
      const double a = s.u(0, 2 * kPi);
      return {r * std::cos(a), r * std::sin(a), s.u(-h, h)};
    }
    auto d = s.sphere_dir();
    return {r * d[0], r * d[1], r * d[2] + (d[2] >= 0 ? h : -h)};
  }
  case 7: { // flat annulus r in [0.4, 0.9], z = 0
    const double rr = std::sqrt(s.u(0.4 * 0.4, 0.9 * 0.9)), a = s.u(0, 2 * kPi);
    return {rr * std::cos(a), rr * std::sin(a), 0.0};
  }
  case 8: { // helix tube: 2.5 turns, radius 0.6, tube 0.08
    const double t = s.u(), a = t * 2.5 * 2 * kPi;
    Vec3 c{0.6 * std::cos(a), 0.6 * std::sin(a), -0.85 + 1.7 * t};
    auto d = s.sphere_dir();
    return {c[0] + 0.08 * d[0], c[1] + 0.08 * d[1], c[2] + 0.08 * d[2]};
  }
  default: { // dumbbell: spheres r 0.35 at z=+-0.55, rod r 0.08
    const double rs = 0.35, rr = 0.08, hl = 0.55 - rs;
    const double sph = 2 * 4 * kPi * rs * rs, rod = 2 * kPi * rr * 2 * hl;
    if (s.u() * (sph + rod) < sph) {
      auto d = s.sphere_dir();
      return {rs * d[0], rs * d[1], rs * d[2] + (s.u() < 0.5 ? 0.55 : -0.55)};
    }
    const double a = s.u(0, 2 * kPi);
    return {rr * std::cos(a), rr * std::sin(a), s.u(-hl, hl)};
  }
  }
}

struct Sample {
  at::Tensor points; // [P, 3] float64 in [-1, 1)
  int label = 0;
};

Sample make_sample(int cls, uint64_t seed, int npoints) {
  Sampler s(seed);
  // Random rotation (uniform quaternion), anisotropic scale, translation.
  const double w0 = s.n(), x0 = s.n(), y0 = s.n(), z0 = s.n();
  const double qn = std::sqrt(w0 * w0 + x0 * x0 + y0 * y0 + z0 * z0);
  const double w = w0 / qn, x = x0 / qn, y = y0 / qn, z = z0 / qn;

  const double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                          {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                          {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
  const double g = s.u(0.65, 0.95);
  const Vec3 an{g * s.u(0.85, 1.15), g * s.u(0.85, 1.15), g * s.u(0.85, 1.15)};
  const Vec3 tr{s.u(-0.08, 0.08), s.u(-0.08, 0.08), s.u(-0.08, 0.08)};
  auto pts = torch::empty({npoints, 3}, torch::kFloat64);
  double *p = pts.data_ptr<double>();
  const int outliers = npoints / 50; // 2 %
  for (int i = 0; i < npoints; ++i) {
    Vec3 v;
    if (i < npoints - outliers) {
      auto c = sample_shape(cls, s);
      for (int d = 0; d < 3; ++d)
        c[d] *= an[d];
      for (int d = 0; d < 3; ++d)
        v[d] = R[d][0] * c[0] + R[d][1] * c[1] + R[d][2] * c[2] + tr[d] + 0.01 * s.n();
    } else {
      v = {s.u(-1, 1), s.u(-1, 1), s.u(-1, 1)};
    }
    for (int d = 0; d < 3; ++d)
      p[i * 3 + d] = std::clamp(v[d], -0.999999, 0.999999);
  }
  return {pts, cls};
}

std::vector<Sample> make_split(int per_class, uint64_t data_seed, uint64_t split, int npoints) {
  std::vector<Sample> out;
  for (int i = 0; i < per_class; ++i)
    for (int c = 0; c < kClasses; ++c)
      out.push_back(make_sample(
          c, data_seed * 1000003ULL + split * 7919ULL * 100000ULL + i * kClasses + c, npoints));
  return out;
}

// ---------------------------------------------------------------- representations

// Every representation is precomputed once per sample (static dataset).
struct Rep {
  // uniform: int32 coords [N, 3] in [0, 32); features [N, Cf]
  // adaptive / adaptive_gaussian: coords [N, 3] int64, levels [N] int32, feats [N, Cf]
  // gaussian: feats [G, Cf] (set)
  at::Tensor coords, levels, feats;
};

constexpr int kUniformRes = 32; // voxel 2/32 = 0.0625
constexpr int kAdaptiveDepth = 3; // level 0 cell 0.5, level 3 cell 0.0625
constexpr double kAdaptiveBase = 0.5;

Rep uniform_rep(const at::Tensor &pts) {
  auto c = ((pts + 1.0) * (kUniformRes / 2.0)).floor().to(torch::kInt64).clamp(0, kUniformRes - 1);
  auto key = (c.select(1, 0) * kUniformRes + c.select(1, 1)) * kUniformRes + c.select(1, 2);
  auto [uniq, inv, cnt] = at::_unique2(key, true, true, true);
  auto coords = torch::stack({uniq.div(kUniformRes * kUniformRes, "floor"),
                              uniq.div(kUniformRes, "floor").remainder(kUniformRes),
                              uniq.remainder(kUniformRes)},
                             1);
  auto f = torch::stack({torch::ones_like(cnt, torch::kFloat32),
                         torch::log1p(cnt.to(torch::kFloat32)) * 0.5f},
                        1);
  return {coords.to(torch::kInt32), {}, f};
}

ad::AdaptiveVoxelConfig adaptive_cfg() {
  ad::AdaptiveVoxelConfig cfg;
  cfg.base_voxel_size = {kAdaptiveBase, kAdaptiveBase, kAdaptiveBase};
  cfg.max_depth = kAdaptiveDepth;
  cfg.origin = {-1.0, -1.0, -1.0};
  return cfg;
}

// Leaf features: [log1p(count), log1p(count per finest-cell volume), level / depth].
at::Tensor adaptive_point_features(const ad::AdaptiveSparseTensor &t, const at::Tensor &pts) {
  auto leaf = gs::find_leaves(t, pts);
  auto valid = leaf.ge(0);
  auto cnt = torch::zeros({t.size()}, torch::kFloat64)
                 .index_add(0, leaf.masked_select(valid),
                            torch::ones({valid.sum().item<int64_t>()}, torch::kFloat64));
  auto lv = t.levels().to(torch::kFloat64);
  auto rel_vol = torch::pow(8.0, kAdaptiveDepth - lv); // leaf volume in finest cells
  return torch::stack({torch::log1p(cnt) * 0.5, torch::log1p(cnt / rel_vol), lv / kAdaptiveDepth}, 1)
      .to(torch::kFloat32);
}

Rep adaptive_rep(const at::Tensor &pts, int64_t threshold) {
  auto t = ad::AdaptiveSparseTensor::from_points(pts, torch::ones({pts.size(0), 1}, torch::kFloat64),
                                                 adaptive_cfg());
  t.refine_by_density(threshold);
  return {t.coordinates(), t.levels(), adaptive_point_features(t, pts)};
}

// Adaptive Gaussian mixture fit: start from 0.25 cells, split a cell (down to
// 0.0625) while its points are not planar (smallest eigenvalue share > tau)
// and it holds >= 16 points. One Gaussian per final cell with >= 3 points.
struct GaussianFit {
  at::Tensor mean;   // [G, 3]
  at::Tensor cov;    // [G, 3, 3]
  at::Tensor count;  // [G]
  at::Tensor cell;   // [G] fitting cell edge
};

GaussianFit fit_gaussians(const at::Tensor &pts) {
  const double *p = pts.data_ptr<double>();
  const int64_t P = pts.size(0);
  struct Cell {
    std::vector<int64_t> idx;
    double size;
    std::array<int64_t, 3> c;
  };
  std::vector<Cell> work;
  {
    const double s0 = 0.25;
    std::map<std::array<int64_t, 3>, std::vector<int64_t>> g;
    for (int64_t i = 0; i < P; ++i) {
      std::array<int64_t, 3> c{};
      for (int d = 0; d < 3; ++d)
        c[d] = static_cast<int64_t>(std::floor((p[i * 3 + d] + 1.0) / s0));
      g[c].push_back(i);
    }
    for (auto &[c, v] : g)
      work.push_back({std::move(v), s0, c});
  }
  std::vector<double> means, covs, counts, cells;
  const double tau = 0.015;
  while (!work.empty()) {
    Cell cell = std::move(work.back());
    work.pop_back();
    const int64_t n = static_cast<int64_t>(cell.idx.size());
    if (n < 3)
      continue; // isolated outliers carry no shape
    double m[3] = {0, 0, 0};
    for (int64_t i : cell.idx)
      for (int d = 0; d < 3; ++d)
        m[d] += p[i * 3 + d];
    for (double &v : m)
      v /= static_cast<double>(n);
    double C[3][3] = {};
    for (int64_t i : cell.idx)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          C[a][b] += (p[i * 3 + a] - m[a]) * (p[i * 3 + b] - m[b]);
    const double reg = std::pow(0.05 * cell.size, 2);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        C[a][b] = C[a][b] / static_cast<double>(n) + (a == b ? reg : 0.0);
    auto ct = torch::from_blob(&C[0][0], {3, 3}, torch::kFloat64).clone();
    auto ev = std::get<0>(at::linalg_eigh(ct));
    const double lmin = ev[0].item<double>(), tr = ev.sum().item<double>();
    if (cell.size > 0.0625 * 1.5 && n >= 16 && lmin / tr > tau) {
      std::map<std::array<int64_t, 3>, std::vector<int64_t>> g;
      const double hs = 0.5 * cell.size;
      for (int64_t i : cell.idx) {
        std::array<int64_t, 3> c{};
        for (int d = 0; d < 3; ++d)
          c[d] = static_cast<int64_t>(std::floor((p[i * 3 + d] + 1.0) / hs));
        g[c].push_back(i);
      }
      for (auto &[c, v] : g)
        work.push_back({std::move(v), hs, c});
      continue;
    }
    for (double v : m)
      means.push_back(v);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        covs.push_back(C[a][b]);
    counts.push_back(static_cast<double>(n));
    cells.push_back(cell.size);
  }
  const int64_t G = static_cast<int64_t>(counts.size());
  auto o = torch::kFloat64;
  return {torch::from_blob(means.data(), {G, 3}, o).clone(),
          torch::from_blob(covs.data(), {G, 3, 3}, o).clone(),
          torch::from_blob(counts.data(), {G}, o).clone(),
          torch::from_blob(cells.data(), {G}, o).clone()};
}

// Per-Gaussian descriptors shared by both Gaussian methods (float64):
// normal |e_min| (3), sqrt eigenvalues / 0.25 (3), log1p(count) / 4 (1).
struct GaussianDesc {
  at::Tensor normal, sig, logc, R; // R: [G,3,3] eigenvectors (columns), det +1
};

GaussianDesc describe(const GaussianFit &f) {
  auto [ev, V] = at::linalg_eigh(f.cov);
  auto det = at::linalg_det(V);
  // flip the last column where det < 0 so V is a rotation
  auto sign = torch::where(det < 0, -torch::ones_like(det), torch::ones_like(det));
  V = V.clone();
  V.select(2, 2).mul_(sign.unsqueeze(1));
  return {V.select(2, 0).abs(), ev.clamp_min(0).sqrt() / 0.25, torch::log1p(f.count) / 4.0, V};
}

// Rotation matrices [G,3,3] -> quaternions (w, x, y, z) [G,4].
at::Tensor matrix_to_quaternion(const at::Tensor &R) {
  auto Rc = R.contiguous();
  const int64_t G = Rc.size(0);
  auto q = torch::empty({G, 4}, torch::kFloat64);
  const double *r = Rc.data_ptr<double>();
  double *o = q.data_ptr<double>();
  for (int64_t i = 0; i < G; ++i) {
    const double *m = r + i * 9;
    const double m00 = m[0], m01 = m[1], m02 = m[2], m10 = m[3], m11 = m[4], m12 = m[5],
                 m20 = m[6], m21 = m[7], m22 = m[8];
    const double tr = m00 + m11 + m22;
    double w, x, y, z;
    if (tr > 0) {
      const double s = std::sqrt(tr + 1.0) * 2;
      w = 0.25 * s, x = (m21 - m12) / s, y = (m02 - m20) / s, z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
      const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2;
      w = (m21 - m12) / s, x = 0.25 * s, y = (m01 + m10) / s, z = (m02 + m20) / s;
    } else if (m11 > m22) {
      const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2;
      w = (m02 - m20) / s, x = (m01 + m10) / s, y = 0.25 * s, z = (m12 + m21) / s;
    } else {
      const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2;
      w = (m10 - m01) / s, x = (m02 + m20) / s, y = (m12 + m21) / s, z = 0.25 * s;
    }
    o[i * 4 + 0] = w, o[i * 4 + 1] = x, o[i * 4 + 2] = y, o[i * 4 + 3] = z;
  }
  return q;
}

// Gaussian set features: mean (3), normal (3), sigma (3), log count (1),
// fitting cell / 0.25 (1) = 11 channels.
Rep gaussian_rep(const at::Tensor &pts) {
  auto f = fit_gaussians(pts);
  auto d = describe(f);
  auto feats = torch::cat({f.mean, d.normal, d.sig, d.logc.unsqueeze(1), (f.cell / 0.25).unsqueeze(1)}, 1);
  return {{}, {}, feats.to(torch::kFloat32)};
}

// Bridge: the same Gaussians -> adaptive leaves. Leaf features =
// [mass, normal(3) (as "colour"), sigma(3), log count(1), level / depth].
Rep adaptive_gaussian_rep(const at::Tensor &pts) {
  auto f = fit_gaussians(pts);
  auto d = describe(f);
  const int64_t G = f.count.size(0);
  auto opacity = (f.count / 32.0).clamp_max(1.0).unsqueeze(1);
  auto scales = (d.sig * 0.25).clamp_min(1e-4);
  gs::GaussianTensor g(f.mean, matrix_to_quaternion(d.R), scales, opacity, d.normal,
                       torch::cat({d.sig, d.logc.unsqueeze(1)}, 1));
  gs::GaussianToAdaptiveOptions o;
  o.cell_per_sigma = std::sqrt(12.0); // a box-filling Gaussian maps to its own cell size
  o.splat = gs::GaussianSplat::Center;
  auto t = gs::gaussians_to_adaptive(g, adaptive_cfg(), o);
  (void)G;
  auto lv = t.levels().to(torch::kFloat64).unsqueeze(1) / kAdaptiveDepth;
  return {t.coordinates(), t.levels(), torch::cat({t.features(), lv}, 1).to(torch::kFloat32)};
}

// ---------------------------------------------------------------- batching

struct Batch {
  // uniform: SparseTensor input built from coords/feats; adaptive: tensor; gaussian: feats
  at::Tensor coords; // uniform: [N, 4] int32 (b, x, y, z); gaussian: [N] int64 sample id
  at::Tensor feats;
  std::shared_ptr<ad::AdaptiveSparseTensor> tensor;
  at::Tensor labels;
  int64_t B = 0;
  // Keeps the uniform engine's coordinate manager alive until backward is done.
  std::shared_ptr<CoordinateManager> manager;
};

constexpr int64_t kSampleStrideCells = 6; // level-0 cells between batch items along x

Batch make_batch(const std::string &method, const std::vector<Rep> &reps,
                 const std::vector<Sample> &data, const std::vector<int64_t> &idx,
                 const torch::Device &dev) {
  Batch b;
  b.B = static_cast<int64_t>(idx.size());
  std::vector<at::Tensor> cs, fs, ls;
  std::vector<int64_t> labels;
  for (int64_t k = 0; k < b.B; ++k) {
    const Rep &r = reps[idx[k]];
    labels.push_back(data[idx[k]].label);
    const int64_t n = r.feats.size(0);
    if (method == "uniform") {
      cs.push_back(torch::cat({torch::full({n, 1}, k, torch::kInt32), r.coords}, 1));
    } else if (method == "gaussian") {
      cs.push_back(torch::full({n}, k, torch::kInt64));
    } else {
      auto c = r.coords.clone();
      auto shift = (torch::pow(2.0, r.levels.to(torch::kFloat64)) * (k * kSampleStrideCells))
                       .to(torch::kInt64);
      c.select(1, 0).add_(shift);
      cs.push_back(c);
      ls.push_back(r.levels);
    }
    fs.push_back(r.feats);
  }
  b.labels = torch::tensor(labels, torch::kInt64).to(dev);
  auto feats = torch::cat(fs, 0);
  if (method == "uniform" || method == "gaussian") {
    b.coords = torch::cat(cs, 0).to(dev);
    b.feats = feats.to(dev);
  } else {
    auto t = ad::AdaptiveSparseTensor::from_voxels(torch::cat(cs, 0), torch::cat(ls, 0), feats,
                                                   adaptive_cfg());
    b.tensor = std::make_shared<ad::AdaptiveSparseTensor>(t.to(dev));
  }
  return b;
}

// Sample id of every leaf from its center (items are kSampleStrideCells apart).
at::Tensor sample_ids(const ad::AdaptiveSparseTensor &t, const torch::Device &dev) {
  auto cx = t.centers().select(1, 0);
  return ((cx + 1.0) / (kSampleStrideCells * kAdaptiveBase)).floor().to(torch::kInt64).to(dev);
}

// [B, 2C]: per-sample mean and max.
at::Tensor pool_avg_max(const at::Tensor &x, const at::Tensor &sid, int64_t B) {
  const int64_t C = x.size(1);
  auto sum = torch::zeros({B, C}, x.options()).index_add(0, sid, x);
  auto cnt = torch::zeros({B}, x.options()).index_add(0, sid, torch::ones({x.size(0)}, x.options()));
  auto mean = sum / cnt.clamp_min(1).unsqueeze(1);
  auto mx = torch::zeros({B, C}, x.options())
                .scatter_reduce(0, sid.unsqueeze(1).expand({x.size(0), C}), x, "amax",
                                /*include_self=*/false);
  return torch::cat({mean, mx}, 1);
}

// ---------------------------------------------------------------- models

const int kWidths[4] = {16, 32, 64, 128};

struct UniformNetImpl : torch::nn::Module {
  std::vector<MinkowskiConvolution> convs;
  std::vector<MinkowskiBatchNorm> bns;
  MinkowskiReLU relu;
  torch::nn::Linear head{nullptr};
  explicit UniformNetImpl(int cin) {
    auto add = [&](int a, int b, int ks, int stride) {
      convs.push_back(register_module("conv" + std::to_string(convs.size()),
                                      MinkowskiConvolution(a, b, ks, stride, 1, false, false,
                                                           ConvolutionMode::DEFAULT, 3)));
      bns.push_back(register_module("bn" + std::to_string(bns.size()), MinkowskiBatchNorm(b)));
    };
    add(cin, kWidths[0], 3, 1);
    add(kWidths[0], kWidths[0], 3, 1);
    for (int s = 1; s < 4; ++s) {
      add(kWidths[s - 1], kWidths[s], 2, 2);
      add(kWidths[s], kWidths[s], 3, 1);
    }
    relu = register_module("relu", MinkowskiReLU());
    head = register_module("head", torch::nn::Linear(2 * kWidths[3], kClasses));
  }
  at::Tensor forward(const SparseTensor &in, int64_t B) {
    SparseTensor x = in;
    for (size_t i = 0; i < convs.size(); ++i)
      x = relu->forward(bns[i]->forward(convs[i]->forward(x)));
    auto sid = x.C().select(1, 0).to(torch::kInt64).to(x.F().device());
    return head->forward(pool_avg_max(x.F(), sid, B));
  }
};
TORCH_MODULE(UniformNet);

struct AdaptiveNetImpl : torch::nn::Module {
  std::vector<at::Tensor> weights;
  std::vector<torch::nn::BatchNorm1d> bns;
  torch::nn::Linear head{nullptr};
  explicit AdaptiveNetImpl(int cin) {
    auto add = [&](int a, int b) {
      const size_t i = weights.size();
      weights.push_back(register_parameter("w" + std::to_string(i),
                                           torch::randn({27, a, b}) * std::sqrt(2.0 / (27.0 * a))));
      bns.push_back(register_module("bn" + std::to_string(i), torch::nn::BatchNorm1d(b)));
    };
    add(cin, kWidths[0]);
    add(kWidths[0], kWidths[0]);
    for (int s = 1; s < 4; ++s) {
      add(kWidths[s - 1], kWidths[s]);
      add(kWidths[s], kWidths[s]);
    }
    head = register_module("head", torch::nn::Linear(2 * kWidths[3], kClasses));
  }
  at::Tensor forward(const ad::AdaptiveSparseTensor &in, int64_t B) {
    ad::AdaptiveKernelConfig kc; // 3^3, CenterSampled
    auto layer = [&](const ad::AdaptiveSparseTensor &x, size_t i) {
      auto y = ad::adaptive_convolution(x, weights[i], kc);
      return y.with_features(torch::relu(bns[i]->forward(y.features())));
    };
    auto x = layer(layer(in, 0), 1);
    size_t i = 2;
    for (int s = 1; s < 4; ++s) {
      // Level kAdaptiveDepth - s: 2, 1, 0 (cell 0.125, 0.25, 0.5) like stride 2, 4, 8.
      x = ad::adaptive_pool(x, static_cast<uint32_t>(kAdaptiveDepth - s), ad::Reduction::Mean);
      x = layer(layer(x, i), i + 1);
      i += 2;
    }
    return head->forward(pool_avg_max(x.features(), sample_ids(x, x.features().device()), B));
  }
};
TORCH_MODULE(AdaptiveNet);

struct GaussianSetNetImpl : torch::nn::Module {
  torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr}, l4{nullptr}, head{nullptr};
  torch::nn::BatchNorm1d b1{nullptr}, b2{nullptr}, b3{nullptr}, b4{nullptr};
  explicit GaussianSetNetImpl(int cin) {
    l1 = register_module("l1", torch::nn::Linear(cin, 64));
    b1 = register_module("b1", torch::nn::BatchNorm1d(64));
    l2 = register_module("l2", torch::nn::Linear(64, 128));
    b2 = register_module("b2", torch::nn::BatchNorm1d(128));
    l3 = register_module("l3", torch::nn::Linear(128, 256));
    b3 = register_module("b3", torch::nn::BatchNorm1d(256));
    l4 = register_module("l4", torch::nn::Linear(512, 128));
    b4 = register_module("b4", torch::nn::BatchNorm1d(128));
    head = register_module("head", torch::nn::Linear(128, kClasses));
  }
  at::Tensor forward(const at::Tensor &f, const at::Tensor &sid, int64_t B) {
    auto x = torch::relu(b1->forward(l1->forward(f)));
    x = torch::relu(b2->forward(l2->forward(x)));
    x = torch::relu(b3->forward(l3->forward(x)));
    auto g = pool_avg_max(x, sid, B);
    return head->forward(torch::relu(b4->forward(l4->forward(g))));
  }
};
TORCH_MODULE(GaussianSetNet);

// ---------------------------------------------------------------- memory

double peak_rss_mb() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
#endif
  return -1.0;
}

double cuda_peak_mb(bool reserved = false) {
#ifdef USE_CUDA
  if (torch::cuda::is_available()) {
    auto st = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
    const auto &s = reserved ? st.reserved_bytes : st.allocated_bytes;
    // c10::CachingDeviceAllocator names StatType in LibTorch 2.5 (CUDA 12.4
    // profile) and re-exports c10::CachingAllocator's in 2.15.
    const auto agg = static_cast<size_t>(c10::CachingDeviceAllocator::StatType::AGGREGATE);
    return static_cast<double>(s[agg].peak) / (1024.0 * 1024.0);
  }
#endif
  (void)reserved;
  return -1.0;
}

void reset_cuda_peak() {
#ifdef USE_CUDA
  if (torch::cuda::is_available())
    c10::cuda::CUDACachingAllocator::resetPeakStats(0);
#endif
}

void sync(const torch::Device &d) {
  if (d.is_cuda())
    torch::cuda::synchronize();
}

double median(std::vector<double> v) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

} // namespace

int main(int argc, char **argv) {
  std::string method = "uniform", output = "ml_run.json", device_s = "cuda";
  int seed = 1, epochs = 12, train_per_class = 80, test_per_class = 40, npoints = 2048,
      batch_size = 16;
  uint64_t data_seed = 2026;
  int64_t density_threshold = 12;
  double lr = 1e-3;
  if ((argc - 1) % 2 != 0)
    throw std::invalid_argument("Arguments must be option/value pairs");
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--method")
      method = v;
    else if (a == "--seed")
      seed = std::stoi(v);
    else if (a == "--epochs")
      epochs = std::stoi(v);
    else if (a == "--train-per-class")
      train_per_class = std::stoi(v);
    else if (a == "--test-per-class")
      test_per_class = std::stoi(v);
    else if (a == "--points")
      npoints = std::stoi(v);
    else if (a == "--batch")
      batch_size = std::stoi(v);
    else if (a == "--data-seed")
      data_seed = std::stoull(v);
    else if (a == "--density-threshold")
      density_threshold = std::stoll(v);
    else if (a == "--lr")
      lr = std::stod(v);
    else if (a == "--device")
      device_s = v;
    else if (a == "--output")
      output = v;
    else
      throw std::invalid_argument("unknown argument " + a);
  }
  if (method != "uniform" && method != "adaptive" && method != "gaussian" &&
      method != "adaptive_gaussian")
    throw std::invalid_argument("--method uniform|adaptive|gaussian|adaptive_gaussian");
  if (batch_size < 2)
    throw std::invalid_argument("--batch must be at least 2 for training with BatchNorm");
  if (device_s != "cpu" && device_s != "cuda" && device_s != "auto")
    throw std::invalid_argument("--device cpu|cuda|auto");
  const bool cuda_available = device_s != "cpu" && torch::cuda::is_available();
  if (device_s == "cuda" && !cuda_available)
    throw std::runtime_error("--device cuda requested but CUDA is unavailable; use --device cpu "
                             "or --device auto only when CPU execution is intended");
  const torch::Device dev = device_s != "cpu" && cuda_available
                                ? torch::Device(torch::kCUDA)
                                : torch::Device(torch::kCPU);
  const bool adaptive_like = method == "adaptive" || method == "adaptive_gaussian";

  // ---- data + representation (CPU, timed per sample)
  const double t_data0 = now_s();
  auto train = make_split(train_per_class, data_seed, 0, npoints);
  auto test = make_split(test_per_class, data_seed, 1, npoints);
  const double data_s = now_s() - t_data0;
  auto convert = [&](const at::Tensor &pts) -> Rep {
    if (method == "uniform")
      return uniform_rep(pts);
    if (method == "adaptive")
      return adaptive_rep(pts, density_threshold);
    if (method == "gaussian")
      return gaussian_rep(pts);
    return adaptive_gaussian_rep(pts);
  };
  std::vector<double> conv_ms;
  auto build = [&](const std::vector<Sample> &d) {
    std::vector<Rep> r;
    r.reserve(d.size());
    for (const auto &s : d) {
      const auto t0 = Clock::now();
      r.push_back(convert(s.points));
      conv_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return r;
  };
  auto train_rep = build(train);
  auto test_rep = build(test);
  std::vector<double> active;
  std::vector<double> level_hist(kAdaptiveDepth + 1, 0.0);
  for (const auto &r : train_rep) {
    active.push_back(static_cast<double>(r.feats.size(0)));
    if (r.levels.defined()) {
      auto h = torch::bincount(r.levels.to(torch::kInt64), {}, kAdaptiveDepth + 1);
      for (int l = 0; l <= kAdaptiveDepth; ++l)
        level_hist[l] += static_cast<double>(h[l].item<int64_t>());
    }
  }
  const double active_mean = std::accumulate(active.begin(), active.end(), 0.0) / active.size();
  const int cin = static_cast<int>(train_rep[0].feats.size(1));

  // ---- model
  torch::manual_seed(seed);
  if (dev.is_cuda())
    torch::cuda::manual_seed_all(seed);
  UniformNet un{nullptr};
  AdaptiveNet an{nullptr};
  GaussianSetNet gn{nullptr};
  torch::nn::Module *model = nullptr;
  if (method == "uniform") {
    un = UniformNet(cin);
    un->to(dev);
    model = un.get();
  } else if (adaptive_like) {
    an = AdaptiveNet(cin);
    an->to(dev);
    model = an.get();
  } else {
    gn = GaussianSetNet(cin);
    gn->to(dev);
    model = gn.get();
  }
  int64_t nparams = 0;
  for (const auto &p : model->parameters())
    nparams += p.numel();
  torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(lr).weight_decay(1e-4));

  auto forward = [&](Batch &b) -> at::Tensor {
    if (method == "uniform") {
      b.manager = std::make_shared<CoordinateManager>(
          3, dev.is_cuda() ? CoordinateMapBackend::CUDA : CoordinateMapBackend::CPU);
      SparseTensor x(b.feats, b.coords, b.manager);
      return un->forward(x, b.B);
    }
    if (adaptive_like)
      return an->forward(*b.tensor, b.B);
    return gn->forward(b.feats, b.coords, b.B);
  };

  // ---- evaluation
  auto evaluate = [&](const std::vector<Rep> &reps, const std::vector<Sample> &data,
                      std::vector<std::vector<int64_t>> *confusion, double *seconds) {
    model->eval();
    torch::NoGradGuard ng;
    std::vector<std::vector<int64_t>> cm(kClasses, std::vector<int64_t>(kClasses, 0));
    sync(dev);
    const double t0 = now_s();
    for (size_t i = 0; i < data.size(); i += batch_size) {
      std::vector<int64_t> idx;
      for (size_t j = i; j < std::min(data.size(), i + batch_size); ++j)
        idx.push_back(static_cast<int64_t>(j));
      auto b = make_batch(method, reps, data, idx, dev);
      auto pred = forward(b).argmax(1).cpu();
      auto lab = b.labels.cpu();
      for (int64_t k = 0; k < b.B; ++k)
        ++cm[lab[k].item<int64_t>()][pred[k].item<int64_t>()];
    }
    sync(dev);
    if (seconds)
      *seconds = now_s() - t0;
    int64_t correct = 0, total = 0;
    for (int a = 0; a < kClasses; ++a)
      for (int p = 0; p < kClasses; ++p) {
        total += cm[a][p];
        if (a == p)
          correct += cm[a][p];
      }
    if (confusion)
      *confusion = cm;
    model->train();
    return static_cast<double>(correct) / std::max<int64_t>(total, 1);
  };

  // ---- training
  reset_cuda_peak();
  std::mt19937_64 shuffle_rng(static_cast<uint64_t>(seed) * 9973ULL);
  json epochs_j = json::array();
  sync(dev);
  const double t_train0 = now_s();
  model->train();
  for (int e = 0; e < epochs; ++e) {
    const double te0 = now_s();
    std::vector<int64_t> order(train.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), shuffle_rng);
    double loss_sum = 0;
    int64_t correct = 0, seen = 0, nb = 0;
    for (size_t i = 0; i < order.size(); i += batch_size) {
      std::vector<int64_t> idx(order.begin() + i,
                               order.begin() + std::min(order.size(), i + batch_size));
      if (idx.size() < 2)
        continue; // BatchNorm needs > 1 sample
      auto b = make_batch(method, train_rep, train, idx, dev);
      auto logits = forward(b);
      auto loss = torch::nn::functional::cross_entropy(logits, b.labels);
      opt.zero_grad();
      loss.backward();
      opt.step();
      loss_sum += loss.item<double>();
      correct += logits.argmax(1).eq(b.labels).sum().item<int64_t>();
      seen += b.B;
      ++nb;
    }
    sync(dev);
    const double epoch_s = now_s() - te0;
    const double test_acc = evaluate(test_rep, test, nullptr, nullptr);
    epochs_j.push_back({{"epoch", e + 1},
                        {"train_loss", loss_sum / std::max<int64_t>(nb, 1)},
                        {"train_acc", static_cast<double>(correct) / std::max<int64_t>(seen, 1)},
                        {"test_acc", test_acc},
                        {"epoch_train_s", epoch_s}});
    std::cout << method << " seed " << seed << " epoch " << e + 1 << " loss "
              << loss_sum / std::max<int64_t>(nb, 1) << " test_acc " << test_acc << " ("
              << epoch_s << " s)" << std::endl;
  }
  // Training time excludes the per-epoch test evaluation.
  double train_s = 0;
  for (const auto &ej : epochs_j)
    train_s += ej["epoch_train_s"].get<double>();
  // These legacy *_train output fields cover the whole epoch loop, including evaluation.
  const double train_peak_alloc = cuda_peak_mb(false), train_peak_res = cuda_peak_mb(true);
  (void)t_train0;

  // ---- final evaluation
  std::vector<std::vector<int64_t>> cm;
  double eval_s = 0;
  const double acc = evaluate(test_rep, test, &cm, &eval_s);
  double p_sum = 0, r_sum = 0, f_sum = 0;
  json per_class = json::array();
  for (int c = 0; c < kClasses; ++c) {
    int64_t tp = cm[c][c], fp = 0, fn = 0;
    for (int k = 0; k < kClasses; ++k)
      if (k != c) {
        fp += cm[k][c];
        fn += cm[c][k];
      }
    const double pr = tp + fp ? static_cast<double>(tp) / (tp + fp) : 0.0;
    const double rc = tp + fn ? static_cast<double>(tp) / (tp + fn) : 0.0;
    const double f1 = pr + rc > 0 ? 2 * pr * rc / (pr + rc) : 0.0;
    p_sum += pr, r_sum += rc, f_sum += f1;
    per_class.push_back({{"class", kClassNames[c]}, {"precision", pr}, {"recall", rc}, {"f1", f1}});
  }

  // ---- inference latency (batch 1, includes batch assembly from the precomputed
  // representation, excludes points->representation conversion) and throughput
  std::vector<double> lat;
  {
    model->eval();
    torch::NoGradGuard ng;
    for (int i = 0; i < 60 && i < static_cast<int>(test.size()); ++i) {
      sync(dev);
      const auto t0 = Clock::now();
      auto b = make_batch(method, test_rep, test, {i}, dev);
      auto out = forward(b);
      sync(dev);
      if (i >= 10) // warm-up
        lat.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
  }

  json j = {{"method", method},
            {"seed", seed},
            {"data_seed", data_seed},
            {"device", dev.is_cuda() ? "cuda" : "cpu"},
            {"epochs", epochs},
            {"batch_size", batch_size},
            {"lr", lr},
            {"train_samples", train.size()},
            {"test_samples", test.size()},
            {"points_per_sample", npoints},
            {"density_threshold", density_threshold},
            {"input_channels", cin},
            {"parameters", nparams},
            {"accuracy", acc},
            {"macro_precision", p_sum / kClasses},
            {"macro_recall", r_sum / kClasses},
            {"macro_f1", f_sum / kClasses},
            {"per_class", per_class},
            {"confusion", cm},
            {"train_time_s", train_s},
            {"data_generation_s", data_s},
            {"conversion_ms_median", median(conv_ms)},
            {"conversion_ms_total", std::accumulate(conv_ms.begin(), conv_ms.end(), 0.0)},
            {"inference_latency_ms_median_b1", median(lat)},
            {"inference_latency_ms_min_b1", lat.empty() ? 0.0 : *std::min_element(lat.begin(), lat.end())},
            {"inference_latency_ms_max_b1", lat.empty() ? 0.0 : *std::max_element(lat.begin(), lat.end())},
            {"throughput_samples_per_s", static_cast<double>(test.size()) / eval_s},
            {"peak_cuda_alloc_mb_train", train_peak_alloc},
            {"peak_cuda_reserved_mb_train", train_peak_res},
            {"peak_rss_mb", peak_rss_mb()},
            {"active_elements_mean", active_mean},
            {"active_elements_min", *std::min_element(active.begin(), active.end())},
            {"active_elements_max", *std::max_element(active.begin(), active.end())},
            {"level_histogram_train", level_hist},
            {"curve", epochs_j}};
  std::ofstream result_file(output);
  if (!result_file)
    throw std::runtime_error("Cannot open result output: " + output);
  result_file << j.dump(1) << "\n";
  result_file.close();
  if (!result_file)
    throw std::runtime_error("Failed to write result output: " + output);
  std::cout << method << " seed " << seed << " acc " << acc << " macroF1 " << f_sum / kClasses
            << " train_s " << train_s << " active " << active_mean << " params " << nparams
            << std::endl;
  return 0;
}
