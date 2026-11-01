#include "gaussian_adaptive.hpp"

#include "../adaptive/voxel_key.hpp"

#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace minkowski {
namespace gaussian {

using adaptive::AdaptiveSparseTensor;
using adaptive::AdaptiveVoxelConfig;

namespace {

using Key = adaptive::AdaptiveVoxelKey<3>;
using KeyHash = adaptive::AdaptiveVoxelKeyHash<3>;
using KeySet = std::unordered_set<Key, KeyHash>;

// Integer cell coordinates must stay exactly representable in double at the
// finest level (2^52), mirroring the adaptive coordinate range rules.
constexpr double kCoordLimit = 4503599627370496.0; // 2^52

bool key_less(const Key &a, const Key &b) {
  if (a.level != b.level)
    return a.level < b.level;
  return a.coordinates < b.coordinates;
}

Key parent_of(const Key &k) {
  Key p = k;
  p.level = k.level - 1;
  for (int d = 0; d < 3; ++d)
    p.coordinates[d] = k.coordinates[d] >> 1; // arithmetic shift == floor div
  return p;
}

// P(a <= Z < b) for a standard normal Z. erfc in the tails and erf near the
// mean keep relative accuracy both for far cells and for cells much smaller
// than the Gaussian (|a|, |b| << 1, e.g. huge splats).
double normal_segment(double a, double b) {
  constexpr double r = 0.70710678118654752440; // 1/sqrt(2)
  if (!(b > a))
    return 0.0;
  if (a >= 1.0)
    return 0.5 * (std::erfc(a * r) - std::erfc(b * r));
  if (b <= -1.0)
    return 0.5 * (std::erfc(-b * r) - std::erfc(-a * r));
  return 0.5 * (std::erf(b * r) - std::erf(a * r));
}

struct Grid {
  std::array<double, 3> origin{};
  std::array<double, 3> base{};
  uint32_t max_depth = 0;

  double size(uint32_t level, int d) const {
    return base[d] * std::ldexp(1.0, -static_cast<int>(level));
  }
  // |c| bound at `level` equivalent to |c_finest| < 2^52 at max_depth.
  double limit(uint32_t level) const {
    return std::ldexp(kCoordLimit, -static_cast<int>(max_depth - level));
  }
};

struct HostGaussian {
  std::array<double, 3> mean{};
  std::array<double, 3> axis_std{}; // sqrt(Sigma_dd), floored
  uint32_t level = 0;
};

struct Entry {
  int64_t g;
  Key key;
  double w;
};

Grid make_grid(const AdaptiveVoxelConfig &config) {
  config.validate();
  if (config.dimension() != 3)
    throw std::invalid_argument(
        "gaussians_to_adaptive: AdaptiveVoxelConfig must be 3-dimensional");
  Grid g;
  for (int d = 0; d < 3; ++d) {
    g.base[d] = config.base_voxel_size[d];
    g.origin[d] = config.origin.empty() ? 0.0 : config.origin[d];
  }
  g.max_depth = config.max_depth;
  return g;
}

void validate_options(const GaussianToAdaptiveOptions &o, const Grid &grid,
                      uint32_t &lmin, uint32_t &lmax) {
  if (!(o.cell_per_sigma > 0.0) || !std::isfinite(o.cell_per_sigma))
    throw std::invalid_argument("gaussians_to_adaptive: cell_per_sigma must be positive");
  if (!(o.support_sigma > 0.0) || !std::isfinite(o.support_sigma))
    throw std::invalid_argument("gaussians_to_adaptive: support_sigma must be positive");
  if (o.max_cells_per_axis < 1)
    throw std::invalid_argument("gaussians_to_adaptive: max_cells_per_axis must be >= 1");
  if (o.max_entries < 1)
    throw std::invalid_argument("gaussians_to_adaptive: max_entries must be >= 1");
  if (!(o.min_weight >= 0.0 && o.min_weight < 1.0))
    throw std::invalid_argument("gaussians_to_adaptive: min_weight must be in [0, 1)");
  const int64_t hi = o.max_level < 0 ? static_cast<int64_t>(grid.max_depth) : o.max_level;
  if (o.min_level < 0 || o.min_level > hi || hi > static_cast<int64_t>(grid.max_depth))
    throw std::invalid_argument(
        "gaussians_to_adaptive: need 0 <= min_level <= max_level <= config.max_depth");
  lmin = static_cast<uint32_t>(o.min_level);
  lmax = static_cast<uint32_t>(hi);
}

// Host copies (double) of means, per-axis marginal std and target levels.
std::vector<HostGaussian> host_gaussians(const GaussianTensor &gs, const Grid &grid,
                                         double cell_per_sigma, uint32_t lmin,
                                         uint32_t lmax) {
  const int64_t N = gs.size();
  std::vector<HostGaussian> out(static_cast<size_t>(N));
  if (N == 0)
    return out;
  auto pos = gs.positions().detach().to(at::kCPU, at::kDouble).contiguous();
  auto rot = gs.rotations().detach().to(at::kCPU, at::kDouble).contiguous();
  auto scl = gs.scales().detach().to(at::kCPU, at::kDouble).contiguous();
  const double *p = pos.data_ptr<double>();
  const double *q = rot.data_ptr<double>();
  const double *s = scl.data_ptr<double>();
  const double max_base = std::max({grid.base[0], grid.base[1], grid.base[2]});
  double finest = std::numeric_limits<double>::infinity();
  for (int d = 0; d < 3; ++d)
    finest = std::min(finest, grid.size(grid.max_depth, d));
  const double std_floor = 1e-12 * finest;

  std::atomic<bool> failed{false};
  std::string first_error;
  std::mutex error_mutex;
  at::parallel_for(0, N, 2048, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e && !failed.load(); ++i) {
      auto fail = [&](const std::string &msg) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!failed.exchange(true))
          first_error = msg + " (Gaussian " + std::to_string(i) + ")";
      };
      HostGaussian &h = out[static_cast<size_t>(i)];
      double smax = 0.0;
      for (int d = 0; d < 3; ++d) {
        const double v = s[i * 3 + d];
        if (!std::isfinite(v) || v < 0.0)
          return fail("gaussians_to_adaptive: scales must be finite and nonnegative");
        smax = std::max(smax, v);
        h.mean[d] = p[i * 3 + d];
        if (!std::isfinite(h.mean[d]))
          return fail("gaussians_to_adaptive: positions must be finite");
        const double c = (h.mean[d] - grid.origin[d]) / grid.size(grid.max_depth, d);
        if (!(std::fabs(c) < kCoordLimit - 2.0))
          return fail("gaussians_to_adaptive: position out of the representable range "
                      "(|coordinate| >= 2^52 cells at config.max_depth)");
      }
      double qw = q[i * 4 + 0], qx = q[i * 4 + 1], qy = q[i * 4 + 2], qz = q[i * 4 + 3];
      const double qn = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
      if (!(qn > 0.0) || !std::isfinite(qn))
        return fail("gaussians_to_adaptive: quaternions must have finite nonzero norm");
      qw /= qn, qx /= qn, qy /= qn, qz /= qn;
      const double R[3][3] = {
          {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
          {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
          {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)}};
      for (int d = 0; d < 3; ++d) {
        double var = 0.0;
        for (int k = 0; k < 3; ++k)
          var += R[d][k] * R[d][k] * s[i * 3 + k] * s[i * 3 + k];
        h.axis_std[d] = std::max(std::sqrt(var), std_floor);
      }
      // round(log2(base / (k sigma))); sigma == 0 -> +inf -> finest allowed level.
      const double lv = std::round(std::log2(max_base / (cell_per_sigma * smax)));
      double clamped = std::isnan(lv) ? static_cast<double>(lmax) : lv;
      clamped = std::min(std::max(clamped, static_cast<double>(lmin)), static_cast<double>(lmax));
      h.level = static_cast<uint32_t>(clamped);
    }
  });
  if (failed.load())
    throw std::invalid_argument(first_error);
  return out;
}

Key locate(const Grid &grid, const std::array<double, 3> &x, uint32_t level) {
  Key k{};
  k.level = level;
  for (int d = 0; d < 3; ++d)
    k.coordinates[d] =
        static_cast<int64_t>(std::floor((x[d] - grid.origin[d]) / grid.size(level, d)));
  return k;
}

// Deposit entries of every Gaussian at its target level.
std::vector<Entry> deposits(const std::vector<HostGaussian> &hg, const Grid &grid,
                            const GaussianToAdaptiveOptions &o) {
  const int64_t N = static_cast<int64_t>(hg.size());
  std::vector<Entry> out;
  if (o.splat == GaussianSplat::Center) {
    out.resize(static_cast<size_t>(N));
    at::parallel_for(0, N, 4096, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i)
        out[static_cast<size_t>(i)] = Entry{i, locate(grid, hg[i].mean, hg[i].level), 1.0};
    });
    return out;
  }

  // BoxIntegral: per-axis cell ranges and 1-D weights; entries = outer product.
  struct Range {
    std::array<int64_t, 3> lo{};
    std::array<int64_t, 3> n{};
  };
  std::vector<Range> ranges(static_cast<size_t>(N));
  std::vector<int64_t> offset(static_cast<size_t>(N) + 1, 0);
  for (int64_t i = 0; i < N; ++i) {
    const HostGaussian &h = hg[i];
    Range &r = ranges[i];
    int64_t count = 1;
    for (int d = 0; d < 3; ++d) {
      const double s = grid.size(h.level, d);
      const double rel = (h.mean[d] - grid.origin[d]) / s;
      const double ext = o.support_sigma * h.axis_std[d] / s;
      const double c_mean = std::floor(rel);
      double c_lo = std::floor(rel - ext), c_hi = std::floor(rel + ext);
      if (!std::isfinite(c_lo) || !std::isfinite(c_hi) ||
          c_hi - c_lo + 1.0 > static_cast<double>(o.max_cells_per_axis)) {
        c_lo = c_mean - static_cast<double>((o.max_cells_per_axis - 1) / 2);
        c_hi = c_lo + static_cast<double>(o.max_cells_per_axis - 1);
      }
      const double lim = grid.limit(h.level);
      if (!(std::fabs(c_lo) < lim - 1.0) || !(std::fabs(c_hi) < lim - 1.0))
        throw std::invalid_argument(
            "gaussians_to_adaptive: splat support out of the representable range "
            "(|coordinate| >= 2^52 cells at config.max_depth)");
      r.lo[d] = static_cast<int64_t>(c_lo);
      r.n[d] = static_cast<int64_t>(c_hi - c_lo) + 1;
      count *= r.n[d];
    }
    offset[i + 1] = offset[i] + count;
    if (offset[i + 1] > o.max_entries)
      throw std::length_error("gaussians_to_adaptive: BoxIntegral splat needs more than "
                              "max_entries Gaussian-cell entries; lower support_sigma, "
                              "max_cells_per_axis or raise max_entries");
  }
  out.resize(static_cast<size_t>(offset[N]));
  at::parallel_for(0, N, 256, [&](int64_t b, int64_t e) {
    std::array<std::vector<double>, 3> wa;
    for (int64_t i = b; i < e; ++i) {
      const HostGaussian &h = hg[i];
      const Range &r = ranges[i];
      bool degenerate = false;
      for (int d = 0; d < 3; ++d) {
        const double s = grid.size(h.level, d);
        wa[d].assign(static_cast<size_t>(r.n[d]), 0.0);
        double sum = 0.0;
        for (int64_t c = 0; c < r.n[d]; ++c) {
          const double lo = grid.origin[d] + static_cast<double>(r.lo[d] + c) * s;
          const double v = normal_segment((lo - h.mean[d]) / h.axis_std[d],
                                          (lo + s - h.mean[d]) / h.axis_std[d]);
          wa[d][c] = v;
          sum += v;
        }
        if (!(sum > 0.0)) {
          degenerate = true;
          break;
        }
        for (double &v : wa[d])
          v /= sum;
      }
      int64_t k = offset[i];
      if (degenerate) { // fall back to the mean's cell; remaining slots get zero weight
        const Key km = locate(grid, h.mean, h.level);
        for (; k < offset[i + 1]; ++k)
          out[k] = Entry{i, km, 0.0};
        out[offset[i]].w = 1.0;
        continue;
      }
      for (int64_t a = 0; a < r.n[0]; ++a)
        for (int64_t bb = 0; bb < r.n[1]; ++bb)
          for (int64_t c = 0; c < r.n[2]; ++c) {
            Entry &en = out[k++];
            en.g = i;
            en.key.level = h.level;
            en.key.coordinates = {r.lo[0] + a, r.lo[1] + bb, r.lo[2] + c};
            en.w = wa[0][a] * wa[1][bb] * wa[2][c];
          }
    }
  });
  out.erase(std::remove_if(out.begin(), out.end(), [](const Entry &e) { return !(e.w > 0.0); }),
            out.end());
  return out;
}

// Pushes deposits on subdivided cells (cells containing a finer deposit) down to
// leaves, splitting by the Gaussian's axis-aligned integral over each child.
std::vector<Entry> resolve(std::vector<Entry> entries, const std::vector<HostGaussian> &hg,
                           const Grid &grid) {
  KeySet targets;
  targets.reserve(entries.size());
  uint32_t min_level = std::numeric_limits<uint32_t>::max(), max_level = 0;
  for (const Entry &e : entries) {
    targets.insert(e.key);
    min_level = std::min(min_level, e.key.level);
    max_level = std::max(max_level, e.key.level);
  }
  if (targets.empty() || min_level == max_level)
    return entries; // one level: no deposit cell can contain another

  // Strict ancestors (down to the coarsest deposit level) of every deposit cell.
  KeySet ancestors;
  for (const Key &k : targets) {
    Key a = k;
    while (a.level > min_level) {
      a = parent_of(a);
      if (!ancestors.insert(a).second)
        break; // its ancestors are already in
    }
  }

  std::vector<Entry> out;
  out.reserve(entries.size());
  std::vector<Entry> stack;
  for (const Entry &e : entries) {
    if (!ancestors.count(e.key)) {
      out.push_back(e);
      continue;
    }
    stack.push_back(e);
    while (!stack.empty()) {
      const Entry cur = stack.back();
      stack.pop_back();
      const HostGaussian &h = hg[cur.g];
      // Per-axis share of the lower / upper half of the cell.
      std::array<std::array<double, 2>, 3> share{};
      for (int d = 0; d < 3; ++d) {
        const double s = grid.size(cur.key.level, d);
        const double lo = grid.origin[d] + static_cast<double>(cur.key.coordinates[d]) * s;
        const double a = (lo - h.mean[d]) / h.axis_std[d];
        const double m = (lo + 0.5 * s - h.mean[d]) / h.axis_std[d];
        const double b = (lo + s - h.mean[d]) / h.axis_std[d];
        const double low = normal_segment(a, m), high = normal_segment(m, b);
        const double tot = low + high;
        if (tot > 0.0 && std::isfinite(tot))
          share[d] = {low / tot, high / tot};
        else
          share[d] = {0.5, 0.5};
      }
      for (uint32_t c = 0; c < 8; ++c) {
        Entry child{cur.g, Key{}, cur.w};
        child.key.level = cur.key.level + 1;
        for (int d = 0; d < 3; ++d) {
          const uint32_t bit = (c >> d) & 1u;
          child.key.coordinates[d] = 2 * cur.key.coordinates[d] + bit;
          child.w *= share[d][bit];
        }
        if (!(child.w > 0.0))
          continue;
        if (ancestors.count(child.key))
          stack.push_back(child);
        else
          out.push_back(child);
      }
    }
  }
  return out;
}

// Drops entries below min_weight (keeping each Gaussian's largest) and
// renormalizes every Gaussian's kept weights to sum to one.
void prune_weights(std::vector<Entry> &entries, int64_t n_gaussians, double min_weight) {
  if (!(min_weight > 0.0) || entries.empty())
    return;
  std::vector<double> wmax(static_cast<size_t>(n_gaussians), 0.0);
  for (const Entry &e : entries)
    wmax[e.g] = std::max(wmax[e.g], e.w);
  bool any = false;
  for (const Entry &e : entries)
    if (e.w < min_weight && e.w < wmax[e.g]) {
      any = true;
      break;
    }
  if (!any)
    return;
  entries.erase(std::remove_if(entries.begin(), entries.end(),
                               [&](const Entry &e) {
                                 return e.w < min_weight && e.w < wmax[e.g];
                               }),
                entries.end());
  std::vector<double> sum(static_cast<size_t>(n_gaussians), 0.0);
  for (const Entry &e : entries)
    sum[e.g] += e.w;
  for (Entry &e : entries)
    e.w /= sum[e.g];
}

} // namespace

AdaptiveSparseTensor gaussians_to_adaptive(const GaussianTensor &gaussians,
                                           const AdaptiveVoxelConfig &config,
                                           const GaussianToAdaptiveOptions &options,
                                           GaussianAdaptiveMap *map) {
  const Grid grid = make_grid(config);
  uint32_t lmin = 0, lmax = 0;
  validate_options(options, grid, lmin, lmax);

  const auto hg = host_gaussians(gaussians, grid, options.cell_per_sigma, lmin, lmax);
  auto entries = resolve(deposits(hg, grid, options), hg, grid);
  prune_weights(entries, static_cast<int64_t>(hg.size()), options.min_weight);

  // Leaves in the adaptive export order (level, coordinates).
  std::vector<Key> leaves;
  {
    KeySet seen;
    seen.reserve(entries.size());
    for (const Entry &e : entries)
      if (seen.insert(e.key).second)
        leaves.push_back(e.key);
  }
  std::sort(leaves.begin(), leaves.end(), key_less);
  std::unordered_map<Key, int64_t, KeyHash> row;
  row.reserve(leaves.size());
  for (size_t j = 0; j < leaves.size(); ++j)
    row.emplace(leaves[j], static_cast<int64_t>(j));

  const int64_t L = static_cast<int64_t>(leaves.size());
  const int64_t E = static_cast<int64_t>(entries.size());
  auto coords = torch::empty({L, 3}, torch::kInt64);
  auto levels = torch::empty({L}, torch::kInt32);
  {
    auto *c = coords.data_ptr<int64_t>();
    auto *l = levels.data_ptr<int32_t>();
    for (int64_t j = 0; j < L; ++j) {
      for (int d = 0; d < 3; ++d)
        c[j * 3 + d] = leaves[j].coordinates[d];
      l[j] = static_cast<int32_t>(leaves[j].level);
    }
  }
  auto gidx = torch::empty({E}, torch::kInt64);
  auto lidx = torch::empty({E}, torch::kInt64);
  auto wts = torch::empty({E}, torch::kFloat64);
  {
    auto *gp = gidx.data_ptr<int64_t>();
    auto *lp = lidx.data_ptr<int64_t>();
    auto *wp = wts.data_ptr<double>();
    for (int64_t m = 0; m < E; ++m) {
      gp[m] = entries[m].g;
      lp[m] = row.at(entries[m].key);
      wp[m] = entries[m].w;
    }
  }

  // Differentiable aggregation on the Gaussians' device and dtype.
  const auto dev = gaussians.device();
  const auto opts = gaussians.positions().options();
  auto g_d = gidx.to(dev);
  auto l_d = lidx.to(dev);
  auto w_d = wts.to(opts.dtype()).to(dev);
  auto mass_i = gaussians.opacities().select(1, 0);
  if (options.weight_by_volume)
    mass_i = mass_i * gaussians.scales().prod(1) * 15.749609945722419 /* (2 pi)^{3/2} */;
  auto contrib = mass_i.index_select(0, g_d) * w_d; // [E]
  auto mass = at::zeros({L}, opts).index_add(0, l_d, contrib);
  std::vector<at::Tensor> parts{gaussians.colors()};
  if (options.include_features && gaussians.feature_channels() > 0)
    parts.push_back(gaussians.features());
  auto attrs = parts.size() == 1 ? parts[0] : at::cat(parts, 1);
  auto num = at::zeros({L, attrs.size(1)}, opts)
                 .index_add(0, l_d, attrs.index_select(0, g_d) * contrib.unsqueeze(1));
  if (options.mean_attributes) {
    auto nonzero = mass != 0;
    auto den = at::where(nonzero, mass, at::ones_like(mass));
    num = at::where(nonzero.unsqueeze(1), num / den.unsqueeze(1), at::zeros_like(num));
  }
  auto feats = at::cat({mass.unsqueeze(1), num}, 1);

  auto out = AdaptiveSparseTensor::from_voxels(coords, levels, feats, config);

  if (map) {
    auto gl = torch::empty({static_cast<int64_t>(hg.size())}, torch::kInt32);
    auto *glp = gl.data_ptr<int32_t>();
    for (size_t i = 0; i < hg.size(); ++i)
      glp[i] = static_cast<int32_t>(hg[i].level);
    map->gaussian_level = gl;
    map->gaussian_index = gidx;
    map->leaf_index = lidx;
    map->weight = wts;
  }
  return out;
}

GaussianTensor adaptive_to_gaussians(const AdaptiveSparseTensor &tensor,
                                     const AdaptiveToGaussiansOptions &options) {
  if (tensor.dimension() != 3)
    throw std::invalid_argument("adaptive_to_gaussians: tensor must be 3-dimensional");
  if (!(options.sigma_per_cell > 0.0) || !std::isfinite(options.sigma_per_cell))
    throw std::invalid_argument("adaptive_to_gaussians: sigma_per_cell must be positive");
  auto f = tensor.features();
  if (!f.is_floating_point())
    f = f.to(at::kFloat);
  const int64_t C = f.size(1);
  if (options.mass_channel >= C || options.mass_channel < -1)
    throw std::invalid_argument("adaptive_to_gaussians: mass_channel out of range");
  if (options.color_channel >= 0 ? options.color_channel + 3 > C : options.color_channel < -1)
    throw std::invalid_argument("adaptive_to_gaussians: color_channel needs 3 channels");
  if (options.feature_channel > C || options.feature_channel < -1)
    throw std::invalid_argument("adaptive_to_gaussians: feature_channel out of range");

  const auto dev = f.device();
  const auto opts = f.options();
  auto positions = tensor.centers().to(opts.dtype()).to(dev);
  auto scales = (tensor.voxel_sizes() * options.sigma_per_cell).to(opts.dtype()).to(dev);
  const bool has_mass = options.mass_channel >= 0;
  if (has_mass) {
    auto keep = at::nonzero(f.select(1, options.mass_channel) > options.min_mass).reshape({-1});
    if (keep.numel() != f.size(0)) {
      f = f.index_select(0, keep);
      positions = positions.index_select(0, keep);
      scales = scales.index_select(0, keep);
    }
  }
  const int64_t K = f.size(0);
  auto rotations = at::zeros({K, 4}, opts);
  rotations.select(1, 0).fill_(1.0);
  at::Tensor opacities;
  if (has_mass && options.opacity == AdaptiveOpacity::FromMass)
    // -expm1(-m) == 1 - exp(-m) without cancellation: tiny masses (far splat
    // tails) keep a nonzero opacity instead of rounding to 0.
    opacities = -at::expm1(-f.narrow(1, options.mass_channel, 1).clamp_min(0.0));
  else
    opacities = at::full({K, 1}, options.constant_opacity, opts);
  auto colors = options.color_channel >= 0 ? f.narrow(1, options.color_channel, 3)
                                           : at::full({K, 3}, options.default_color, opts);
  auto features = options.feature_channel >= 0
                      ? f.narrow(1, options.feature_channel, C - options.feature_channel)
                      : at::empty({K, 0}, opts);
  return GaussianTensor(positions, rotations, scales, opacities, colors, features);
}

at::Tensor find_leaves(const AdaptiveSparseTensor &tensor, const at::Tensor &points) {
  if (tensor.dimension() != 3)
    throw std::invalid_argument("find_leaves: tensor must be 3-dimensional");
  if (points.dim() != 2 || points.size(1) != 3)
    throw std::invalid_argument("find_leaves: points must have shape [P, 3]");
  auto pts = points.detach().to(at::kCPU, at::kDouble).contiguous();
  const int64_t P = pts.size(0);
  auto out = torch::full({P}, -1, torch::kInt64);
  if (P == 0 || tensor.size() == 0)
    return out;
  const auto &cfg = tensor.config();
  Grid grid;
  for (int d = 0; d < 3; ++d) {
    grid.base[d] = cfg.base_voxel_size[d];
    grid.origin[d] = cfg.origin.empty() ? 0.0 : cfg.origin[d];
  }
  grid.max_depth = cfg.max_depth;
  std::vector<uint32_t> present;
  const auto counts = tensor.counts_by_level();
  for (size_t l = 0; l < counts.size(); ++l)
    if (counts[l] > 0)
      present.push_back(static_cast<uint32_t>(l));
  const double *p = pts.data_ptr<double>();
  int64_t *o = out.data_ptr<int64_t>();
  at::parallel_for(0, P, 1024, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      for (uint32_t l : present) {
        std::array<int64_t, 3> c{};
        bool ok = true;
        for (int d = 0; d < 3; ++d) {
          const double v = std::floor((p[i * 3 + d] - grid.origin[d]) / grid.size(l, d));
          if (!std::isfinite(v) || std::fabs(v) > 4e18) {
            ok = false;
            break;
          }
          c[d] = static_cast<int64_t>(v);
        }
        if (!ok)
          break;
        const int64_t j = tensor.index_of(c, l);
        if (j >= 0) {
          o[i] = j;
          break;
        }
      }
    }
  });
  return out;
}

AdaptiveSparseTensor refine_by_gaussian_density(const AdaptiveSparseTensor &tensor,
                                                const GaussianTensor &gaussians,
                                                int64_t max_gaussians_per_leaf, int max_level) {
  if (tensor.dimension() != 3)
    throw std::invalid_argument("refine_by_gaussian_density: tensor must be 3-dimensional");
  if (max_gaussians_per_leaf < 0)
    throw std::invalid_argument("refine_by_gaussian_density: threshold must be >= 0");
  const auto &cfg = tensor.config();
  if (max_level > static_cast<int>(cfg.max_depth))
    throw std::invalid_argument("refine_by_gaussian_density: max_level > config.max_depth");
  const uint32_t top = max_level < 0 ? cfg.max_depth : static_cast<uint32_t>(max_level);
  AdaptiveSparseTensor out = tensor.with_features(tensor.features());
  if (tensor.size() == 0 || gaussians.size() == 0 || top == 0)
    return out;
  Grid grid;
  for (int d = 0; d < 3; ++d) {
    grid.base[d] = cfg.base_voxel_size[d];
    grid.origin[d] = cfg.origin.empty() ? 0.0 : cfg.origin[d];
  }
  grid.max_depth = cfg.max_depth;

  // Gaussian-mean counts per cell for every level that may still be split.
  auto pos = gaussians.positions().detach().to(at::kCPU, at::kDouble).contiguous();
  const double *p = pos.data_ptr<double>();
  const int64_t N = gaussians.size();
  std::vector<std::unordered_map<Key, int64_t, KeyHash>> counts(top);
  for (uint32_t l = 0; l < top; ++l) {
    auto &m = counts[l];
    m.reserve(static_cast<size_t>(std::min<int64_t>(N, int64_t{1} << std::min(3 * l, 24u))));
    for (int64_t i = 0; i < N; ++i) {
      Key k{};
      k.level = l;
      bool ok = true;
      for (int d = 0; d < 3; ++d) {
        const double v = std::floor((p[i * 3 + d] - grid.origin[d]) / grid.size(l, d));
        if (!std::isfinite(v) || std::fabs(v) > 4e18) {
          ok = false;
          break;
        }
        k.coordinates[d] = static_cast<int64_t>(v);
      }
      if (ok)
        ++m[k];
    }
  }
  out.refine_if([&](int64_t, const std::array<double, 4> &lo, const std::array<double, 4> &hi,
                    uint32_t level) {
    if (level >= top)
      return false;
    Key k{};
    k.level = level;
    for (int d = 0; d < 3; ++d) // the center avoids rounding at the box faces
      k.coordinates[d] = static_cast<int64_t>(
          std::floor((0.5 * (lo[d] + hi[d]) - grid.origin[d]) / grid.size(level, d)));
    const auto it = counts[level].find(k);
    return it != counts[level].end() && it->second > max_gaussians_per_leaf;
  });
  return out;
}

AdaptiveVoxelConfig adaptive_config_for(const GaussianTensor &gaussians, uint32_t max_depth,
                                        double cell_per_sigma, double fine_quantile) {
  if (gaussians.size() == 0)
    throw std::invalid_argument("adaptive_config_for: no Gaussians");
  if (!(cell_per_sigma > 0.0) || !std::isfinite(cell_per_sigma))
    throw std::invalid_argument("adaptive_config_for: cell_per_sigma must be positive");
  if (!(fine_quantile >= 0.0 && fine_quantile <= 1.0))
    throw std::invalid_argument("adaptive_config_for: fine_quantile must be in [0, 1]");
  if (max_depth > AdaptiveVoxelConfig::kMaxSupportedDepth)
    throw std::invalid_argument("adaptive_config_for: max_depth too large");
  auto smax = std::get<0>(gaussians.scales().detach().to(at::kCPU, at::kDouble).max(1));
  auto sorted = std::get<0>(smax.sort());
  const int64_t n = sorted.numel();
  const int64_t qi = std::min<int64_t>(n - 1, static_cast<int64_t>(fine_quantile * (n - 1) + 0.5));
  double sigma = sorted[qi].item<double>();
  if (!(sigma > 0.0)) { // degenerate quantile: smallest positive scale, else scene extent
    auto pos = sorted.masked_select(sorted > 0);
    sigma = pos.numel() > 0 ? pos.min().item<double>() : 0.0;
  }
  auto p = gaussians.positions().detach().to(at::kCPU, at::kDouble);
  auto lo = std::get<0>(p.min(0));
  auto hi = std::get<0>(p.max(0));
  double finest = cell_per_sigma * sigma;
  if (!(finest > 0.0) || !std::isfinite(finest)) {
    const double extent = (hi - lo).max().item<double>();
    finest = extent > 0.0 ? std::ldexp(extent, -static_cast<int>(max_depth)) : 1.0;
  }
  AdaptiveVoxelConfig cfg;
  const double base = std::ldexp(finest, static_cast<int>(max_depth));
  cfg.base_voxel_size = {base, base, base};
  cfg.min_voxel_size = {finest, finest, finest};
  cfg.origin = {lo[0].item<double>(), lo[1].item<double>(), lo[2].item<double>()};
  cfg.max_depth = max_depth;
  cfg.validate();
  return cfg;
}

} // namespace gaussian
} // namespace minkowski
