#include "ops.hpp"
#include "adaptive_topology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace minkowski {
namespace adaptive {

using detail::DVec;
using detail::Key;
using detail::Topology;

namespace {

void check_weight(const at::Tensor &weight, const AdaptiveKernelMap &m, int64_t cin) {
  if (weight.dim() != 3 || weight.size(0) != m.num_kernels || weight.size(1) != cin)
    throw std::invalid_argument("weight must have shape [kernel_size^D, in_channels, out_channels]");
}

// Gather -> (optional per-pair weight) -> GEMM per kernel slot -> scatter-add.
// `from`/`to` select the direction so the transposed convolution reuses the map.
at::Tensor apply_map(const at::Tensor &x, const at::Tensor &weight, const AdaptiveKernelMap &m,
                     const at::Tensor &from, const at::Tensor &to, int64_t n_out,
                     bool weighted) {
  const auto dev = x.device();
  const int64_t cout = weight.size(2);
  auto from_d = from.to(dev);
  auto to_d = to.to(dev);
  at::Tensor w_d;
  if (weighted)
    w_d = m.weight.to(dev).to(x.scalar_type());
  auto wt = weight.to(x.scalar_type());
  auto y = torch::zeros({n_out, cout}, x.options());
  for (int64_t k = 0; k < m.num_kernels; ++k) {
    const int64_t b = m.kernel_offsets[k], e = m.kernel_offsets[k + 1];
    if (b == e)
      continue;
    auto xs = x.index_select(0, from_d.slice(0, b, e));
    if (weighted)
      xs = xs * w_d.slice(0, b, e).unsqueeze(1);
    y.index_add_(0, to_d.slice(0, b, e), xs.matmul(wt[k]));
  }
  return y;
}

// InterpMode::ContinuousMultilinear: weights phi_i(q) / sum_j phi_j(q) with the tent
// phi_i(q) = prod_d max(0, 1 - |q_d - c_id| / s_id) of every leaf i (centre c_i, edge s_i).
// At level l the tents that reach q belong to the cells overlapping the open box q +- s_l / 2:
// with a = (q - origin) / s_l - 1/2 per axis these are cells floor(a) and floor(a) + 1, with
// tent factors 1 - frac(a) and frac(a). The descent starts at the coarsest present level and
// stops below a level where none of the (at most 2^D) candidate cells is an ancestor of a
// leaf: every finer candidate lies inside a candidate of the level above.
at::Tensor interpolate_tent(const AdaptiveSparseTensor &input, const at::Tensor &points) {
  const Topology &t = *input.topology();
  auto pts = points.to(torch::kFloat64).cpu().contiguous();
  if (pts.dim() != 2 || pts.size(1) != t.D)
    throw std::invalid_argument("points must have shape [M, D]");
  const int64_t M = pts.size(0);
  const int D = t.D;
  const double *p = pts.data_ptr<double>();

  struct Pairs {
    std::vector<int64_t> pt_idx, leaf_idx;
    std::vector<double> w;
  };
  int64_t n_levels = 0;
  if (M > 0 && t.size() > 0) {
    t.node_table(); // build the shared lookup table once, before the workers start
    n_levels = static_cast<int64_t>(t.present_levels.back() - t.present_levels.front()) + 1;
  }
  const int blocks = detail::parallel_block_count(M, M * (n_levels << D));
  std::vector<Pairs> parts(static_cast<size_t>(blocks));
  detail::run_parallel_blocks(M, blocks, [&](int b, int64_t m_begin, int64_t m_end) {
    auto &pt_idx = parts[b].pt_idx, &leaf_idx = parts[b].leaf_idx;
    auto &w = parts[b].w;
    if (n_levels == 0)
      return;
    const Topology::NodeTable &nodes = t.node_table();
    const uint32_t lmin = t.present_levels.front(), lmax = t.present_levels.back();
    const uint32_t corners = 1u << D;
    std::vector<std::pair<int64_t, double>> hits;
    for (int64_t m = m_begin; m < m_end; ++m) {
      DVec q{};
      for (int d = 0; d < D; ++d)
        q[d] = p[m * D + d];
      const int64_t j = t.find_containing(q);
      if (j < 0)
        continue;
      hits.clear();
      double total = 0.0;
      for (uint32_t l = lmin; l <= lmax; ++l) {
        const DVec s = t.size_at(l);
        std::array<int64_t, detail::kMaxDim> i0{};
        std::array<std::array<double, 2>, detail::kMaxDim> tent{};
        bool ok = true;
        for (int d = 0; d < D; ++d) {
          const double a = (q[d] - t.origin[d]) / s[d] - 0.5;
          const double fl = std::floor(a);
          if (!std::isfinite(fl) || std::fabs(fl) > 4e18) {
            ok = false; // not representable here, nor at any finer level
            break;
          }
          i0[d] = static_cast<int64_t>(fl);
          const double f = a - fl;
          tent[d] = {1.0 - f, f};
        }
        if (!ok)
          break;
        bool internal = false;
        Key k{};
        k.level = l;
        for (uint32_t c = 0; c < corners; ++c) {
          double phi = 1.0;
          for (int d = 0; d < D; ++d) {
            const uint32_t bit = (c >> d) & 1u;
            k.coordinates[d] = i0[d] + bit;
            phi *= tent[d][bit];
          }
          const int64_t v = nodes.lookup(k);
          if (v == Topology::NodeTable::kInternal) {
            internal = true;
          } else if (v >= 0 && phi > 0.0) {
            hits.emplace_back(v, phi);
            total += phi;
          }
        }
        if (!internal)
          break;
      }
      if (!(total > 0.0)) { // unreachable in exact arithmetic (phi_j >= 2^-D); be safe
        hits.assign(1, {j, 1.0});
        total = 1.0;
      }
      std::sort(hits.begin(), hits.end(),
                [](const auto &x, const auto &y) { return x.first < y.first; });
      for (auto &h : hits) {
        pt_idx.push_back(m);
        leaf_idx.push_back(h.first);
        w.push_back(h.second / total);
      }
    }
  });
  std::vector<int64_t> pt_idx, leaf_idx;
  std::vector<double> w;
  for (auto &part : parts) {
    pt_idx.insert(pt_idx.end(), part.pt_idx.begin(), part.pt_idx.end());
    leaf_idx.insert(leaf_idx.end(), part.leaf_idx.begin(), part.leaf_idx.end());
    w.insert(w.end(), part.w.begin(), part.w.end());
  }

  const auto &x = input.features();
  auto y = torch::zeros({M, x.size(1)}, x.options());
  const int64_t n = static_cast<int64_t>(pt_idx.size());
  if (n == 0) { // empty input or no point inside a leaf: zeros still connected to the features
    auto none = torch::empty({0}, torch::TensorOptions(torch::kInt64).device(x.device()));
    return y.index_add_(0, none, x.index_select(0, none));
  }
  auto pt_t = torch::from_blob(pt_idx.data(), {n}, torch::kInt64).clone().to(x.device());
  auto lf_t = torch::from_blob(leaf_idx.data(), {n}, torch::kInt64).clone().to(x.device());
  auto w_t = torch::from_blob(w.data(), {n}, torch::kFloat64)
                 .clone()
                 .to(x.scalar_type())
                 .to(x.device())
                 .unsqueeze(1);
  y.index_add_(0, pt_t, x.index_select(0, lf_t) * w_t);
  return y;
}

} // namespace

AdaptiveSparseTensor adaptive_convolution(const AdaptiveSparseTensor &input,
                                          const at::Tensor &weight,
                                          const AdaptiveKernelConfig &config) {
  auto entry = KernelMapCache::instance().get_or_build(input, config);
  check_weight(weight, entry.map, input.channels());
  auto y = apply_map(input.features(), weight, entry.map, entry.map.in_index,
                     entry.map.out_index, entry.out_topology->size(),
                     config.mode != AdaptiveConvMode::CenterSampled);
  return AdaptiveSparseTensor::from_topology(entry.out_topology, y);
}

AdaptiveSparseTensor adaptive_convolution_transpose(const AdaptiveSparseTensor &input,
                                                    const at::Tensor &weight,
                                                    const AdaptiveKernelConfig &config,
                                                    const AdaptiveSparseTensor &reference) {
  // The forward convolution applied to `reference` defines the pairs.
  auto entry = KernelMapCache::instance().get_or_build(reference, config);
  if (!detail::same_topology(*entry.out_topology, *input.topology()))
    throw std::invalid_argument(
        "transposed convolution input does not match the forward convolution output topology");
  check_weight(weight, entry.map, input.channels());
  auto y = apply_map(input.features(), weight, entry.map, entry.map.out_index,
                     entry.map.in_index, reference.size(),
                     config.mode != AdaptiveConvMode::CenterSampled);
  return AdaptiveSparseTensor::from_topology(reference.topology(), y);
}

at::Tensor adaptive_global_avg_pool(const AdaptiveSparseTensor &input, bool volume_weighted) {
  const auto &f = input.features();
  if (input.size() == 0)
    return torch::zeros({f.size(1)}, f.options());
  if (!volume_weighted)
    return f.mean(0);
  auto v = input.volumes().to(f.device()).to(f.scalar_type());
  return (f * v.unsqueeze(1)).sum(0) / v.sum();
}

AdaptiveSparseTensor adaptive_pool(const AdaptiveSparseTensor &input, uint32_t target_level,
                                   Reduction reduction) {
  AdaptiveSparseTensor out = input;
  out.coarsen(AxisAlignedRegion::everything(static_cast<size_t>(input.dimension())), target_level,
              reduction);
  return out;
}

at::Tensor interpolate(const AdaptiveSparseTensor &input, const at::Tensor &points,
                       InterpMode mode) {
  if (mode == InterpMode::ContinuousMultilinear)
    return interpolate_tent(input, points);
  const Topology &t = *input.topology();
  auto pts = points.to(torch::kFloat64).cpu().contiguous();
  if (pts.dim() != 2 || pts.size(1) != t.D)
    throw std::invalid_argument("points must have shape [M, D]");
  const int64_t M = pts.size(0);
  const double *p = pts.data_ptr<double>();

  // Points are split into contiguous blocks handled on separate threads; concatenating the
  // blocks in order gives exactly the serial (point, then leaf) order.
  struct Pairs {
    std::vector<int64_t> pt_idx, leaf_idx;
    std::vector<double> w;
  };
  if (M > 0 && t.size() > 0)
    t.node_table(); // build the shared lookup table once, before the workers start
  const int blocks = detail::parallel_block_count(
      M, M * (mode == InterpMode::Nearest ? 1 : 8));
  std::vector<Pairs> parts(static_cast<size_t>(blocks));
  detail::run_parallel_blocks(M, blocks, [&](int b, int64_t m_begin, int64_t m_end) {
    auto &pt_idx = parts[b].pt_idx, &leaf_idx = parts[b].leaf_idx;
    auto &w = parts[b].w;
    std::vector<std::pair<int64_t, double>> hits;
    for (int64_t m = m_begin; m < m_end; ++m) {
      DVec q{};
      for (int d = 0; d < t.D; ++d)
        q[d] = p[m * t.D + d];
      const int64_t j = t.find_containing(q);
      if (j < 0)
        continue;
      if (mode == InterpMode::Nearest) {
        pt_idx.push_back(m);
        leaf_idx.push_back(j);
        w.push_back(1.0);
        continue;
      }
      const DVec s = t.size_at(t.keys[j].level);
      DVec lo{}, hi{};
      for (int d = 0; d < t.D; ++d) {
        lo[d] = q[d] - 0.5 * s[d];
        hi[d] = q[d] + 0.5 * s[d];
      }
      hits.clear();
      double total = 0.0;
      t.for_each_overlap(lo, hi, [&](int64_t leaf, double ov) {
        hits.emplace_back(leaf, ov);
        total += ov;
      });
      for (auto &h : hits) {
        pt_idx.push_back(m);
        leaf_idx.push_back(h.first);
        w.push_back(h.second / total);
      }
    }
  });
  std::vector<int64_t> pt_idx, leaf_idx;
  std::vector<double> w;
  if (blocks == 1) {
    pt_idx.swap(parts[0].pt_idx);
    leaf_idx.swap(parts[0].leaf_idx);
    w.swap(parts[0].w);
  } else {
    for (auto &part : parts) {
      pt_idx.insert(pt_idx.end(), part.pt_idx.begin(), part.pt_idx.end());
      leaf_idx.insert(leaf_idx.end(), part.leaf_idx.begin(), part.leaf_idx.end());
      w.insert(w.end(), part.w.begin(), part.w.end());
    }
  }

  const auto &x = input.features();
  auto y = torch::zeros({M, x.size(1)}, x.options());
  if (pt_idx.empty())
    return y;
  const int64_t n = static_cast<int64_t>(pt_idx.size());
  auto pt_t = torch::from_blob(pt_idx.data(), {n}, torch::kInt64).clone().to(x.device());
  auto lf_t = torch::from_blob(leaf_idx.data(), {n}, torch::kInt64).clone().to(x.device());
  auto w_t = torch::from_blob(w.data(), {n}, torch::kFloat64)
                 .clone()
                 .to(x.scalar_type())
                 .to(x.device())
                 .unsqueeze(1);
  y.index_add_(0, pt_t, x.index_select(0, lf_t) * w_t);
  return y;
}

} // namespace adaptive
} // namespace minkowski
