#include "ops.hpp"
#include "adaptive_topology.hpp"

#include <stdexcept>

namespace minkowski {
namespace adaptive {

using detail::DVec;
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
  bool any = false;
  for (int64_t k = 0; k < m.num_kernels; ++k) {
    const int64_t b = m.kernel_offsets[k], e = m.kernel_offsets[k + 1];
    if (b == e)
      continue;
    any = true;
    auto xs = x.index_select(0, from_d.slice(0, b, e));
    if (weighted)
      xs = xs * w_d.slice(0, b, e).unsqueeze(1);
    y.index_add_(0, to_d.slice(0, b, e), xs.matmul(wt[k]));
  }
  // No pairs (e.g. an empty input): keep y connected to x and weight so a training step on an
  // empty batch gets zero gradients instead of a backward() error.
  if (!any) {
    auto none = from_d.slice(0, 0, 0);
    y.index_add_(0, none, x.index_select(0, none).matmul(wt[0]));
  }
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
    return f.sum(0); // zeros, but still part of the autograd graph
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
  const Topology &t = *input.topology();
  auto pts = points.to(torch::kFloat64).cpu().contiguous();
  if (pts.dim() != 2 || pts.size(1) != t.D)
    throw std::invalid_argument("points must have shape [M, D]");
  const int64_t M = pts.size(0);
  const double *p = pts.data_ptr<double>();

  std::vector<int64_t> pt_idx, leaf_idx;
  std::vector<double> w;
  std::vector<std::pair<int64_t, double>> hits;
  for (int64_t m = 0; m < M; ++m) {
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

  const auto &x = input.features();
  auto y = torch::zeros({M, x.size(1)}, x.options());
  if (pt_idx.empty()) { // no query hits a leaf: zeros that stay connected to the features
    auto none = torch::empty({0}, torch::TensorOptions(torch::kInt64).device(x.device()));
    return y.index_add_(0, none, x.index_select(0, none));
  }
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
