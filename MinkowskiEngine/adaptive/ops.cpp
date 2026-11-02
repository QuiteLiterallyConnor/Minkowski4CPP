#include "ops.hpp"
#include "adaptive_topology.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

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
// `transpose` swaps the direction so the transposed convolution reuses the map.
// The map's index/weight arrays come from the entry's device cache (uploaded once per
// device/dtype). On CUDA one gather, one weight multiply and one scatter-add cover all kernel
// slots; only the GEMM runs per slot, on row blocks of the gathered features. The pairs are sorted by
// slot, so the scatter-add sees the same rows in the same order as one scatter per slot did.
// split_with_sizes / unbind (not per-slot slice / select) keep the backward pass to one
// concatenation each instead of one full-size zero-filled gradient per slot.
at::Tensor apply_map(const at::Tensor &x, const at::Tensor &weight, const KernelMapEntry &entry,
                     bool transpose, int64_t n_out, bool weighted) {
  const AdaptiveKernelMap &m = entry.map;
  const int64_t cout = weight.size(2);
  auto arrays = entry.device->get(m, x.device(), x.scalar_type(), weighted);
  const at::Tensor &from_d = transpose ? arrays.out_index : arrays.in_index;
  const at::Tensor &to_d = transpose ? arrays.in_index : arrays.out_index;
  auto y = torch::zeros({n_out, cout}, x.options());
  if (!x.is_cuda()) {
    // Host path: the per-slot formulation. Its temporaries stay small (one slot at a time),
    // which measured faster on the CPU than materialising the full [P, C] gather/products
    // (large fresh allocations), and it keeps CPU results bit-identical to earlier versions.
    auto wt = weight.to(x.scalar_type());
    bool any = false;
    for (int64_t k = 0; k < m.num_kernels; ++k) {
      const int64_t b = m.kernel_offsets[k], e = m.kernel_offsets[k + 1];
      if (b == e)
        continue;
      any = true;
      auto xs = x.index_select(0, from_d.slice(0, b, e));
      if (weighted)
        xs = xs * arrays.weight.slice(0, b, e).unsqueeze(1);
      y.index_add_(0, to_d.slice(0, b, e), xs.matmul(wt[k]));
    }
    if (!any) { // keep y connected to x and weight (empty batch -> zero gradients)
      auto none = from_d.slice(0, 0, 0);
      y.index_add_(0, none, x.index_select(0, none).matmul(wt[0]));
    }
    return y;
  }
  // Device path: launches per call drop from ~4 per slot to ~1 per slot + 4.
  const std::vector<at::Tensor> wk = weight.to(x.scalar_type()).unbind(0);
  auto xs = x.index_select(0, from_d);
  if (weighted)
    xs = xs * arrays.weight.unsqueeze(1);
  std::vector<int64_t> sizes(static_cast<size_t>(m.num_kernels));
  for (int64_t k = 0; k < m.num_kernels; ++k)
    sizes[k] = m.kernel_offsets[k + 1] - m.kernel_offsets[k];
  const std::vector<at::Tensor> blocks = xs.split_with_sizes(sizes, 0);
  std::vector<at::Tensor> parts;
  parts.reserve(static_cast<size_t>(m.num_kernels));
  for (int64_t k = 0; k < m.num_kernels; ++k)
    if (sizes[k] > 0)
      parts.push_back(blocks[k].matmul(wk[k]));
  // No pairs (e.g. an empty input): a zero-row product keeps y connected to x and weight, so
  // backward() on an empty batch yields zero gradients instead of an error.
  if (parts.empty())
    parts.push_back(xs.matmul(wk[0]));
  y.index_add_(0, to_d, parts.size() == 1 ? parts[0] : torch::cat(parts, 0));
  return y;
}

} // namespace

AdaptiveSparseTensor adaptive_convolution(const AdaptiveSparseTensor &input,
                                          const at::Tensor &weight,
                                          const AdaptiveKernelConfig &config) {
  auto entry = KernelMapCache::instance().get_or_build(input, config);
  check_weight(weight, entry.map, input.channels());
  auto y = apply_map(input.features(), weight, entry, /*transpose=*/false,
                     entry.out_topology->size(), config.mode != AdaptiveConvMode::CenterSampled);
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
  auto y = apply_map(input.features(), weight, entry, /*transpose=*/true, reference.size(),
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

namespace {

// InterpFallback::CoarserAncestor support. A cell "holds data" when at least one active leaf
// lies strictly inside it. Cells at or below the coarsest present level are answered by the
// topology's node table (kInternal); coarser cells by `coarse`, the set of leaf ancestors
// above that level (small: at most one entry per occupied cell of each coarse level).
struct AncestorIndex {
  const Topology &t;
  std::unordered_set<detail::Key, detail::KeyHash> coarse;

  explicit AncestorIndex(const Topology &topo) : t(topo) {
    if (t.keys.empty() || t.present_levels.front() == 0)
      return;
    const uint32_t lmin = t.present_levels.front();
    for (const auto &k : t.keys) {
      detail::Key a = detail::ancestor_at(k, lmin - 1);
      while (coarse.insert(a).second && a.level > 0)
        a = detail::parent_key(a);
    }
  }

  bool holds_data(const detail::Key &k) const {
    if (k.level < t.present_levels.front())
      return coarse.count(k) != 0;
    return t.node_table().lookup(k) == Topology::NodeTable::kInternal;
  }

  // The finest cell that contains q and holds data; false when no cell does (q is in an
  // unoccupied level-0 cell or outside the representable range). Holding data is
  // inherited by every coarser cell, so the walk stops at the first cell without data.
  bool deepest(const DVec &q, detail::Key &out) const {
    bool found = false;
    detail::Key k;
    for (uint32_t l = 0; l <= t.max_depth; ++l) {
      if (!t.locate(q, l, k) || !holds_data(k))
        break;
      out = k;
      found = true;
    }
    return found;
  }
};

} // namespace

at::Tensor interpolate(const AdaptiveSparseTensor &input, const at::Tensor &points,
                       InterpMode mode, InterpFallback fallback) {
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
    std::vector<int64_t> miss_pt;      // CoarserAncestor: uncovered points ...
    std::vector<detail::Key> miss_key; // ... and the cell whose leaves they average
  };
  if (M > 0 && t.size() > 0)
    t.node_table(); // build the shared lookup table once, before the workers start
  const bool use_fallback =
      fallback == InterpFallback::CoarserAncestor && M > 0 && t.size() > 0;
  std::unique_ptr<AncestorIndex> ancestors;
  if (use_fallback)
    ancestors = std::make_unique<AncestorIndex>(t);
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
      if (j < 0) {
        detail::Key a;
        if (use_fallback && ancestors->deepest(q, a)) {
          parts[b].miss_pt.push_back(m);
          parts[b].miss_key.push_back(a);
        }
        continue;
      }
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
  if (use_fallback) {
    // Uncovered point -> volume-weighted mean of the leaves inside its cell. Cells get rows
    // in first-use order (serial point order), leaves contribute in index order: the result
    // does not depend on the number of worker blocks.
    std::unordered_map<detail::Key, int64_t, detail::KeyHash> row_of;
    std::vector<detail::Key> cells;
    std::vector<int64_t> miss_pt, miss_row;
    uint32_t lo_level = AdaptiveVoxelConfig::kMaxSupportedDepth, hi_level = 0;
    for (auto &part : parts)
      for (size_t i = 0; i < part.miss_pt.size(); ++i) {
        const detail::Key &k = part.miss_key[i];
        auto it = row_of.try_emplace(k, static_cast<int64_t>(cells.size())).first;
        if (it->second == static_cast<int64_t>(cells.size())) {
          cells.push_back(k);
          lo_level = std::min(lo_level, k.level);
          hi_level = std::max(hi_level, k.level);
        }
        miss_pt.push_back(part.miss_pt[i]);
        miss_row.push_back(it->second);
      }
    if (!cells.empty()) {
      std::vector<int64_t> cell_row, cell_leaf;
      std::vector<double> cell_w, total(cells.size(), 0.0);
      const double *vol = t.volumes.data_ptr<double>();
      for (int64_t i = 0; i < t.size(); ++i) {
        const detail::Key &k = t.keys[static_cast<size_t>(i)];
        if (k.level <= lo_level)
          continue; // a cell only averages leaves strictly inside it
        detail::Key a = detail::ancestor_at(k, std::min(hi_level, k.level - 1));
        while (true) {
          auto it = row_of.find(a);
          if (it != row_of.end()) {
            cell_row.push_back(it->second);
            cell_leaf.push_back(i);
            cell_w.push_back(vol[i]);
            total[static_cast<size_t>(it->second)] += vol[i];
          }
          if (a.level <= lo_level)
            break;
          a = detail::parent_key(a);
        }
      }
      for (size_t e = 0; e < cell_w.size(); ++e)
        cell_w[e] /= total[static_cast<size_t>(cell_row[e])];
      const auto to_dev = [&](std::vector<int64_t> &v) {
        return torch::from_blob(v.data(), {static_cast<int64_t>(v.size())}, torch::kInt64)
            .clone()
            .to(x.device());
      };
      auto w_c = torch::from_blob(cell_w.data(), {static_cast<int64_t>(cell_w.size())},
                                  torch::kFloat64)
                     .clone()
                     .to(x.scalar_type())
                     .to(x.device())
                     .unsqueeze(1);
      auto means = torch::zeros({static_cast<int64_t>(cells.size()), x.size(1)}, x.options())
                       .index_add(0, to_dev(cell_row), x.index_select(0, to_dev(cell_leaf)) * w_c);
      y = y.index_add(0, to_dev(miss_pt), means.index_select(0, to_dev(miss_row)));
    }
  }
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
