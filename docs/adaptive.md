---
title: Adaptive voxelization
description: Refine regions of interest while preserving coarse context in a mixed-resolution sparse tensor.
permalink: /adaptive/
---

{% raw %}
Include `<adaptive/adaptive.hpp>` after installation, or
`"MinkowskiEngine/adaptive/adaptive.hpp"` within the repository.
The namespace is `minkowski::adaptive`; the uniform `SparseTensor` API is
unchanged.

## Hierarchy and geometry

Level 0 uses `base_voxel_size`. Each level halves the size, so a valid
configuration satisfies `base / min == 2^max_depth` per axis. A leaf stores
its level and integer coordinates. At size `s`, its world-space box is
`[c*s, (c+1)*s)`; bounds are half-open.

```cpp
using namespace minkowski::adaptive;
AdaptiveVoxelConfig cfg;
cfg.base_voxel_size = {8.0, 8.0, 8.0};
cfg.min_voxel_size = {0.5, 0.5, 0.5};
cfg.max_depth = 4;
```

Topologies are immutable and versioned internally. Tensor topology-edit
methods update the tensor to a new topology.

## Construct and inspect

```cpp
auto points = torch::rand({20000, 3}, torch::kFloat64) * 64.0;
auto feats = torch::randn({20000, 4}, torch::kFloat64);
auto t = AdaptiveSparseTensor::from_points(points, feats, cfg);

auto coordinates = t.coordinates(); // [N,3] int64 leaf cell indices
auto levels = t.levels();           // [N] int32, 0 = coarsest
auto centers = t.centers();         // [N,3] double world-space centers
auto counts = t.counts_by_level();  // max_depth + 1 entries
```

Unlike uniform sparse coordinates, these examples use spatial coordinates
without a leading batch column. Consult the public API before introducing
batching assumptions from `SparseTensor`.

For explicitly specified leaves:

```cpp
auto coords = torch::tensor({{0, 0, 0}, {2, 0, 0}, {2, 1, 0}}, torch::kInt64);
auto leaf_levels = torch::tensor({0, 1, 1}, torch::kInt32);
auto values = torch::randn({3, 8});
auto explicit_tensor = AdaptiveSparseTensor::from_voxels(
    coords, leaf_levels, values, cfg);
```

Leaves must not overlap.

## Refinement and coarsening

```cpp
AxisAlignedRegion roi{{0, 0, 0}, {16, 16, 16}};
t.refine(roi, 4);
t.coarsen(roi, 2, Reduction::Mean);
```

| Policy | Semantics |
|---|---|
| `RefinePolicy::Replicate` | Copy the parent's feature to all `2^D` children |
| `RefinePolicy::Split` | Divide the parent feature by `2^D`, conserving a summed quantity |
| `Reduction::Mean` | Average occupied children |
| `Reduction::Sum` | Sum occupied children |
| `Reduction::Max` | Maximum over occupied children |

A parent merges only if its whole box lies inside the requested region.
Coarsening reduces occupied children only. Choose replicate/mean for
intensive quantities and split/sum when conserving extensive quantities is
the intended model.

Point-driven edits require a tensor constructed with `from_points`, which
retains source points:

```cpp
t.refine_by_density(32);
t.refine_by_variance(0.5);
t.revoxelize_from_points();
t.refine_if([](int64_t, const std::array<double, 4>& lo,
               const std::array<double, 4>& hi, uint32_t level) {
  return level < 3 && std::abs(0.5 * (lo[0] + hi[0])) < 10.0;
});
```

Topology decisions are not differentiable. Feature computations are built
from torch operations and support autograd.

## Mixed-resolution convolution

For each output leaf, the kernel samples around its center at that leaf's
own voxel scale: `center + offset * output_voxel_size`.

```cpp
AdaptiveKernelConfig kc;
kc.mode = AdaptiveConvMode::OverlapWeighted;
auto weight = torch::randn({27, 4, 16}, torch::kFloat64)
                  .set_requires_grad(true);
auto output = adaptive_convolution(t, weight, kc);
output.features().sum().backward();
```

Weights have shape `[K^D, Cin, Cout]`; the default 3x3x3 kernel has 27 offsets,
with the first axis varying slowest.

| Mode | Interpretation |
|---|---|
| `CenterSampled` | Sample the neighbor at the kernel query center |
| `VolumeWeighted` | Volume-aware contribution |
| `OverlapWeighted` | Weight by overlap volume / output volume |

Coarsen output resolution with `output_level_shift = -1`:

```cpp
AdaptiveKernelConfig down;
down.output_level_shift = -1;
auto coarse = adaptive_convolution(t, weight, down);
auto up_weight = torch::randn({27, 16, 4}, torch::kFloat64);
auto restored = adaptive_convolution_transpose(coarse, up_weight, down, t);
```

Transpose convolution uses the reference tensor's leaves.

Kernel maps are built on the host once per (topology, kernel size, mode, shift) and cached;
their index and weight arrays are uploaded once per device and dtype and reused by later
calls. `KernelMapCache::instance().clear()` releases both. On CUDA the scatter-add uses
`index_add_`, whose floating-point atomics make results vary in the last bits between runs;
call `torch::globalContext().setDeterministicAlgorithms(true, false)` for bit-reproducible
outputs (slower), as for any PyTorch scatter.
`output_level_shift > 0` is currently unsupported.

## Pooling, interpolation, and resampling

```cpp
auto global = adaptive_global_avg_pool(t); // [C], volume-weighted
auto pooled = adaptive_pool(t, 1, Reduction::Max);
auto query = torch::tensor({{1.0, 2.0, 3.0}, {40.0, 5.0, 5.0}},
                           torch::kFloat64);
auto nearest = interpolate(t, query);
auto smooth = interpolate(t, query, InterpMode::Multilinear);
auto uniform = t.resample_uniform(2);
```

Nearest interpolation returns the containing leaf's value. Multilinear
interpolation averages the leaves overlapping a box the size of the containing
leaf, centred on the query: it is continuous inside a level and across faces
between equal-level leaves, but the box size changes where the containing
leaf's level changes, so values can jump there. Queries outside every leaf
return zeros, or with
`interpolate(t, query, mode, InterpFallback::CoarserAncestor)` the
volume-weighted mean of the leaves inside the finest cell that contains the
query and at least one leaf (useful after `drop_empty` refinement, which leaves
point-free children uncovered). `other.resample_to(t)` aligns another tensor to
`t`'s leaves; addition requires identical topology.

Uniform resampling can expand a sparse hierarchy dramatically. The
benchmark skips expansions above five million voxels; estimate the output
size before using it in an application.

## Complete example

```cpp
#include <adaptive/adaptive.hpp>
#include <iostream>
using namespace minkowski::adaptive;

int main() {
  AdaptiveVoxelConfig cfg;
  cfg.base_voxel_size = {8.0, 8.0, 8.0};
  cfg.min_voxel_size = {0.5, 0.5, 0.5};
  cfg.max_depth = 4;
  auto points = torch::rand({20000, 3}, torch::kFloat64) * 64.0;
  auto features = torch::randn({20000, 4}, torch::kFloat64);
  auto t = AdaptiveSparseTensor::from_points(points, features, cfg);
  t.refine(AxisAlignedRegion{{0, 0, 0}, {16, 16, 16}}, 4);

  auto w1 = torch::randn({27, 4, 16}, torch::kFloat64)
                .mul_(0.1).set_requires_grad(true);
  auto w2 = torch::randn({27, 16, 2}, torch::kFloat64)
                .mul_(0.1).set_requires_grad(true);
  AdaptiveKernelConfig kc;
  auto hidden = adaptive_convolution(t, w1, kc);
  hidden = hidden.with_features(torch::relu(hidden.features()));
  auto output = adaptive_convolution(hidden, w2, kc);
  auto logits = adaptive_global_avg_pool(output);
  logits.sum().backward();
  std::cout << t.size() << " leaves, logits " << logits << '\n';
}
```

## CUDA and caching

`t.to(torch::kCUDA)` moves feature tensors to CUDA. Move convolution weights
to that device as well. Hierarchy and kernel-map construction remain
host-side; CUDA feature execution does not imply a fully GPU-resident
topology pipeline.

Kernel maps are cached per topology. Inspect
`KernelMapCache::instance().stats()` and use `.clear()` when intentionally
resetting the cache.

## Limits and measurements

The README's 20K-point CPU benchmark reported 41,376 adaptive voxels versus
about 35 million uniform voxels at the finest resolution: kernel-map build
221 ms, forward 14 ms, and forward/backward 50 ms. These are local
measurements, not guarantees.

There is no uniform fast path, no ROCm backend, and overlap-weighted cost
grows with the level gap. The legacy all-ones comparison established row
count and total-sum agreement, not per-row equivalence.

See [tests and the sphere-volume benchmark](testing.md) for reproducible
commands and accuracy interpretation.
{% endraw %}
