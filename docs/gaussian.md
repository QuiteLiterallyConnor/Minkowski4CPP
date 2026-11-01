---
title: Gaussian splats
description: Gaussian geometry, batched tensor operations, PLY interchange, and optimization.
permalink: /gaussian/
---

Gaussian support is enabled by default. Disable it with
`-DMINKOWSKI_ENABLE_GAUSSIAN=OFF`. The namespace is
`minkowski::gaussian`; installed applications can include `<minkowski.hpp>`
or individual `<gaussian/...>` headers.

## Three representations

| Type | Use it for |
|---|---|
| `Gaussian3D<T>` | Host-side geometry and probability calculations |
| `GaussianTensor` | Activated batched scene data and filtering/transforms |
| `GaussianModel` | Trainable raw parameters, spherical harmonics, rendering |

`Gaussian3Df` and `Gaussian3Dd` alias float and double value types.
Orientation quaternions use **`(w, x, y, z)`**, not `(x, y, z, w)`.
Covariance is `R diag(scale^2) R^T`.

The value API exposes covariance, inverse covariance, determinant, density,
Mahalanobis distance, pairwise density-product integral, sampling, and
confidence-radius axis-aligned bounds.

```cpp
minkowski::gaussian::Gaussian3Dd g;
g.mean = {1.0, 2.0, 3.0};
g.scale = {0.5, 1.0, 2.0};
auto covariance = g.covariance();
```

## Batched tensor layout

| Field | Shape | Convention |
|---|---|---|
| Positions | `[N,3]` | World-space centers |
| Rotations | `[N,4]` | Quaternions, w/x/y/z |
| Scales | `[N,3]` | Positive axis scales |
| Opacities | `[N,1]` | Activated opacity |
| Colors | `[N,3]` | RGB |
| Features | Optional | Additional per-Gaussian attributes |
| Velocities | Optional | Per-Gaussian motion data |

`GaussianTensor` supports float32/float64 on CPU or CUDA. Covariance and
rigid transforms use dedicated CUDA kernels for inference tensors; when
gradients are required, LibTorch operations preserve the autograd graph.

Rendering has a stricter contract: float32 tensors on one device. See
[rendering](rendering.md).

## Load and inspect a scene

```cpp
#include <minkowski.hpp>
#include <iostream>
using namespace minkowski::gaussian;

GaussianTensor scene = load_gaussian_ply("scene.ply");
std::cout << scene.size() << " Gaussians\n";
auto covariances = scene.covariances(); // [N,3,3]
auto visible = scene.filter_opacity(0.5);
```

`load_gaussian_ply` detects standard 3DGS or SuperSplat/PlayCanvas compressed
PLY from the header. It loads float32 CPU tensors and throws
`std::runtime_error` for missing or malformed input.

## PLY conventions

Supported input is **binary little-endian**, not arbitrary PLY variants.
For the standard 3DGS layout:

| Stored PLY fields | Activated tensor values |
|---|---|
| `x`, `y`, `z` | Positions |
| `rot_*` | Normalized quaternion |
| `scale_*` | `exp(raw_scale)` |
| `opacity` | `sigmoid(raw_opacity)` |
| `f_dc_*` | `0.5 + C0 * f_dc` colors |

`load_gaussian_ply` skips spherical-harmonic rest coefficients.
Use **`load_gaussian_model_ply`** to preserve SH and raw trainable parameters.
`save_gaussian_ply(model, path)` writes a standard 3DGS layout with SH.

In compressed files, packed colors are already RGB; conversion to the model
uses `f_dc = (rgb - 0.5) / C0`. Compressed SH-byte decoding has **not been
verified against reference output**. Validate interchange against your
source renderer before relying on view-dependent color fidelity.

## Filter, select, and merge

```cpp
auto lo = std::get<0>(scene.positions().min(0));
auto hi = std::get<0>(scene.positions().max(0));
auto half = scene.filter_bounds(lo, 0.5 * (lo + hi));
auto indices = at::arange(0, scene.size(), 2, at::kLong);
auto even = scene.select(indices);
auto rest = scene.remove(indices);
auto merged = even.append(rest);
```

Bounds filtering uses Gaussian centers, not a full covariance-envelope
intersection test. Choose indices and auxiliary tensors compatible with
the scene's device.

## Rigid transforms and CUDA

```cpp
auto q = at::tensor({0.9238795f, 0.f, 0.3826834f, 0.f});
auto translation = at::tensor({0.5f, 1.0f, -1.0f});
auto moved = scene.transform(q, translation);

if (torch::cuda::is_available()) {
  auto gpu = scene.to(torch::kCUDA);
  auto gpu_cov = gpu.covariances();
  auto gpu_moved = gpu.transform(q.cuda(), translation.cuda());
}
```

Transforms accept shared or per-Gaussian quaternions/translations.
CUDA execution also requires a CUDA-enabled Minkowski build.

## Optimize tensor fields

`GaussianOptimizer` wraps LibTorch Adam with RGB/depth reconstruction losses,
opacity/scale regularizers, and projection of scale/opacity constraints.
You provide prediction and target tensors; the utility does not itself
render a scene.

```cpp
GaussianTensor work = scene.clone();
GaussianOptimizer optimizer(work, 1e-2);
auto target = at::zeros_like(work.colors());
for (int i = 0; i < 100; ++i) {
  auto loss = GaussianOptimizer::rgb_reconstruction_loss(
      work.colors(), target);
  optimizer.step(loss);
}
```

The optimizer detaches selected fields into trainable leaf tensors and
updates `work` in place. **The tensor must outlive the optimizer.**
This example optimizes color toward black; it is not a full image
reconstruction pipeline.

For posed-image reconstruction with densification, use
[GaussianModel and GaussianTrainer](rendering.md).

## Current limitations

- No Gaussian-to-sparse-voxel bridge or adaptive spatial index.
- No shared SE(3) abstraction; transforms accept quaternions directly.
- Compressed SH decoding remains unverified.
- The renderer's fused CUDA path covers tile compositing; projection and
  SH evaluation remain LibTorch operations.

See [Gaussian tests and benchmarks](testing.md#gaussian-tests-and-benchmarks)
for fixture paths, parity checks, and local measurements.
