# Gaussian foundation

This directory contains the opt-in native Gaussian foundation. Enable it with
`-DMINKOWSKI_ENABLE_GAUSSIAN=ON` (the default). Its public headers are available
through `MinkowskiEngine/minkowski.hpp` when the feature is enabled.

## Current API

`Gaussian3D<T>` is a value type with a world-space mean, a `(w, x, y, z)`
quaternion, positive axis scales, opacity, RGB color, and optional feature,
velocity, timestamp, semantic-class, and object identifiers. Covariance is
represented by `R diag(scale^2) R^T`; the full matrix is computed on demand.
The value type provides covariance, inverse covariance, determinant, density,
Mahalanobis distance, pairwise density-product integral, sampling, and
confidence-radius axis-aligned bounds.

`GaussianTensor` stores positions, rotations, scales, opacity, color, features,
and optional velocity as separate LibTorch tensors. It supports empty allocation,
covariance construction, selection, removal, opacity and center-region filters,
concatenation, cloning, device transfer, and batched rigid transforms. Supported
dtypes are float32 and float64. CUDA inference tensors use dedicated kernels for
covariance construction and rigid transforms; tensors requiring gradients use
LibTorch operations so their autograd graph is preserved.

`GaussianOptimizer` wraps LibTorch Adam for trainable Gaussian fields and
provides RGB/depth reconstruction losses, opacity/scale regularizers, and an
objective helper. The caller supplies the prediction tensor; in the absence of
a rasterizer this utility does not itself render RGB or depth.

```cpp
#include "minkowski.hpp"

minkowski::gaussian::Gaussian3Dd g;
g.mean = {1.0, 2.0, 3.0};
g.scale = {0.5, 1.0, 2.0};
auto covariance = g.covariance();
```

For optimization, construct a `GaussianOptimizer` from a `GaussianTensor`, form
a loss against your predicted and target RGB tensors, then call `step(loss)`.
The optimizer detaches the selected fields into trainable leaf tensors and
updates the tensor's underlying fields in place.

Build and run the focused CPU/CUDA-device-independent tests with:

```powershell
cmake --build build --target gaussian_tests
ctest --test-dir build -R gaussian_tests --output-on-failure
```

Build and run the benchmark with:

```powershell
cmake --build build --target gaussian_bench --config Release
.\build\gaussian_bench.exe --output gaussian-results.json
.\build\gaussian_bench.exe --count 1000000 --output gaussian-million.json
```

The benchmark measures covariance construction, rigid transforms, and repeated
RGB optimization steps. It reports warm-up-adjusted median and p95 latency,
standard deviation, throughput in Gaussians per second, and an estimated tensor
footprint in JSON. CUDA timings use events on the current stream. The footprint
is explicitly an estimate of tensor inputs and the largest output, not
allocator-reported peak memory. Do not compare benchmark records across
different hardware/builds as if they were equivalent.

The working-tree [benchmark results](../../benchmarks/gaussian_benchmark_results.json)
were measured in a Release build with LibTorch 2.15.0 on Windows, an AMD
Family 25 CPU (12 reported hardware threads), and an NVIDIA GeForce RTX 3080.
At 100,000 Gaussians, covariance construction measured 6.559 ms on CPU and
0.180 ms on CUDA; rigid transforms measured 10.715 ms on CPU and 0.412 ms on
CUDA. The RGB optimization objective decreased from 0.334 to 0.228 over the
benchmark's 12 Adam steps (2 warm-up and 10 timed). These are local
measurements, not general performance guarantees.

## Scope and limitations

This remains a partial Gaussian subsystem. There is not yet a
Gaussian-to-sparse-voxel bridge, adaptive spatial index, camera projection,
rasterizer, custom rendering backward pass, or dedicated Gaussian CUDA
rasterizer. The CUDA kernels cover covariance construction and rigid transforms
only. `GaussianTensor::transform` accepts quaternion transforms directly; this
repository currently has no shared SE(3) module to integrate with.

## PLY loading

`load_gaussian_ply(path)` (`gaussian_ply.hpp`) loads binary little-endian 3DGS PLY files and PlayCanvas/SuperSplat compressed PLY files into a float32 CPU `GaussianTensor` (SH rest coefficients are skipped). `tests/gaussian/test_gaussian_ply.cpp` exercises `tests/data/test.ply` (override with `MINK_TEST_DATA_DIR`); `gaussian_ply_bench` benchmarks loading and operations on it (`--input`, `--output`, `--iterations`).

