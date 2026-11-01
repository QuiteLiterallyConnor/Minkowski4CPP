---
title: Tests & benchmarks
description: Focused test targets, training examples, Gaussian timings, and sphere-volume accuracy.
permalink: /testing/
---

Use the same dependency versions and Release configuration when comparing
results. GPU parity tests require a usable CUDA device; compiling a CUDA
build without one does not establish runtime correctness.

## Focused tests

```bash
cmake --build build --target adaptive_tests gaussian_tests --parallel 4
ctest --test-dir build -R 'adaptive_tests|gaussian_tests' --output-on-failure
```

For multi-configuration generators, add `--config Release` to the build and
`-C Release` to CTest.

### Adaptive tests

The README reports 66 tests covering geometry, topology, numerical behavior,
gradients, regressions, and sphere behavior. CUDA agreement is checked when
CUDA is available.

```powershell
cmake --build build --target adaptive_tests adaptive_bench sphere
.\build\adaptive_tests.exe
.\build\adaptive_bench.exe
.\build\sphere.exe
```

`sphere` exports JSON with leaf grid coordinates, refinement level,
world-space bounds, centers, sizes, and occupancy. It is not the separate
volume-accuracy benchmark described below.

### Gaussian tests and benchmarks

The README reports 35 Gaussian tests covering value math, tensor operations,
optimizer behavior, CUDA parity, PLY, SH, rendering finite-difference
gradients, and training.

```powershell
cmake --build build --target gaussian_tests gaussian_bench gaussian_ply_bench
.\build\gaussian_tests.exe
.\build\gaussian_bench.exe --count 100000 --output gaussian-results.json
.\build\gaussian_ply_bench.exe `
  --input tests\data\test.ply `
  --input tests\data\test_compressed.ply `
  --iterations 10
```

Linux executable paths omit `.exe`.
PLY tests use `tests/data/test.ply` and `tests/data/test_compressed.ply`.
Override the fixture directory with `MINK_TEST_DATA_DIR`. Tests skip when
fixture files are missing, so inspect skips rather than interpreting a
successful exit as complete PLY coverage.

`gaussian_bench` measures covariance, transforms, and optimization on
synthetic data. JSON includes warm-up-adjusted median/p95 latency, standard
deviation, throughput, and an estimated tensor footprint. The footprint
is **not allocator-reported peak memory**. CUDA timings use events on the
current stream.

`gaussian_ply_bench` measures cold/warm loading, file throughput, scene
operations, transfers, model loading with SH, rendering, and a full
training step.

## Reported local measurements

These values come from the README's RTX 3080 measurements; they are neither
performance guarantees nor comparable across arbitrary builds.

| Workload | Reported latency |
|---|---|
| 139,410-Gaussian uncompressed PLY, warm load | About 37 ms, 32.9 MB |
| Same scene compressed PLY, warm load | About 17 ms, 8.5 MB |
| Scene covariance, CPU / CUDA | About 7.8 ms / 0.3 ms |
| Full scene, 512x512 fused CUDA render | About 6 ms |
| Full scene, fused CUDA training step | About 28 ms |
| Full scene, LibTorch fallback render / train | About 772 ms / 3.2 s |
| 5,000 Gaussians, 128x128 CPU render / train | About 48 ms / 197 ms |

The synthetic 100,000-Gaussian benchmark notes separately report covariance
6.559 ms CPU / 0.180 ms CUDA and transform 10.715 ms CPU / 0.412 ms CUDA.
Do not mix synthetic and scene-based measurements into a single comparison.

For adaptive voxelization, the README's 20K-point depth-4 CPU run reports
41,376 leaves versus roughly 35 million uniform finest-level voxels, a
221 ms kernel-map build, 14 ms forward, and 50 ms forward/backward.

## Sphere-volume accuracy benchmark

The `benchmarks/` CMake project links an **installed** MinkowskiEngine
package; it does not build the library sources itself.

```powershell
cmake -G Ninja -S benchmarks -B build-sphere-benchmark `
  -DMinkowskiEngine_DIR="D:\Minkowski4CPP\install\lib\cmake\MinkowskiEngine" `
  -DCMAKE_PREFIX_PATH="D:\libtorch_cu134" `
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\nvcc.exe" `
  -DCUDAToolkit_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
cmake --build build-sphere-benchmark --config Release
.\build-sphere-benchmark\sphere_volume_benchmark.exe 1000
.\build-sphere-benchmark\sphere_volume_benchmark.exe --accuracy 99
.\build-sphere-benchmark\sphere_volume_benchmark.exe --optimal
```

Use the toolkit matching the installed library. Linux uses the equivalent
paths and executable without `.exe`.

### How to interpret results

- The reference is the analytic volume of a radius-100 m sphere.
- The estimate includes a voxel's full volume if its center lies inside.
- Uniform grids choose the nearest cubic count: a request for 100 becomes
  `5^3 = 125`; 1000 becomes `10^3`.
- The adaptive octree refines boundary cells, coarsest first, under the
  same requested budget. Each split increases leaf count by seven, so the
  actual count can finish below the target.
- Count arguments range from 1 to 10,000,000; default is 1000.
- `--accuracy` accepts percentages strictly between 0 and 100 and searches
  up to ten million voxels.
- Uniform accuracy is not monotonic: a larger grid can have worse error.
- `--optimal` reports latency/accuracy tradeoffs and a Pareto frontier;
  accuracy digits are `-log10(relative error)`.

Output includes error/bias, actual voxel counts, boundary cells, adaptive
level distribution, construction/evaluation timing, and end-to-end timing.
Typical runs use one warm-up and the median of seven repetitions; above
200,000 voxels the benchmark uses a single repetition. Sub-0.1 ms timings
are noisy; repeat measurements.

These are geometric approximation and CPU timing results, not a universal
claim about sparse convolution performance.

## Training example

`train` is the repository's sparse 4D CNN trainer:

```powershell
cmake --build build --target train
.\build\train.exe --help
.\build\train.exe --data_dir tests\data --epochs 10 --batch_size 1
```

| Flag | Default | Meaning |
|---|---|---|
| `--data_dir` | Required | Training data directory |
| `--epochs` | `10` | Epoch count |
| `--batch_size` | `1` | Batches |
| `--radius` | `2` | Labeling radius in voxels |
| `--pos_weight` | `12.0` | BCE positive-class weight |
| `--lr` | `1e-3` | Learning rate |
| `--weight_decay` | `1e-4` | Weight decay |
| `--save_dir` | `checkpoints` | Output directory |
| `--clip_len` | `10` | Frames per clip |
| `--c_mid` | `32` | Mid-level channel count |

It expects the repository's training-data conventions, not any arbitrary
point-cloud directory. Inspect
[the data loader]({{ site.repository_url }}/blob/main/tests/train.cpp) when
adapting a dataset.

## Builder regression tests

```powershell
.\tests\builders\test-builders.ps1
```

```bash
bash tests/builders/test-builders.sh
CHECK_DOWNLOADS=1 bash tests/builders/test-builders.sh
CHECK_CUDA=1 bash tests/builders/test-builders.sh
```

Docker and native packaging commands are mocked in the regression suite.
It verifies profiles, arguments, output naming, default selection,
publishing aliases, toolkit mismatch errors, and CPU fallback.
`CHECK_DOWNLOADS` probes pinned upstream files; `CHECK_CUDA` compiles a
small NVTX compatibility fixture with installed nvcc.

Passing mock tests or Dockerfile checks is **not** a substitute for building,
installing, and executing a real image/package.
