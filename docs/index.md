---
title: Native sparse computing in C++
description: Build sparse neural networks, mixed-resolution voxel pipelines, and differentiable Gaussian renderers without a Python runtime.
permalink: /
---

Minkowski4CPP ports NVIDIA's [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
to C++, using LibTorch for tensors, neural-network modules, and automatic
differentiation. It supports CPU execution and CUDA acceleration on Windows and
Linux.

## Choose your workflow

| Workflow | Representation | Start here |
|---|---|---|
| Sparse classification and segmentation | Uniform integer-grid `SparseTensor` | [Quick start](quickstart.md), [sparse networks](sparse-tensors.md) |
| Detail only where it matters | Mixed-resolution `AdaptiveSparseTensor` | [Adaptive voxelization](adaptive.md) |
| Gaussian-splat scenes and image reconstruction | `GaussianTensor` and trainable `GaussianModel` | [Gaussian splats](gaussian.md), [rendering and training](rendering.md) |

The adaptive API is separate from the uniform sparse API. The Gaussian
subsystem is enabled by default but can be disabled at build time. Neither
subsystem currently provides a Gaussian-to-voxel bridge.

## Get started

1. [Install the library and matching LibTorch/CUDA dependencies](installation.md).
2. [Build a small downstream C++ application](quickstart.md).
3. Follow the guide for your representation.
4. [Run tests and measure performance](testing.md) on your hardware.

Prefer an isolated toolchain? The [Docker and Debian builders](builders.md)
select matched CUDA, Ubuntu, and LibTorch versions, including legacy CUDA 12.4.

## Platform profiles

| Platform | CUDA | LibTorch | README status |
|---|---|---|---|
| Ubuntu 22.04 | 12.4.0 | 2.5.1 / cu124 | Verified |
| Ubuntu 24.04 builders | 13.2.1 | 2.15.0 / cu132 | Builder profile |
| Ubuntu 24.04 builders, default | 13.4.3 | 2.15.0 / cu134 | Builder profile |
| Windows | 13.2 or 13.4 | 2.15.0, matching CUDA | Verified |

"Builder profile" identifies a configured dependency pairing, not a claim
that every operation has been validated on that platform. macOS and ROCm are
not documented supported targets here.

## What is included

- Sparse convolution and transpose convolution, pooling, normalization,
  nonlinearities, linear layers, pruning, interpolation, and coordinate maps.
- LibTorch module registration, optimizers, autograd, and serialization.
- Adaptive refinement/coarsening, mixed-resolution kernels, interpolation,
  resampling, and topology-aware caching.
- Gaussian geometry, batched scene operations, standard and compressed PLY
  input, spherical harmonics, differentiable rasterization, and 3DGS-style
  training with densification and pruning.
- Focused tests and JSON-producing benchmarks.

See the [API reference](api.md) for entry points and
[known limitations](troubleshooting.md#known-limitations) before designing a
production pipeline.
