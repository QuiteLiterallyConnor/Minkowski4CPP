---
title: API reference
description: Public entry points, tensor contracts, and source-header links.
permalink: /api/
---

This is a curated public API reference, not generated Doxygen documentation.
Use the linked headers for complete overloads and default arguments.
Header paths below are relative to `MinkowskiEngine/` in the source tree.
Installed includes omit that prefix.

## Entry points

| Include | Namespace / scope |
|---|---|
| `<minkowski.hpp>` | Uniform sparse modules, utilities, serialization, and enabled Gaussian APIs |
| `<adaptive/adaptive.hpp>` | `minkowski::adaptive` |
| `<gaussian/gaussian_tensor.hpp>` | Batched Gaussian scene data |
| `<gaussian/gaussian_model.hpp>` | Trainable Gaussian model |
| `<gaussian/gaussian_render.hpp>` | Rasterizer and render output |
| `<gaussian/gaussian_trainer.hpp>` | 3DGS-style trainer |

## Uniform sparse API

### SparseTensor

[Source header]({{ site.repository_url }}/blob/main/MinkowskiEngine/sparse_tensor.hpp)

```cpp
SparseTensor(at::Tensor features,
             at::Tensor coordinates,
             std::shared_ptr<CoordinateManager> manager = nullptr,
             SparseTensorQuantizationMode::Type quantization_mode =
                 SparseTensorQuantizationMode::RANDOM_SUBSAMPLE,
             GPUMemoryAllocatorBackend::Type alloc_type =
                 GPUMemoryAllocatorBackend::PYTORCH,
             MinkowskiAlgorithm::Mode algo = MinkowskiAlgorithm::DEFAULT);

SparseTensor(at::Tensor features,
             CoordinateMapKey coordinate_map_key,
             std::shared_ptr<CoordinateManager> manager);
```

Use `[N,C]` floating features and `[N,D+1]` integer coordinates with batch
first. The key-based constructor preserves an existing coordinate map.

| API | Contract |
|---|---|
| `F()`, `C()` | Features and coordinates |
| `coordinate_map_key()`, `coordinate_manager()` | Map identity and manager |
| `tensor_stride()`, `dimension()` | Spatial metadata |
| `decomposed_*()` | Batch decomposition |
| `sparse()` | Torch sparse COO conversion |
| `to(dtype)`, `float_()`, `double_()` | Feature dtype conversion |
| `detach()` | Detached features with the existing map |
| Tensor arithmetic | Same manager and key required |
| `dense(...)`, `to(device)` | Declared but currently throw; not implemented |

### Coordinate management and kernels

- [CoordinateManager]({{ site.repository_url }}/blob/main/MinkowskiEngine/coordinate_manager.hpp):
  construct with spatial dimension and CPU/CUDA backend.
- [KernelGenerator]({{ site.repository_url }}/blob/main/MinkowskiEngine/kernel_generator.hpp):
  kernel configuration.
- [Types]({{ site.repository_url }}/blob/main/MinkowskiEngine/types.hpp):
  public coordinate backend and shared type definitions.

### TensorField

[Source header]({{ site.repository_url }}/blob/main/MinkowskiEngine/tensor_field.hpp)

```cpp
TensorField(at::Tensor features,
            at::Tensor coordinates,
            std::shared_ptr<CoordinateManager> manager = nullptr,
            SparseTensorQuantizationMode::Type quantization_mode =
                SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE);
```

`sparse(...)` voxelizes, `splat()` splats, and `inverse_mapping(key)` exposes
the field-to-sparse mapping. Inspect `F()`, `C()`, `device()`, `dtype()`,
`dimension()`, and `coordinate_field_map_key()`.

### Neural modules

| Family | Header |
|---|---|
| Convolution / transpose | [convolution.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/modules/convolution.hpp) |
| Local/global pooling | [pooling.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/modules/pooling.hpp) |
| Normalization | [normalization.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/modules/normalization.hpp) |
| Activations | [nonlinearity.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/modules/nonlinearity.hpp) |
| Linear | [linear.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/modules/linear.hpp) |
| Pruning | [pruning.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/modules/pruning.hpp) |
| Interpolation | [interpolation.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/modules/interpolation.hpp) |
| Functional helpers | [functional.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/functional.hpp) |

See [sparse networks](sparse-tensors.md) and the [classifier](quickstart.md)
for registration and `forward` composition.

### Serialization

[Source header]({{ site.repository_url }}/blob/main/MinkowskiEngine/serialization.hpp)

| Function | Purpose |
|---|---|
| `save_model`, `load_model` | LibTorch module persistence |
| `save_tensors`, `load_tensors` | Named tensor collections |
| `save_sparse_tensor`, `load_sparse_tensor` | Sparse tensor persistence |
| `load_sparse_tensor_data` | Coordinates, features, stride, dimension |
| `reconstruct_sparse_tensor` | Rebuild from data, optionally with a manager |
| `save_sparse_tensors`, `load_sparse_tensors_data` | Multiple sparse records |

## Adaptive API

[Public entry header]({{ site.repository_url }}/blob/main/MinkowskiEngine/adaptive/adaptive.hpp)

| API | Purpose |
|---|---|
| `AdaptiveVoxelConfig` | Base/min voxel sizes and depth |
| `AxisAlignedRegion` | Half-open spatial region |
| `AdaptiveSparseTensor::from_points` | Construct and retain source points |
| `AdaptiveSparseTensor::from_voxels` | Explicit nonoverlapping leaves |
| `features`, `coordinates`, `levels`, `centers`, `counts_by_level` | Inspect representation |
| `refine`, `coarsen` | Region topology edits |
| `refine_if`, `refine_by_density`, `refine_by_variance` | Rule-driven edits |
| `revoxelize_from_points` | Recompute features from retained points |
| `with_features`, `to` | Replace features or transfer tensors |
| `AdaptiveKernelConfig`, `AdaptiveConvMode` | Mixed-resolution kernel behavior |
| `adaptive_convolution`, `adaptive_convolution_transpose` | Differentiable feature kernels |
| `adaptive_pool`, `adaptive_global_avg_pool` | Local/global reductions |
| `interpolate`, `InterpMode` | Nearest, multilinear or continuous multilinear queries |
| `resample_uniform`, `resample_to` | Resolution/topology alignment |
| `KernelMapCache::instance()` | Cache statistics and clearing |

Convolution weights are `[K^D,Cin,Cout]`. See
[adaptive geometry and limitations](adaptive.md) for policies and mode semantics.

## Gaussian API

| Type / function | Header | Key contract |
|---|---|---|
| `Gaussian3D<T>` | [gaussian3d.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian3d.hpp) | Host math; w/x/y/z orientation |
| `GaussianTensor` | [gaussian_tensor.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_tensor.hpp) | Float32/64 CPU/CUDA activated data |
| `GaussianOptimizer` | [gaussian_optimizer.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_optimizer.hpp) | In-place Adam; tensor must outlive optimizer |
| PLY load/save | [gaussian_ply.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_ply.hpp) | Binary little-endian 3DGS/compressed input |
| `GaussianModel` | [gaussian_model.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_model.hpp) | Raw parameters, SH degree 0-3 |
| `Camera` | [gaussian_camera.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_camera.hpp) | OpenCV pinhole camera |
| SH evaluation | [gaussian_sh.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_sh.hpp) | View-dependent color |
| `render`, `render_gaussians` | [gaussian_render.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_render.hpp) | Float32, one device, differentiable outputs |
| `GaussianTrainer` | [gaussian_trainer.hpp]({{ site.repository_url }}/blob/main/MinkowskiEngine/gaussian/gaussian_trainer.hpp) | Model must outlive trainer |

### RenderOptions

`background` is an optional `[3]` tensor. `sh_degree=-1` uses the model's
degree. Defaults are `tile_size=16`, `chunk_size=1024`, and
`use_cuda_kernels=true`. Fused CUDA ignores tile/chunk tuning.

### GaussianTrainer

`step(camera, target)` returns `StepResult { loss, gaussians, iteration }`.
Targets are float32 RGB `[H,W,3]`. Additional entry points include
`render_image`, `iteration`, `active_sh_degree`, `config`,
`accumulate_statistics`, `densify_and_prune`, and `reset_opacity`.

See [render output shapes and trainer defaults](rendering.md).
