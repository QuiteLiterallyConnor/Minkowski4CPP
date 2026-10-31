---
title: Sparse tensors & networks
description: Coordinate maps, feature layouts, sparse modules, segmentation, and persistence.
permalink: /sparse-tensors/
---

{% raw %}
## Representation

A uniform `SparseTensor` contains feature rows and a coordinate map, not a
dense array of every grid cell.

| Value | Shape / type | Meaning |
|---|---|---|
| Features | `[N, C]`, floating tensor | Channels at occupied coordinates |
| Coordinates | `[N, D+1]`, int32 | Batch index, then spatial integer coordinates |
| Manager | `std::shared_ptr<CoordinateManager>` | Backend and coordinate/kernel-map ownership |
| Key | `CoordinateMapKey` | Identity of the tensor's map within its manager |
| Tensor stride | `stride_type` | Grid stride at this stage of the network |

For 3D, a coordinate row is `[batch, x, y, z]`. Spatial dimension is **3**,
not 4; the batch column is not a spatial axis.

```cpp
auto manager = std::make_shared<minkowski::CoordinateManager>(
    3, minkowski::CoordinateMapBackend::CPU);
auto coords = torch::tensor(
    {{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 1, 0}}, torch::kInt32);
auto features = torch::randn({3, 8});
minkowski::SparseTensor input(features, coords, manager);
```

Coordinate duplicates are subject to quantization. The constructor defaults
to `SparseTensorQuantizationMode::RANDOM_SUBSAMPLE`; select the appropriate
mode explicitly when multiple points map to one voxel. Do not assume that
the output feature rows remain in the original point order.

## Inspection and conversion

| Method | Result |
|---|---|
| `F()` | Feature tensor reference |
| `C()` | Coordinates |
| `dimension()`, `tensor_stride()` | Spatial dimension and grid stride |
| `coordinate_map_key()`, `coordinate_manager()` | Coordinate identity and owner |
| `device()`, `dtype()`, `requires_grad()` | Feature tensor metadata |
| `decomposed_coordinates()` / `decomposed_features()` | Per-batch tensors |
| `decomposed_coordinates_and_features()` | Per-batch coordinate/feature pairs |
| `sparse()` | Torch sparse COO conversion |
| `to(dtype)` | Feature dtype conversion |
| `detach()` | Tensor with detached feature gradients |

`dense(...)` and `to(device)` are declared but currently throw
`std::runtime_error` because they are not fully implemented. To use another
device, construct input features/coordinates on that device with a matching
manager rather than calling `SparseTensor::to(device)`.
Sparse COO conversion needs coordinates suitable for nonnegative tensor
indices; do not treat it as an arbitrary signed-coordinate export.

## Continuous point fields

`TensorField` represents features attached to continuous coordinates. Its
default quantization is `UNWEIGHTED_AVERAGE`.

- `sparse(tensor_stride, coordinate_map_key, quantization_mode)` creates a
  voxelized sparse representation.
- `splat()` constructs a splatted sparse representation.
- `inverse_mapping(sparse_key)` exposes the mapping back to the field.

Preserve these mappings when supervising per-point segmentation: the number
and order of voxel rows can differ from the input points.

## Network modules

Sparse modules are LibTorch modules. Register them with `register_module`,
use `TORCH_MODULE` holders, and pass `SparseTensor` through `forward`.

```cpp
auto conv = minkowski::MinkowskiConvolution(
    /*in_channels=*/8, /*out_channels=*/32,
    /*kernel_size=*/3, /*stride=*/1, /*dilation=*/1,
    /*bias=*/false, /*expand_coordinates=*/false,
    minkowski::ConvolutionMode::DEFAULT, /*dimension=*/3);
auto output = conv->forward(input);
```

The convolution constructor's dimension must match the input coordinate
dimension. Stride changes resolution; dilation changes kernel spacing.

| Family | Public header | Role |
|---|---|---|
| Convolution | `modules/convolution.hpp` | Convolution and transpose convolution |
| Pooling | `modules/pooling.hpp` | Local and global aggregation |
| Normalization | `modules/normalization.hpp` | Batch and instance normalization |
| Nonlinearities | `modules/nonlinearity.hpp` | Activations such as `MinkowskiReLU` |
| Linear | `modules/linear.hpp` | Feature-space linear layers |
| Pruning | `modules/pruning.hpp` | Remove selected coordinates |
| Interpolation | `modules/interpolation.hpp` | Query features |
| Broadcast / union | `modules/broadcast.hpp`, `modules/union_op.hpp` | Coordinate-aware combination |
| Blocks | `modules/resnet_block.hpp`, `modules/senet_block.hpp` | Reusable network components |

The [quick start](quickstart.md) gives a complete registered classifier.

## Residual connections and segmentation

A typical 3D segmentation architecture consists of:

1. A stem convolution block.
2. Encoder residual blocks separated by strided convolutions.
3. A bottleneck at the coarsest scale.
4. Transpose convolutions to restore encoder resolutions.
5. Coordinate-aligned skip connections.
6. A kernel-size-1 convolution producing class logits.

Residual addition (`out + input`) requires the **same manager and map key**.
Cross-key addition, subtraction, multiplication, and division currently
throw rather than constructing a union. Preserve these relationships and
verify coordinates before attaching labels. Features from different
coordinate sets must be aligned explicitly; equal row counts are insufficient.

For classification, global pooling produces one row per batch. For
segmentation, the head produces rows at the output coordinate map; those
rows must be mapped to the correct point/voxel targets before cross entropy.
The README's toy UNet illustrates the architecture, not a substitute for
label alignment in a real dataset.

## Training workflow

Use a fresh coordinate manager for a new forward-pass input, as in the
README's training example. Keep all tensors and the coordinate backend on
compatible devices.

```cpp
model->train();
optimizer.zero_grad();
auto output = model->forward(input);
auto loss = torch::nn::functional::cross_entropy(output.F(), aligned_labels);
loss.backward();
optimizer.step();
```

`aligned_labels` must be `int64`, with one label for each output logit row.
For inference use `model->eval()` and `torch::NoGradGuard`.

The repository's `train` executable provides data loading, checkpoints, and
evaluation. See [training tests and commands](testing.md#training-example).

## Save and load

```cpp
minkowski::save_model(model, "network.pt");
minkowski::load_model(model, "network.pt");

minkowski::save_sparse_tensor(input, "input.pt");
auto restored = minkowski::load_sparse_tensor("input.pt");
```

Instantiate the same model architecture before loading weights.
`save_tensors` / `load_tensors` handle named tensor collections.
For explicit sparse reconstruction, use `load_sparse_tensor_data` and
`reconstruct_sparse_tensor(data, manager)`. Data includes coordinates,
features, tensor stride, and dimension.

See [API reference](api.md#uniform-sparse-api) for the corresponding headers.
{% endraw %}
