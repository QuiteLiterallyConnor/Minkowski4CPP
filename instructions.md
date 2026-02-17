# MinkowskiEngine → Pure C++ (LibTorch + CUDA) Conversion Roadmap

## Overview

MinkowskiEngine has a mature C++ backend (`mink/`) that already implements:
- Coordinate map management (CPU + GPU)
- Sparse convolution forward/backward kernels
- Pooling (local, global, transpose, max) forward/backward kernels
- Broadcast forward/backward kernels
- Pruning forward/backward kernels
- Interpolation forward/backward kernels
- SpMM (sparse matrix–matrix multiply) via cuSPARSE
- Quantization (voxelisation / dedup)

The Python layer (`MinkowskiEngine/MinkowskiEngine/*.py`) adds:
- `torch.autograd.Function` subclasses that call forward/backward C++ kernels
- `torch.nn.Module` subclasses (convolution, pooling, normalization, activation, etc.)
- `SparseTensor` / `TensorField` data containers
- `KernelGenerator` (kernel region geometry)
- `CoordinateManager` (Python wrapper around C++ `CoordinateMapManager`)
- Various utilities (collation, quantization helpers, serialization)

**Goal:** Rewrite all Python-layer logic in pure C++ using LibTorch (`torch::nn`, `torch::autograd`, `at::Tensor`), so the library can be consumed from C++ without any Python dependency. The existing `mink/` C++ kernels are reused as-is.

---

## Architecture at a Glance

```
┌─────────────────────────── NEW C++ LAYER ───────────────────────────┐
│                                                                     │
│  SparseTensor  TensorField   (data containers, operator overloads)  │
│       │              │                                              │
│  CoordinateManager   KernelGenerator   (coordinate/kernel logic)    │
│       │                    │                                        │
│  Autograd Functions  ←─────┘  (ConvolutionFn, PoolingFn, etc.)     │
│       │                                                             │
│  nn::Module layers   (MinkowskiConvolution, BatchNorm, ReLU, etc.) │
│       │                                                             │
│  Utilities  (sparse_collate, sparse_quantize, kaiming_init, etc.)  │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ calls
┌──────────────────────────────▼──────────────────────────────────────┐
│                    EXISTING mink/ C++ KERNELS                       │
│  ConvolutionForward/BackwardCPU/GPU, LocalPoolingForward/Backward,  │
│  GlobalPoolingForward/Backward, BroadcastForward/Backward,          │
│  PruningForward/Backward, InterpolationForward/Backward,            │
│  CoordinateMapManager, coo_spmm, quantize, direct_max_pool, etc.   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Build System & Project Skeleton

### 0.1 CMake build system
- Create a top-level `CMakeLists.txt` that:
  - Finds LibTorch (`find_package(Torch REQUIRED)`)
  - Finds CUDA toolkit
  - Compiles all `mink/*.cpp` and `mink/*.cu` files into a library target
  - Compiles the new C++ layer into a second library/target that links against the mink kernel library and LibTorch
  - Optionally builds a test/example executable
- Support both CPU-only (`-DCPU_ONLY`) and GPU builds
- Link `cusparse` for GPU builds

### 0.2 Directory layout
```
mink/                         ← existing kernel code (unchanged)
mink_cpp/                     ← NEW pure-C++ layer
  mink_cpp/types.hpp          ← re-export mink/types.hpp + any new type aliases
  mink_cpp/sparse_tensor.hpp / .cpp
  mink_cpp/tensor_field.hpp / .cpp
  mink_cpp/coordinate_manager.hpp / .cpp
  mink_cpp/kernel_generator.hpp / .cpp
  mink_cpp/autograd/          ← autograd functions
    convolution_fn.hpp / .cpp
    pooling_fn.hpp / .cpp
    broadcast_fn.hpp / .cpp
    pruning_fn.hpp / .cpp
    interpolation_fn.hpp / .cpp
    spmm_fn.hpp / .cpp
    instance_norm_fn.hpp / .cpp
  mink_cpp/modules/           ← nn::Module wrappers
    convolution.hpp / .cpp
    pooling.hpp / .cpp
    normalization.hpp / .cpp
    nonlinearity.hpp / .cpp
    linear.hpp / .cpp
    pruning.hpp / .cpp
    broadcast.hpp / .cpp
    union_op.hpp / .cpp
    ops.hpp / .cpp             ← cat, to_dense, to_sparse, etc.
  mink_cpp/utils/
    collation.hpp / .cpp
    quantization.hpp / .cpp
    init.hpp / .cpp
  mink_cpp/functional.hpp / .cpp  ← functional API (optional)
  mink_cpp/minkowski.hpp          ← convenience "include everything" header
CMakeLists.txt
tests/                         ← C++ test programs
examples/
  example_unet.cpp             ← port of example_trainer.py
```

### 0.3 Remove pybind11 dependency from kernel code
- The existing `mink/coordinate_map_manager.hpp` returns `py::object` in several methods (`insert_and_map`, `insert_field`, etc.). These need to be refactored:
  - Add non-pybind overloads that return `std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>` (or similar C++ types) instead of `py::object`.
  - Gate the existing `py::object`-returning methods behind `#ifdef MINK_WITH_PYBIND` so the core library compiles without pybind11.
  - The `py_stride()` method → replace with a pure-C++ `stride()` that returns `CoordinateMapKey` directly.

---

## Phase 1 — Core Data Structures

### 1.1 `KernelGenerator` (port of `MinkowskiKernelGenerator.py`)

**What it does (Python):**
- Stores kernel_size, stride, dilation, region_type, region_offsets, axis_types, dimension
- `get_kernel_volume()` — computes number of kernel elements for HYPER_CUBE / HYPER_CROSS / CUSTOM
- `convert_region_type()` — resolves axis_types / hybrid into final region_type + offset tensor
- `get_kernel(tensor_stride, is_transpose)` — returns `(kernel_size, kernel_stride, kernel_dilation, region_type, region_offset, expand_coordinates, D)` tuple; caches by tensor_stride

**C++ plan:**
```cpp
class KernelGenerator {
public:
  KernelGenerator(
    torch::IntArrayRef kernel_size,
    torch::IntArrayRef stride = {1},
    torch::IntArrayRef dilation = {1},
    bool is_transpose = false,
    RegionType::Type region_type = RegionType::HYPER_CUBE,
    at::Tensor region_offsets = {},
    bool expand_coordinates = false,
    std::vector<RegionType::Type> axis_types = {},
    int dimension = -1);

  int kernel_volume() const;
  stride_type const& kernel_size() const;
  stride_type const& kernel_stride() const;
  stride_type const& kernel_dilation() const;
  RegionType::Type region_type() const;
  at::Tensor const& region_offsets() const;
  bool expand_coordinates() const;
  bool requires_strided_coordinates() const;

  struct KernelParams {
    stride_type kernel_size, kernel_stride, kernel_dilation;
    RegionType::Type region_type;
    at::Tensor region_offsets;
    bool expand_coordinates;
    int D;
  };
  KernelParams get_kernel(stride_type const& tensor_stride, bool is_transpose) const;

private:
  // Cache: tensor_stride -> KernelParams
  mutable std::unordered_map<stride_type, KernelParams, ...> cache_;
  // ... member fields ...
};
```

- Port `get_kernel_volume()` and `convert_region_type()` as free functions or static methods.
- The HYBRID region type logic (expanding axis_types into CUSTOM offsets) must be ported.

### 1.2 `CoordinateManager` (port of `MinkowskiCoordinateManager.py`)

**What it does (Python):**
- Thin wrapper around the C++ `CoordinateMapManager{CPU|GPU_default|GPU_c10}`
- Selects the backend based on `coordinate_map_type` / `allocator_type`
- Exposes `insert_and_map`, `stride`, `origin`, `kernel_map`, `origin_map`, `union_map`, `get_coordinates`, `size`, `interpolation_map_weight`, etc.

**C++ plan:**
```cpp
class CoordinateManager {
public:
  CoordinateManager(
    int D,
    CoordinateMapBackend::Type map_type = CoordinateMapBackend::CUDA,
    GPUMemoryAllocatorBackend::Type alloc_type = GPUMemoryAllocatorBackend::PYTORCH,
    MinkowskiAlgorithm::Mode algo = MinkowskiAlgorithm::DEFAULT);

  // Core coordinate operations — delegate to internal manager_
  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
    insert_and_map(at::Tensor const& coordinates, stride_type tensor_stride,
                   std::string string_id = "");

  CoordinateMapKey stride(CoordinateMapKey const& in_key,
                          stride_type const& kernel_stride,
                          std::string string_id = "");

  std::pair<CoordinateMapKey, bool> origin();
  at::Tensor get_coordinates(CoordinateMapKey const& key);
  uint32_t size(CoordinateMapKey const& key);

  // Kernel maps
  // Returns const reference to cached kernel map
  auto const& kernel_map(CoordinateMapKey const& in_key,
                         CoordinateMapKey const& out_key,
                         stride_type const& kernel_size,
                         stride_type const& kernel_stride,
                         stride_type const& kernel_dilation,
                         RegionType::Type region_type,
                         at::Tensor const& offsets,
                         bool is_transpose,
                         bool is_pool);

  auto const& origin_map(CoordinateMapKey const& key);
  std::vector<at::Tensor> union_map(
    std::vector<CoordinateMapKey> const& in_keys,
    CoordinateMapKey& out_key);

  std::vector<at::Tensor> interpolation_map_weight(
    at::Tensor const& tfield, CoordinateMapKey const& key);

  // Field operations (for TensorField)
  CoordinateMapKey insert_field(at::Tensor const& coordinates,
                                stride_type tensor_stride, std::string string_id = "");
  std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>
    field_to_sparse_insert_and_map(CoordinateMapKey const& field_key,
                                   stride_type sparse_stride, std::string string_id = "");

  // Access internal manager (for passing to kernel functions)
  template <typename ManagerType>
  ManagerType* manager_ptr();

  int D() const;

private:
  int D_;
  CoordinateMapBackend::Type map_type_;
  GPUMemoryAllocatorBackend::Type alloc_type_;
  // Type-erased internal manager (variant or shared_ptr<void> + cast)
  // One of: cpu_manager_type<int32_t>*, gpu_default_manager_type<int32_t>*, gpu_c10_manager_type<int32_t>*
  std::variant<
    std::unique_ptr<cpu_manager_type<int32_t>>,
    std::unique_ptr<gpu_default_manager_type<int32_t>>,
    std::unique_ptr<gpu_c10_manager_type<int32_t>>
  > manager_;
};
```

The key challenge is that the existing `CoordinateMapManager` is heavily templated. The C++ `CoordinateManager` wrapper needs to type-erase or use `std::variant` to hold whichever concrete manager type applies.

**Refactoring needed in `mink/coordinate_map_manager.hpp`:**
- Add pure-C++ return types for methods currently returning `py::object`:
  - `insert_and_map` → return `std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>`
  - `insert_field` → return `CoordinateMapKey`
  - `field_to_sparse_insert_and_map` → return `std::pair<CoordinateMapKey, std::pair<at::Tensor, at::Tensor>>`
  - `py_stride` → use existing `stride()` which already returns C++ types
  - `get_coordinate_map_keys` → return `std::vector<CoordinateMapKey>`
  - `field_to_sparse_keys` → return `std::vector<CoordinateMapKey>`

### 1.3 `SparseTensor` (port of `MinkowskiSparseTensor.py` + `MinkowskiTensor.py`)

**What it does (Python):**
- Holds `_F` (feature tensor), `coordinate_map_key`, `_manager` (CoordinateManager)
- Lazily fetches coordinates `_C` from the manager
- Constructor either creates new coordinate map (`insert_and_map`) or reuses existing key
- Handles quantization modes during initialization
- Operator overloads (`+`, `-`, `*`, `/`) with same-key and cross-key (union) support
- Batch decomposition (`decomposed_coordinates`, `decomposed_features`)
- Conversion to dense, to sparse torch tensor

**C++ plan:**
```cpp
class SparseTensor {
public:
  // Construct from coordinates + features (inserts into manager)
  SparseTensor(
    at::Tensor features,
    at::Tensor coordinates,
    std::shared_ptr<CoordinateManager> manager = nullptr,
    SparseTensorQuantizationMode quantization_mode = SparseTensorQuantizationMode::RANDOM_SUBSAMPLE,
    GPUMemoryAllocatorBackend::Type alloc_type = GPUMemoryAllocatorBackend::PYTORCH,
    MinkowskiAlgorithm::Mode algo = MinkowskiAlgorithm::DEFAULT);

  // Construct from existing key + features (wrap existing coordinate map)
  SparseTensor(
    at::Tensor features,
    CoordinateMapKey coordinate_map_key,
    std::shared_ptr<CoordinateManager> manager);

  // Accessors
  at::Tensor const& F() const;             // features
  at::Tensor C() const;                     // coordinates (fetched from manager)
  CoordinateMapKey const& coordinate_map_key() const;
  std::shared_ptr<CoordinateManager> coordinate_manager() const;
  stride_type tensor_stride() const;
  int dimension() const;
  torch::Device device() const;
  c10::ScalarType dtype() const;
  bool requires_grad() const;

  // Batch decomposition
  std::vector<at::Tensor> decomposed_coordinates() const;
  std::vector<at::Tensor> decomposed_features() const;
  std::vector<std::pair<at::Tensor,at::Tensor>> decomposed_coordinates_and_features() const;

  // Conversions
  at::Tensor dense(c10::optional<std::vector<int64_t>> shape = c10::nullopt,
                   c10::optional<at::Tensor> min_coordinate = c10::nullopt) const;
  at::Tensor sparse() const;  // torch.sparse COO

  // Operator overloads
  SparseTensor operator+(SparseTensor const& other) const;
  SparseTensor operator-(SparseTensor const& other) const;
  SparseTensor operator*(SparseTensor const& other) const;
  SparseTensor operator/(SparseTensor const& other) const;
  SparseTensor& operator+=(SparseTensor const& other);

  // Scalar overloads
  SparseTensor operator+(at::Scalar s) const;
  SparseTensor operator*(at::Scalar s) const;

  // Type casting
  SparseTensor to(torch::Device device) const;
  SparseTensor to(c10::ScalarType dtype) const;
  SparseTensor float_() const;
  SparseTensor double_() const;
  SparseTensor detach() const;

private:
  at::Tensor features_;
  mutable c10::optional<at::Tensor> coordinates_;  // lazy-cached
  CoordinateMapKey key_;
  std::shared_ptr<CoordinateManager> manager_;
  int D_;
};
```

**Key implementation details:**
- Quantization during construction: when coordinates are provided, call `manager->insert_and_map()` to get `unique_index` and `inverse_mapping`. Then apply quantization:
  - `RANDOM_SUBSAMPLE` → `features_[unique_index]`
  - `UNWEIGHTED_AVERAGE` → `MinkowskiSPMMAverageFunction::apply(inverse, unique, size, features)`
  - `UNWEIGHTED_SUM` → `MinkowskiSPMMFunction::apply(inverse, unique, ones, size, features)`
  - `MAX_POOL` → `scatter_max` via torch ops
  - `NO_QUANTIZATION` → use features as-is (assertion: no duplicates)
- Binary operators with different keys need union_map from the manager.

### 1.4 `TensorField` (port of `MinkowskiTensorField.py`)

**What it does (Python):**
- Like SparseTensor but for continuous (float) coordinates
- Uses `insert_field()` instead of `insert_and_map()`
- `sparse()` method converts to SparseTensor via `field_to_sparse_insert_and_map()`
- `splat()` for linear interpolation to 2^D neighbors
- Inverse mapping tracking

**C++ plan:**
```cpp
class TensorField {
public:
  TensorField(
    at::Tensor features,
    at::Tensor coordinates,           // float coordinates
    std::shared_ptr<CoordinateManager> manager = nullptr,
    SparseTensorQuantizationMode quantization_mode = SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE);

  // Convert to SparseTensor
  SparseTensor sparse(
    stride_type tensor_stride = {1},
    c10::optional<CoordinateMapKey> coordinate_map_key = c10::nullopt,
    SparseTensorQuantizationMode quantization_mode = SparseTensorQuantizationMode::UNWEIGHTED_AVERAGE) const;

  SparseTensor splat() const;  // linear interpolation to 2^D neighbors

  at::Tensor inverse_mapping(CoordinateMapKey const& sparse_key) const;

  // Accessors (same pattern as SparseTensor)
  at::Tensor const& F() const;
  at::Tensor C() const;
  CoordinateMapKey const& coordinate_field_map_key() const;
  std::shared_ptr<CoordinateManager> coordinate_manager() const;

private:
  at::Tensor features_;
  CoordinateMapKey field_key_;
  std::shared_ptr<CoordinateManager> manager_;
  int D_;
  mutable std::unordered_map<CoordinateMapKey, at::Tensor, ...> inverse_mappings_;
};
```

---

## Phase 2 — Autograd Functions

Each Python `torch.autograd.Function` subclass becomes a C++ `torch::autograd::Function<Derived>` with static `forward()` and `backward()` methods. The C++ autograd machinery uses `AutogradContext*` instead of Python's `ctx`.

### 2.1 `ConvolutionFunction` / `ConvolutionTransposeFunction`

**Python behavior:**
- `forward(ctx, input_features, kernel_weights, kernel_generator, convolution_mode, in_key, out_key, coord_manager)`:
  - Calls `get_minkowski_function("ConvolutionForward", input_features)` → dispatches to CPU or GPU C++ function
  - Passes kernel_size, stride, dilation, region_type, region_offsets, expand_coordinates, convolution_mode, and the C++ manager
  - Saves `input_features`, `kernel_weights`, kernel params, keys, manager for backward
- `backward(ctx, grad_output)`:
  - Calls `ConvolutionBackward{GPU|CPU}` with saved state
  - Returns `(grad_input_features, grad_kernel, None, None, None, None, None)`

**C++ plan:**
```cpp
class ConvolutionFunction : public torch::autograd::Function<ConvolutionFunction> {
public:
  static at::Tensor forward(
    torch::autograd::AutogradContext* ctx,
    at::Tensor input_features,
    at::Tensor kernel,
    // Kernel params
    stride_type kernel_size, stride_type kernel_stride, stride_type kernel_dilation,
    RegionType::Type region_type, at::Tensor region_offsets,
    bool expand_coordinates,
    ConvolutionMode::Type conv_mode,
    CoordinateMapKey in_key, CoordinateMapKey out_key,
    std::shared_ptr<CoordinateManager> manager);

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_outputs);
};
```

**Dispatch pattern** (replaces Python `get_minkowski_function`):
```cpp
inline bool is_gpu(at::Tensor const& t) { return t.is_cuda(); }

// In forward():
at::Tensor out;
if (is_gpu(input_features)) {
  #ifndef CPU_ONLY
  out = ConvolutionForwardGPU<int32_t, detail::default_allocator>(
    input_features, kernel, kernel_size, kernel_stride, kernel_dilation,
    region_type, region_offsets, expand_coordinates, conv_mode,
    &in_key, &out_key, manager->gpu_manager_ptr());
  #endif
} else {
  out = ConvolutionForwardCPU<int32_t>(
    input_features, kernel, kernel_size, kernel_stride, kernel_dilation,
    region_type, region_offsets, expand_coordinates, conv_mode,
    &in_key, &out_key, manager->cpu_manager_ptr());
}
```

**Context saving:** Use `ctx->save_for_backward({input_features, kernel})` for tensors, and save non-tensor state via `ctx->saved_data["key"] = IValue(...)` or a custom struct stored via `save_for_backward`.

### 2.2 `LocalPoolingFunction` / `LocalPoolingTransposeFunction`

Same pattern as convolution. Forward calls `LocalPoolingForward{CPU|GPU}`, backward calls `LocalPoolingBackward{CPU|GPU}`.

The forward returns `pair<Tensor, Tensor>` (output, num_nonzero). The num_nonzero tensor must be saved for backward.

### 2.3 `GlobalPoolingFunction`

Forward calls `GlobalPoolingForward{CPU|GPU}` → returns `tuple<Tensor, Tensor>` (output, num_nonzero).
Backward calls `GlobalPoolingBackward{CPU|GPU}`.

### 2.4 `DirectMaxPoolingFunction`

Forward calls `direct_max_pool_fw(in_map, out_map, in_feat, out_nrows, is_sorted)` → returns `pair<Tensor, Tensor>` (output, mask).
Backward calls `direct_max_pool_bw(grad_output, mask_index, in_nrows)`.

### 2.5 `BroadcastFunction`

Forward calls `BroadcastForward{CPU|GPU}(in_feat, in_feat_global, op_type, in_key, glob_key, manager)`.
Backward calls `BroadcastBackward{CPU|GPU}`.

### 2.6 `PruningFunction`

Forward calls `PruningForward{CPU|GPU}(in_feat, keep_mask, in_key, out_key, manager)`.
Backward calls `PruningBackward{CPU|GPU}(grad_out, in_key, out_key, manager)`.

### 2.7 `InterpolationFunction`

Forward calls `InterpolationForward{CPU|GPU}(in_feat, tfield, in_key, manager)` → returns `vector<Tensor>` (out_feat, in_map, out_map, weights).
Backward calls `InterpolationBackward{CPU|GPU}(grad_out, in_map, out_map, weights, in_key, manager)`.

### 2.8 `SPMMFunction` / `SPMMAverageFunction`

**SPMMFunction:**
- Forward: GPU → `coo_spmm_int32(rows, cols, vals, M, N, mat, alg, is_sorted)`. CPU → build torch sparse COO and matmul.
- Backward: transpose spmm (swap rows/cols).

**SPMMAverageFunction:**
- Forward: GPU → `coo_spmm_average_int32(rows, cols, M, N, mat, alg)` → returns `vector<Tensor>` (result, COO_vals). CPU → compute 1/count weights, sparse matmul.
- Backward: transpose spmm with saved weights.

### 2.9 `InstanceNormFunction`

This one is more complex — implemented in Python using other C++ ops:
- Calls `GlobalPoolingForward` to compute mean
- Calls `BroadcastForward` to subtract mean
- Computes variance manually
- Normalizes, scales, shifts

Port this logic to C++ autograd directly, calling the same C++ kernel functions.

---

## Phase 3 — nn::Module Layers

All modules use LibTorch's `torch::nn::Module` (via `TORCH_MODULE` macro or `torch::nn::Cloneable`).

### 3.1 `MinkowskiConvolution` / `MinkowskiConvolutionTranspose` / `MinkowskiGenerativeConvolutionTranspose`

**Python behavior:**
- Constructor: creates `KernelGenerator`, allocates `kernel` Parameter (shape `[kernel_volume, in_ch, out_ch]` or `[in_ch, out_ch]` if kernel_volume==1), optional `bias` Parameter
- `forward(input: SparseTensor, coords=None) -> SparseTensor`:
  - If `kernel_volume == 1` and stride == 1: shortcut via `input.F.mm(kernel.squeeze()) + bias`
  - Otherwise: call `ConvolutionFunction::apply(...)` or `ConvolutionTransposeFunction::apply(...)`
  - Wrap output in new SparseTensor with the output key

**C++ plan:**
```cpp
struct MinkowskiConvolutionOptions {
  int in_channels, out_channels;
  std::vector<int> kernel_size = {3};
  std::vector<int> stride = {1};
  std::vector<int> dilation = {1};
  bool bias = true;
  int dimension = 3;
  ConvolutionMode::Type convolution_mode = ConvolutionMode::DEFAULT;
  // ... region_type, expand_coordinates, etc.
};

class MinkowskiConvolutionImpl : public torch::nn::Cloneable<MinkowskiConvolutionImpl> {
public:
  explicit MinkowskiConvolutionImpl(MinkowskiConvolutionOptions options);
  void reset() override;
  SparseTensor forward(SparseTensor const& input,
                       c10::optional<at::Tensor> coordinates = c10::nullopt);

  MinkowskiConvolutionOptions options_;
  torch::Tensor kernel_{nullptr};
  torch::Tensor bias_{nullptr};
  KernelGenerator kernel_generator_;
};
TORCH_MODULE(MinkowskiConvolution);
```

Repeat for `MinkowskiConvolutionTranspose` (sets `is_transpose=true`) and `MinkowskiGenerativeConvolutionTranspose` (sets `expand_coordinates=true`).

### 3.2 `MinkowskiChannelwiseConvolution`

**Python behavior:** Depthwise conv — kernel shape `[kernel_volume, in_channels]`, Python-loop over kernel map entries.

Port the loop to C++. Consider a fused CUDA kernel later for performance, but the Python-loop-equivalent (iterate kernel map entries, multiply + accumulate) is sufficient initially.

### 3.3 Pooling Modules

| Module | Autograd Function used | PoolingMode |
|---|---|---|
| `MinkowskiSumPooling` | `LocalPoolingFunction` | `LOCAL_SUM_POOLING` |
| `MinkowskiAvgPooling` | `LocalPoolingFunction` | `LOCAL_AVG_POOLING` |
| `MinkowskiMaxPooling` | `LocalPoolingFunction` | `LOCAL_MAX_POOLING` |
| `MinkowskiPoolingTranspose` | `LocalPoolingTransposeFunction` | `LOCAL_SUM_POOLING` |
| `MinkowskiGlobalPooling` | `GlobalPoolingFunction` | configurable |
| `MinkowskiGlobalSumPooling` | `GlobalPoolingFunction` | `GLOBAL_SUM_POOLING_*` |
| `MinkowskiGlobalAvgPooling` | `GlobalPoolingFunction` | `GLOBAL_AVG_POOLING_*` |
| `MinkowskiGlobalMaxPooling` | `GlobalPoolingFunction` | `GLOBAL_MAX_POOLING_*` |

Each is a thin module that stores `KernelGenerator` + `PoolingMode`, and calls the corresponding autograd function in `forward()`.

### 3.4 Normalization Modules

**`MinkowskiBatchNorm`:**
- Wraps `torch::nn::BatchNorm1d` internally
- `forward`: extract `input.F()`, apply BN, wrap result in SparseTensor with same key
- Must handle `training` vs `eval` mode (running stats)

**`MinkowskiSyncBatchNorm`:**
- Same, wraps `torch::nn::SyncBatchNorm` (for multi-GPU DDP)

**`MinkowskiInstanceNorm`:**
- Uses `InstanceNormFunction` autograd function
- Learnable `weight` and `bias` parameters

**`MinkowskiStableInstanceNorm`:**
- Module-only implementation using `MinkowskiGlobalAvgPooling` + `MinkowskiBroadcastAddition/Multiplication`
- No custom autograd; composed from other modules

### 3.5 Nonlinearity Modules

All activations are trivial wrappers:
```cpp
template <typename TorchModule>
class MinkowskiNonlinearity : public torch::nn::Cloneable<MinkowskiNonlinearity<TorchModule>> {
  TorchModule module_;
public:
  SparseTensor forward(SparseTensor const& input) {
    return SparseTensor(module_->forward(input.F()),
                        input.coordinate_map_key(),
                        input.coordinate_manager());
  }
};

using MinkowskiReLU = MinkowskiNonlinearity<torch::nn::ReLU>;
using MinkowskiLeakyReLU = MinkowskiNonlinearity<torch::nn::LeakyReLU>;
using MinkowskiELU = MinkowskiNonlinearity<torch::nn::ELU>;
using MinkowskiGELU = MinkowskiNonlinearity<torch::nn::GELU>;
using MinkowskiSigmoid = MinkowskiNonlinearity<torch::nn::Sigmoid>;
using MinkowskiTanh = MinkowskiNonlinearity<torch::nn::Tanh>;
using MinkowskiSoftmax = MinkowskiNonlinearity<torch::nn::Softmax>;
using MinkowskiDropout = MinkowskiNonlinearity<torch::nn::Dropout>;
// ... etc for all 28 activations
```

### 3.6 `MinkowskiLinear`

```cpp
class MinkowskiLinearImpl : public torch::nn::Cloneable<MinkowskiLinearImpl> {
  torch::nn::Linear linear_{nullptr};
public:
  MinkowskiLinearImpl(int in_features, int out_features, bool bias = true);
  SparseTensor forward(SparseTensor const& input);
};
```

### 3.7 `MinkowskiPruning`

```cpp
class MinkowskiPruningImpl : public torch::nn::Cloneable<MinkowskiPruningImpl> {
public:
  SparseTensor forward(SparseTensor const& input, at::Tensor mask);
};
```
Creates a new `CoordinateMapKey` for the output, calls `PruningFunction::apply()`.

### 3.8 `MinkowskiUnion`

```cpp
class MinkowskiUnionImpl : public torch::nn::Cloneable<MinkowskiUnionImpl> {
public:
  SparseTensor forward(std::vector<SparseTensor> const& inputs);
};
```

Port the `MinkowskiUnionFunction` logic:
- Get union maps from `manager->union_map(in_keys, out_key)`
- Zero-init output, scatter each input's features to output positions
- Backward: scatter gradient back to each input

### 3.9 `MinkowskiBroadcast` / `MinkowskiBroadcastAddition` / `MinkowskiBroadcastMultiplication`

- The C++-backed versions call `BroadcastFunction::apply()`
- The pure-torch `MinkowskiBroadcast` uses `origin_map` to expand global features — port this gather logic to C++

### 3.10 `MinkowskiInterpolation`

Calls `InterpolationFunction::apply()`, wraps result.

---

## Phase 4 — Utility Functions

### 4.1 Collation (`utils/collation.py`)

**`batched_coordinates(coords_list)`:**
- Prepends batch index column to each coordinate array
- Concatenates into single tensor `[sum(N_i), D+1]`

```cpp
at::Tensor batched_coordinates(
  std::vector<at::Tensor> const& coords_list,
  c10::ScalarType dtype = at::kInt,
  torch::Device device = torch::kCPU);
```

**`sparse_collate(coords_list, feats_list, labels_list)`:**
- Calls `batched_coordinates` on coords
- Concatenates features and labels
- Returns tuple `(batched_coords, batched_feats, batched_labels)`

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor>
sparse_collate(
  std::vector<at::Tensor> const& coords,
  std::vector<at::Tensor> const& feats,
  std::vector<at::Tensor> const& labels = {});
```

### 4.2 Quantization (`utils/quantization.py`)

**`sparse_quantize(coordinates, features, labels, quantization_size, device)`:**
- Scales coordinates by 1/quantization_size, floors to int
- Deduplicates via the C++ `quantize_th` or via a CoordinateMapManager
- Returns (unique_coords, [unique_feats], [unique_labels], [unique_index], [inverse_mapping])

The underlying C++ functions `quantize_th` and `quantize_label_th` already exist in `mink/quantization.cpp`. Wrap them.

```cpp
struct SparseQuantizeResult {
  at::Tensor coordinates;
  c10::optional<at::Tensor> features;
  c10::optional<at::Tensor> labels;
  at::Tensor unique_index;
  at::Tensor inverse_mapping;
};

SparseQuantizeResult sparse_quantize(
  at::Tensor coordinates,
  c10::optional<at::Tensor> features = c10::nullopt,
  c10::optional<at::Tensor> labels = c10::nullopt,
  float quantization_size = 1.0f,
  torch::Device device = torch::kCPU);
```

### 4.3 Weight Initialization (`utils/init.py`)

**`kaiming_normal_(tensor, a, mode, nonlinearity)`:**
- For 3D tensors `[kernel_volume, in_ch, out_ch]`: computes fan_in/fan_out per kernel_volume, then fills with normal.

```cpp
void kaiming_normal_(at::Tensor& tensor, double a = 0,
                     std::string mode = "fan_in",
                     std::string nonlinearity = "leaky_relu");
```

### 4.4 Operations (`MinkowskiOps.py`)

- `cat(tensors...)` → concatenate features (same key required)
- `to_sparse(dense_tensor)` → extract nonzero coordinates, build SparseTensor
- `to_sparse_all(dense_tensor, coordinates)` → build SparseTensor from all specified coordinates
- `dense_coordinates(shape)` → generate dense grid coordinate tensor
- `MinkowskiToSparseTensor`, `MinkowskiToDenseTensor`, `MinkowskiToFeature` modules

```cpp
SparseTensor cat(std::vector<SparseTensor> const& tensors);
SparseTensor sum(std::vector<SparseTensor> const& tensors);
SparseTensor mean(std::vector<SparseTensor> const& tensors);
SparseTensor to_sparse(at::Tensor dense_tensor, ...);
at::Tensor dense_coordinates(at::IntArrayRef shape);
```

### 4.5 Functional API (optional, `MinkowskiFunctional.py`)

Port functional versions of activations, normalization, loss functions. These all operate on `SparseTensor.F()` using `torch::nn::functional`, then wrap the result.

This is lower priority since the module versions cover all training/inference needs.

---

## Phase 5 — Predefined Network Blocks (`modules/`)

### 5.1 ResNet Blocks

```cpp
class BasicBlockImpl : public torch::nn::Cloneable<BasicBlockImpl> {
  // 2x MinkowskiConvolution + MinkowskiBatchNorm + residual
  static constexpr int expansion = 1;
  SparseTensor forward(SparseTensor x, c10::optional<SparseTensor> residual = c10::nullopt);
};

class BottleneckImpl : public torch::nn::Cloneable<BottleneckImpl> {
  // 1x1 -> 3x3 -> 1x1 conv + BN + residual
  static constexpr int expansion = 4;
  SparseTensor forward(SparseTensor x, c10::optional<SparseTensor> residual = c10::nullopt);
};
```

### 5.2 SE (Squeeze-and-Excitation) Blocks

```cpp
class SELayerImpl : public torch::nn::Cloneable<SELayerImpl> {
  // MinkowskiGlobalPooling -> Linear -> ReLU -> Linear -> Sigmoid -> Broadcast
};
```

---

## Phase 6 — Serialization & Model I/O

### 6.1 Model saving/loading
- Use LibTorch's built-in `torch::save()` / `torch::load()` for `nn::Module` state dicts
- TorchScript (`torch::jit::save/load`) for full model export (optional)

### 6.2 SparseTensor serialization
- Need a custom format to serialize SparseTensor (coordinates + features + metadata)
- Simple approach: save coordinates and features as separate tensors, reconstruct SparseTensor on load

---

## Phase 7 — Training Infrastructure

These are LibTorch equivalents of what PyTorch Python provides.

### 7.1 Optimizers
- LibTorch provides `torch::optim::Adam`, `torch::optim::SGD`, `torch::optim::AdamW`, etc.
- Use them directly with `model->parameters()`.

### 7.2 Loss Functions
- LibTorch provides `torch::nn::BCEWithLogitsLoss`, `torch::nn::CrossEntropyLoss`, etc.
- Apply to `output_sparse_tensor.F()` (extracted features) and labels.

### 7.3 Data Loading
- LibTorch has `torch::data::datasets::Dataset` and `torch::data::DataLoaderOptions`
- Port `Sparse4DClipDataset` to C++:
  - Read JSON files (use a C++ JSON library like nlohmann/json)
  - Build coordinate/feature/label arrays
  - Custom collate function using `sparse_collate()`

### 7.4 Training Loop
```cpp
model->train();
for (auto& batch : *data_loader) {
  auto [coords, feats, labels] = batch;
  auto stensor = SparseTensor(feats, coords, manager);
  auto logits = model->forward(stensor);
  auto loss = criterion(logits.F(), labels);
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();
}
```

### 7.5 Gradient Accumulation / Mixed Precision
- LibTorch supports `torch::autocast` for AMP
- Gradient scaling via `torch::cuda::amp::GradScaler` (if needed)

---

## Phase 8 — Testing & Validation

### 8.1 Unit Tests
- Test each autograd function: forward pass shape/value correctness, backward gradient check via `torch::autograd::gradcheck`
- Test each module: forward pass, parameter count, serialization round-trip
- Test SparseTensor construction, operations, conversions
- Test quantization and collation utilities

### 8.2 Integration Tests
- Port one of the example networks (UNet from `example_trainer.py`) to C++
- Verify identical output given identical weights and input
- Cross-validate with Python implementation if possible

### 8.3 Gradient Checking
LibTorch provides `torch::autograd::gradcheck()` — use this to verify all custom autograd functions.

---

## Implementation Order (Recommended)

The dependencies form a DAG. Build from the bottom up:

```
Phase 0: Build system + refactor mink/ to remove py::object deps
   ↓
Phase 1.1: KernelGenerator
Phase 1.2: CoordinateManager
   ↓
Phase 2.8: SPMMFunction (needed for SparseTensor quantization)
   ↓
Phase 1.3: SparseTensor
Phase 1.4: TensorField
   ↓
Phase 2.1: ConvolutionFunction
Phase 2.2: LocalPoolingFunction
Phase 2.3: GlobalPoolingFunction
Phase 2.4: DirectMaxPoolingFunction
Phase 2.5: BroadcastFunction
Phase 2.6: PruningFunction
Phase 2.7: InterpolationFunction
Phase 2.9: InstanceNormFunction
   ↓
Phase 3: All nn::Module layers (can be done in parallel)
   ↓
Phase 4: Utilities (collation, quantization, init)
   ↓
Phase 5: Predefined blocks (ResNet, SE)
   ↓
Phase 6: Serialization
   ↓
Phase 7: Training infrastructure
   ↓
Phase 8: Testing
```

---

## Key Challenges & Decisions

### C1: Type-erasing the CoordinateMapManager
The manager is templated on `<coordinate_type, allocator_type, CoordinateMapType>`. In Python, the right one is chosen at runtime. In C++, we need `std::variant` or a virtual base class to hold any of the three manager types (CPU, GPU_default, GPU_c10). `std::variant` + `std::visit` is the cleanest approach.

### C2: Removing py::object from the kernel layer
Several `CoordinateMapManager` methods return `py::object`. These need alternative pure-C++ signature versions. Guard existing pybind code behind `#ifdef MINK_WITH_PYBIND`.

### C3: Context saving in autograd
Python autograd stores arbitrary Python objects in `ctx`. C++ autograd uses `ctx->save_for_backward()` for tensors and `ctx->saved_data` (an `IValue` dict) for other state. Non-tensor state like `CoordinateMapKey`, `KernelGenerator` params, and `shared_ptr<CoordinateManager>` need to be packed into IValues or stored via a custom mechanism (e.g., `ctx->saved_data["manager_ptr"] = reinterpret_cast<int64_t>(manager.get())`).

**Recommended approach:** Store metadata in `ctx->saved_data` as int64 (pointers cast to int) and stride/size vectors as `IValue(std::vector<int64_t>)`. Use a shared reference-counted holder to prevent dangling.

### C4: SparseTensor is not an nn::Module
`SparseTensor` is a data container, not a module. It doesn't participate in `parameters()` — it just holds features (which are regular tensors that *do* participate in autograd) and coordinate metadata. This is straightforward in C++.

### C5: Thread safety
The `CoordinateManager` is often shared across layers. In Python this is fine due to the GIL. In C++ multi-threaded scenarios (e.g., data loading), ensure the manager's internal maps are thread-safe or document single-threaded usage requirements.

### C6: Device management
The Python code uses `get_postfix(tensor)` to decide CPU vs GPU dispatch at runtime. In C++ this becomes `tensor.is_cuda()` checks. Ensure all dispatch points handle both paths.

---

## Summary: File-by-File Conversion Map

| Python file | C++ target | Phase |
|---|---|---|
| `MinkowskiCommon.py` | `mink_cpp/types.hpp` (extend existing `mink/types.hpp`) | 0 |
| `MinkowskiKernelGenerator.py` | `mink_cpp/kernel_generator.hpp/.cpp` | 1.1 |
| `MinkowskiCoordinateManager.py` | `mink_cpp/coordinate_manager.hpp/.cpp` | 1.2 |
| `MinkowskiTensor.py` + `MinkowskiSparseTensor.py` | `mink_cpp/sparse_tensor.hpp/.cpp` | 1.3 |
| `MinkowskiTensorField.py` | `mink_cpp/tensor_field.hpp/.cpp` | 1.4 |
| `MinkowskiConvolution.py` (Functions) | `mink_cpp/autograd/convolution_fn.hpp/.cpp` | 2.1 |
| `MinkowskiConvolution.py` (Modules) | `mink_cpp/modules/convolution.hpp/.cpp` | 3.1 |
| `MinkowskiChannelwiseConvolution.py` | `mink_cpp/modules/convolution.hpp/.cpp` | 3.2 |
| `MinkowskiPooling.py` (Functions) | `mink_cpp/autograd/pooling_fn.hpp/.cpp` | 2.2–2.4 |
| `MinkowskiPooling.py` (Modules) | `mink_cpp/modules/pooling.hpp/.cpp` | 3.3 |
| `MinkowskiNormalization.py` | `mink_cpp/autograd/instance_norm_fn.hpp/.cpp` + `mink_cpp/modules/normalization.hpp/.cpp` | 2.9, 3.4 |
| `MinkowskiNonlinearity.py` | `mink_cpp/modules/nonlinearity.hpp/.cpp` | 3.5 |
| `MinkowskiOps.py` | `mink_cpp/modules/ops.hpp/.cpp` | 3.6, 4.4 |
| `MinkowskiBroadcast.py` (Functions) | `mink_cpp/autograd/broadcast_fn.hpp/.cpp` | 2.5 |
| `MinkowskiBroadcast.py` (Modules) | `mink_cpp/modules/broadcast.hpp/.cpp` | 3.9 |
| `MinkowskiPruning.py` | `mink_cpp/autograd/pruning_fn.hpp/.cpp` + `mink_cpp/modules/pruning.hpp/.cpp` | 2.6, 3.7 |
| `MinkowskiUnion.py` | `mink_cpp/modules/union_op.hpp/.cpp` | 3.8 |
| `MinkowskiInterpolation.py` | `mink_cpp/autograd/interpolation_fn.hpp/.cpp` + `mink_cpp/modules/interpolation.hpp/.cpp` | 2.7, 3.10 |
| `MinkowskiNetwork.py` | Not needed (just use `torch::nn::Module` directly) | — |
| `MinkowskiFunctional.py` | `mink_cpp/functional.hpp/.cpp` (optional) | 4.5 |
| `sparse_matrix_functions.py` | `mink_cpp/autograd/spmm_fn.hpp/.cpp` | 2.8 |
| `utils/collation.py` | `mink_cpp/utils/collation.hpp/.cpp` | 4.1 |
| `utils/quantization.py` | `mink_cpp/utils/quantization.hpp/.cpp` | 4.2 |
| `utils/init.py` | `mink_cpp/utils/init.hpp/.cpp` | 4.3 |
| `modules/resnet_block.py` | `mink_cpp/modules/resnet_block.hpp/.cpp` | 5.1 |
| `modules/senet_block.py` | `mink_cpp/modules/senet_block.hpp/.cpp` | 5.2 |
