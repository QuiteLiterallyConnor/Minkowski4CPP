# Phase 1 Complete - Core Data Structures

## Summary

Phase 1 of the MinkowskiEngine → Pure C++ conversion has been successfully completed. All core data structures have been implemented in pure C++ using LibTorch.

## Completed Components

### 1.1 KernelGenerator ✓
**Files:** `mink_cpp/kernel_generator.hpp`, `mink_cpp/kernel_generator.cpp`

Implemented functionality:
- Kernel parameter storage (size, stride, dilation)
- Region type handling (HYPER_CUBE, HYPER_CROSS, CUSTOM)
- Kernel volume computation
- Kernel parameter caching based on tensor stride
- Support for transpose convolutions
- Region type conversion utilities

Key features:
- Efficient caching of kernel parameters per tensor stride
- Proper validation of kernel configurations
- Support for different region types

### 1.2 CoordinateManager ✓
**Files:** `mink_cpp/coordinate_manager.hpp`, `mink_cpp/coordinate_manager.cpp`

Implemented functionality:
- Wrapper around C++ backend coordinate managers
- `std::variant` based type-erasure for CPU/GPU managers
- Core operations:
  - `insert_and_map()` - Insert coordinates and get unique/inverse maps
  - `insert_field()` - Insert continuous (float) coordinates
  - `field_to_sparse_insert_and_map()` - Convert field to sparse
  - `stride()` - Generate strided coordinate maps
  - `origin()` - Get origin coordinate map
  - `get_coordinates()` - Fetch coordinates by key
  - `size()` - Get coordinate map size
  - `union_map()` - Union multiple coordinate maps
  - `interpolation_map_weight()` - Get interpolation weights

Backend support:
- CPU backend
- GPU backend with CUDA allocator
- GPU backend with PyTorch c10 allocator

### 1.3 SparseTensor ✓
**Files:** `mink_cpp/sparse_tensor.hpp`, `mink_cpp/sparse_tensor.cpp`

Implemented functionality:
- Construction from coordinates + features with quantization
- Construction from existing CoordinateMapKey
- Quantization modes:
  - `RANDOM_SUBSAMPLE` - Take first occurrence
  - `UNWEIGHTED_AVERAGE` - Average duplicate coordinates
  - `UNWEIGHTED_SUM` - Sum duplicate coordinates
  - `MAX_POOL` - Max pooling for duplicates
  - `NO_QUANTIZATION` - No duplicates expected

Core accessors:
- `F()` - Get features
- `C()` - Get coordinates (lazy-fetched)
- `coordinate_map_key()` - Get key
- `coordinate_manager()` - Get manager
- `tensor_stride()` - Get stride
- `dimension()` - Get spatial dimension

Operations:
- Arithmetic operators (`+`, `-`, `*`, `/`, `+=`)
- Scalar operations
- Batch decomposition (`decomposed_coordinates`, `decomposed_features`)
- Conversions (`sparse()` to PyTorch COO, `dense()` stub)
- Type conversions (`to()`, `float_()`, `double_()`, `detach()`)

### 1.4 TensorField ✓
**Files:** `mink_cpp/tensor_field.hpp`, `mink_cpp/tensor_field.cpp`

Implemented functionality:
- Construction from continuous (float) coordinates + features
- Conversion to SparseTensor with quantization
- Linear interpolation splatting to 2^D neighbors
- Inverse mapping tracking

Core operations:
- `sparse()` - Convert to SparseTensor at given stride
- `splat()` - Linear interpolation to all 2^D corners
- `inverse_mapping()` - Get field-to-sparse mapping
- Coordinate and feature accessors

## Architecture

```
┌─────────────── Phase 1: Core Data Structures ───────────────┐
│                                                              │
│  SparseTensor          TensorField                          │
│      │                     │                                │
│      ├─── CoordinateManager ────┤                           │
│      │           │                                          │
│      └─ KernelGenerator                                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────── Existing mink/ C++ Kernels ─────────────────┐
│  CoordinateMapManager, Convolution, Pooling, etc.           │
└──────────────────────────────────────────────────────────────┘
```

## Usage Example

```cpp
#include "mink_cpp/minkowski.hpp"

using namespace minkowski;

// Create coordinates and features
auto coords = torch::tensor({{0, 0, 0}, {0, 1, 0}, {0, 0, 1}}, torch::kInt32);
auto feats = torch::randn({3, 128});

// Create SparseTensor
auto sparse_tensor = SparseTensor(
    feats, coords, nullptr,
    SparseTensorQuantizationMode::RANDOM_SUBSAMPLE);

// Access properties
auto coordinates = sparse_tensor.C();
auto features = sparse_tensor.F();
int dimension = sparse_tensor.dimension();

// Arithmetic operations
auto doubled = sparse_tensor * 2.0;
auto sum = sparse_tensor + sparse_tensor;

// Create TensorField with continuous coordinates
auto field_coords = torch::tensor({{0, 0.5, 0.3}, {0, 1.2, 0.8}}, torch::kFloat);
auto field_feats = torch::randn({2, 64});
auto tensor_field = TensorField(field_feats, field_coords);

// Convert to sparse with quantization
auto sparse_from_field = tensor_field.sparse();

// Splat with linear interpolation
auto splatted = tensor_field.splat();
```

## Next Steps - Phase 2: Autograd Functions

The next phase involves implementing autograd functions for:
- Convolution (forward/backward)
- Pooling (local, global, transpose)
- Broadcast operations
- Pruning
- Interpolation
- Sparse matrix operations (SpMM)
- Instance normalization

Each will use `torch::autograd::Function` and dispatch to existing `mink/` C++ kernels.

## Testing

To test Phase 1 components:
1. Ensure LibTorch is properly linked
2. Compile with the CMake build system from Phase 0
3. Create test programs in `tests/` directory
4. Verify coordinate insertion, quantization, and basic operations

## Notes

- All core data structures are now available in pure C++
- No Python dependencies required
- Compatible with existing mink/ backend kernels
- Ready for Phase 2 autograd implementation
- Some advanced features (e.g., full `dense()` conversion) are marked as stubs for later completion
