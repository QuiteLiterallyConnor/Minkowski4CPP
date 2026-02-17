# Phase 0 Completion Summary

## Overview
Phase 0 of the MinkowskiEngine C++ conversion has been completed successfully. This phase focused on establishing the build system, project structure, and removing pybind11 dependencies from the core kernel code.

## Completed Tasks

### 1. Directory Structure ✅
Created the complete directory layout for the C++ layer:
```
mink_cpp/
  ├── autograd/          # Autograd function implementations (Phase 2)
  ├── modules/           # nn::Module layer implementations (Phase 3)
  ├── utils/             # Utility functions (Phase 4)
  ├── types.hpp          # Type definitions and enums
  ├── kernel_generator.hpp    # KernelGenerator class
  ├── coordinate_manager.hpp  # CoordinateManager wrapper
  ├── sparse_tensor.hpp       # SparseTensor class
  ├── tensor_field.hpp        # TensorField class
  └── minkowski.hpp           # Convenience header

tests/                   # Test executables (future)
examples/                # Example programs (future)
```

### 2. CMake Build System ✅
Created comprehensive `CMakeLists.txt` with:
- LibTorch integration via `find_package(Torch REQUIRED)`
- CUDA support with conditional compilation
- CPU-only build mode via `-DCPU_ONLY` option
- Kernel library target (`minkowski_kernel`) from existing `mink/` sources
- New C++ layer library target (`minkowski_cpp`) 
- OpenMP support for parallel processing
- cuSPARSE linking for GPU builds
- Configurable CUDA architectures (61, 70, 75, 80, 86)
- Optional test and example targets
- Installation rules

### 3. Core Header Stubs ✅
Created initial header files with class declarations:

#### `types.hpp`
- Re-exports kernel layer types
- Defines enums: `SparseTensorQuantizationMode`, `RegionType`, `ConvolutionMode`, `PoolingMode`, `BroadcastMode`
- Backend selection: `CoordinateMapBackend`, `GPUMemoryAllocatorBackend`, `MinkowskiAlgorithm`

#### `kernel_generator.hpp`
- `KernelGenerator` class encapsulates kernel geometry configuration
- Methods: `get_kernel()`, `kernel_volume()`, accessors for kernel params
- Caching mechanism for computed kernel parameters

#### `coordinate_manager.hpp`
- `CoordinateManager` wrapper around templated C++ backend
- Type-erased manager storage using `std::variant`
- Methods for coordinate insertion, stride, kernel maps, union operations
- Both C++ and field operations support

#### `sparse_tensor.hpp`
- `SparseTensor` data container class
- Coordinate/feature storage with lazy coordinate fetching
- Quantization support during construction
- Operator overloads (+, -, *, /)
- Batch decomposition methods
- Dense/sparse conversion

#### `tensor_field.hpp`
- `TensorField` for continuous (float) coordinates
- Conversion to `SparseTensor` via quantization
- Splat operation for interpolation
- Inverse mapping tracking

### 4. Pybind11 Dependency Removal ✅
Refactored `mink/coordinate_map_manager.hpp` and `.cpp`:

#### Added conditional compilation guards:
```cpp
#ifndef MINK_NO_PYBIND
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif
```

#### Created C++ versions of pybind-dependent methods:
- `insert_field()` → `insert_field_cpp()` returns `CoordinateMapKey`
- `field_to_sparse_insert_and_map()` → `_cpp()` version returns `std::pair<CoordinateMapKey, ...>`
- `insert_and_map()` → `_cpp()` version returns `std::pair<CoordinateMapKey, ...>`
- `field_to_sparse_keys()` → `_cpp()` version returns `std::vector<CoordinateMapKey>`
- `get_coordinate_map_keys()` → `_cpp()` version returns `std::vector<CoordinateMapKey>`

#### Gated Python-specific methods:
- `py_stride()`, `py_origin()`, `py_origin_field()` wrapped in `#ifndef MINK_NO_PYBIND`
- Original `py::object`-returning methods now call `_cpp()` versions and wrap results

### 5. Build Documentation ✅
Created `BUILD.md` with:
- Prerequisites (LibTorch, CUDA, CMake)
- Build instructions for GPU and CPU-only modes
- Platform-specific guidance (Linux, Windows)
- Configuration options
- Usage example
- Current status and next steps

## Build Compilation Flag
The kernel library now compiles with `-DMINK_NO_PYBIND`, ensuring all Python dependencies are excluded when building the pure C++ library.

## Key Technical Decisions

### Type Erasure for CoordinateMapManager
Used `std::variant` to hold different manager types (CPU, GPU_default, GPU_c10) in a type-safe manner:
```cpp
std::variant<
  std::unique_ptr<cpu_manager_type<int32_t>>,
  std::unique_ptr<gpu_default_manager_type<int32_t>>,
  std::unique_ptr<gpu_c10_manager_type<int32_t>>
> manager_;
```

### Separation of Concerns
- Pybind bindings remain available when `MINK_NO_PYBIND` is not defined
- New C++ API is independent and can be used standalone
- Python bindings can be rebuilt on top of C++ layer if needed

## Files Modified
- `mink/coordinate_map_manager.hpp` - Added C++ method declarations, gated pybind code
- `mink/coordinate_map_manager.cpp` - Implemented C++ methods, gated pybind implementations

## Files Created
- `CMakeLists.txt` - Main build system
- `BUILD.md` - Build documentation
- `mink_cpp/types.hpp` - Type definitions
- `mink_cpp/kernel_generator.hpp` - KernelGenerator stub
- `mink_cpp/coordinate_manager.hpp` - CoordinateManager wrapper
- `mink_cpp/sparse_tensor.hpp` - SparseTensor class
- `mink_cpp/tensor_field.hpp` - TensorField class
- `mink_cpp/minkowski.hpp` - Main include header

## Next Steps: Phase 1
With Phase 0 complete, the foundation is in place to implement Phase 1:

1. **Implement KernelGenerator** (`mink_cpp/kernel_generator.cpp`)
   - Port region type conversion logic
   - Implement kernel volume calculation
   - Implement kernel parameter caching

2. **Implement CoordinateManager** (`mink_cpp/coordinate_manager.cpp`)
   - Constructor with backend selection
   - Coordinate insertion and mapping
   - Kernel map generation
   - Union operations

3. **Implement SparseTensor** (`mink_cpp/sparse_tensor.cpp`)
   - Construction with quantization
   - Operator overloads
   - Batch decomposition
   - Dense/sparse conversions

4. **Implement TensorField** (`mink_cpp/tensor_field.cpp`)
   - Field to sparse conversion
   - Splat operation

This will complete the core data structures needed before implementing autograd functions in Phase 2.

## Testing Phase 0
To verify Phase 0 completion:
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```

Expected: Successful compilation of both `minkowski_kernel` and `minkowski_cpp` libraries (though the latter will contain only stubs until Phase 1 is implemented).

## Summary
✅ Build system established
✅ Project structure created  
✅ Pybind11 dependencies isolated
✅ Core class interfaces defined
✅ Ready for Phase 1 implementation
