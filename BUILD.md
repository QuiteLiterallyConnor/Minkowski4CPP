# MinkowskiEngine C++ Build Instructions

## Prerequisites

- CMake 3.18+
- C++17 compiler (GCC 7+, Clang 7+, MSVC 2019+)
- LibTorch (download from https://pytorch.org/)
- CUDA Toolkit 11.0+ (optional, for GPU support)
- OpenMP (usually included with compiler)

## LibTorch Setup

Download LibTorch for your platform:
```bash
# Example for Linux with CUDA 11.8
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
```

## Building

### GPU Build (default)

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release -j8
```

### CPU-Only Build

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCPU_ONLY=ON ..
cmake --build . --config Release -j8
```

### Windows

```powershell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=C:\path\to\libtorch -G "Visual Studio 16 2019" ..
cmake --build . --config Release
```

## Configuration Options

- `-DCPU_ONLY=ON` - Build without CUDA support
- `-DBUILD_TESTS=OFF` - Skip building tests
- `-DBUILD_EXAMPLES=OFF` - Skip building examples
- `-DCMAKE_CUDA_ARCHITECTURES="61;70;75;80;86"` - Specify CUDA compute capabilities

## Installation

```bash
cmake --install . --prefix /usr/local
```

## Usage Example

```cpp
#include <minkowski.hpp>

int main() {
  using namespace minkowski;
  
  // Create a coordinate manager
  auto manager = std::make_shared<CoordinateManager>(
    3,  // dimension
    CoordinateMapBackend::CUDA
  );
  
  // Create coordinates and features
  auto coords = torch::rand({1000, 4}).to(torch::kInt32);  // [N, D+1] with batch
  auto feats = torch::rand({1000, 128});
  
  // Create sparse tensor
  SparseTensor sparse(feats, coords, manager);
  
  // Use in network...
  
  return 0;
}
```

## Current Status (Phase 0 Complete)

- ✅ CMake build system
- ✅ Directory structure
- ✅ Core header stubs created
- ⏳ Pybind removal from kernel code (in progress)
- ⏳ Implementation of C++ layer classes (Phase 1+)

## Next Steps

See [instructions.md](instructions.md) for the full conversion roadmap.
