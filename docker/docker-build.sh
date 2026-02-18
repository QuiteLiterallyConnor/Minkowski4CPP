#!/bin/bash
# Build script for MinkowskiEngine C++ in Docker

set -e

echo "==================================="
echo "Building MinkowskiEngine C++ (CUDA)"
echo "==================================="

# Clean and create build directory
rm -rf build
mkdir -p build
cd build

# Show CUDA version
nvcc --version

# Configure with CMake with explicit CUDA path
echo "Configuring with CMake..."
cmake -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/libtorch \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86" \
    ..

# Build
echo "Building..."
cmake --build . --config Release -j$(nproc)

echo ""
echo "==================================="
echo "Build complete!"
echo "==================================="
echo ""
echo "Executables:"
ls -lh phase1_test phase0_test 2>/dev/null || echo "  (test executables not found)"
echo ""
