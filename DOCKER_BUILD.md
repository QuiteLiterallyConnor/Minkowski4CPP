# Building with Docker

This guide explains how to build MinkowskiEngine C++ using Docker on Windows, which avoids all Windows-specific build issues.

## Prerequisites

1. **Docker Desktop for Windows** with WSL2 backend
   - Download: https://www.docker.com/products/docker-desktop
   - Enable WSL2 integration in Docker Desktop settings

2. **NVIDIA Container Toolkit** (optional, for GPU support at runtime)
   - Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Quick Start

### Option 1: Automated Build (PowerShell)

```powershell
# Build image and compile the project
.\docker-run-build.ps1

# Run the Phase 1 test
docker run --rm -v "${PWD}:/workspace" minkowski-build /workspace/build/phase1_test
```

### Option 2: Manual Steps

```powershell
# 1. Build the Docker image
docker build -t minkowski-build .

# 2. Run the build script inside the container
docker run --rm -v "${PWD}:/workspace" minkowski-build bash /workspace/docker-build.sh

# 3. Run tests
docker run --rm -v "${PWD}:/workspace" minkowski-build /workspace/build/phase1_test
docker run --rm -v "${PWD}:/workspace" minkowski-build /workspace/build/phase0_test
```

### Option 3: Interactive Development

```powershell
# Start an interactive shell in the container
docker run --rm -it -v "${PWD}:/workspace" minkowski-build bash

# Inside the container:
mkdir -p build && cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/workspace/libtorch ..
cmake --build . -j$(nproc)
./phase1_test
```

## With GPU Support

If you have NVIDIA GPU and Container Toolkit installed:

```powershell
# Run with GPU access
docker run --rm --gpus all -v "${PWD}:/workspace" minkowski-build /workspace/build/phase1_test
```

## Troubleshooting

### "docker: command not found"
- Install Docker Desktop and restart PowerShell

### "Cannot connect to Docker daemon"
- Start Docker Desktop application
- Ensure Docker Desktop is running in the system tray

### Build errors about LibTorch
- The Dockerfile automatically downloads LibTorch compatible with CUDA 11.8
- If you have a different version locally, it will be used instead

### Slow build on first run
- The first build downloads the CUDA image (~2-3 GB) and installs dependencies
- Subsequent builds are much faster due to Docker layer caching

## Clean Up

```powershell
# Remove the Docker image
docker rmi minkowski-build

# Clean build artifacts
Remove-Item -Recurse -Force build
```

## Notes

- The container uses CUDA 11.8 by default (compatible with most recent GPUs)
- Build artifacts (build/ directory) are shared with the host through volume mounting
- You can modify code on Windows and rebuild in the container
- For CUDA 12.x, modify the Dockerfile base image to `nvidia/cuda:12.1.0-devel-ubuntu22.04`
