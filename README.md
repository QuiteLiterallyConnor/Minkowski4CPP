[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-minkowski4cpp-0db7ed?logo=docker&logoColor=white&style=for-the-badge)](https://hub.docker.com/r/acaicia/minkowski4cpp)

# Minkowski4CPP

**A C++ port of NVIDIA's MinkowskiEngine with full CUDA support for Windows and Linux.**

Minkowski4CPP brings the power of sparse tensor operations from [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) into native C++ workflows, with first-class CUDA acceleration on both major platforms — something the original Python-based library does not support.

---

## Platform Support

| Platform | CUDA Version | Torch Version | Status |
|---|---|---|---|
| Ubuntu 22.04 | 12.4.0 | 2.5.1 | Verified |
| Windows | 13.1 | 2.10.0 | Verified |

---

## Prerequisites

### Windows

- [CUDA Toolkit 13.1](https://developer.nvidia.com/cuda-downloads)
- [LibTorch 2.10.0](https://pytorch.org/get-started/locally/) (pre-built, extract only)
- [Ninja](https://ninja-build.org/) (extract only)
- [CMake](https://cmake.org/download/) (install)

### Linux

- [CUDA Toolkit 12.4.0](https://developer.nvidia.com/cuda-downloads)
- [LibTorch 2.5.1](https://pytorch.org/get-started/locally/) (pre-built, extract only)
- `ninja-build` — install via your package manager
- `cmake` — install via your package manager

---

## Building

### Windows

Run the provided PowerShell build script, passing paths to your CUDA installation, LibTorch directory, and Ninja executable:

```powershell
scripts\build-all-windows.ps1 `
  -CudaPath    [PATH TO CUDA] `
  -LibTorchPath [PATH TO TORCH] `
  -NinjaPath   [PATH TO NINJA EXECUTABLE]
```

**Example:**

```powershell
scripts\build-all-windows.ps1 `
  -CudaPath     "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1" `
  -LibTorchPath "E:\Code\libtorch" `
  -NinjaPath    "C:\Program Files\Ninja\ninja.exe"
```

---

### Linux

Navigate to the project root, create a clean build directory, configure with CMake, and build with Ninja:

```bash
cd /path/to/minkowski4cpp/
mkdir build && cd build

cmake -G Ninja \
  -DCMAKE_PREFIX_PATH=[PATH TO LIBTORCH] \
  -DCMAKE_CUDA_COMPILER=[PATH TO NVCC] \
  ..

ninja -j4
```

**Example:**

```bash
cd /path/to/minkowski4cpp/
mkdir build && cd build

cmake -G Ninja \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  ..

ninja -j4
```

---

## About

MinkowskiEngine is a library for sparse tensor operations commonly used in 3D deep learning (e.g., point cloud processing). Minkowski4CPP ports this functionality to C++, enabling integration into native applications and pipelines that cannot depend on a Python runtime, while preserving GPU acceleration via CUDA.

---

Credit to:

    4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19, [pdf]

@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}


## License

This project is a derivative work of [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) by NVIDIA. Please refer to the original project's license for terms governing the underlying implementation.
