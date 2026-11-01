---
title: Installation
description: Configure a C++20 toolchain and install a LibTorch-backed CPU or CUDA build.
permalink: /installation/
---

## Prebuilt binaries

Download available binaries from
[GitHub Releases](https://github.com/QuiteLiterallyConnor/Minkowski4CPP/releases).
Match the asset to your OS and CUDA version, and follow the release-specific
instructions. For an unsupported asset combination, build from source below
or use the [container and package builders](builders.md).

## Requirements

The current CMake project requires **CMake 3.24+ and C++20**. CUDA translation
units also use C++20. LibTorch is required in both CPU and CUDA builds.

| Component | Windows | Linux |
|---|---|---|
| Compiler | Visual Studio 2022, C++ workload | C++20 compiler supported by your CUDA toolkit |
| Build system | Ninja; MSVC Developer PowerShell | Ninja and CMake |
| LibTorch | Matching MSVC/CUDA distribution | Matching Linux **CXX11 ABI** distribution |
| CUDA | 13.2 or 13.4 for documented Windows profiles | 12.4, 13.2, or 13.4 builder profiles |
| cuDNN | README lists 9.27.0; supply matching paths | CUDA 13 container profiles include cuDNN |

Do not mix a CUDA 12 LibTorch archive with a CUDA 13 toolkit. A GPU is not
required to compile a CUDA build; GPU execution needs a compatible device,
driver, and runtime libraries.

```bash
git clone https://github.com/QuiteLiterallyConnor/Minkowski4CPP.git
cd Minkowski4CPP
```

## Windows with the build helper

Run from Visual Studio Developer PowerShell. Provide explicit dependency paths
instead of relying on machine-specific defaults:

```powershell
.\scripts\build-all-windows.ps1 `
  -CudaVersion 134 `
  -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4" `
  -LibTorchPath "D:\libtorch_cu134" `
  -NinjaPath "C:\Program Files\Ninja\ninja.exe" `
  -BuildDir build-cu134 `
  -Jobs 4 `
  -Install `
  -InstallPrefix "D:\Minkowski4CPP\install"
```

Use `-CudaVersion 132` with matching paths for CUDA 13.2. The helper also
accepts a 13.1 selector, but it is not one of the README's documented verified
profiles. `-CudnnPath` overrides cuDNN discovery.

| Option | Purpose |
|---|---|
| `-BuildType Release` | Build configuration; Release is the default |
| `-BuildDir` | Separate build trees for different toolchains |
| `-CpuOnly` | Disable Minkowski CUDA kernels |
| `-Clean` | Clean the selected build directory before configuring |
| `-Install`, `-InstallPrefix` | Install library, headers, and CMake exports |
| `-Jobs` | Limit compilation parallelism to fit available memory |
| `-Verbose` | More detailed helper output |

When switching CUDA or LibTorch versions, choose a new build directory or use
`-Clean`. Do not point cleanup at a directory containing unrelated files.

## Linux from source

Install basic tools:

```bash
sudo apt update
sudo apt install build-essential ninja-build libomp-dev
```

Install CMake and extract a matching LibTorch distribution separately.
The container builders pin CMake 4.2.3 for CUDA 13 and 3.28.6 for the
LibTorch 2.5.1 legacy profile.

LibTorch can replace `CMAKE_CUDA_ARCHITECTURES` with its own nvcc flags. Set
**both** architecture variables; this example targets Ampere SM 86:

```bash
export TORCH_CUDA_ARCH_LIST='8.6'
cmake -S . -B build-cu134 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/libtorch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build build-cu134 --parallel 4
sudo cmake --install build-cu134
```

For multiple targets, use equivalent lists such as
`TORCH_CUDA_ARCH_LIST='8.6;8.9;9.0'` and
`-DCMAKE_CUDA_ARCHITECTURES='86;89;90'`. CUDA 13 cannot compile targets below
SM 75. Use CUDA 12.4 for those older GPU architectures.

## CPU-only build

Use a CPU LibTorch distribution if you want to avoid CUDA runtime dependencies:

```bash
cmake -S . -B build-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/libtorch-cpu \
  -DCPU_ONLY=ON
cmake --build build-cpu --parallel 4
cmake --install build-cpu --prefix "$HOME/.local"
```

`CPU_ONLY=ON` disables Minkowski kernels, **not** CUDA dependencies embedded in
a CUDA-enabled LibTorch distribution. Applications must use the CPU coordinate
backend when built without Minkowski CUDA support.

## Build options

| CMake option | Default | Effect |
|---|---|---|
| `CPU_ONLY` | `OFF` | Disable Minkowski CUDA kernels |
| `MINK_BUILD_TESTS` | `ON` | Build the `train` executable |
| `MINK_BUILD_ADAPTIVE_TESTS` | `ON` | Build adaptive tests, benchmark, and sphere exporter |
| `MINKOWSKI_ENABLE_GAUSSIAN` | `ON` | Include Gaussian sources and public API |
| `MINK_BUILD_GAUSSIAN_BENCHMARKS` | `ON` | Build Gaussian benchmark executables when Gaussian support is enabled |
| `MINK_ENABLE_OPENMP` | `ON` | Enable OpenMP where found |
| `MINK_USE_CCACHE` | `ON` | Use ccache/sccache when available |
| `MINK_COPY_TORCH_DLLS` | `ON` | Copy LibTorch DLLs beside Windows build executables |

Test switches are separate: disabling `MINK_BUILD_TESTS` alone does not
disable adaptive tests.

## Installed files and downstream use

With `CMAKE_INSTALL_LIBDIR=lib`, installation produces:

| Path under the installation prefix | Contents |
|---|---|
| `lib/libminkowski_cpp.a` or Windows archive | Static library |
| `include/minkowski/` | Public headers and required support headers |
| `lib/cmake/MinkowskiEngine/` | CMake config, version, and exported targets |

The installed target is **`MinkowskiEngine::minkowski_cpp`**.
Within this repository it is **`minkowski_cpp`**.

The package config includes the export but does not discover dependencies
automatically. Find LibTorch, OpenMP when required, and CUDA when required
before consuming it. See the complete [downstream example](quickstart.md).
