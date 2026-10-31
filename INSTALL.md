Installation
============

Prerequisites
- CMake >= 3.24
- A C++20 toolchain
- LibTorch (download and set `CMAKE_PREFIX_PATH` to LibTorch directory)
- Ninja
- CUDA toolkit (optional, required for GPU build)

For current platform profiles, complete build commands, and downstream CMake
integration, see the [installation guide](docs/installation.md) and
[quick start](docs/quickstart.md) in the GitHub Pages documentation.

Build and install (Linux / macOS)

```bash
git clone <repo>
cd mink_rewrite_2
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..
cmake --build . -- -j
cmake --install .
```

Build and install (Windows - Visual Studio / MSVC)

```powershell
git clone <repo>
cd mink_rewrite_2
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="C:/path/to/libtorch" -B build
cmake --build build --config Release
cmake --install build --config Release
```

Or use the provided helper (from a Developer PowerShell / MSVC environment):

```powershell
.
\scripts\build-and-install.bat build "C:\Program Files\MinkowskiEngine" Release
```

This will install headers to `/usr/local/include/minkowski` and libraries to `/usr/local/lib`, and will drop CMake package files to `/usr/local/lib/cmake/MinkowskiEngine` so downstream projects can `find_package(MinkowskiEngine CONFIG)`.

Docker
------
Use `docker/docker-build.sh` or `docker\docker-build.ps1` to build images.
The default is CUDA 13.4; select `--cuda-version 132` / `-CudaVersion 132`
for CUDA 13.2, or `124` for legacy CUDA 12.4.
See [docker/README.md](docker/README.md) for the matched Ubuntu/LibTorch profiles.

Windows
-------
Experimental Windows support has been added. Please ensure you have a matching LibTorch build for MSVC and that you run the build from a Visual Studio Developer PowerShell or with the MSVC environment set.
