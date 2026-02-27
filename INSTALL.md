Installation
============

Prerequisites
- CMake >= 3.18
- A C++17 toolchain
- LibTorch (download and set `CMAKE_PREFIX_PATH` to LibTorch directory)
- Ninja
- CUDA toolkit (optional, required for GPU build)

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
The `docker/` folder contains Dockerfile(s). Use `docker/build-and-push.sh` to build and push images.

Windows
-------
Experimental Windows support has been added. Please ensure you have a matching LibTorch build for MSVC and that you run the build from a Visual Studio Developer PowerShell or with the MSVC environment set.

