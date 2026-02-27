# apt — Debian Package Builder

Produces `libminkowski-cpp-dev_<VERSION>_amd64.deb` for **Ubuntu 22.04 + CUDA 12.4**.  Two build paths are provided:

| Script | Host | Requires |
|---|---|---|
| `build-apt.ps1` | Windows | Docker Desktop (Linux containers mode) |
| `build-apt.sh` | Linux | CMake, Ninja, LibTorch, CUDA toolkit |

---

## Prerequisites

### Windows (`build-apt.ps1`)

- Docker Desktop with Linux containers mode enabled (BuildKit on by default since v23)
- ~20 GB free disk space (CUDA base image ~6 GB, LibTorch download ~2 GB)

### Linux (`build-apt.sh`)

```bash
sudo apt install build-essential cmake ninja-build dpkg-dev libomp-dev
```

- CUDA Toolkit 12.4+ (`nvcc` on `PATH`) — optional; falls back to CPU-only if absent
- LibTorch 2.5.1 (CXX11 ABI, CUDA 12.4) extracted somewhere (e.g. `/opt/libtorch`)

---

## Quick Start

### Windows — via Docker

```powershell
.\apt\build-apt.ps1
```

### Linux — native (no Docker)

```bash
chmod +x apt/build-apt.sh
./apt/build-apt.sh --libtorch /opt/libtorch
```

The finished package is written to the `apt/` directory in both cases.

---

## Parameters

### `build-apt.ps1`

| Parameter | Default | Description |
|---|---|---|
| `-Version` | `1.0.0` | Semver string embedded in the package filename and metadata |
| `-Jobs` | `4` | Parallel ninja compile jobs inside the container |
| `-OutputDir` | `apt\` | Directory to write the .deb into |
| `-ImageTag` | `minkowski4cpp-deb-builder:latest` | Docker image tag (used for layer caching) |
| `-KeepImage` | _(switch)_ | Keep the intermediate Docker image after the build |

**Example:**

```powershell
.\apt\build-apt.ps1 -Version 1.2.0 -Jobs 8 -OutputDir .\dist
```

---

### `build-apt.sh`

| Flag | Default | Description |
|---|---|---|
| `--libtorch PATH` | _(required)_ | Path to LibTorch root directory |
| `--version STRING` | `1.0.0` | Semver string embedded in the package |
| `--jobs N` | `nproc` | Parallel ninja compile jobs |
| `--build-dir PATH` | `<project>/build-deb` | CMake build directory |
| `--output PATH` | `apt/` | Directory to write the .deb into |
| `--keep-build` | _(flag)_ | Keep the CMake build tree after packaging |

**Example:**

```bash
./apt/build-apt.sh --libtorch /opt/libtorch --version 1.2.0 --jobs 8 --output ./dist
```

---

## What the Build Does

### Windows (Docker path)

1. **Dockerfile stage `builder`** — starts from `nvidia/cuda:12.4.0-devel-ubuntu22.04`, installs build tools, downloads LibTorch 2.5.1 + CUDA 12.4, copies the project source, and runs `cmake --install` into a staging tree.
2. **Package assembly** — writes `DEBIAN/control` and `DEBIAN/postinst`, then calls `dpkg-deb --build`.
3. **Dockerfile stage `artifact`** — `FROM scratch` layer that holds only the .deb, enabling `--output type=local` to stream it directly to the host without running a container.

### Linux (native path)

1. **Configure & build** — runs `cmake` + `ninja` against the project source using the provided LibTorch path.
2. **Stage** — `cmake --install` into a temp directory, then flattens `libminkowski_cpp.a` with `ld -r`.
3. **Package** — writes `DEBIAN/control` and `DEBIAN/postinst`, then calls `dpkg-deb --build`.
4. **Cleanup** — removes the build tree (unless `--keep-build` is passed) and the temp staging directory.

---

## What the Package Installs

| Path | Contents |
|---|---|
| `/usr/local/lib/libminkowski_cpp.a` | Static library |
| `/usr/local/include/minkowski/` | Public C++ and CUDA headers |
| `/usr/local/lib/cmake/MinkowskiEngine/` | CMake package config (`find_package` support) |

After install, `ldconfig` is run automatically via `DEBIAN/postinst`.

---

## Installing on a Target Machine

```bash
sudo dpkg -i libminkowski-cpp-dev_1.0.0_amd64.deb
sudo ldconfig
```

**Dependencies** — the package declares `libgomp1` and `libstdc++6` as runtime dependencies (normally already present).  LibTorch is **not** bundled; install it separately and tell CMake where it is:

```cmake
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
```

---

## Files in This Directory

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage Docker build: compile → package → extract artifact |
| `build-apt.ps1` | Windows PowerShell script that drives the Docker build |
| `build-apt.sh` | Linux shell script for a native (no Docker) build |
| `README.md` | This file |
