---
title: Docker & Debian packages
description: Reproducible amd64 builders for CUDA 13.4, CUDA 13.2, and legacy CUDA 12.4.
permalink: /builders/
---

## Dependency profiles

| Selector | CUDA base | Ubuntu | LibTorch | CMake |
|---|---|---|---|---|
| `134` (default) | 13.4.3, cuDNN development | 24.04 | 2.15.0 / cu134 | 4.2.3 |
| `132` | 13.2.1, cuDNN development | 24.04 | 2.15.0 / cu132 | 4.2.3 |
| `124` (legacy) | 12.4.0 development | 22.04 | 2.5.1 / cu124 | 3.28.6 |

All profiles use the Linux CXX11 ABI. The CUDA 13 LibTorch filenames omit the
older `cxx11-abi` prefix. The shared builder helper chooses the correct URL.

Container builds target **linux/amd64**; the pinned LibTorch archives and
CMake binaries are x86_64. The modern NVIDIA base images having arm64
manifests does not make these builders arm64-capable.

## Build a development image

From the repository root:

```powershell
.\docker\docker-build.ps1
.\docker\docker-build.ps1 -CudaVersion 132 -Jobs 4
.\docker\docker-build.ps1 -CudaVersion 124 -Tag legacy
```

```bash
./docker/docker-build.sh
./docker/docker-build.sh --cuda-version 132 --jobs 4
./docker/docker-build.sh --cuda-version 124 --tag legacy
```

The image includes the CUDA development toolchain, LibTorch, and the installed
Minkowski library. Its default working directory is `/workspace`; its default
command is Bash.

| PowerShell | Bash | Default / meaning |
|---|---|---|
| `-CudaVersion` | `--cuda-version` | `134`; accepts `124`, `132`, `134` |
| `-CudaArchitectures` | `--cuda-architectures` | `86`; semicolon-separated SM numbers |
| `-Jobs` | `--jobs` | `4` parallel jobs |
| `-Tag` | `--tag` | `cu134`, `cu132`, or `cu124` |
| `-User` | `--user` | Registry username prefix; also reads `DOCKER_USER` |
| `-Repo` | `--repo` | `minkowski4cpp`; also reads `DOCKER_REPO` |
| `-Push` | `--push` | Push after a successful build |

Only CUDA 13.4 builds also push `latest`. Explicit tags are honored; do not
reuse a custom tag across incompatible CUDA profiles.

```bash
docker build --platform linux/amd64 -f docker/Dockerfile \
  --build-arg CUDA_VERSION=132 \
  --build-arg CUDA_ARCHITECTURES='86;89;90' \
  --build-arg JOBS=4 -t minkowski4cpp:cu132 .
docker run --rm -it minkowski4cpp:cu132
```

To execute GPU code, install the NVIDIA Container Toolkit on the host and
start the image with `--gpus all`. Building an image itself does not need a GPU.

## Build a Debian package with Docker

On Windows, Docker Desktop must be running in Linux-container mode with
BuildKit enabled. Allow at least 30 GB of free space per profile for images,
LibTorch, compilation, and cached layers.

```powershell
.\apt\build-apt.ps1 -CudaVersion 134 -Jobs 4 -OutputDir .\dist
.\apt\build-apt.ps1 -CudaVersion 132 -Jobs 4 -OutputDir .\dist
.\apt\build-apt.ps1 -CudaVersion 124 -Jobs 4 -OutputDir .\dist
```

| Option | Default | Purpose |
|---|---|---|
| `-Version` | `1.3.0` | Debian metadata version |
| `-CudaVersion` | `134` | Matched profile |
| `-CudaArchitectures` | `86` | GPU targets |
| `-Jobs` | `4` | Parallel build jobs |
| `-OutputDir` | `apt/` | Artifact directory |
| `-ImageTag` | `minkowski4cpp-deb-builder:cu<profile>` | Build tag |
| `-KeepImage` | Not set | Skip intermediate-image cleanup |

The artifact stage is `FROM scratch` and exports only the package. You can
also invoke it directly on Linux:

```bash
docker build --platform linux/amd64 -f apt/Dockerfile \
  --build-arg CUDA_VERSION=134 --build-arg VERSION=1.3.0 \
  --target artifact --output type=local,dest=./dist .
```

Output names include the profile:
`libminkowski-cpp-dev_1.3.0_cu134_amd64.deb`.

## Native Linux packaging

The native script uses an existing toolkit and LibTorch installation:

```bash
./apt/build-apt.sh --libtorch /opt/libtorch \
  --cuda-version 134 --jobs 4 --output ./dist
```

| Flag | Default | Purpose |
|---|---|---|
| `--libtorch PATH` | Required | LibTorch root |
| `--version` | `1.0.0` | Package version |
| `--cuda-version` | Detect from nvcc | Require a matching `124`, `132`, or `134` toolkit |
| `--cuda-architectures` | `86` | GPU targets |
| `--jobs` | `nproc` | Compilation parallelism |
| `--build-dir` | `build-deb-cu<profile>` | Separate CMake tree |
| `--output` | `apt/` | Artifact directory |
| `--keep-build` | Not set | Keep the build tree after packaging |

If nvcc is absent and no profile was requested, the script builds CPU-only
and uses `_cpu_amd64.deb`. An explicit profile without nvcc, or with a
mismatched toolkit, is an error. CPU-only builds use `build-deb-cpu`.

## Install and compatibility

```bash
sudo dpkg -i dist/libminkowski-cpp-dev_1.3.0_cu134_amd64.deb
sudo ldconfig
```

Install on the matching Ubuntu release with matching CUDA and LibTorch.
The package declares `libgomp1` and `libstdc++6`; **CUDA and LibTorch are not
bundled**. The package installs a static archive, headers, and CMake exports.
Its post-install script refreshes the linker cache.

CUDA-specific filenames prevent artifact overwrites. All variants retain
the same Debian package name and are not co-installable.

## GPU architecture selection

Use SM-number lists, for example `86;89;90`, optionally with an `a` suffix
when supported by the toolkit and LibTorch. These builders do not accept
CMake's `native` or `all` architecture keywords. They convert the list into
`TORCH_CUDA_ARCH_LIST`, preventing headless LibTorch discovery from selecting
unsupported old targets.

CUDA 13 requires SM 75 or newer. Legacy hardware needs the CUDA 12.4 profile
and an explicit compatible architecture list.

See [troubleshooting](troubleshooting.md) for compiler, dependency, and runtime
errors, and [tests](testing.md#builder-regression-tests) for builder validation.
