# Docker builders

Build from the repository root using either host script:

```powershell
.\docker\docker-build.ps1
.\docker\docker-build.ps1 -CudaVersion 132
.\docker\docker-build.ps1 -CudaVersion 124 -Jobs 8
```

```bash
./docker/docker-build.sh
./docker/docker-build.sh --cuda-version 132
./docker/docker-build.sh --cuda-version 124 --jobs 8
```

| Profile | CUDA toolkit | Ubuntu | LibTorch | Default image tag |
|---|---|---|---|---|
| `134` (default) | 13.4.3 + cuDNN | 24.04 | 2.15.0 / cu134 | `cu134` |
| `132` | 13.2.1 + cuDNN | 24.04 | 2.15.0 / cu132 | `cu132` |
| `124` (legacy) | 12.4.0 | 22.04 | 2.5.1 / cu124 | `cu124` |

The pairings are defined in [cuda-profile.sh](cuda-profile.sh). All Linux
LibTorch downloads use the CXX11 ABI; the CUDA 13 archives no longer include
`cxx11-abi` in their filenames.

The scripts accept `-Push` / `--push`, `-Tag` / `--tag`, `-User` / `--user`,
and `-Repo` / `--repo`. Only CUDA 13.4 builds also publish the `latest` alias.
Explicit tags are honored; avoid reusing a tag across CUDA profiles.

GPU code defaults to Ampere SM 86. Use `-CudaArchitectures "86;89;90"` or
`--cuda-architectures '86;89;90'` to change it. CUDA 13 no longer supports
compiling pre-Turing architectures (below SM 75); use the legacy profile for
older GPUs. `-Jobs` / `--jobs` defaults to 4.
Use semicolon-separated SM numbers (optionally with an `a` suffix), not
CMake's `native`/`all` keywords. The builders also set `TORCH_CUDA_ARCH_LIST`
to the equivalent targets so LibTorch cannot override the requested GPU list
with unsupported headless auto-detection defaults.

Direct Docker builds use the same defaults:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile \
  --build-arg CUDA_VERSION=132 --build-arg CUDA_ARCHITECTURES=86 \
  --build-arg JOBS=4 -t minkowski4cpp:cu132 .
```

Images are **linux/amd64** only: both the pinned LibTorch archives and CMake
binary are x86_64. Building CUDA code does not require a GPU on the builder;
running it requires a compatible NVIDIA driver and GPU. CI builds all three
profiles and publishes CUDA-specific tags, with `latest` reserved for CUDA 13.4.
CMake is pinned to 4.2.3 for CUDA 13 and 3.28.6 for the legacy profile
(LibTorch 2.5.1 requires the old NvToolsExt CMake targets).

For Debian packages, see [apt/README.md](../apt/README.md).

Builder regression tests (Docker commands are mocked):

```powershell
.\tests\builders\test-builders.ps1
```

```bash
bash tests/builders/test-builders.sh
# Also check that the pinned upstream downloads exist:
CHECK_DOWNLOADS=1 bash tests/builders/test-builders.sh
# Also compile the NVTX compatibility fixture when nvcc is installed:
CHECK_CUDA=1 bash tests/builders/test-builders.sh
```

Linux uses the toolkit's NVTX C++ wrapper when available, preserving profiling
without mixing the bundled legacy wrapper with CUDA 13's versioned namespaces.
