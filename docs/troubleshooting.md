---
title: Troubleshooting
description: Dependency discovery, CUDA targets, runtime libraries, coordinate alignment, and known limits.
permalink: /troubleshooting/
---

## CMake cannot find Torch

Point `CMAKE_PREFIX_PATH` at the extracted **LibTorch root**, not its `lib`
directory. Multiple prefixes use a quoted semicolon-separated list.
Confirm you downloaded the right OS, ABI, and CUDA build.

## CMake cannot find MinkowskiEngine

Install the library first. Add its installation prefix to
`CMAKE_PREFIX_PATH`, or set `MinkowskiEngine_DIR` to the directory containing
`MinkowskiEngineConfig.cmake`, commonly `lib/cmake/MinkowskiEngine`.
Discover dependencies before loading the package as shown in the
[quick start](quickstart.md).

## Wrong CMake version or C++ standard

Current source requires CMake **3.24+** and C++20. Older standalone
installation notes mentioning CMake 3.18/C++17 do not describe the current
project. CUDA 13 builders use CMake 4.2.3; the legacy LibTorch 2.5.1 profile
keeps CMake 3.28.6 for its older CUDA target expectations.

## Unsupported GPU architecture

CUDA 13 cannot compile SM targets below 75. Use the legacy 12.4 profile for
older GPUs. In headless builds, LibTorch may choose an old target list even
when `CMAKE_CUDA_ARCHITECTURES` was supplied. Set the equivalent
`TORCH_CUDA_ARCH_LIST`; the builders do this automatically.

Example: SM 86 uses `CMAKE_CUDA_ARCHITECTURES=86` and
`TORCH_CUDA_ARCH_LIST=8.6`. Check the actual "Added CUDA NVCC flags" line
in configuration output, not just the initial CMake architecture message.

## CUDA/LibTorch mismatch or stale cache

Use matching distributions. Choose a new build directory when changing
toolkit, compiler, LibTorch, ABI, or CPU/CUDA mode. The Windows helper's
`-Clean` option cleans its selected tree; do not use it on unrelated data.
The native package builder rejects explicit toolkit-version mismatches.

## NVTX errors on CUDA 13

Modern NVTX's versioned C++ namespaces conflict with the bundled legacy
wrapper if both are included. The Linux compatibility header uses the
toolkit's `<nvtx3/nvtx3.hpp>` when available, retaining profiling without
mixing wrappers. The existing MSVC CUDA stub is preserved.

Use `CHECK_CUDA=1 bash tests/builders/test-builders.sh` to compile the
focused compatibility fixture.

## Windows executable exits without useful output

Missing DLLs can cause exit code `0xC0000135`. The repository normally copies
LibTorch DLLs beside build executables. Downstream applications must do the
same or add the matching LibTorch `lib` directory to `PATH`.
Also ensure CUDA/cuDNN runtime DLLs are discoverable.

## OpenMP runtime conflict on Windows

LibTorch's `libiomp5md` can conflict with a second OpenMP runtime. The
repository includes runtime-link handling for MSVC; do not add another
OpenMP runtime as a workaround. Preserve the existing adaptive host
threading approach instead of introducing conflicting OpenMP execution.

## Docker build or export failure

- Confirm Docker Desktop is running in Linux-container mode.
- Ensure BuildKit is enabled for `--output`.
- Use an amd64 build; current downloads are x86_64-only.
- Check available disk space and reduce `-Jobs` / `--jobs` for memory pressure.
- Inspect the first failed download, CMake, compiler, or packaging step.
- Verify the **exact expected profile-specific package** exists; another
  `.deb` in the output directory does not establish success.

Do not purge shared Docker images or caches indiscriminately.

## Sparse labels or residuals do not align

Coordinates are `[batch, spatial...]`. Quantization and striding can change
row count and order. Align labels through coordinate mappings.
Arithmetic requires the same manager and coordinate-map key; cross-key
arithmetic currently throws. A transpose-convolution output having the
same row count as an encoder output does not prove matching keys.

## Gaussian input or render failures

- The PLY loader supports binary little-endian layouts; inspect the header.
- Quaternion order is w/x/y/z.
- Use `load_gaussian_model_ply` when SH-rest coefficients must be preserved.
- Rendering requires float32 inputs on one device, unlike general
  `GaussianTensor` operations which also accept float64.
- Supply calibrated cameras and `[H,W,3]` float32 target images.
- Keep tensor/model owners alive for optimizer/trainer lifetimes.
- Cap Gaussian count for long densification runs.

## Known limitations

| Area | Current limitation |
|---|---|
| Uniform sparse conversion | `dense(...)` and `to(device)` throw as not fully implemented |
| Uniform arithmetic | Cross-key operators require an unimplemented union path |
| Adaptive execution | Hierarchy and kernel maps remain host-side |
| Adaptive kernels | No uniform fast path; positive output-level shift unsupported |
| Adaptive overlap | Cost grows with level gap |
| Uniform resampling | Potentially enormous output |
| Legacy/adaptive comparison | Per-row numerical equivalence was not established |
| Gaussian integration | No sparse-voxel bridge or adaptive index |
| Compressed Gaussian PLY | SH decode not verified against reference output |
| Gaussian CUDA rasterizer | Fixed 16x16 tiles; tile/chunk options ignored |
| Backend support | No ROCm backend; macOS is not a documented supported profile |

## Reporting an issue

Include OS, GPU/driver, CUDA and LibTorch versions, compiler/CMake versions,
build options, the first error, and a minimal reproducer. For performance
reports, include scene size, shapes/dtypes, device, Release/Debug mode,
warm-up, timing method, and whether fused CUDA kernels were enabled.
Avoid attaching credentials or private datasets.
