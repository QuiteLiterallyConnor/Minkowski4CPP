# MinkowskiEngine C++ — MSI Installer

This folder contains tooling to produce a Windows Installer (`.msi`) package for
MinkowskiEngine C++.

## Prerequisites

| Tool | Install |
|------|---------|
| **.NET SDK 6+** | <https://dotnet.microsoft.com/download> |
| **WiX Toolset v4** | `dotnet tool install --global wix` |
| **WiX UI extension** | `wix extension add WixToolset.UI.wixext` |
| **CMake 3.18+** | <https://cmake.org/download/> |

> You must have already **built** MinkowskiEngine with CMake before generating
> the MSI.  See [INSTALL.md](../INSTALL.md) for build instructions.

## Quick Start

```powershell
# 1. Build the project (if not already done)
cd <repo-root>
cmake -B build -DCMAKE_PREFIX_PATH="E:\Code\libtorch" -G Ninja
cmake --build build --config Release

# 2. Generate the MSI
cd msi
.\build-msi.ps1 -BuildDir ..\build -Version 1.0.0
```

The MSI will be created in this folder as
`MinkowskiEngine-<version>-<variant>-x64.msi`.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-Version` | `1.0.0` | Semantic version (Major.Minor.Patch) embedded in the MSI. |
| `-BuildDir` | `..\build` | Path to the CMake build directory. |
| `-BuildType` | `Release` | CMake configuration used during the build. |
| `-StagingDir` | `_staging` (inside `msi/`) | Temporary directory for `cmake --install`. |
| `-OutputPath` | auto | Output `.msi` path.  Defaults to `MinkowskiEngine-<ver>-<variant>-x64.msi`. |
| `-SkipStaging` | off | Reuse an existing staging directory instead of re-running `cmake --install`. |
| `-CpuOnly` | off | Tag the MSI as a CPU-only build (affects product name). |

## What the MSI Installs

The installer places files under `C:\Program Files\MinkowskiEngine\` with the
following layout:

```
MinkowskiEngine\
  lib\
    minkowski_cpp.lib          # static library
    cmake\
      MinkowskiEngine\
        MinkowskiEngineConfig.cmake
        MinkowskiEngineConfigVersion.cmake
        MinkowskiEngineTargets.cmake
        MinkowskiEngineTargets-release.cmake
  include\
    minkowski\
      *.hpp / *.h / *.cuh      # public headers (preserving subdirectory structure)
```

It also sets the **system** environment variable `MinkowskiEngine_DIR` pointing
to the CMake config directory so that downstream projects can use:

```cmake
find_package(MinkowskiEngine CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE MinkowskiEngine::MinkowskiEngine)
```

## Upgrade Behaviour

The MSI uses a fixed `UpgradeCode` GUID.  Installing a newer version will
automatically remove the previous one (major upgrade).  Downgrading is blocked.

## Troubleshooting

- **"wix not found"** — Run `dotnet tool install --global wix` and make sure
  `~/.dotnet/tools` is on your `PATH`.
- **"WixToolset.UI.wixext" errors** — Run `wix extension add WixToolset.UI.wixext`.
- **Empty staging directory** — Make sure the CMake build completed successfully
  and that `cmake --install` produces output to the staging path.
- **Iterate on WiX source without rebuilding** — Use `-SkipStaging` to skip the
  cmake install step and reuse the existing staging directory.
