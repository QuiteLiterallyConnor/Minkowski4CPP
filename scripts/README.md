# Windows Build Scripts

These PowerShell scripts help build MinkowskiEngine C++ on Windows with MSVC and CUDA.

## Prerequisites

1. **Visual Studio 2022** with C++ development tools
2. **CUDA Toolkit 13.1+**
3. **CMake 3.18+**
4. **Ninja build system**
5. **LibTorch** (PyTorch C++ distribution)

## Quick Start

### Option 1: All-in-One Build

```powershell
# Run from Visual Studio Developer PowerShell (x64 Native Tools)
.\scripts\build-all-windows.ps1 -Clean
```

This will configure and build in one command.

### Option 2: Step-by-Step

```powershell
# 1. Configure
.\scripts\configure-windows.ps1 -Clean

# 2. Build
.\scripts\build-windows.ps1

# 3. Install (optional)
cmake --install build --config Release --prefix "C:\Program Files\MinkowskiEngine"
```

## Script Details

### `configure-windows.ps1`
Configures the CMake build system.

**Parameters:**
- `-LibTorchPath` - Path to LibTorch (default: `E:\Code\libtorch`)
- `-BuildDir` - Build directory (default: `build`)
- `-BuildType` - Build configuration (default: `Release`)
- `-CudaPath` - CUDA installation path
- `-NinjaPath` - Path to ninja.exe
- `-CpuOnly` - Build without CUDA support
- `-Clean` - Clean build directory before configuring

**Example:**
```powershell
.\scripts\configure-windows.ps1 `
  -LibTorchPath "D:\lib\libtorch" `
  -CudaPath "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0" `
  -Clean
```

### `build-windows.ps1`
Builds the project (run after configure).

**Parameters:**
- `-BuildDir` - Build directory (default: `build`)
- `-BuildType` - Build configuration (default: `Release`)
- `-Jobs` - Number of parallel jobs (default: auto)
- `-Verbose` - Verbose build output
- `-Target` - Specific target to build

**Example:**
```powershell
.\scripts\build-windows.ps1 -Jobs 8 -Verbose
```

### `build-all-windows.ps1`
Complete build pipeline: configure + build + install.

**Parameters:**
- All parameters from `configure-windows.ps1`
- All parameters from `build-windows.ps1`
- `-Install` - Install after successful build
- `-InstallPrefix` - Installation directory

**Example:**
```powershell
.\scripts\build-all-windows.ps1 `
  -Clean `
  -Install `
  -InstallPrefix "C:\MinkowskiEngine" `
  -Jobs 8
```

## Environment Setup

**IMPORTANT:** All scripts must be run from a Visual Studio Developer environment.

### Method 1: Use Developer PowerShell
Open "x64 Native Tools Command Prompt for VS 2022" from Start Menu

### Method 2: Initialize in Existing PowerShell
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```

### Method 3: Add to PowerShell Profile
Add this to your PowerShell profile to create an alias:
```powershell
function Enter-VSDevShell {
    & "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
}
Set-Alias vsdev Enter-VSDevShell
```

Then just run `vsdev` in any PowerShell session.

## Troubleshooting

### "Not in Visual Studio Developer environment"
Run from VS Developer PowerShell or initialize the environment first.

### "CMake not found"
Ensure CMake is installed and in PATH. Visual Studio 2022 includes CMake.

### "Ninja not found"
Install Ninja and specify path with `-NinjaPath` parameter, or add to PATH.

### "LibTorch not found"
Download LibTorch from pytorch.org and specify correct path with `-LibTorchPath`.

### "CUDA not found"
Either install CUDA Toolkit, use `-CpuOnly` flag, or specify correct path with `-CudaPath`.

## CPU-Only Build

To build without CUDA:

```powershell
.\scripts\build-all-windows.ps1 -CpuOnly -Clean
```

## See Also

- `build-notes.md` - Detailed technical notes about the build process
- `INSTALL.md` - General installation instructions
