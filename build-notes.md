# MinkowskiEngine C++ Build Notes

## Prerequisites

### Required Tools
- **CMake** 4.2+ (installed at: `C:\Program Files\CMake\bin`)
- **Ninja** build system (installed at: `C:\Program Files\Ninja\ninja.exe`)
- **Visual Studio 2022** with MSVC 19.44.35222.0
  - Path: `C:\Program Files\Microsoft Visual Studio\2022\Community`
  - MSVC Toolchain: `14.44.35207`
- **CUDA Toolkit 13.1.115**
  - Path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
  - NVCC: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe`
- **LibTorch** (PyTorch C++ distribution)
  - Path: `E:\Code\libtorch`

### Environment Requirements
**CRITICAL**: Must run from a **Visual Studio Developer PowerShell** to get proper MSVC environment (cl.exe, link.exe, rc.exe, mt.exe in PATH).

To set up environment:
```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64
```

## CMake Configuration (Successful)

### Working Configuration Command
```powershell
cmake -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_PREFIX_PATH="E:\Code\libtorch" `
  -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" `
  -DCMAKE_MAKE_PROGRAM="C:\Program Files\Ninja\ninja.exe" `
  -B build
```

### Configuration Results
- CXX Compiler: MSVC 19.44.35222.0
- CUDA Compiler: NVIDIA 13.1.115 with host compiler MSVC 19.44.35222.0
- CUDAToolkit: Found version 13.1.115
- Autodetected CUDA architecture: 8.6
- Target CUDA architectures: 75, 80, 86
- OpenMP: Found version 2.0
- CPU_ONLY: OFF (CUDA enabled)

### Key CMake Fixes Applied
1. Fixed `INTERFACE_INCLUDE_DIRECTORIES` to use generator expressions:
   - `$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/...>`
   - `$<INSTALL_INTERFACE:include/minkowski/...>`
   - This prevents "prefixed in source directory" errors during generation

## Build Issues Encountered

### Issue 1: NVCC Fatal Error
**Error**: "A single input file is required for a non-link phase when an outputfile is specified"

**Context**: All CUDA files (.cu) fail with this error when using:
- `-rdc=true` (relocatable device code)
- Multiple architecture targets
- Complex `-Xcompiler` flags including PDB file specifications

**Potential causes**:
- Conflict between `-Xcompiler=-MD` appearing twice in command line
- Issue with `-Xcompiler=-FdCMakeFiles\...\minkowski_kernel.pdb,-FS` syntax
- NVCC argument parsing issue with comma-separated Xcompiler flags

### Issue 2: MSVC Cannot Find Standard Headers
**Error**: "Cannot open include file: 'algorithm'" (and other STL headers like 'cassert', 'math.h')

**Context**: All C++ files fail to find standard library headers despite being in VS Dev environment.

**Potential causes**:
- MSVC include paths not properly set
- Conflict with `/permissive-` flag + `/Zc:preprocessor`
- External include path issues (`-external:I` flags)

## Build Command Used
```powershell
cmake --build build --config Release
```

## Next Steps
1. Resolve NVCC command-line argument issues
2. Ensure MSVC standard library headers are accessible
3. May need to adjust CMake CUDA compilation flags
4. Consider testing CPU_ONLY build first to isolate CUDA vs MSVC issues

## Notes
- Initial build attempt from `/workspace/` (Docker) caused CMakeCache mismatch
- Required clearing build directory: `Remove-Item -Path "build\*" -Recurse -Force`
- Ninja wasn't initially in PATH, required explicit `-DCMAKE_MAKE_PROGRAM` specification
- GCC from msys64 was initially picked up, needed explicit MSVC compiler in environment
