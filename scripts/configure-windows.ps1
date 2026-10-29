
# MinkowskiEngine C++ - Windows Configuration
# CUDA 13.4 / cuDNN 9.27 / Visual Studio 2026
# Run from Visual Studio Developer PowerShell

param(
    [string]$LibTorchPath = "D:\libtorch_cu131",
    [string]$BuildDir = "build",
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$BuildType = "Release",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4",
    [string]$CudnnPath = "C:\Program Files\NVIDIA\CUDNN\v9.27",
    [string]$NinjaPath = "C:\Program Files\Ninja\ninja.exe",
    [string]$CudaArch = "86",
    [switch]$CpuOnly,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Fail {
    param([string]$Message)

    Write-Host ""
    Write-Host "[ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Resolve-BuildPath {
    param([string]$Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath(
        (Join-Path $ProjectRoot $Path)
    )
}

function Section {
    param([string]$Title)

    Write-Host ""
    Write-Host ("=" * 80) -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor Cyan
    Write-Host ("=" * 80) -ForegroundColor Cyan
}

Section "MinkowskiEngine C++ - Windows Configuration"

# ============================================================
# Visual Studio
# ============================================================

if (-not $env:VSINSTALLDIR) {
    Fail "Visual Studio Developer PowerShell is required."
}

Write-Host "[OK] Visual Studio: $env:VSINSTALLDIR" -ForegroundColor Green

$compiler = Get-Command cl.exe -ErrorAction SilentlyContinue

if (-not $compiler) {
    Fail "MSVC compiler cl.exe not found."
}

Write-Host "[OK] MSVC: $($compiler.Source)" -ForegroundColor Green

# ============================================================
# Build tools
# ============================================================

$cmake = Get-Command cmake.exe -ErrorAction SilentlyContinue

if (-not $cmake) {
    Fail "CMake not found in PATH."
}

if (-not (Test-Path -LiteralPath $NinjaPath -PathType Leaf)) {
    $ninja = Get-Command ninja.exe -ErrorAction SilentlyContinue

    if (-not $ninja) {
        Fail "Ninja not found."
    }

    $NinjaPath = $ninja.Source
}

Write-Host "[OK] CMake: $($cmake.Source)" -ForegroundColor Green
Write-Host "[OK] Ninja: $NinjaPath" -ForegroundColor Green

# ============================================================
# Paths
# ============================================================

$BuildDir = Resolve-BuildPath $BuildDir
$LibTorchPath = Resolve-BuildPath $LibTorchPath

$torchConfig = Join-Path $LibTorchPath `
    "share\cmake\Torch\TorchConfig.cmake"

if (-not (Test-Path -LiteralPath $torchConfig -PathType Leaf)) {
    Fail "LibTorch configuration not found: $torchConfig"
}

Write-Host "[OK] LibTorch: $LibTorchPath" -ForegroundColor Green

# ============================================================
# CUDA and cuDNN
# ============================================================

$cudaArgs = @()

if (-not $CpuOnly) {

    Section "CUDA 13.4"

    $CudaPath = (Resolve-BuildPath $CudaPath).Replace('\', '/')
    $nvcc = "$CudaPath/bin/nvcc.exe"

    if (-not (Test-Path -LiteralPath $nvcc -PathType Leaf)) {
        Fail "NVCC not found: $nvcc"
    }

    $versionOutput = (& $nvcc --version 2>&1) -join "`n"

    if ($LASTEXITCODE -ne 0) {
        Fail "NVCC failed to execute."
    }

    if ($versionOutput -notmatch 'release\s+(\d+\.\d+)') {
        Fail "Unable to determine CUDA version."
    }

    $cudaVersion = $Matches[1]

    Write-Host "[OK] CUDA: $cudaVersion" -ForegroundColor Green
    Write-Host "     $CudaPath"

    # Explicitly select one CUDA installation. Forward slashes avoid CMake
    # "Invalid character escape" errors in Torch's FindCUDA.
    $CudaPath = $CudaPath.Replace('\', '/')
    $nvcc = $nvcc.Replace('\', '/')
    $env:CUDA_PATH = $CudaPath
    $env:CUDA_HOME = $CudaPath
    $env:CUDAToolkit_ROOT = $CudaPath
    $env:CUDACXX = $nvcc
    $cudnnInclude = $cudnnInclude.Replace('\', '/')
    $cudnnLib = $cudnnLib.Replace('\', '/')
    $CudnnPath = $CudnnPath.Replace('\', '/')
    $LibTorchPath = $LibTorchPath.Replace('\', '/')
    $NinjaPath = $NinjaPath.Replace('\', '/')

    # Remove other CUDA toolkit bin directories from PATH.
    $cudaBinPattern = '[\\/]NVIDIA GPU Computing Toolkit[\\/]CUDA[\\/]v[^\\/]+[\\/]bin(?:[\\/]x64)?[\\/]?$'

    $filteredPath = @(
        $env:PATH -split ';' |
        Where-Object {
            $_ -and ($_ -notmatch $cudaBinPattern)
        }
    )

    $env:PATH = (@("$CudaPath\bin") + $filteredPath) -join ';'

    # --------------------------------------------------------
    # cuDNN
    # --------------------------------------------------------

    Section "cuDNN 9.27"

    $CudnnPath = Resolve-BuildPath $CudnnPath

    if (-not (Test-Path -LiteralPath $CudnnPath -PathType Container)) {
        Fail "cuDNN directory not found: $CudnnPath"
    }

    $cudnnLib = Join-Path $CudnnPath `
        "lib\$cudaVersion\x64\cudnn.lib"

    if (-not (Test-Path -LiteralPath $cudnnLib -PathType Leaf)) {
        Fail "cuDNN import library not found: $cudnnLib"
    }

    # Find the CUDA-version-specific include directory.
    $includeCandidates = @(
        (Join-Path $CudnnPath "include\$cudaVersion"),
        (Join-Path $CudnnPath "include")
    )

    $cudnnInclude = $null

    foreach ($candidate in $includeCandidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate "cudnn.h")) {
            $cudnnInclude = $candidate
            break
        }
    }

    if (-not $cudnnInclude) {
        Fail "cudnn.h not found in the expected cuDNN include directories."
    }

    $cudnnBin = Join-Path $CudnnPath "bin\$cudaVersion\x64"

    Write-Host "[OK] cuDNN library: $cudnnLib" -ForegroundColor Green
    Write-Host "[OK] cuDNN include: $cudnnInclude" -ForegroundColor Green

    if (Test-Path -LiteralPath $cudnnBin -PathType Container) {
        $env:PATH = "$cudnnBin;$env:PATH"
        Write-Host "[OK] cuDNN runtime directory: $cudnnBin" -ForegroundColor Green
    } else {
        Write-Host "[WARNING] cuDNN runtime directory not found: $cudnnBin" -ForegroundColor Yellow
    }

    $env:CUDNN_ROOT = $CudnnPath
    $env:CUDNN_INCLUDE_PATH = $cudnnInclude
    $env:CUDNN_LIBRARY_PATH = $cudnnLib
    $env:CUDNN_INCLUDE_DIR = $cudnnInclude
    $env:CUDNN_LIBRARY = $cudnnLib

    # LibTorch/Caffe2 also uses CUDA architecture discovery.
    $archList = ($CudaArch -split ';' | ForEach-Object {
        $arch = $_.Trim()
        if ($arch -match '^\d{2,3}$') {
            $arch.Insert($arch.Length - 1, '.')
        } else {
            $arch
        }
    }) -join ';'

    $env:TORCH_CUDA_ARCH_LIST = $archList

    $cudaArgs = @(
        "-DCPU_ONLY=OFF"
        "-DCUDAToolkit_ROOT=$CudaPath"
        "-DCUDA_TOOLKIT_ROOT_DIR=$CudaPath"
        "-DCMAKE_CUDA_COMPILER=$nvcc"
        "-DCUDA_NVCC_EXECUTABLE=$nvcc"
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
        "-DCUDNN_ROOT=$CudnnPath"
        "-DCUDNN_INCLUDE_PATH=$cudnnInclude"
        "-DCUDNN_LIBRARY_PATH=$cudnnLib"
        "-DCUDNN_INCLUDE_DIR=$cudnnInclude"
        "-DCUDNN_LIBRARY=$cudnnLib"
    )

} else {

    Write-Host "[INFO] CPU-only build selected." -ForegroundColor Yellow

    $cudaArgs = @(
        "-DCPU_ONLY=ON"
    )
}

# ============================================================
# Build directory and cache
# ============================================================

Section "Build Directory"

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
    Write-Host "Cleaning $BuildDir" -ForegroundColor Yellow
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}

if (-not (Test-Path -LiteralPath $BuildDir)) {
    New-Item -ItemType Directory `
        -Path $BuildDir -Force | Out-Null
}

$cache = Join-Path $BuildDir "CMakeCache.txt"

if (Test-Path -LiteralPath $cache) {
    Write-Host "[INFO] Existing CMake cache detected." -ForegroundColor Yellow
    Write-Host "       Use -Clean if changing CUDA or compiler versions."
}

# ============================================================
# Configure
# ============================================================

Section "CMake Configuration"

Write-Host "Project:      $ProjectRoot"
Write-Host "Build:        $BuildDir"
Write-Host "Build type:   $BuildType"
Write-Host "LibTorch:     $LibTorchPath"
Write-Host "CPU only:     $CpuOnly"
Write-Host "Architecture: $CudaArch"

$cmakeArgs = @(
    "-S", $ProjectRoot,
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_PREFIX_PATH=$LibTorchPath",
    "-DCMAKE_MAKE_PROGRAM=$NinjaPath",
    "-Wno-dev",
    "-Wno-unused-cli"
)

$cmakeArgs += $cudaArgs

Write-Host ""
Write-Host "Running CMake..." -ForegroundColor Cyan
Write-Host ""

& $cmake.Source @cmakeArgs

$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    Fail "CMake configuration failed with exit code $exitCode"
}

Section "Configuration Successful"

Write-Host "[OK] CMake configuration completed." -ForegroundColor Green
Write-Host ""
Write-Host "Build with:"
Write-Host "  cmake --build `"$BuildDir`" --parallel 12"

exit 0
