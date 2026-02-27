# MinkowskiEngine C++ - Windows Build Configuration Script
# Must be run from Visual Studio Developer PowerShell

param(
    [string]$LibTorchPath = "E:\Code\libtorch",
    [string]$BuildDir = "build",
    [string]$BuildType = "Release",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [string]$NinjaPath = "C:\Program Files\Ninja\ninja.exe",
    [switch]$CpuOnly = $false,
    [switch]$Clean = $false
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "MinkowskiEngine C++ - Windows Configuration" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in a VS Developer environment
if (-not $env:VSINSTALLDIR) {
    Write-Host "ERROR: Not in Visual Studio Developer environment!" -ForegroundColor Red
    Write-Host "Please run from 'x64 Native Tools Command Prompt for VS 2022' or run:" -ForegroundColor Yellow
    Write-Host '  & "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64' -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Visual Studio environment detected" -ForegroundColor Green
Write-Host "    VSINSTALLDIR: $env:VSINSTALLDIR" -ForegroundColor Gray

# Check for required tools
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) {
    Write-Host "ERROR: CMake not found in PATH" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] CMake: $($cmake.Source)" -ForegroundColor Green

if (-not (Test-Path $NinjaPath)) {
    Write-Host "ERROR: Ninja not found at: $NinjaPath" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Ninja: $NinjaPath" -ForegroundColor Green

if (-not (Test-Path $LibTorchPath)) {
    Write-Host "ERROR: LibTorch not found at: $LibTorchPath" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] LibTorch: $LibTorchPath" -ForegroundColor Green

if (-not $CpuOnly) {
    $nvccPath = Join-Path $CudaPath "bin\nvcc.exe"
    if (-not (Test-Path $nvccPath)) {
        Write-Host "WARNING: CUDA not found at: $CudaPath" -ForegroundColor Yellow
        Write-Host "         Use -CpuOnly switch for CPU-only build" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[OK] CUDA: $CudaPath" -ForegroundColor Green
}

# Clean build directory if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-Host ""
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Path "$BuildDir\*" -Recurse -Force
    Write-Host "[OK] Build directory cleaned" -ForegroundColor Green
}

Write-Host ""
Write-Host "Configuration Options:" -ForegroundColor Cyan
Write-Host "  Build Type: $BuildType" -ForegroundColor Gray
Write-Host "  CPU Only: $CpuOnly" -ForegroundColor Gray
Write-Host "  Build Directory: $BuildDir" -ForegroundColor Gray
Write-Host ""

# Build CMake command
$cmakeArgs = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DCMAKE_PREFIX_PATH=`"$LibTorchPath`"",
    "-DCMAKE_MAKE_PROGRAM=`"$NinjaPath`"",
    "-Wno-dev",
    "--no-warn-unused-cli",
    "-B", $BuildDir
)

if ($CpuOnly) {
    $cmakeArgs += "-DCPU_ONLY=ON"
} else {
    $nvccPath = Join-Path $CudaPath "bin\nvcc.exe"
    $cmakeArgs += "-DCMAKE_CUDA_COMPILER=`"$nvccPath`""
}

Write-Host "Running CMake configuration..." -ForegroundColor Cyan
Write-Host "Command: cmake $($cmakeArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

& cmake @cmakeArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Configuration successful!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To build, run:" -ForegroundColor Cyan
    Write-Host "  cmake --build $BuildDir --config $BuildType" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or use the build script:" -ForegroundColor Cyan
    Write-Host "  .\scripts\build-windows.ps1" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "Configuration failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}
