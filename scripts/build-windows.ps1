# MinkowskiEngine C++ - Windows Build Script
# Must be run from Visual Studio Developer PowerShell
# Run configure-windows.ps1 first!

param(
    [string]$BuildDir = "build",
    [string]$BuildType = "Release",
    [int]$Jobs = 2,
    [switch]$Verbose = $false,
    [string]$Target = ""
)

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "MinkowskiEngine C++ - Windows Build" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if build directory exists
if (-not (Test-Path $BuildDir)) {
    Write-Host "ERROR: Build directory not found: $BuildDir" -ForegroundColor Red
    Write-Host "Run configure-windows.ps1 first to generate build files" -ForegroundColor Yellow
    exit 1
}

# Check if we're in a VS Developer environment
if (-not $env:VSINSTALLDIR) {
    Write-Host "ERROR: Not in Visual Studio Developer environment!" -ForegroundColor Red
    Write-Host "Please run from 'x64 Native Tools Command Prompt for VS 2022' or run:" -ForegroundColor Yellow
    Write-Host '  & "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64' -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Visual Studio environment detected" -ForegroundColor Green

# Build CMake command
$cmakeArgs = @(
    "--build", $BuildDir,
    "--config", $BuildType,
    "--parallel", $Jobs
)

# Suppress warnings
$env:NINJA_STATUS = "[%f/%t] "

if ($Verbose) {
    $cmakeArgs += "--verbose"
}

if ($Target) {
    $cmakeArgs += "--target", $Target
}

Write-Host ""
Write-Host "Build Options:" -ForegroundColor Cyan
Write-Host "  Build Directory: $BuildDir" -ForegroundColor Gray
Write-Host "  Build Type: $BuildType" -ForegroundColor Gray
Write-Host "  Parallel Jobs: $Jobs" -ForegroundColor Gray
if ($Target) {
    Write-Host "  Target: $Target" -ForegroundColor Gray
}
Write-Host ""

Write-Host "Starting build..." -ForegroundColor Cyan
Write-Host "Command: cmake $($cmakeArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

& cmake @cmakeArgs

$stopwatch.Stop()

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "Time elapsed: $($stopwatch.Elapsed.ToString('mm\:ss'))" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To install, run:" -ForegroundColor Cyan
    Write-Host "  cmake --install $BuildDir --config $BuildType" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
    Write-Host "Time elapsed: $($stopwatch.Elapsed.ToString('mm\:ss'))" -ForegroundColor Gray
    exit $LASTEXITCODE
}
