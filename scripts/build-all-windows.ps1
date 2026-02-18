# MinkowskiEngine C++ - Complete Windows Build Script
# Configures, builds, and optionally installs in one command
# Must be run from Visual Studio Developer PowerShell

param(
    [string]$LibTorchPath = "E:\Code\libtorch",
    [string]$BuildDir = "build",
    [string]$BuildType = "Release",
    [string]$InstallPrefix = "",
    [string]$CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [string]$NinjaPath = "C:\Program Files\Ninja\ninja.exe",
    [switch]$CpuOnly = $false,
    [switch]$Clean = $false,
    [switch]$Install = $false,
    [switch]$Verbose = $false,
    [int]$Jobs = 12
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================================================================" -ForegroundColor Cyan
Write-Host "MinkowskiEngine C++ - Complete Windows Build" -ForegroundColor Cyan
Write-Host "========================================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in a VS Developer environment
if (-not $env:VSINSTALLDIR) {
    Write-Host "ERROR: Not in Visual Studio Developer environment!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run from 'x64 Native Tools Command Prompt for VS 2022' or run:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host '  & "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64' -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

#===================================================================================================
# STEP 1: CONFIGURE
#===================================================================================================
Write-Host "STEP 1: Configuration" -ForegroundColor Cyan
Write-Host "========================================================================================================" -ForegroundColor Cyan
Write-Host ""

& "$PSScriptRoot\configure-windows.ps1" `
    -LibTorchPath $LibTorchPath `
    -BuildDir $BuildDir `
    -BuildType $BuildType `
    -CudaPath $CudaPath `
    -NinjaPath $NinjaPath `
    -CpuOnly:$CpuOnly `
    -Clean:$Clean

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Configuration failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

#===================================================================================================
# STEP 2: BUILD
#===================================================================================================
Write-Host ""
Write-Host ""
Write-Host "STEP 2: Build" -ForegroundColor Cyan
Write-Host "========================================================================================================" -ForegroundColor Cyan
Write-Host ""

$buildArgs = @{
    BuildDir = $BuildDir
    BuildType = $BuildType
    Verbose = $Verbose
    Jobs = $Jobs
}

& "$PSScriptRoot\build-windows.ps1" @buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

#===================================================================================================
# STEP 3: INSTALL (Optional)
#===================================================================================================
if ($Install) {
    Write-Host ""
    Write-Host ""
    Write-Host "STEP 3: Install" -ForegroundColor Cyan
    Write-Host "========================================================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    $installArgs = @(
        "--install", $BuildDir,
        "--config", $BuildType
    )
    
    if ($InstallPrefix) {
        $installArgs += "--prefix", $InstallPrefix
    }
    
    Write-Host "Installing..." -ForegroundColor Cyan
    Write-Host "Command: cmake $($installArgs -join ' ')" -ForegroundColor Gray
    Write-Host ""
    
    & cmake @installArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[OK] Installation successful" -ForegroundColor Green
        if ($InstallPrefix) {
            Write-Host "     Installed to: $InstallPrefix" -ForegroundColor Gray
        }
    } else {
        Write-Host ""
        Write-Host "Installation failed!" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

#===================================================================================================
# COMPLETE
#===================================================================================================
Write-Host ""
Write-Host ""
Write-Host "========================================================================================================" -ForegroundColor Green
Write-Host "BUILD COMPLETE!" -ForegroundColor Green
Write-Host "========================================================================================================" -ForegroundColor Green
Write-Host ""

if (-not $Install) {
    Write-Host "To install, run:" -ForegroundColor Cyan
    Write-Host "  cmake --install $BuildDir --config $BuildType [--prefix <path>]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or use:" -ForegroundColor Cyan
    Write-Host "  .\scripts\build-all-windows.ps1 -Install [-InstallPrefix <path>]" -ForegroundColor Yellow
}

Write-Host ""
