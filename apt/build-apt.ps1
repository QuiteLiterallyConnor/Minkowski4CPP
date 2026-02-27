# =============================================================================
# MinkowskiEngine C++ — Debian Package Builder (Windows host)
# =============================================================================
#
# Builds a .deb package for Ubuntu 22.04 + CUDA 12.4 by spinning up a
# Docker build and exporting the finished package to the local machine.
#
# Prerequisites
# -------------
#   - Docker Desktop (Linux containers mode, BuildKit enabled)
#     BuildKit is on by default in Docker Desktop 23+.
#
# Usage
# -----
#   .\apt\build-apt.ps1
#   .\apt\build-apt.ps1 -Version 1.2.0 -Jobs 8 -OutputDir .\dist
#
# =============================================================================

[CmdletBinding()]
param(
    # Semver string embedded in the .deb filename and package metadata.
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version = "1.0.0",

    # Parallel compile jobs passed to ninja inside the container.
    [int]$Jobs = 4,

    # Where to write the finished .deb.  Defaults to the apt\ directory.
    [string]$OutputDir = "",

    # Docker image tag used for the build cache; does not appear in the output.
    [string]$ImageTag = "minkowski4cpp-deb-builder:latest",

    # Keep the intermediate Docker image after the build completes.
    # By default the image is removed to save disk space.
    [switch]$KeepImage
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$Dockerfile  = Join-Path $ScriptDir "Dockerfile"

if (-not $OutputDir) {
    $OutputDir = $ScriptDir
}
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path (Get-Location) $OutputDir
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$ExpectedDeb = Join-Path $OutputDir "libminkowski-cpp-dev_${Version}_amd64.deb"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " MinkowskiEngine C++ - Debian Package Builder" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Version      : $Version"
Write-Host "  Jobs         : $Jobs"
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Dockerfile   : $Dockerfile"
Write-Host "  Output dir   : $OutputDir"
Write-Host "  Output file  : $(Split-Path -Leaf $ExpectedDeb)"
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------

Write-Host "[1/3] Checking prerequisites..." -ForegroundColor Cyan

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 'docker' not found on PATH." -ForegroundColor Red
    Write-Host "Install Docker Desktop (Linux containers mode) and try again." -ForegroundColor Yellow
    exit 1
}

# Verify Docker daemon is reachable
try {
    docker info --format "{{.ServerVersion}}" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "ERROR: Docker daemon is not running or not accessible." -ForegroundColor Red
    Write-Host "Start Docker Desktop and ensure Linux container mode is active." -ForegroundColor Yellow
    exit 1
}

Write-Host "  Docker OK" -ForegroundColor Green

if (-not (Test-Path $Dockerfile)) {
    Write-Host "ERROR: Dockerfile not found at: $Dockerfile" -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Build the Docker image and export the .deb via BuildKit --output
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "[2/3] Building Docker image and packaging .deb..." -ForegroundColor Cyan
Write-Host "  This will download LibTorch (~2 GB) on the first run." -ForegroundColor DarkGray
Write-Host "  Subsequent runs reuse the Docker layer cache." -ForegroundColor DarkGray
Write-Host ""

# Force BuildKit (required for --output)
$env:DOCKER_BUILDKIT = "1"

# Build the 'artifact' target stage and export its filesystem to $OutputDir.
# The artifact stage is FROM scratch and contains only /libminkowski-cpp-dev_*.deb.
$buildArgs = @(
    "build",
    "--file", $Dockerfile,
    "--build-arg", "VERSION=$Version",
    "--build-arg", "JOBS=$Jobs",
    "--target", "artifact",
    "--output", "type=local,dest=$OutputDir",
    "--tag", $ImageTag,
    $ProjectRoot
)

Write-Host "  Running: docker $($buildArgs -join ' ')`n" -ForegroundColor DarkGray
& docker @buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Docker build failed (exit code $LASTEXITCODE)." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Verify output and report
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "[3/3] Verifying output..." -ForegroundColor Cyan

if (-not (Test-Path $ExpectedDeb)) {
    # Docker may have written the file with a slightly different path; search
    $found = Get-ChildItem -Path $OutputDir -Filter "libminkowski-cpp-dev_*.deb" -File |
             Select-Object -First 1
    if ($found) {
        $ExpectedDeb = $found.FullName
    } else {
        Write-Host "ERROR: .deb not found in output directory: $OutputDir" -ForegroundColor Red
        Write-Host "Check the Docker build log above for errors." -ForegroundColor Yellow
        exit 1
    }
}

$sizeMb = [math]::Round((Get-Item $ExpectedDeb).Length / 1MB, 2)
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host " Build complete!" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  Package : $(Split-Path -Leaf $ExpectedDeb)"
Write-Host "  Size    : $sizeMb MB"
Write-Host "  Path    : $ExpectedDeb"
Write-Host ""
Write-Host "  To install on a target Ubuntu 22.04 machine:" -ForegroundColor DarkGray
Write-Host "    sudo dpkg -i $(Split-Path -Leaf $ExpectedDeb)" -ForegroundColor DarkGray
Write-Host "    sudo ldconfig" -ForegroundColor DarkGray
Write-Host "================================================================" -ForegroundColor Green

# ---------------------------------------------------------------------------
# Clean up intermediate image (opt-out with -KeepImage)
# ---------------------------------------------------------------------------

if (-not $KeepImage) {
    Write-Host ""
    Write-Host "  Removing intermediate build image '$ImageTag'..." -ForegroundColor DarkGray
    docker rmi $ImageTag 2>&1 | Out-Null
}
