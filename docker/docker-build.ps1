<#
.SYNOPSIS
    Build and push the Minkowski4CPP Docker image to Docker Hub.

.DESCRIPTION
    Builds the Docker image from the project root using docker/Dockerfile.
    Optionally pushes to Docker Hub.

.PARAMETER Push
    Push the image to Docker Hub after building.

.PARAMETER Tag
    Image tag (default: "latest").

.PARAMETER User
    Docker Hub username. If omitted, the image is tagged without a username prefix.

.PARAMETER Repo
    Docker Hub repository name (default: "minkowski4cpp").

.EXAMPLE
    .\docker-build.ps1
    .\docker-build.ps1 -Push
    .\docker-build.ps1 -Push -Tag "1.0.0" -User "myuser"
#>

param(
    [switch]$Push,
    [string]$Tag  = "latest",
    [string]$User = $env:DOCKER_USER,
    [string]$Repo = $(if ($env:DOCKER_REPO) { $env:DOCKER_REPO } else { "minkowski4cpp" })
)

$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path

# ── Resolve image name ───────────────────────────────────────────────────────
if ($User) {
    $ImageName = "${User}/${Repo}:${Tag}"
} else {
    $ImageName = "${Repo}:${Tag}"
}

Write-Host "============================================"
Write-Host " Minkowski4CPP Docker Build"
Write-Host "============================================"
Write-Host " Image:   $ImageName"
Write-Host " Context: $ProjectRoot"
Write-Host " Push:    $Push"
Write-Host "============================================"

# ── Build ─────────────────────────────────────────────────────────────────────
docker build `
    -t $ImageName `
    -f (Join-Path $ScriptDir "Dockerfile") `
    $ProjectRoot

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build failed."
    exit 1
}

Write-Host ""
Write-Host "Successfully built: $ImageName"

# ── Push (optional) ──────────────────────────────────────────────────────────
if ($Push) {
    Write-Host "Pushing $ImageName to Docker Hub..."
    docker push $ImageName

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker push failed."
        exit 1
    }

    # If a version tag was given, also tag and push as 'latest'
    if ($Tag -ne "latest") {
        $LatestName = ($ImageName -replace ":[^:]+$", ":latest")
        docker tag $ImageName $LatestName
        docker push $LatestName
        Write-Host "Also pushed: $LatestName"
    }

    Write-Host "Push complete."
}
