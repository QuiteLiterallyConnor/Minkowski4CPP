# PowerShell script to build and run in Docker

Write-Host "Building Docker image..." -ForegroundColor Cyan
docker build -t minkowski-build .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nRunning build in Docker container..." -ForegroundColor Cyan
docker run --rm -v "${PWD}:/workspace" minkowski-build bash /workspace/docker-build.sh

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nBuild successful! Executables are in ./build/" -ForegroundColor Green
    Write-Host "`nTo run the Phase 1 test:" -ForegroundColor Yellow
    Write-Host "  docker run --rm -v `"`${PWD}:/workspace`" minkowski-build /workspace/build/phase1_test" -ForegroundColor Yellow
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
}
