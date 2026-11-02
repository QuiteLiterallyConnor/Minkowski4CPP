$ErrorActionPreference='Stop'
$out='C:\Users\User\Work\mink-lab\results\round10\gaussian-atomics\sanitize'
New-Item -ItemType Directory -Force $out | Out-Null
. C:\Users\User\Work\mink-lab\tools\mink-env.ps1 -Quiet
$env:PYTORCH_NO_CUDA_MEMORY_CACHING='1'
$san=Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe'
Set-Location C:\Users\User\Work\mink-build\gauss-cull-cuda
$receipt=[ordered]@{started=(Get-Date -Format o);status='running';sha256=(Get-FileHash .\gaussian_tests.exe).Hash;exclusions=@();tools=@()}
try {
  foreach ($tool in 'memcheck','racecheck','initcheck') {
    & $san --tool $tool --error-exitcode 99 --print-limit 20 --show-backtrace no .\gaussian_tests.exe gaussian_parallel_atomics *> "$out\$tool.txt"
    $rc=$LASTEXITCODE
    $receipt.tools+=@{tool=$tool;exit=$rc}
    $receipt | ConvertTo-Json -Depth 8 | Set-Content "$out\receipt.json"
    if ($rc) {throw "$tool failed: $rc"}
  }
  $receipt.status='passed'
} catch {$receipt.status='failed';$receipt.error=$_.Exception.Message}
$receipt.completed=(Get-Date -Format o)
$receipt | ConvertTo-Json -Depth 8 | Set-Content "$out\receipt.json"
if ($receipt.status -ne 'passed') {throw $receipt.error}
