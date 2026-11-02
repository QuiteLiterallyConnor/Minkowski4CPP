$ErrorActionPreference = 'Stop'
$lab = 'C:\Users\User\Work\mink-lab'
$src = 'C:\Users\User\Work\mink-wt\gauss-cull'
$out = "$lab\results\round10\gaussian-atomics\validation"
New-Item -ItemType Directory -Force $out | Out-Null
$receipt=[ordered]@{started=(Get-Date -Format o);head=(& git -C $src rev-parse HEAD).Trim();status='running';configurations=@()}
function Save {$receipt | ConvertTo-Json -Depth 8 | Set-Content "$out\receipt.json"}
Save
try {
  foreach ($kind in 'cpu','cuda') {
    $build="C:\Users\User\Work\mink-build\gauss-cull-$kind"
    $args=@('-NoProfile','-File',"$lab\tools\mink-build.ps1",'-Src',$src,'-Build',$build,'-Who',"gaussian-atomics-$kind")
    if ($kind -eq 'cpu') {$args+='-Cpu'}
    & pwsh @args *> "$out\build-$kind.txt"
    if ($LASTEXITCODE) {throw "$kind build failed"}
    $cmd="pwsh -NoProfile -File $lab\results\round10\gaussian-atomics\run-ctest.ps1 -Kind $kind"
    # Tests are serialized with benchmark activity even for CPU configuration.
    & pwsh -NoProfile -File "$lab\tools\with-lock.ps1" -Kind gpu -Who "gaussian-atomics-$kind-tests" -Command $cmd
    if ($LASTEXITCODE) {throw "$kind CTest failed"}
    $hashes=@('gaussian_tests.exe','adaptive_tests.exe') | ForEach-Object {Get-FileHash -LiteralPath "$build\$_" -Algorithm SHA256 | Select-Object Path,Hash}
    $receipt.configurations+=@{kind=$kind;build_exit=0;ctest_exit=0;binaries=$hashes}
    Save
  }
  $receipt.status='passed'
} catch {$receipt.status='failed';$receipt.error=$_.Exception.Message}
$receipt.completed=(Get-Date -Format o);Save
if ($receipt.status -ne 'passed') {throw $receipt.error}
