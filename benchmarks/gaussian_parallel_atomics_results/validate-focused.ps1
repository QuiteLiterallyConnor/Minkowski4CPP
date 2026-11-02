$ErrorActionPreference = 'Stop'
. C:\Users\User\Work\mink-lab\tools\mink-env.ps1 -Quiet
Set-Location C:\Users\User\Work\mink-build\gauss-cull-cuda
& .\gaussian_tests.exe gaussian_parallel_atomics *> C:\Users\User\Work\mink-lab\results\round10\gaussian-atomics\focused.txt
$rc=$LASTEXITCODE
@{exit=$rc;completed=(Get-Date -Format o);sha256=(Get-FileHash .\gaussian_tests.exe).Hash} | ConvertTo-Json | Set-Content C:\Users\User\Work\mink-lab\results\round10\gaussian-atomics\focused.json
exit $rc
