param([ValidateSet('cpu','cuda')][string]$Kind)
$ErrorActionPreference='Stop'
$out='C:\Users\User\Work\mink-lab\results\round10\gaussian-atomics\validation'
if ($Kind -eq 'cpu') {. C:\Users\User\Work\mink-lab\tools\mink-env.ps1 -Cpu -Quiet}
else {. C:\Users\User\Work\mink-lab\tools\mink-env.ps1 -Quiet}
Set-Location "C:\Users\User\Work\mink-build\gauss-cull-$Kind"
& ctest --output-on-failure --timeout 420 --output-junit "$out\ctest-$Kind.xml" *> "$out\ctest-$Kind.txt"
exit $LASTEXITCODE
