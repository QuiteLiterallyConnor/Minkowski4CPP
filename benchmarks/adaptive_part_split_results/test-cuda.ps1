$ErrorActionPreference = 'Stop'
$splitOut = 'C:\Users\User\Work\mink-lab\results\round10\adaptive'
$splitBuild = 'C:\Users\User\Work\mink-build\device-compose-cuda'
. C:\Users\User\Work\mink-lab\tools\mink-env.ps1 -Quiet
& ctest --test-dir $splitBuild --verbose --output-on-failure --timeout 420 --output-junit "$splitOut\ctest-cuda.xml" *> "$splitOut\ctest-cuda.txt"
if ($LASTEXITCODE -ne 0) { throw "CUDA CTest exit $LASTEXITCODE" }
'passed' | Set-Content "$splitOut\validate-cuda-status.txt"
