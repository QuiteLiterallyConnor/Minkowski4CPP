$ErrorActionPreference = 'Stop'
$splitOut = 'C:\Users\User\Work\mink-lab\results\round10\adaptive'
$splitBuild = 'C:\Users\User\Work\mink-build\device-compose-cuda'
& pwsh -NoProfile -File C:\Users\User\Work\mink-lab\tools\mink-build.ps1 -Src C:\Users\User\Work\mink-wt\device-compose -Build $splitBuild -Targets adaptive_split_tests,adaptive_split_bench,adaptive_tests,adaptive_kmap_bench,gaussian_tests,adaptive_vs_uniform_bench,test_framework_contract -Jobs 2 -Who adaptive-split-cuda *> "$splitOut\build-cuda.log"
if ($LASTEXITCODE -ne 0) { throw "CUDA build exit $LASTEXITCODE" }
& pwsh -NoProfile -File C:\Users\User\Work\mink-lab\tools\with-lock.ps1 -Kind gpu -Who adaptive-split-cuda-tests -Command 'pwsh -NoProfile -File C:\Users\User\Work\mink-lab\results\round10\adaptive\test-cuda.ps1'
if ($LASTEXITCODE -ne 0) { throw "CUDA CTest lock exit $LASTEXITCODE" }
