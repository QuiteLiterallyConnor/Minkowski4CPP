$ErrorActionPreference = 'Stop'
$splitOut = 'C:\Users\User\Work\mink-lab\results\round10\adaptive'
$splitBuild = 'C:\Users\User\Work\mink-build\device-compose-cpu'
& pwsh -NoProfile -File C:\Users\User\Work\mink-lab\tools\mink-build.ps1 -Src C:\Users\User\Work\mink-wt\device-compose -Build $splitBuild -Cpu -Targets adaptive_split_tests,adaptive_split_bench,adaptive_tests,adaptive_kmap_bench,gaussian_tests,adaptive_vs_uniform_bench,test_framework_contract -Jobs 2 -Who adaptive-split-cpu *> "$splitOut\build-cpu-final.log"
if ($LASTEXITCODE -ne 0) { throw "CPU build exit $LASTEXITCODE" }
. C:\Users\User\Work\mink-lab\tools\mink-env.ps1 -Cpu -Quiet
& ctest --test-dir $splitBuild --verbose --output-on-failure --timeout 420 --output-junit "$splitOut\ctest-cpu.xml" *> "$splitOut\ctest-cpu.txt"
if ($LASTEXITCODE -ne 0) { throw "CPU CTest exit $LASTEXITCODE" }
'passed' | Set-Content "$splitOut\validate-cpu-status.txt"
