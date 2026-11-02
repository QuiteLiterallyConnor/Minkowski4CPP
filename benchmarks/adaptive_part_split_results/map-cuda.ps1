$ErrorActionPreference = 'Stop'
$splitLab = 'C:\Users\User\Work\mink-lab'
$splitOut = "$splitLab\results\round10\adaptive"
. "$splitLab\tools\mink-env.ps1" -Quiet
$env:CUDA_MODULE_LOADING = 'EAGER'
& pwsh -NoProfile -File "$splitLab\results\bridge-ml\r3\hostload.ps1" -Log "$splitOut\map-cuda-load.log" -Tag before -Sec 3
& python -I -X utf8 "$splitOut\compare.py" --kind map --device cuda
if ($LASTEXITCODE -ne 0) { throw "Map compare exit $LASTEXITCODE" }
& pwsh -NoProfile -File "$splitLab\results\bridge-ml\r3\hostload.ps1" -Log "$splitOut\map-cuda-load.log" -Tag after -Sec 2
