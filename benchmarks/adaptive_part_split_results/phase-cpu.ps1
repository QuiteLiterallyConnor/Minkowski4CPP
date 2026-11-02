$ErrorActionPreference = 'Stop'
$splitLab = 'C:\Users\User\Work\mink-lab'
$splitOut = "$splitLab\results\round10\adaptive"
. "$splitLab\tools\mink-env.ps1" -Cpu -Quiet
& pwsh -NoProfile -File "$splitLab\results\bridge-ml\r3\hostload.ps1" -Log "$splitOut\phase-load.log" -Tag before -Sec 3
& python -I -X utf8 "$splitOut\compare.py" --kind phase --device cpu
if ($LASTEXITCODE -ne 0) { throw "Phase compare exit $LASTEXITCODE" }
& pwsh -NoProfile -File "$splitLab\results\bridge-ml\r3\hostload.ps1" -Log "$splitOut\phase-load.log" -Tag after -Sec 2
