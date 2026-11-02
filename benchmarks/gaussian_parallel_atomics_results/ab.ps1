param([switch]$Resume)
$ErrorActionPreference = 'Stop'
$lab = 'C:\Users\User\Work\mink-lab'
$src = 'C:\Users\User\Work\mink-wt\gauss-cull'
$out = "$lab\results\round10\gaussian-atomics\ab"
New-Item -ItemType Directory -Force $out | Out-Null
. "$lab\tools\mink-env.ps1" -Quiet
Set-Location C:\Users\User\Work\mink-build\gauss-cull-cuda
$receipt = [ordered]@{started=(Get-Date -Format o);status='running';head=(& git -C $src rev-parse HEAD).Trim();rounds=10;runs=@();
  source_diff=(& git -C $src diff --stat);input_sha256=(Get-FileHash "$src\tests\data\test.ply").Hash;
  raster_sha256=(Get-FileHash .\gaussian_raster_backward_bench.exe).Hash;render_sha256=(Get-FileHash .\gaussian_render_bench.exe).Hash}
if ($Resume) {
  $previous = Get-Content "$out\receipt.json" -Raw | ConvertFrom-Json -AsHashtable
  foreach ($key in 'head','raster_sha256','render_sha256','input_sha256') {
    if ($previous[$key] -ne $receipt[$key]) {throw "Resume identity changed: $key"}
  }
  $previous.resume_started = $receipt.started
  $previous.interrupted_error = $previous.error
  [void]$previous.Remove('error')
  [void]$previous.Remove('completed')
  $previous.status = 'running'
  $receipt = $previous
} elseif (Test-Path "$out\receipt.json") {throw 'Receipt exists: use -Resume to preserve completed trials'}
function Save {
  $tmp = "$out\receipt-$([guid]::NewGuid().ToString('N')).tmp"
  try {
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tmp
    for ($attempt=0;$attempt -lt 20;$attempt++) {
      try {[System.IO.File]::Move($tmp, "$out\receipt.json", $true); return}
      catch [System.IO.IOException] {if ($attempt -eq 19) {throw}; Start-Sleep -Milliseconds 50}
    }
  } finally {if (Test-Path -LiteralPath $tmp) {Remove-Item -LiteralPath $tmp}}
}
Save
try {
  & nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv > "$out\gpu-before.txt"
  & pwsh -NoProfile -File "$lab\results\bridge-ml\r3\hostload.ps1" -Log "$out\hostload.log" -Tag before -Sec 3
  for ($round=1;$round -le 10;$round++) {
    $arms=if ($round % 2){@(0,1)}else{@(1,0)}
    foreach ($arm in $arms) {
      foreach ($pattern in 'row-groups','dense','sparse') {
        if (@($receipt.runs | Where-Object {$_.round -eq $round -and $_.arm -eq $arm -and $_.pattern -eq $pattern -and $_.exit -eq 0}).Count) {continue}
        $stem="${pattern}_a${arm}_r${round}"
        $args=@('--pattern',$pattern,'--tiles','32','--per-tile','256','--warmup','5','--iterations','30','--parallel-atomics',"$arm",'--output',"$out\$stem.json")
        & .\gaussian_raster_backward_bench.exe @args *> "$out\$stem.txt"
        $rc=$LASTEXITCODE
        $receipt.runs += [ordered]@{round=$round;arm=$arm;pattern=$pattern;arguments=$args;exit=$rc}
        Save
        if ($rc) {throw "$stem failed: $rc"}
      }
      if (@($receipt.runs | Where-Object {$_.round -eq $round -and $_.arm -eq $arm -and $_.pattern -eq 'render' -and $_.exit -eq 0}).Count) {continue}
      $stem="render_a${arm}_r${round}"
      $args=@('--input',"$src\tests\data\test.ply",'--counts','0','--sizes','512,1024','--devices','cuda','--warmup','3','--iterations','10','--train','--parallel-atomics',"$arm",'--output',"$out\$stem.json")
      & .\gaussian_render_bench.exe @args *> "$out\$stem.txt"
      $rc=$LASTEXITCODE
      $receipt.runs += [ordered]@{round=$round;arm=$arm;pattern='render';arguments=$args;exit=$rc}
      Save
      if ($rc) {throw "$stem failed: $rc"}
    }
    & pwsh -NoProfile -File "$lab\results\bridge-ml\r3\hostload.ps1" -Log "$out\hostload.log" -Tag "round-$round" -Sec 2
  }
  if ((Get-FileHash .\gaussian_raster_backward_bench.exe).Hash -ne $receipt.raster_sha256) {throw 'Raster binary changed'}
  if ((Get-FileHash .\gaussian_render_bench.exe).Hash -ne $receipt.render_sha256) {throw 'Renderer binary changed'}
  $receipt.status='passed'
} catch {$receipt.status='failed';$receipt.error=$_.Exception.Message}
$receipt.completed=(Get-Date -Format o)
Save
if ($receipt.status -ne 'passed') {throw $receipt.error}

