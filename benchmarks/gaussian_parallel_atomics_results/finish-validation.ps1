$ErrorActionPreference='Stop'
$out='C:\Users\User\Work\mink-lab\results\round10\gaussian-atomics'
& pwsh -NoProfile -File "$out\full-validation.ps1"
if ($LASTEXITCODE) {throw 'Full validation failed'}
& pwsh -NoProfile -File C:\Users\User\Work\mink-lab\tools\with-lock.ps1 -Kind gpu -Who gaussian-atomics-sanitize -Command "pwsh -NoProfile -File $out\sanitize.ps1"
if ($LASTEXITCODE) {throw 'Sanitizer validation failed'}
