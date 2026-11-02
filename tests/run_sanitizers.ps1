<#
Run the native test suites under memory-safety tools and summarise the findings.

  pwsh -File tests/run_sanitizers.ps1 -BuildDir <build> -OutDir <dir> -Jobs memcheck:gaussian_tests:cuda,racecheck:adaptive_tests

Each job is "<tool>:<suite>[:<filter>]":
  tool   memcheck | racecheck | initcheck | synccheck  (NVIDIA compute-sanitizer, CUDA builds)
         plain                                         (no tool: CUDA_LAUNCH_BLOCKING=1 + uncached CUDA allocator, so a
                                                         device fault is reported at the faulting kernel; a weak
                                                         fallback where compute-sanitizer is unavailable)
         run                                           (run the suite directly, nothing changed; with
                                                         -KeepCachingAllocator this is a normal timed suite run)
         asan                                          (run the suite directly; the build must use /fsanitize=address,
                                                         see ASan notes below)
  suite  test executable in -BuildDir (adaptive_tests, gaussian_tests, core_tests, ...)
  filter optional substring filter passed to the suite (mink_test::run_all semantics)

compute-sanitizer notes
- PYTORCH_NO_CUDA_MEMORY_CACHING=1 is set (unless -KeepCachingAllocator) so every tensor is its own cudaMalloc:
  with the caching allocator an out-of-bounds access that lands inside a cached block is invisible to memcheck,
  and cached blocks are reported as leaks at exit.
- memcheck runs with --leak-check full. Errors are printed inline with the test output, so an error is attributed
  to the first "[ PASS ]/[ FAIL ]/[ SKIP ]" line that follows it.
- Each job is killed after -TimeoutMin minutes (marked TIMEOUT in the summary).
- Windows (WDDM): compute-sanitizer needs the GPU debugger interface, enabled once by an administrator with
  <CUDA>\compute-sanitizer\EnableDebuggerInterface.bat (HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger
  EnableInterface=1). Without it the suite runs unchecked; such jobs are marked SANITIZER-UNAVAILABLE.

ASan notes (MSVC): LibTorch is not instrumented, so build with /D_DISABLE_VECTOR_ANNOTATION
/D_DISABLE_STRING_ANNOTATION to avoid container-overflow false positives on std::vector/std::string objects that
cross the LibTorch boundary. The asan runtime DLL (clang_rt.asan_dynamic-x86_64.dll) must be on PATH (vcvars64).

Writes <OutDir>/<tool>-<suite>[-<filter>].txt per job and appends a row per job to <OutDir>/summary.csv:
tool,suite,filter,exit,seconds,tests_passed,tests_failed,tests_skipped,sanitizer_errors,leaked_bytes,status
Exit code: 0 when every job finished with zero sanitizer errors and zero failed tests, else 1.
#>
param(
    [Parameter(Mandatory)][string]$BuildDir,
    [Parameter(Mandatory)][string]$OutDir,
    [Parameter(Mandatory)][string[]]$Jobs,
    [string]$Sanitizer = $(if ($env:CUDA_PATH) { Join-Path $env:CUDA_PATH 'compute-sanitizer\compute-sanitizer.exe' } else { 'compute-sanitizer' }),
    [double]$TimeoutMin = 10,
    [switch]$KeepCachingAllocator,
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = 'Stop'
$Jobs = @($Jobs | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ })
New-Item -ItemType Directory -Force $OutDir | Out-Null
$summary = Join-Path $OutDir 'summary.csv'
if (-not (Test-Path $summary)) {
    'tool,suite,filter,exit,seconds,tests_passed,tests_failed,tests_skipped,sanitizer_errors,leaked_bytes,status' |
        Set-Content $summary
}
if (-not $KeepCachingAllocator) { $env:PYTORCH_NO_CUDA_MEMORY_CACHING = '1' }
if (-not $env:ASAN_OPTIONS) { $env:ASAN_OPTIONS = 'halt_on_error=0:detect_leaks=0:print_summary=1' }

function Quote([string]$s) { if ($s -match '[\s"]') { '"' + ($s -replace '"', '\"') + '"' } else { $s } }

$bad = 0
foreach ($job in $Jobs) {
    $parts = $job -split ':', 3
    $tool = $parts[0]; $suite = $parts[1]; $filter = if ($parts.Count -gt 2) { $parts[2] } else { '' }
    $exe = Join-Path $BuildDir "$suite.exe"
    if (-not (Test-Path $exe)) { $exe = Join-Path $BuildDir $suite }
    if (-not (Test-Path $exe)) { throw "suite not found: $suite in $BuildDir" }
    $tag = "$tool-$suite" + $(if ($filter) { "-$filter" } else { '' })
    $log = Join-Path $OutDir "$tag.txt"

    $env:CUDA_LAUNCH_BLOCKING = $(if ($tool -eq 'plain') { '1' } else { $null })
    if ($tool -in 'asan', 'plain', 'run') {
        $file = $exe; $argList = @()
    } else {
        if ($tool -notin 'memcheck', 'racecheck', 'initcheck', 'synccheck') { throw "unknown tool: $tool" }
        $file = $Sanitizer
        $argList = @('--tool', $tool, '--print-limit', '200', '--show-backtrace', 'host')
        if ($tool -eq 'memcheck') { $argList += @('--leak-check', 'full') }
        if ($tool -eq 'racecheck') { $argList += @('--racecheck-report', 'all') }
        $argList += $ExtraArgs
        $argList += $exe
    }
    if ($filter) { $argList += $filter }

    "# $tag  $(Get-Date -Format s)  $file $($argList -join ' ')" | Set-Content $log
    $psi = [Diagnostics.ProcessStartInfo]::new($file, (($argList | ForEach-Object { Quote $_ }) -join ' '))
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.WorkingDirectory = $BuildDir
    $t0 = Get-Date
    $p = [Diagnostics.Process]::Start($psi)
    $outTask = $p.StandardOutput.ReadToEndAsync()
    $errTask = $p.StandardError.ReadToEndAsync()
    $status = 'done'
    if (-not $p.WaitForExit([int]($TimeoutMin * 60000))) {
        $status = 'TIMEOUT'
        # Kill through our own handle: taskkill can fail with "no running instance" for a process that is
        # visible to CIM but not openable from a new process (seen on this host).
        try { $p.Kill($true) } catch { & taskkill /T /F /PID $p.Id *> $null }
        [void]$p.WaitForExit(60000)
    }
    $sec = [int]((Get-Date) - $t0).TotalSeconds
    $text = $outTask.Result + "`n--- stderr ---`n" + $errTask.Result
    Add-Content $log $text
    $rc = $p.ExitCode

    $m = [regex]::Match($text, '(\d+) passed, (\d+) failed, (\d+) skipped')
    $tp, $tf, $ts = if ($m.Success) { $m.Groups[1].Value, $m.Groups[2].Value, $m.Groups[3].Value } else { '', '', '' }
    $errs = if ($tool -eq 'asan') {
        ([regex]::Matches($text, 'ERROR: AddressSanitizer')).Count
    } elseif ($tool -in 'plain', 'run') {
        ([regex]::Matches($text, 'illegal memory access|misaligned address|unspecified launch failure|device-side assert')).Count
    } else {
        $e = [regex]::Match($text, 'ERROR SUMMARY: (\d+) error')
        if ($e.Success) { [int]$e.Groups[1].Value } else { -1 }
    }
    $leak = [regex]::Match($text, 'LEAK SUMMARY: (\d+) bytes leaked')
    $leakBytes = if ($leak.Success) { $leak.Groups[1].Value } else { '' }
    if ($status -eq 'done' -and -not $m.Success) { $status = 'NO-SUMMARY' }
    # On Windows/WDDM compute-sanitizer needs the GPU debugger interface (EnableDebuggerInterface.bat, admin).
    # Without it the tool prints init errors and the suite runs UNCHECKED - never report that as clean.
    if ($tool -notin 'asan', 'plain', 'run' -and $text -match 'Failed to initialize WDDM debugger interface|Device not supported') {
        $status = 'SANITIZER-UNAVAILABLE'
    }
    "$tool,$suite,$filter,$rc,$sec,$tp,$tf,$ts,$errs,$leakBytes,$status" | Add-Content $summary
    Write-Host ("{0,-45} rc={1} {2}s tests {3}/{4}/{5} errors={6} leaked={7} {8}" -f $tag, $rc, $sec, $tp, $tf, $ts, $errs, $leakBytes, $status)
    if ($status -ne 'done' -or $errs -ne 0 -or ($tf -and [int]$tf -gt 0) -or $rc -ne 0) { $bad++ }
}
exit $(if ($bad) { 1 } else { 0 })
