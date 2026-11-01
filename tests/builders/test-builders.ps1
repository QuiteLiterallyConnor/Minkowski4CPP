$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutputDir = Join-Path ([System.IO.Path]::GetTempPath()) ("minkowski-builder-test-" + [guid]::NewGuid())
$Calls = New-Object System.Collections.Generic.List[object]

function docker {
    $Calls.Add(@($args))
    $global:LASTEXITCODE = 0
    if ($args[0] -eq "info") { return "29.9.2" }
    if ($args[0] -eq "build" -and $args -contains "--output") {
        $versionArg = @($args | Where-Object { $_ -like "VERSION=*" })[0]
        $cudaArg = @($args | Where-Object { $_ -like "CUDA_VERSION=*" })[0]
        $destArg = @($args | Where-Object { $_ -like "type=local,dest=*" })[0]
        $version = $versionArg.Substring("VERSION=".Length)
        $cuda = $cudaArg.Substring("CUDA_VERSION=".Length)
        $dest = $destArg.Substring("type=local,dest=".Length)
        [System.IO.File]::WriteAllText((Join-Path $dest "libminkowski-cpp-dev_${version}_cu${cuda}_amd64.deb"), "test package")
    }
}

function Assert-Contains($Call, $Value) {
    if ($Call -notcontains $Value) { throw "Missing argument: $Value" }
}

try {
    foreach ($cuda in @("124", "132", "134")) {
        $Calls.Clear()
        & (Join-Path $ProjectRoot "docker\docker-build.ps1") -CudaVersion $cuda -Repo test -User test -Push -Jobs 2 -CudaArchitectures "86;89"
        $build = $Calls[0]
        Assert-Contains $build "CUDA_VERSION=$cuda"
        Assert-Contains $build "CUDA_ARCHITECTURES=86;89"
        Assert-Contains $build "JOBS=2"
        Assert-Contains $build "linux/amd64"
        Assert-Contains $build "test/test:cu$cuda"
        $expectedCalls = if ($cuda -eq "134") { 4 } else { 2 }
        if ($Calls.Count -ne $expectedCalls) { throw "Unexpected latest publishing for cu$cuda" }

        $Calls.Clear()
        & (Join-Path $ProjectRoot "apt\build-apt.ps1") -CudaVersion $cuda -Version 1.3.0 -OutputDir $OutputDir -KeepImage -Jobs 2 -CudaArchitectures "86;89"
        $build = $Calls[1]
        Assert-Contains $build "CUDA_VERSION=$cuda"
        Assert-Contains $build "CUDA_ARCHITECTURES=86;89"
        Assert-Contains $build "JOBS=2"
        Assert-Contains $build "linux/amd64"
        if (-not (Test-Path (Join-Path $OutputDir "libminkowski-cpp-dev_1.3.0_cu${cuda}_amd64.deb"))) {
            throw "Missing cu$cuda package"
        }
    }

    $Calls.Clear()
    & (Join-Path $ProjectRoot "docker\docker-build.ps1") -Repo test -User test
    Assert-Contains $Calls[0] "CUDA_VERSION=134"
    Assert-Contains $Calls[0] "test/test:cu134"

    $Calls.Clear()
    & (Join-Path $ProjectRoot "apt\build-apt.ps1") -OutputDir $OutputDir -KeepImage
    Assert-Contains $Calls[1] "CUDA_VERSION=134"

    $Calls.Clear()
    & (Join-Path $ProjectRoot "docker\docker-build.ps1") -Repo test -User test -CudaVersion 124 -Tag legacy
    Assert-Contains $Calls[0] "test/test:legacy"

    foreach ($builder in @("apt\build-apt.ps1", "docker\docker-build.ps1")) {
        $rejected = $false
        try { & (Join-Path $ProjectRoot $builder) -CudaVersion 999 } catch [System.Management.Automation.ParameterBindingException] { $rejected = $true }
        if (-not $rejected) { throw "Invalid profile accepted by $builder" }
    }
    Write-Host "PowerShell builder tests passed."
} finally {
    if (Test-Path $OutputDir) { Remove-Item -LiteralPath $OutputDir -Recurse -Force }
}
