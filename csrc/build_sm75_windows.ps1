param(
    [string]$CudaRoot = "",
    [string]$VisualStudioRoot = "",
    [string]$OutputName = "star7_sla_sm75_v7.dll",
    [int]$MaxRegisters = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $PSScriptRoot "sla_sm75_sparse.cu"
$outputDir = Join-Path $projectRoot "bin\win_amd64"
$output = Join-Path $outputDir $OutputName

if (-not $CudaRoot) {
    $nvccCommand = Get-Command nvcc.exe -ErrorAction SilentlyContinue
    if ($nvccCommand) {
        $CudaRoot = Split-Path -Parent (Split-Path -Parent $nvccCommand.Source)
    } else {
        $candidates = Get-ChildItem `
            "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" `
            -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
        if (-not $candidates) { throw "CUDA Toolkit was not found" }
        $CudaRoot = $candidates[0].FullName
    }
}

if (-not $VisualStudioRoot) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path -LiteralPath $vswhere)) { throw "vswhere.exe was not found" }
    # CUDA 13.0 accepts VS 2019/2022 but rejects the newer VS 2026 host
    # compiler.  Prefer the newest installation from the supported range.
    $VisualStudioRoot = & $vswhere -latest -version "[16.0,18.0)" -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $VisualStudioRoot) { throw "Visual Studio C++ Build Tools were not found" }
}

$vcvars = Join-Path $VisualStudioRoot "VC\Auxiliary\Build\vcvars64.bat"
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path -LiteralPath $vcvars)) { throw "vcvars64.bat was not found: $vcvars" }
if (-not (Test-Path -LiteralPath $nvcc)) { throw "nvcc.exe was not found: $nvcc" }
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$registerFlag = if ($MaxRegisters -gt 0) { "--maxrregcount=$MaxRegisters" } else { "" }
$command = 'call "{0}" && "{1}" -shared -O3 --use_fast_math -std=c++17 -arch=sm_75 --cudart static -Xcompiler=/MT -Xcompiler=/O2 -Xptxas=-v {2} -o "{3}" "{4}"' -f `
    $vcvars, $nvcc, $registerFlag, $output, $source
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) { throw "NVCC failed with exit code $LASTEXITCODE" }
Write-Host "Built $output"
