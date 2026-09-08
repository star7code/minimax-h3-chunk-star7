$ErrorActionPreference = 'Stop'

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "[DLSS5-NR] ERROR: $Message" -ForegroundColor Red
    exit 1
}

$Source = $PSScriptRoot
$Native = Split-Path -Parent $Source
$Root = Split-Path -Parent $Native
$Bin = Join-Path $Native 'bin'
$CallerOut = Join-Path $Root 'runtime\caller'

Write-Host "[DLSS5-NR] PowerShell native builder v0.3.0"
Write-Host "[DLSS5-NR] No Developer Command Prompt is required."

$pf86 = [Environment]::GetFolderPath('ProgramFilesX86')
$pf = [Environment]::GetFolderPath('ProgramFiles')
$programRoots = @($pf86, $pf) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path $_) } | Select-Object -Unique
if (-not $programRoots -or $programRoots.Count -eq 0) { Fail 'Could not determine Program Files directories.' }

# Find the newest installed x64 MSVC compiler. Supports Build Tools and
# full Visual Studio installations on local machines and GitHub Actions.
$clCandidates = @()
foreach ($programRoot in $programRoots) {
    $vsRoot = Join-Path $programRoot 'Microsoft Visual Studio'
    if (-not (Test-Path $vsRoot)) { continue }
    $clCandidates += Get-ChildItem -Path $vsRoot -Filter cl.exe -File -Recurse -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match '\\VC\\Tools\\MSVC\\[^\\]+\\bin\\Hostx64\\x64\\cl\.exe$' } |
        Select-Object -ExpandProperty FullName
}
$clCandidates = $clCandidates | Select-Object -Unique
if (-not $clCandidates -or $clCandidates.Count -eq 0) {
    Fail 'MSVC x64 compiler cl.exe was not found. Install Desktop development with C++.'
}

# Sort by the toolset version directory where possible.
$cl = $clCandidates | Sort-Object {
    if ($_ -match '\\MSVC\\([^\\]+)\\bin\\Hostx64') {
        try { [version]$Matches[1] } catch { [version]'0.0' }
    } else { [version]'0.0' }
} -Descending | Select-Object -First 1

$msvcVersionDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $cl)))
# cl = ...\MSVC\14.xx\bin\Hostx64\x64\cl.exe; four parents => ...\MSVC\14.xx
if (-not (Test-Path (Join-Path $msvcVersionDir 'include'))) {
    # Defensive fallback derived from regex.
    if ($cl -match '^(.*\\MSVC\\[^\\]+)\\bin\\Hostx64\\x64\\cl\.exe$') {
        $msvcVersionDir = $Matches[1]
    }
}
$msvcInclude = Join-Path $msvcVersionDir 'include'
$msvcLib = Join-Path $msvcVersionDir 'lib\x64'
if (-not (Test-Path (Join-Path $msvcInclude 'vector'))) { Fail "MSVC include directory is invalid: $msvcInclude" }
if (-not (Test-Path $msvcLib)) { Fail "MSVC library directory is invalid: $msvcLib" }

# Find the newest Windows 10/11 SDK installed in Windows Kits\10.
$sdkRoot = $null
foreach ($programRoot in $programRoots) {
    $candidateSdk = Join-Path $programRoot 'Windows Kits\10'
    if (Test-Path (Join-Path $candidateSdk 'Include')) { $sdkRoot = $candidateSdk; break }
}
if (-not $sdkRoot) { Fail 'Windows Kits\10 was not found.' }
$sdkIncludeRoot = Join-Path $sdkRoot 'Include'
$sdkLibRoot = Join-Path $sdkRoot 'Lib'
if (-not (Test-Path $sdkIncludeRoot)) { Fail "Windows SDK Include directory not found: $sdkIncludeRoot" }

$sdkVersions = Get-ChildItem -Path $sdkIncludeRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        (Test-Path (Join-Path $_.FullName 'um\windows.h')) -and
        (Test-Path (Join-Path $sdkLibRoot ($_.Name + '\um\x64\d3d12.lib'))) -and
        (Test-Path (Join-Path $sdkLibRoot ($_.Name + '\um\x64\d3d11.lib')))
    } |
    Sort-Object {
        try { [version]$_.Name } catch { [version]'0.0' }
    } -Descending

if (-not $sdkVersions -or $sdkVersions.Count -eq 0) {
    Fail 'A usable Windows SDK with windows.h, d3d12.lib and d3d11.lib was not found.'
}
$sdkVersion = $sdkVersions[0].Name
$sdkIncBase = Join-Path $sdkIncludeRoot $sdkVersion
$sdkLibBase = Join-Path $sdkLibRoot $sdkVersion

$includeDirs = @(
    $msvcInclude,
    (Join-Path $sdkIncBase 'ucrt'),
    (Join-Path $sdkIncBase 'shared'),
    (Join-Path $sdkIncBase 'um'),
    (Join-Path $sdkIncBase 'winrt'),
    (Join-Path $sdkIncBase 'cppwinrt')
) | Where-Object { Test-Path $_ }

$libDirs = @(
    $msvcLib,
    (Join-Path $sdkLibBase 'ucrt\x64'),
    (Join-Path $sdkLibBase 'um\x64')
) | Where-Object { Test-Path $_ }

$clDir = Split-Path -Parent $cl
$env:PATH = "$clDir;$env:PATH"
$env:INCLUDE = ($includeDirs -join ';')
$env:LIB = ($libDirs -join ';')

Write-Host "[DLSS5-NR] Compiler: $cl"
Write-Host "[DLSS5-NR] MSVC root: $msvcVersionDir"
Write-Host "[DLSS5-NR] Windows SDK: $sdkVersion"

New-Item -ItemType Directory -Force -Path $Bin | Out-Null
New-Item -ItemType Directory -Force -Path $CallerOut | Out-Null

function Run-Cl([string[]]$Arguments, [string]$StepName) {
    Write-Host ""
    Write-Host $StepName
    & $cl @Arguments
    if ($LASTEXITCODE -ne 0) {
        Fail "$StepName failed with cl.exe exit code $LASTEXITCODE."
    }
}

$common = @('/nologo','/std:c++17','/EHsc','/MT','/LD')
foreach ($inc in $includeDirs) { $common += ('/I' + $inc) }

$callerCpp = Join-Path $Source 'caller_shim.cpp'
$callerDll = Join-Path $CallerOut 'nvngx.dll_comfy.dll'
if (-not (Test-Path -LiteralPath $callerDll)) {
    Run-Cl ($common + @('/Od', $callerCpp, '/link', ('/OUT:' + $callerDll)) + ($libDirs | ForEach-Object { '/LIBPATH:' + $_ })) '[1/2] Building caller shim...'
} else {
    Write-Host '[1/2] Existing caller shim retained (safe while ComfyUI is running).'
}

$bridgeCpp = Join-Path $Source 'dlss5nr_bridge.cpp'
$nvofCpp = Join-Path $Source 'nvof_flow.cpp'
$simdCpp = Join-Path $Source 'simd_convert.cpp'
$simdObj = Join-Path $Source 'simd_convert.obj'
$bridgeDll = Join-Path $Bin 'dlss5nr_bridge_v2.dll'
$simdCompile = @('/nologo','/std:c++17','/EHsc','/MT','/O2','/c','/arch:AVX2',('/Fo' + $simdObj))
foreach ($inc in $includeDirs) { $simdCompile += ('/I' + $inc) }
Run-Cl ($simdCompile + @($simdCpp)) '[2/3] Building optional F16C transfer accelerator...'
Run-Cl ($common + @('/O2', $bridgeCpp, $nvofCpp, $simdObj, '/link', ('/OUT:' + $bridgeDll), 'd3d12.lib', 'd3d11.lib', 'dxgi.lib', 'ole32.lib') + ($libDirs | ForEach-Object { '/LIBPATH:' + $_ })) '[3/3] Building in-process ComfyUI bridge + NVIDIA Optical Flow...'

# Remove intermediary build products created next to the invocation working directory / source.
Get-ChildItem -Path $Root -Filter '*.obj' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Root -Filter '*.exp' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Root -Filter '*.lib' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Source -Filter '*.obj' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Source -Filter '*.exp' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $Source -Filter '*.lib' -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host '[DLSS5-NR] Build complete.' -ForegroundColor Green
Write-Host "  Bridge: $bridgeDll"
Write-Host "  Shim:   $callerDll"
Write-Host ""
Write-Host 'Place nvngx_dlssnr.dll in:'
Write-Host "  $(Join-Path $Root 'runtime\nvngx_dlssnr.dll')"
Write-Host 'then restart ComfyUI.'
exit 0
