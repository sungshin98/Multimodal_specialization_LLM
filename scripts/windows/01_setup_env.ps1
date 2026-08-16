param(
    [string]$EnvName = "karina",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$Requirements = Join-Path $RepoRoot "requirements\karina-windows.txt"

function Resolve-CondaCommand {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return (Resolve-Path $env:CONDA_EXE).Path
    }
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) {
        if ($command.CommandType -eq "Application") { return $command.Path }
        # In an initialized Anaconda PowerShell, conda is commonly a Function
        # whose Source/Path is empty. The invocation operator still resolves it.
        return "conda"
    }
    $candidates = @()
    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe")
        $candidates += (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe")
    }
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe")
        $candidates += (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe")
    }
    $candidates += "C:\ProgramData\anaconda3\Scripts\conda.exe"
    $candidates += "C:\ProgramData\miniconda3\Scripts\conda.exe"
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    throw "Conda를 찾지 못했습니다. Anaconda Prompt에서 실행하거나 Anaconda/Miniconda 설치 경로를 확인하세요."
}

function Invoke-Checked {
    param([string]$Command, [string[]]$CommandArgs)
    Write-Host "`n> $Command $($CommandArgs -join ' ')" -ForegroundColor Cyan
    & $Command @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "명령 실패(exit=$LASTEXITCODE): $Command $($CommandArgs -join ' ')"
    }
}

$Conda = Resolve-CondaCommand
$envJson = (& $Conda env list --json) | Out-String | ConvertFrom-Json
$exists = @($envJson.envs | ForEach-Object { Split-Path $_ -Leaf }) -contains $EnvName

Write-Host "Repository: $RepoRoot" -ForegroundColor Green
Write-Host "Conda: $Conda" -ForegroundColor Green

if (-not $exists) {
    Invoke-Checked $Conda @(
        "create", "-n", $EnvName, "python=3.11", "pip", "ffmpeg", "git",
        "-c", "conda-forge", "--override-channels", "-y"
    )
} else {
    Write-Host "기존 Conda 환경을 재사용합니다: $EnvName" -ForegroundColor Yellow
}

$PythonVersionCode = @'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "KARINA requires Python 3.11, but this existing environment uses "
        + sys.version.split()[0]
        + ". Choose a new -EnvName or recreate that environment explicitly."
    )
print("Python version check: 3.11")
'@
Invoke-Checked $Conda @(
    "run", "-n", $EnvName, "--no-capture-output", "python", "-c", $PythonVersionCode
)
if ($exists) {
    Invoke-Checked $Conda @(
        "install", "-n", $EnvName, "pip", "ffmpeg", "git",
        "-c", "conda-forge", "--override-channels", "-y"
    )
}

Invoke-Checked $Conda @(
    "run", "-n", $EnvName, "--no-capture-output",
    "python", "-m", "pip", "install", "--upgrade", "pip"
)

if ($CpuOnly) {
    $TorchIndex = "https://download.pytorch.org/whl/cpu"
    $ExpectedTorchTag = "2.6.0+cpu|0.21.0+cpu"
} else {
    # Reproducible official Windows wheel; RTX 4070 Ti is supported with a current driver.
    $TorchIndex = "https://download.pytorch.org/whl/cu124"
    $ExpectedTorchTag = "2.6.0+cu124|0.21.0+cu124"
}
$TorchVersionCode = @'
from importlib import metadata
def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"
print("KARINA_TORCH=" + version("torch") + "|" + version("torchvision"))
'@
$TorchOutput = & $Conda run -n $EnvName python -c $TorchVersionCode
if ($LASTEXITCODE -ne 0) { throw "기존 PyTorch 버전을 확인하지 못했습니다." }
$TorchMarker = $TorchOutput | Where-Object { $_ -like "KARINA_TORCH=*" } | Select-Object -Last 1
$InstalledTorchTag = if ($TorchMarker) { $TorchMarker.Substring("KARINA_TORCH=".Length).Trim() } else { "unknown" }
if ($InstalledTorchTag -ne $ExpectedTorchTag) {
    Write-Host "PyTorch wheel 교체: $InstalledTorchTag -> $ExpectedTorchTag" -ForegroundColor Yellow
    Invoke-Checked $Conda @(
        "run", "-n", $EnvName, "--no-capture-output",
        "python", "-m", "pip", "uninstall", "-y", "torch", "torchvision"
    )
    Invoke-Checked $Conda @(
        "run", "-n", $EnvName, "--no-capture-output",
        "python", "-m", "pip", "install",
        "torch==2.6.0", "torchvision==0.21.0", "--index-url", $TorchIndex
    )
} else {
    Write-Host "요청한 PyTorch wheel이 이미 설치되어 있습니다: $InstalledTorchTag" -ForegroundColor Green
}
Invoke-Checked $Conda @(
    "run", "-n", $EnvName, "--no-capture-output",
    "python", "-m", "pip", "install", "-r", $Requirements
)

$VerifyCode = @'
import sys, torch
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM(GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
'@
Invoke-Checked $Conda @(
    "run", "-n", $EnvName, "--no-capture-output", "python", "-c", $VerifyCode
)
if (-not $CpuOnly) {
    $CudaRequiredCode = @'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA PyTorch was installed, but no usable CUDA GPU was detected. "
        "Update the NVIDIA driver, or rerun setup with -CpuOnly."
    )
print("CUDA execution check:", torch.cuda.get_device_name(0))
'@
    Invoke-Checked $Conda @(
        "run", "-n", $EnvName, "--no-capture-output", "python", "-c", $CudaRequiredCode
    )
}

$envPath = @($envJson.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName })
if ($envPath.Count -eq 0) {
    $envJson = (& $Conda env list --json) | Out-String | ConvertFrom-Json
    $envPath = @($envJson.envs | Where-Object { (Split-Path $_ -Leaf) -eq $EnvName })
}
$Interpreter = if ($envPath.Count -gt 0) { Join-Path $envPath[0] "python.exe" } else { "conda env: $EnvName" }

Write-Host "`n환경 구성이 완료되었습니다." -ForegroundColor Green
Write-Host "PyCharm Interpreter: $Interpreter" -ForegroundColor Green
