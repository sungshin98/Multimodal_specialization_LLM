param(
    [string]$EnvName = "karina",
    [ValidateSet("all", "preflight", "data", "offline", "pretrained", "full")]
    [string]$Stage = "all",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [int]$Limit = 20,
    [switch]$RequireFull
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
Set-Location $RepoRoot

function Resolve-CondaCommand {
    if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
        return (Resolve-Path $env:CONDA_EXE).Path
    }
    $command = Get-Command conda -ErrorAction SilentlyContinue
    if ($command) {
        if ($command.CommandType -eq "Application") { return $command.Path }
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
    throw "Conda를 찾지 못했습니다. Anaconda Prompt에서 실행해 주세요."
}

$Conda = Resolve-CondaCommand

$RunArgs = @(
    "run", "-n", $EnvName, "--no-capture-output",
    "python", "-m", "experiments.karina_local.run",
    "--config", "configs/karina_local_experiment.yaml",
    "--stage", $Stage,
    "--device", $Device,
    "--limit", "$Limit"
)
if ($RequireFull) { $RunArgs += "--require-full" }

Write-Host "> $Conda $($RunArgs -join ' ')" -ForegroundColor Cyan
& $Conda @RunArgs
if ($LASTEXITCODE -ne 0) {
    throw "KARINA 실험 실행이 실패했습니다. reports\local_experiment\latest.json을 확인하세요."
}

Write-Host "`n실행 완료: reports\local_experiment\latest.json" -ForegroundColor Green
