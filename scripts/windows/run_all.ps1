param(
    [string]$EnvName = "karina",
    [int]$Limit = 20,
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",
    [switch]$CpuOnly,
    [switch]$RequireFull
)

$ErrorActionPreference = "Stop"
$Setup = Join-Path $PSScriptRoot "01_setup_env.ps1"
$Run = Join-Path $PSScriptRoot "02_run_experiment.ps1"

& $Setup -EnvName $EnvName -CpuOnly:$CpuOnly
if ($LASTEXITCODE -ne 0) { throw "환경 구성 실패" }

& $Run -EnvName $EnvName -Stage all -Device $Device -Limit $Limit -RequireFull:$RequireFull
if ($LASTEXITCODE -ne 0) { throw "실험 실행 실패" }

