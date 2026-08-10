param(
    [string]$ProjectDir = "$HOME\PycharmProjects\KARINA_Integration"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repository = "https://github.com/sungshin98/Multimodal_specialization_LLM.git"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $true)][string[]]$CommandArgs
    )

    Write-Host "`n> $Command $($CommandArgs -join ' ')" -ForegroundColor Cyan
    & $Command @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit=$LASTEXITCODE): $Command $($CommandArgs -join ' ')"
    }
}

function Resolve-CondaCommand {
    $existing = Get-Command conda -ErrorAction SilentlyContinue
    if ($existing) {
        return "conda"
    }

    $candidates = @(
        $env:CONDA_EXE,
        (Join-Path $HOME "anaconda3\Scripts\conda.exe"),
        (Join-Path $HOME "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:USERPROFILE "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    ) | Where-Object { $_ -and $_.Trim() }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    throw "Conda was not found."
}

function Find-SourcePython {
    param([Parameter(Mandatory = $true)][string]$CondaCommand)

    $jsonText = (& $CondaCommand env list --json) | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list Conda environments."
    }

    $envInfo = $jsonText | ConvertFrom-Json
    foreach ($envPath in $envInfo.envs) {
        $candidate = Join-Path $envPath "python.exe"
        if (-not (Test-Path $candidate)) {
            continue
        }

        & $candidate -c "import sys, numpy; assert sys.version_info >= (3, 10); print(sys.version.split()[0], numpy.__version__)" *> $null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }

    throw "No existing Conda environment has Python 3.10+ and NumPy."
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found."
}

$CondaCommand = Resolve-CondaCommand
$SourcePython = Find-SourcePython -CondaCommand $CondaCommand
$ProjectDir = [IO.Path]::GetFullPath($ProjectDir)
$ParentDir = Split-Path -Parent $ProjectDir
New-Item -ItemType Directory -Force -Path $ParentDir | Out-Null

Write-Host "Using source Python: $SourcePython" -ForegroundColor DarkCyan

if (-not (Test-Path $ProjectDir)) {
    Invoke-Checked -Command "git" -CommandArgs @("clone", "--no-single-branch", $Repository, $ProjectDir)
}

if (-not (Test-Path (Join-Path $ProjectDir ".git"))) {
    throw "Project directory is not a Git repository: $ProjectDir"
}

Set-Location $ProjectDir

Write-Host "`n[1/5] Refreshing all role branches" -ForegroundColor Green
Invoke-Checked -Command "git" -CommandArgs @("fetch", "origin", "joyechan", "sungshin", "haneul")
Invoke-Checked -Command "git" -CommandArgs @("reset", "--hard")
Invoke-Checked -Command "git" -CommandArgs @("switch", "-C", "integration", "origin/sungshin")
Invoke-Checked -Command "git" -CommandArgs @("checkout", "origin/joyechan", "--", "modular_encoder")
Invoke-Checked -Command "git" -CommandArgs @("checkout", "origin/haneul", "--", "evidence_decoder")

Write-Host "`n[2/5] Creating an isolated venv without downloads" -ForegroundColor Green
$VenvDir = Join-Path $ProjectDir ".venv"
if (Test-Path $VenvDir) {
    Remove-Item $VenvDir -Recurse -Force
}
Invoke-Checked -Command $SourcePython -CommandArgs @("-m", "venv", $VenvDir, "--system-site-packages")

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Venv Python was not created: $VenvPython"
}
Invoke-Checked -Command $VenvPython -CommandArgs @("-c", "import sys, numpy; assert sys.version_info >= (3, 10); print('Python', sys.version); print('NumPy', numpy.__version__)")

Write-Host "`n[3/5] Compiling all sources" -ForegroundColor Green
Invoke-Checked -Command $VenvPython -CommandArgs @("-m", "compileall", "adaptive_rag", "evidence_decoder", "modular_encoder")

Write-Host "`n[4/5] Running all offline tests" -ForegroundColor Green
$Tests = @(
    @("-m", "adaptive_rag.test_pipeline_V4"),
    @("-m", "adaptive_rag.test_role2_pipeline"),
    @("-m", "evidence_decoder.test_offline"),
    @("-m", "adaptive_rag.integration_offline_test")
)

foreach ($Test in $Tests) {
    Invoke-Checked -Command $VenvPython -CommandArgs $Test
}

Write-Host "`n[5/5] Completed" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "All KARINA offline integration tests passed." -ForegroundColor Green
Write-Host "Workspace: $ProjectDir" -ForegroundColor Green
Write-Host "Venv Python: $VenvPython" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
