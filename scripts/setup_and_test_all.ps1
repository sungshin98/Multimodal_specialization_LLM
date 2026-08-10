param(
    [string]$ProjectDir = "$HOME\PycharmProjects\KARINA_Integration",
    [string]$EnvName = "KARINA",
    [switch]$ReuseExisting,
    [switch]$InstallEncoderDependencies,
    [switch]$RunRealEncoderTest
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

function Assert-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Command not found: $Name"
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

    throw "Conda was not found. Expected a command named 'conda' or an installation under USERPROFILE\anaconda3 or USERPROFILE\miniconda3."
}

function Test-CondaEnvironment {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CondaCommand
    )

    $jsonText = (& $CondaCommand env list --json) | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to list Conda environments."
    }
    $json = $jsonText | ConvertFrom-Json
    $suffix = [IO.Path]::DirectorySeparatorChar + $Name
    return [bool]($json.envs | Where-Object { $_.EndsWith($suffix, [StringComparison]::OrdinalIgnoreCase) })
}

Assert-Command git
$CondaCommand = Resolve-CondaCommand
Write-Host "Using Conda: $CondaCommand" -ForegroundColor DarkCyan

$ProjectDir = [IO.Path]::GetFullPath($ProjectDir)
$ParentDir = Split-Path -Parent $ProjectDir
New-Item -ItemType Directory -Force -Path $ParentDir | Out-Null

Write-Host "`n[1/6] Preparing repository and all role branches" -ForegroundColor Green
if (Test-Path $ProjectDir) {
    if ($ReuseExisting) {
        if (-not (Test-Path (Join-Path $ProjectDir ".git"))) {
            throw "-ReuseExisting was specified, but this is not a Git repository: $ProjectDir"
        }
        Write-Host "Reusing existing integration workspace: $ProjectDir" -ForegroundColor Yellow
        Set-Location $ProjectDir
    } else {
        $backup = "${ProjectDir}_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Write-Host "Moving the existing directory to: $backup" -ForegroundColor Yellow
        Move-Item -Path $ProjectDir -Destination $backup
        Invoke-Checked -Command "git" -CommandArgs @("clone", "--no-single-branch", $Repository, $ProjectDir)
        Set-Location $ProjectDir
    }
} else {
    Invoke-Checked -Command "git" -CommandArgs @("clone", "--no-single-branch", $Repository, $ProjectDir)
    Set-Location $ProjectDir
}
Invoke-Checked -Command "git" -CommandArgs @("fetch", "origin", "joyechan", "sungshin", "haneul")

Write-Host "`n[2/6] Building the integration workspace" -ForegroundColor Green
Invoke-Checked -Command "git" -CommandArgs @("switch", "-C", "integration", "origin/sungshin")
Invoke-Checked -Command "git" -CommandArgs @("checkout", "origin/joyechan", "--", "modular_encoder")
Invoke-Checked -Command "git" -CommandArgs @("checkout", "origin/haneul", "--", "evidence_decoder")

Write-Host "Integrated directories:" -ForegroundColor DarkCyan
Write-Host "  modular_encoder : joyechan" -ForegroundColor DarkCyan
Write-Host "  adaptive_rag    : sungshin" -ForegroundColor DarkCyan
Write-Host "  evidence_decoder: haneul" -ForegroundColor DarkCyan

Write-Host "`n[3/6] Preparing the Conda environment" -ForegroundColor Green
if (-not (Test-CondaEnvironment -Name $EnvName -CondaCommand $CondaCommand)) {
    Invoke-Checked -Command $CondaCommand -CommandArgs @("create", "-n", $EnvName, "python=3.11", "-y")
} else {
    Write-Host "Reusing the existing Conda environment: $EnvName" -ForegroundColor Yellow
}
Invoke-Checked -Command $CondaCommand -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Command $CondaCommand -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "pip", "install", "numpy", "python-dotenv")

if ($InstallEncoderDependencies -or $RunRealEncoderTest) {
    Write-Host "`nInstalling encoder dependencies. This may take a while." -ForegroundColor Yellow
    Invoke-Checked -Command $CondaCommand -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "pip", "install", "-r", "modular_encoder\requirements.txt")
}

Write-Host "`n[4/6] Compiling all source files" -ForegroundColor Green
Invoke-Checked -Command $CondaCommand -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "compileall", "adaptive_rag", "evidence_decoder", "modular_encoder")

Write-Host "`n[5/6] Running module and end-to-end offline tests" -ForegroundColor Green
$Tests = @(
    @("python", "-m", "adaptive_rag.test_pipeline_V4"),
    @("python", "-m", "adaptive_rag.test_role2_pipeline"),
    @("python", "-m", "evidence_decoder.test_offline"),
    @("python", "-m", "adaptive_rag.integration_offline_test")
)

foreach ($Test in $Tests) {
    $CondaArgs = @("run", "-n", $EnvName, "--no-capture-output") + $Test
    Invoke-Checked -Command $CondaCommand -CommandArgs $CondaArgs
}

Write-Host "`n[6/6] Checking the optional real encoder test" -ForegroundColor Green
if ($RunRealEncoderTest) {
    $RequiredFiles = @(
        "modular_encoder\checkpoints\resplit_135\joint_finetuned_best.pt",
        "modular_encoder\checkpoints\video_added\video_projection_best.pt",
        "modular_encoder\data\processed\frames\charade\charade_00000.jpg",
        "modular_encoder\data\processed\clips\charade\charade_00000.mp4"
    )
    $Missing = @($RequiredFiles | Where-Object { -not (Test-Path $_) })
    if ($Missing.Count -gt 0) {
        Write-Host "The real encoder test files are not stored in Git." -ForegroundColor Yellow
        Write-Host "Place the following files and rerun with -ReuseExisting:" -ForegroundColor Yellow
        $Missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
        throw "Missing real encoder test files."
    }

    Push-Location modular_encoder
    try {
        Invoke-Checked -Command $CondaCommand -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "tests\test_final_router.py")
    }
    finally {
        Pop-Location
    }
} else {
    Write-Host "The real encoder model test was skipped." -ForegroundColor Yellow
    Write-Host "After placing checkpoints and sample data, rerun with -ReuseExisting -RunRealEncoderTest." -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "All KARINA integration offline tests passed." -ForegroundColor Green
Write-Host "Workspace: $ProjectDir" -ForegroundColor Green
Write-Host "Conda environment: $EnvName" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
