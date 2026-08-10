param(
    [string]$ProjectDir = "$HOME\PycharmProjects\KARINA_Integration",
    [string]$EnvName = "KARINA",
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
        throw "명령 실패(exit=$LASTEXITCODE): $Command $($CommandArgs -join ' ')"
    }
}

function Assert-Command {
    param([Parameter(Mandatory = $true)][string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "'$Name' 명령을 찾지 못했습니다. Anaconda PowerShell Prompt에서 실행하고 Git을 설치해 주세요."
    }
}

function Test-CondaEnvironment {
    param([Parameter(Mandatory = $true)][string]$Name)
    $json = (& conda env list --json) | Out-String | ConvertFrom-Json
    $suffix = [IO.Path]::DirectorySeparatorChar + $Name
    return [bool]($json.envs | Where-Object { $_.EndsWith($suffix, [StringComparison]::OrdinalIgnoreCase) })
}

Assert-Command git
Assert-Command conda

$ProjectDir = [IO.Path]::GetFullPath($ProjectDir)
$ParentDir = Split-Path -Parent $ProjectDir
New-Item -ItemType Directory -Force -Path $ParentDir | Out-Null

if (Test-Path $ProjectDir) {
    $backup = "${ProjectDir}_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Write-Host "기존 폴더를 보존하기 위해 이동합니다: $backup" -ForegroundColor Yellow
    Move-Item -Path $ProjectDir -Destination $backup
}

Write-Host "`n[1/6] 저장소와 세 담당 브랜치 다운로드" -ForegroundColor Green
Invoke-Checked -Command "git" -CommandArgs @("clone", "--no-single-branch", $Repository, $ProjectDir)
Set-Location $ProjectDir
Invoke-Checked -Command "git" -CommandArgs @("fetch", "origin", "joyechan", "sungshin", "haneul")

Write-Host "`n[2/6] 통합 작업공간 구성" -ForegroundColor Green
Invoke-Checked -Command "git" -CommandArgs @("switch", "-C", "integration", "origin/sungshin")
Invoke-Checked -Command "git" -CommandArgs @("checkout", "origin/joyechan", "--", "modular_encoder")
Invoke-Checked -Command "git" -CommandArgs @("checkout", "origin/haneul", "--", "evidence_decoder")

Write-Host "통합 폴더:" -ForegroundColor DarkCyan
Write-Host "  modular_encoder : joyechan" -ForegroundColor DarkCyan
Write-Host "  adaptive_rag    : sungshin" -ForegroundColor DarkCyan
Write-Host "  evidence_decoder: haneul" -ForegroundColor DarkCyan

Write-Host "`n[3/6] Conda 환경 준비" -ForegroundColor Green
if (-not (Test-CondaEnvironment $EnvName)) {
    Invoke-Checked -Command "conda" -CommandArgs @("create", "-n", $EnvName, "python=3.11", "-y")
} else {
    Write-Host "기존 Conda 환경을 재사용합니다: $EnvName" -ForegroundColor Yellow
}
Invoke-Checked -Command "conda" -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked -Command "conda" -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "pip", "install", "numpy", "python-dotenv")

if ($InstallEncoderDependencies -or $RunRealEncoderTest) {
    Write-Host "`nEncoder 의존성 설치를 시작합니다. Torch/Transformers 설치로 시간이 걸릴 수 있습니다." -ForegroundColor Yellow
    Invoke-Checked -Command "conda" -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "pip", "install", "-r", "modular_encoder\requirements.txt")
}

Write-Host "`n[4/6] 전체 소스 문법 검사" -ForegroundColor Green
Invoke-Checked -Command "conda" -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "-m", "compileall", "adaptive_rag", "evidence_decoder", "modular_encoder")

Write-Host "`n[5/6] 담당 모듈 및 전체 연결 오프라인 테스트" -ForegroundColor Green
$Tests = @(
    @("python", "-m", "adaptive_rag.test_pipeline_V4"),
    @("python", "-m", "adaptive_rag.test_role2_pipeline"),
    @("python", "-m", "evidence_decoder.test_offline"),
    @("python", "-m", "adaptive_rag.integration_offline_test")
)

foreach ($Test in $Tests) {
    $CondaArgs = @("run", "-n", $EnvName, "--no-capture-output") + $Test
    Invoke-Checked -Command "conda" -CommandArgs $CondaArgs
}

Write-Host "`n[6/6] 실제 Encoder 테스트 확인" -ForegroundColor Green
if ($RunRealEncoderTest) {
    $RequiredFiles = @(
        "modular_encoder\checkpoints\resplit_135\joint_finetuned_best.pt",
        "modular_encoder\checkpoints\video_added\video_projection_best.pt",
        "modular_encoder\data\processed\frames\charade\charade_00000.jpg",
        "modular_encoder\data\processed\clips\charade\charade_00000.mp4"
    )
    $Missing = @($RequiredFiles | Where-Object { -not (Test-Path $_) })
    if ($Missing.Count -gt 0) {
        Write-Host "실제 Encoder 테스트에 필요한 체크포인트/샘플 데이터가 Git에 포함되어 있지 않습니다." -ForegroundColor Yellow
        Write-Host "다음 경로에 파일을 배치한 뒤 같은 명령을 다시 실행하세요:" -ForegroundColor Yellow
        $Missing | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
        throw "실제 Encoder 테스트 파일 누락"
    }

    Push-Location modular_encoder
    try {
        Invoke-Checked -Command "conda" -CommandArgs @("run", "-n", $EnvName, "--no-capture-output", "python", "tests\test_final_router.py")
    }
    finally {
        Pop-Location
    }
} else {
    Write-Host "실제 Encoder 모델 테스트는 생략했습니다." -ForegroundColor Yellow
    Write-Host "체크포인트와 샘플 데이터를 배치한 뒤 -RunRealEncoderTest 옵션으로 실행할 수 있습니다." -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "KARINA 통합 오프라인 테스트를 모두 통과했습니다." -ForegroundColor Green
Write-Host "작업 폴더: $ProjectDir" -ForegroundColor Green
Write-Host "Conda 환경: $EnvName" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
