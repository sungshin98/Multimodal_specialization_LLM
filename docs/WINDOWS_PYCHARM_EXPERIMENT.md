# KARINA Windows + Anaconda + PyCharm 실험 가이드

대상 환경은 Windows 10/11, Anaconda 또는 Miniconda, PyCharm, NVIDIA RTX 4070 Ti 12GB, RAM 32GB이다. PowerShell 명령은 저장소 최상위 폴더에서 실행한다.

## 1. 저장소 준비

```powershell
git clone https://github.com/sungshin98/Multimodal_specialization_LLM.git
cd Multimodal_specialization_LLM
git switch main
```

현재 작업 폴더를 이미 PyCharm에서 열었다면 다시 clone하지 않는다. 설치 스크립트는 `git reset`, branch 변경, 기존 폴더 삭제를 수행하지 않는다.

## 2. 한 번에 환경 구성부터 모델 smoke test까지

PowerShell이 로컬 스크립트 실행을 차단할 때 현재 창에만 허용한다.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

빠른 최초 실행은 split당 MovieQA 20개만 정규화한다.

```powershell
.\scripts\windows\run_all.ps1 -EnvName karina -Device auto -Limit 20
```

이 명령은 다음을 순서대로 실행한다.

1. Python 3.11 Conda 환경 생성
2. FFmpeg와 Git 설치
3. 공식 PyTorch 2.6.0 CUDA 12.4 wheel 설치
4. Transformers·SentenceTransformers·OpenCV 등 설치
5. MovieQA 공식 공개 metadata 다운로드
6. MovieQA 공통 JSONL 정규화
7. 데이터·Role-2·Role-3 오프라인 회귀 테스트
8. 실제 MiniLM·CLIP·VideoMAE backbone 다운로드 및 CUDA 실행
9. checkpoint 존재 시 learned KARINA `InputRouter` 실행
10. `reports/local_experiment/latest.json` 결과 저장

최초 backbone 다운로드에는 수 GB의 저장 공간과 시간이 필요하다. Hugging Face cache는 저장소의 `.cache/huggingface/`에 생성된다.

`Limit=20` 결과는 `data/karina/processed/movieqa_smoke/`에만 기록된다. 이미 만든 전체 `processed/movieqa/` 파일을 줄이거나 덮어쓰지 않는다.

기본 setup은 CUDA wheel 설치 후 GPU 실행 가능 여부까지 검사한다. NVIDIA 드라이버를 바로 정비할 수 없거나 CPU 검증만 원하면 두 옵션을 함께 사용한다.

```powershell
.\scripts\windows\run_all.ps1 -EnvName karina_cpu -CpuOnly -Device cpu -Limit 20
```

같은 이름의 기존 환경이 Python 3.11이 아니면 스크립트는 그 환경을 임의로 변경하지 않고 중단한다. 이때 기존 환경을 지우지 말고 새 `-EnvName`을 지정한다. CPU/CUDA 모드를 바꾸면 요청한 공식 wheel build인지 확인한 뒤 필요한 경우에만 torch·torchvision을 교체한다.

## 3. 환경 구성과 실행을 분리하는 방법

```powershell
.\scripts\windows\01_setup_env.ps1 -EnvName karina
.\scripts\windows\02_run_experiment.ps1 -EnvName karina -Stage all -Device auto -Limit 20
```

전체 MovieQA metadata 14,944개를 변환하려면 `Limit=0`을 사용한다.

```powershell
.\scripts\windows\02_run_experiment.ps1 -EnvName karina -Stage data -Limit 0
```

이때 정식 출력은 `data/karina/processed/movieqa/`이고, 공식 split 개수 `9,848 / 1,958 / 3,138`도 검증한다.

단계별 실행:

```powershell
.\scripts\windows\02_run_experiment.ps1 -Stage preflight
.\scripts\windows\02_run_experiment.ps1 -Stage offline
.\scripts\windows\02_run_experiment.ps1 -Stage pretrained -Device cuda
.\scripts\windows\02_run_experiment.ps1 -Stage full -Device cuda -RequireFull
```

## 4. PyCharm Interpreter 설정

저장소에 남아 있는 `.idea`의 base Anaconda SDK를 그대로 사용하지 않는다.

1. `File → Settings → Project → Python Interpreter`
2. `Add Interpreter → Add Local Interpreter → Conda Environment`
3. `Existing environment` 선택
4. 일반적인 경로: `C:\Users\사용자명\anaconda3\envs\karina\python.exe`
5. Run Configuration의 Working directory를 저장소 최상위 폴더로 지정

PyCharm Run Configuration:

- 실행 형식: `Module name`
- Module name: `experiments.karina_local.run`
- Parameters: `--stage all --device auto --limit 20`
- Working directory: `Multimodal_specialization_LLM` 저장소 최상위 폴더

저장소 최상위가 working directory가 아니면 로컬 `datasets` 패키지 대신 Hugging Face의 동명 패키지를 불러올 수 있다.

## 5. 테스트 단계의 의미

| 단계 | 실제 실행 내용 | 논문 성능 결과 사용 |
|---|---|---|
| `offline` | Dataset adapter, Retrieval Gate, Role-2/Role-3 계약 | 불가, 합성 검증 |
| `pretrained` | 정규화된 MovieQA 1개를 Dataset/adapter로 전달하고 실제 backbone forward; 별도 합성 멀티모달 입력도 실행 | 불가, projection checkpoint 미사용 |
| `full` | 같은 MovieQA adapter 경로와 GitHub `InputRouter`를 learned projection checkpoint로 실행 | 실행 검증만 가능 |
| 별도 benchmark | 실제 MovieQA story/video, 검색 index, 정답 생성 | 논문 성능 평가 가능 |

`pretrained` 단계는 Text 384차원을 zero padding하여 768차원 계약만 검증한다. 학습된 cross-modal alignment가 아니므로 이 단계의 유사도나 정확도를 논문 결과로 사용하면 안 된다.

MovieQA story/video가 설치되지 않은 기본 상태에서는 정규화 샘플 검사도 `metadata_query_only`로 기록된다. 즉 실제 MovieQA 질문은 text backbone까지 들어가지만, 합성 이미지·동영상 검사는 그 질문과 독립된 실행 검사다. 두 결과를 MovieQA end-to-end 추론으로 해석하지 않는다.

## 6. GitHub에 없는 필수 체크포인트

현재 어느 원격 branch에도 다음 파일이 없다.

```text
modular_encoder/checkpoints/
├── resplit_135/joint_finetuned_best.pt
└── video_added/video_projection_best.pt
```

체크포인트가 없으면 `full` 단계는 명확한 사유와 함께 `skipped`로 기록되고 나머지 단계는 계속된다. 체크포인트를 연구실 PC나 기존 학습 서버에서 위 경로로 복사한 후 다음을 실행한다.

```powershell
.\scripts\windows\02_run_experiment.ps1 -Stage full -Device cuda -RequireFull
```

임의의 random checkpoint를 만들어 full model 결과로 사용하지 않는다.

## 7. 실제 데이터 범위

- MovieQA의 `qa.json`, `movies.json`, `splits.json`: 자동 다운로드
- MovieQA story·subtitle·script·video: 공식 benchmark 등록 후 수동 배치
- MovieNet: OpenDataLab 약관 동의 후 선택 배치
- MovieChat-1K: 공식 Hugging Face 접근 조건 확인 후 선택 배치

MovieQA story/video가 없으면 질문과 정답 metadata는 변환되지만 `initial_input`은 비어 있다. 이 상태에서도 배관 테스트는 가능하지만 실제 멀티모달 QA 성능 실험은 아니다.

## 8. 결과 확인

```text
reports/local_experiment/
├── latest.json
└── karina_local_run_YYYYMMDD_HHMMSS.json
```

보고서에는 Python·패키지·CUDA·GPU·데이터·checkpoint 상태, 단계별 성공 여부, encoder shape/norm, latency, peak GPU memory, Retrieval Gate packet 상태가 기록된다. `preflight.status=warning`은 CPU fallback, 누락된 dependency 또는 checkpoint처럼 실행 가능 범위를 제한하는 조건을 `issues`에 함께 기록한다. 단계에 `warning`이나 `skipped`가 하나라도 있으면 최상위 상태는 `passed_with_limitations`이며, `full_model_executed`가 learned router의 실제 실행 여부를 별도로 보여준다.

오류가 발생하면 `latest.json`의 `status`, `fatal_error`, 각 단계의 `reason`을 먼저 확인한다.

PyTorch 조합은 공식 Windows 설치 안내와 공식 이전 버전 wheel 표에 맞춘 Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, CUDA 12.4이다.

- https://docs.pytorch.org/get-started/locally/
- https://docs.pytorch.org/get-started/previous-versions/
