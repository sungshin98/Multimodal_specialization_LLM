# Modular Multimodal Encoder

입력 모달리티에 따라 필요한 인코더만 선택적으로 활성화하는 **Text / Image / Video 모듈형 인코더**입니다.  
최종 평가에서는 Router + Lazy Loading 구조의 효율성과 공통 768차원 임베딩 공간의 교차 모달 검색 성능을 측정했습니다.

## Final architecture

```text
Query / Document ── MiniLM (384) ── Text Projection ──┐
Image            ── CLIP ViT-B/32 (768) ─ Image Projection ─┼─ 768-d shared embedding
Video            ── VideoMAE (768) ─── Video Projection ────┘
                                      + L2 normalization

InputRouter
└─ loads only the encoder(s) required by the current input
```

주요 모델:

| Modality | Encoder |
|---|---|
| Query / Document | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Image | `openai/clip-vit-base-patch32` |
| Video | `MCG-NJU/videomae-base-finetuned-kinetics` |

`EncoderOutput`은 최종적으로 `pooled_embedding`과 `modality`만 반환합니다.

## Experimental results

### 1. Eager loading vs Router + Lazy Loading

| Input | Eager Memory | Lazy Memory | Memory reduction | Eager Time | Lazy Time | Time reduction |
|---|---:|---:|---:|---:|---:|---:|
| Text | 3533.6 MB | 2428.4 MB | 31.28% | 24.504 s | 7.804 s | 68.15% |
| Image | 3552.9 MB | 2042.0 MB | 42.53% | 21.975 s | 6.685 s | 69.58% |
| Video | 4006.2 MB | 997.4 MB | 75.10% | 24.643 s | 4.571 s | 81.45% |
| Mixed | 4020.1 MB | 3280.6 MB | 18.40% | 25.849 s | 20.268 s | 21.59% |

실험 결과 원본 요약은 `evaluation/results/efficiency_text_image_video.json`에 있습니다.

### 2. Text / Image / Video cross-modal retrieval

135개 test scene, 6개 검색 방향 평균:

| Metric | Score |
|---|---:|
| Recall@1 | 0.0210 |
| Recall@5 | 0.0753 |
| Recall@10 | 0.1284 |
| MRR | 0.0656 |
| Cosine margin | 0.0288 |

방향별 결과는 `evaluation/results/retrieval_text_image_video_135.json`에 있습니다.  
Image↔Video 방향이 상대적으로 높은 성능을 보였으며, 전체 결과를 과장하지 않고 최종 구조의 실제 검색 성능으로 보고합니다.

## Repository structure

```text
modular_encoder/
├─ encoders/
│  ├─ query/
│  ├─ document/
│  ├─ image/
│  └─ video/
├─ interfaces/
│  └─ encoder_output.py
├─ models/
│  └─ alignment_model.py
├─ router/
│  └─ input_router.py
├─ training/
│  └─ train_video_projection_overnight.py
├─ evaluation/
│  ├─ evaluate_efficiency_video.py
│  ├─ evaluate_joint_retrieval_135_video.py
│  └─ results/
├─ tests/
├─ legacy/
│  ├─ audio/
│  └─ training/
├─ requirements.txt
└─ .gitignore
```

`legacy/`는 최종 평가에 사용하는 코드가 아닙니다. 초기 Text/Image/Audio 개발 단계와 base checkpoint 생성 과정을 기록하기 위해 보존했습니다. **최종 Router와 최종 논문 실험은 Audio를 사용하지 않습니다.**

## Installation

프로젝트에서 실제로 사용한 로컬 환경과 새 설치 환경의 패키지 버전은 달라질 수 있으므로 `requirements.txt`는 패키지 목록 중심으로 제공합니다.

```powershell
pip install -r requirements.txt
```

대형 모델은 최초 실행 시 Hugging Face에서 다운로드됩니다.

## Checkpoints

체크포인트는 Git에 포함하지 않습니다. 아래 위치에 로컬로 배치합니다.

```text
checkpoints/
├─ resplit_135/
│  └─ joint_finetuned_best.pt
└─ video_added/
   └─ video_projection_best.pt
```

- `joint_finetuned_best.pt`: 기존 개발 단계에서 얻은 Text/Image encoder state를 제공합니다.
- `video_projection_best.pt`: 최종 Text/Image/Video projection state와 학습 완료 표시를 제공합니다.

최종 Router는 과거 체크포인트에 남아 있을 수 있는 `audio_projection.*` 키를 무시하고 **Text/Image/Video projection만 로드**합니다.

## Data layout

데이터 원본은 라이선스와 용량 문제로 저장소에 포함하지 않습니다. 평가 코드는 아래와 같은 장면 단위 구조를 가정합니다.

```text
data/
├─ test_manifest.jsonl
└─ processed/
   ├─ frames/<movie>/<scene_id>.jpg
   └─ clips/<movie>/<scene_id>.mp4
```

`test_manifest.jsonl`에는 최소한 아래 필드가 필요합니다.

```json
{"scene_id":"charade_00000","movie":"charade","text":"...","image_path":"data/processed/frames/charade/charade_00000.jpg","clip_path":"data/processed/clips/charade/charade_00000.mp4"}
```

## Run

### Router example

```python
from router.input_router import InputRouter

router = InputRouter()

outputs = router.route(
    query="A person is moving inside a room.",
    file_paths=[
        "data/processed/frames/charade/charade_00000.jpg",
        "data/processed/clips/charade/charade_00000.mp4",
    ],
)

for modality, items in outputs.items():
    print(modality, items[0].pooled_embedding.shape)
```

모든 최종 projection 출력은 `(1, 768)`이며 L2 normalization을 적용합니다.

### Efficiency evaluation

```powershell
python .\evaluation\evaluate_efficiency_video.py
```

### Retrieval smoke test

```powershell
python .\evaluation\evaluate_joint_retrieval_135_video.py --limit 3
```

### Full 135-scene retrieval evaluation

```powershell
python .\evaluation\evaluate_joint_retrieval_135_video.py
```

## Notes

`MCG-NJU/videomae-base-finetuned-kinetics`를 `VideoMAEModel`로 불러올 때 Transformers 버전에 따라 `UNEXPECTED` / `MISSING` weight warning이 출력될 수 있습니다. 최종 실험에서는 실제 영상 인코딩 출력 `(1, 768)`과 L2 norm, Video Projection 적용을 별도로 확인한 뒤 평가했습니다.

대용량 데이터, 영화 파일, 가상환경, 체크포인트는 `.gitignore`로 제외합니다.
