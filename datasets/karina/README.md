# KARINA movie data pipeline

MovieQA를 주 데이터로 정규화하고, 로컬에 합법적으로 준비된 MovieNet을 외부 근거 풀로, MovieChat-1K를 별도 일반화 평가 split으로 연결한다. 데이터·영상은 Git에 포함하지 않는다.

## 확인한 공식 배포 경로와 접근 조건

확인일은 2026-08-16이다.

| 데이터셋 | 공식 경로 | 이 구현의 다운로드 정책 | 라이선스/접근 주의사항 |
|---|---|---|---|
| MovieQA | [MovieQA benchmark GitHub](https://github.com/makarandtapaswi/MovieQA_benchmark), [benchmark site](http://movieqa.cs.toronto.edu/) | 공개된 `qa.json`, `movies.json`, `splits.json`과 로더는 공식 Git commit `c9367df...`로 자동 설치한다. story와 test 평가 접근은 자동화하지 않는다. | 공식 README가 story 접근과 test 평가를 위해 benchmark 등록을 요구한다. 저장소에는 독립적인 데이터 라이선스 파일이 보이지 않으므로 이용 권한을 임의로 확대 해석하지 않는다. |
| MovieNet | [MovieNet official site](https://movienet.github.io/), [official tools](https://github.com/movienet/movienet-tools) | 자동 다운로드하지 않고 로컬 배치 여부만 검사한다. | 공식 사이트는 영화를 제외한 데이터·annotation·feature를 OpenDataLab 계정, User Service Agreement, Privacy Policy 아래 제공한다고 안내한다. 원본 영화는 저작권 제약으로 공개 다운로드가 제공되지 않는다. 별도 데이터 라이선스가 명시되지 않은 경우 해당 약관을 따른다. |
| MovieChat-1K | [official MovieChat repository](https://github.com/wenhaochai/MovieChat), [official test repository](https://huggingface.co/datasets/Enxin/MovieChat-1K-test), [official train repository](https://huggingface.co/datasets/Enxin/MovieChat-1K_train) | 사용자가 Hugging Face에서 받은 JSON/video만 변환한다. 인증 토큰이나 약관 동의를 우회하지 않는다. | MovieChat 코드 저장소는 BSD-3-Clause지만 이것이 영화/TV clip의 데이터 라이선스를 뜻하지는 않는다. train은 gated access이고, test는 공개 저장소지만 dataset card에 별도 라이선스가 명시되지 않았다. |

MovieChat 공식 README는 test에 video당 global question 3개와 breakpoint question 10개가 있으며, train은 저작권 사유로 feature 배포에서 raw video 공개로 변경된 이력을 명시한다. 실제 접근 화면의 조건을 우선한다.

## 설치

데이터 모듈만 설치할 때:

```bash
pip install -r datasets/karina/requirements.txt
```

기존 encoder/decoder까지 실행하려면 각 모듈의 requirements도 설치하고 `modular_encoder` checkpoint를 준비해야 한다.

공개 MovieQA 메타데이터 설치:

```bash
python -m datasets.karina.cli --config configs/karina_data.yaml setup --dataset movieqa
python -m datasets.karina.cli --config configs/karina_data.yaml status
```

## 사용자가 배치해야 하는 데이터 구조

`movies.json`에 적힌 상대경로를 그대로 보존하는 것이 가장 안전하다.

```text
data/karina/
├── raw/
│   ├── movieqa/benchmark/
│   │   ├── data/{qa.json,movies.json,splits.json}
│   │   └── story/
│   │       ├── plot/<imdb_key>.wiki
│   │       ├── split_plot/<imdb_key>.split.wiki
│   │       ├── subtt/<imdb_key>.srt
│   │       ├── scripts-or-path-from-movies.json/...
│   │       └── video_clips/<imdb_key>/<official_clip_name>.mp4
│   ├── movienet/
│   │   ├── meta/
│   │   ├── subtitle/ 또는 subtitles/
│   │   ├── script/ 또는 scripts/
│   │   ├── synopsis/ 또는 synopses/
│   │   ├── keyframes/ 또는 images/ 또는 posters/
│   │   └── videos/ 또는 clips/ 또는 trailers/
│   └── moviechat_1k/
│       ├── test/gt/gt/*.json
│       └── test/raw_videos/*.mp4
├── cache/frames/<dataset>/<movie>/<video>/frame_*.jpg
└── processed/
    ├── movieqa/{train,val,test}.jsonl
    ├── movieqa_smoke/{train,val,test}.jsonl
    ├── moviechat_1k/generalization_test.jsonl
    ├── moviechat_1k_smoke/generalization_test.jsonl
    └── evidence/movienet.jsonl
```

MovieChat 파일을 Hugging Face CLI로 받는 경우 먼저 웹에서 해당 저장소의 조건을 확인·수락하고 로그인해야 한다. 이 저장소는 토큰을 코드나 config에 저장하지 않는다.

## 정규화와 검증

MovieQA만 준비:

```bash
python -m datasets.karina.cli --config configs/karina_data.yaml prepare --dataset movieqa
python -m datasets.karina.cli --config configs/karina_data.yaml validate
```

MovieNet과 MovieChat-1K를 활성화하려면 `configs/karina_data.yaml`의 각 `enabled`를 `true`로 바꾸고 실행한다.

```bash
python -m datasets.karina.cli --config configs/karina_data.yaml prepare --dataset all
```

GPU/RAM을 적게 쓰는 점검:

```bash
python -m datasets.karina.cli --config configs/karina_data.yaml prepare --dataset movieqa --limit 20 --no-frames
python -m datasets.karina.smoke_test
```

양수 `--limit`은 `processed/movieqa_smoke/`에 기록하므로 기존 전체 JSONL을 축소하지 않는다. `--limit 0` 또는 옵션 생략은 config의 정식 출력 경로에 전체 데이터를 기록한다. MovieChat-1K 제한 변환도 같은 이유로 `processed/moviechat_1k_smoke/`를 사용한다.

프레임은 video당 기본 4장을 균등 sampling하고 JPEG로 캐시한다. OpenCV가 없으면 `ffmpeg/ffprobe` CPU fallback을 사용하며, 재실행 시 기존 캐시를 사용한다. `runtime.batch_size=2`, `num_workers=0`, adapter의 video 상한 1개가 RTX 4070 Ti 12GB/RAM 32GB의 보수적 기본값이다. media는 Dataset worker에서 decode하지 않고 encoder 호출 시 lazy loading한다.

## 공통 포맷

각 JSONL 행은 다음 계약을 따른다.

```json
{
  "sample_id": "movieqa:train:45",
  "split": "train",
  "question": "...",
  "initial_input": {"text": [], "image": [], "video": []},
  "question_condition": null,
  "retrieval_target": [],
  "evidence": [],
  "answer": "...",
  "metadata": {},
  "schema_version": "karina.movie.v1"
}
```

각 modality 원소는 `{source_id, path, role, content_hint, metadata}`이다. `path`는 `data_root` 상대경로라서 다른 컴퓨터에서도 config만 바꾸면 된다. MovieQA answer는 정답 문자열이며 다섯 선택지와 `correct_index`는 metadata에 보존한다. test 정답이 공식 파일에 없으면 빈 문자열로 둔다.

## 기존 KARINA 인터페이스 연결

```python
from torch.utils.data import DataLoader
from datasets.karina.adapter import KARINADataset, KARINAInputAdapter, collate_karina_samples

dataset = KARINADataset("data/karina/processed/movieqa/train.jsonl")
loader = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_karina_samples)
adapter = KARINAInputAdapter("data/karina")

sample = dataset[0]
router_payload = adapter.to_router_payload(sample)
encoder_outputs = adapter.encode(input_router, sample)
rag_packet = adapter.run_role2(role2_pipeline, input_router, sample)
decoder_target = adapter.decoder_target(sample)
```

| 단계 | 필드/tensor | shape와 dtype | batching/padding |
|---|---|---|---|
| Dataset | `question` | `str` | batch에서는 `list[str]`, tokenizer는 QueryEncoder 내부에서 실행 |
| Dataset | `initial_input.*.path` | `str` 경로 | ragged list 유지; worker에서 video decode하지 않음 |
| collate | `modality_mask` | `bool [B,3]` | 순서 text/image/video |
| collate | `correct_index` | `int64 [B]` | 정답이 없으면 `-100` |
| InputRouter | query/document/image/video `pooled_embedding` | 각 `float32 [1,768]` | 샘플별 lazy encoding |
| `collate_encoder_outputs` | modality embeddings | `float32 [B,M_max,768]` | modality별 `M_max`까지 zero padding |
| `collate_encoder_outputs` | `attention_mask` | `bool [B,M_max]` | 실제 embedding만 true |
| InputBridge | `InitialMultimodalContext` | Python dataclass | 경로·hint·embedding_dim을 source별 보존 |
| Question Understanding | `QuestionCondition` | dataclass/JSON | `required_modalities`, task, reasoning, sub-query 등 ragged field |
| Retrieval Gate | `AdaptiveRAGOutput` | dataclass/JSON | evidence score/content/path를 modality별 list로 전달 |
| evidence decoder | Role-2 packet + `answer` target | packet + `str` | modality evidence는 길이가 달라 mask/list로 유지 |

`InputRouter`가 문서 확장자로 `.txt/.pdf/.docx`만 받으므로 MovieQA `.wiki/.srt`는 정규화 때 UTF-8 `.txt` cache로 변환한다. 기존 encoder 코드는 수정하지 않았다.

## 실제 구현과 미구현 경계

실제 구현됨:

- 공식 MovieQA metadata pinned download
- MovieQA split/QA/answer/plot alignment/story path/video path 정규화
- video 균등 sampled frame cache
- MovieNet text/image/video streaming evidence pool과 IMDb alias가 일치하는 MovieQA sample 병합
- MovieChat-1K global/breakpoint QA의 `generalization_test` 변환
- lazy JSONL Dataset, collate, InputRouter/Role-2 adapter
- schema validation, missing asset 검사, smoke test

데이터가 있어야 실행되는 부분:

- MovieQA story/subtitle/script/video 실제 변환
- MovieNet/OpenDataLab 자료의 전체 evidence pool 생성
- MovieChat-1K 17GB급 test video 또는 gated train 자료 변환
- 실제 encoder checkpoint를 사용한 768차원 embedding 통합 테스트

의도만 있고 구현되지 않은 기능은 dataset 자동 우회 다운로드, MovieNet 원본 영화 수집, MovieChat 약관 자동 동의, decoder 학습이다.
