# 2.2.1 질문 이해 및 초기 멀티모달 표현 생성 — 실제 구현 대응

KARINA의 입력 데이터는 `datasets.karina.movieqa.MovieQANormalizer`가 MovieQA의 질문·정답·story·clip alignment를 읽어 `karina.movie.v1` 포맷으로 변환한다. plot, subtitle, script는 `processed/movieqa/text/<imdb_key>/`의 UTF-8 text cache로 통합되며, 질문에 정렬된 video clip은 원본 경로를 유지한다. `datasets.karina.media.FrameCache`는 clip별 균등 sampled frame을 한 번만 생성하여 image modality로 사용한다. 따라서 동일 영상을 매 epoch마다 전부 decode하지 않는다.

학습 또는 평가 시 `datasets.karina.adapter.KARINADataset`은 JSONL byte offset만 메모리에 보존한다. `KARINAInputAdapter.to_router_payload`는 sample에서 실제 존재하는 text/image/video 경로와 content hint를 선택한다. 기존 `modular_encoder.router.input_router.InputRouter`는 질문을 QueryEncoder로, text cache를 DocumentEncoder로, sampled frame을 ImageEncoder로, clip을 VideoEncoder로 처리한다. 각 출력은 `EncoderOutput.pooled_embedding: [1,768]`로 정렬된다.

인코더 출력과 원본 경로는 `adaptive_rag.v4_contracts.InputBridge`에서 `InitialMultimodalContext`로 결합된다. 이어 `adaptive_rag.role2_pipeline.KARINARole2Pipeline`의 Question Understanding Decoder가 원 질문과 초기 입력의 `content_hint`, 사용 가능한 modality, source metadata를 분석하여 `QuestionCondition`을 만든다. 이 값은 Retrieval Gate가 검색 실행 여부, modality, 검색량을 정하는 입력으로 사용한다.

## Python 스타일 슈도코드

```python
def build_initial_representation(config, split):
    dataset = KARINADataset(config.processed_movieqa / f"{split}.jsonl")
    adapter = KARINAInputAdapter(config.data_root)
    router = InputRouter(device=config.device)

    for sample in dataset:                         # lazy JSONL read
        payload = adapter.to_router_payload(sample)
        # payload = question, file_paths, content_hints, source_metadata

        encoder_outputs = router.route(
            query=payload["question"],
            file_paths=payload["file_paths"],      # lazy file/media load
        )
        # each EncoderOutput.pooled_embedding: float32 [1, 768]

        initial_context = InputBridge.from_router_output(
            encoder_outputs=encoder_outputs,
            file_paths=payload["file_paths"],
            modality_summaries=payload["content_hints"],
            source_metadata=payload["source_metadata"],
        )

        question_condition = question_decoder.analyze(
            question=payload["question"],
            initial_context=initial_context,
        )

        yield {
            "sample_id": sample["sample_id"],
            "encoder_outputs": encoder_outputs,
            "initial_context": initial_context,
            "question_condition": question_condition,
            "answer": sample["answer"],
            "retrieval_target": sample["retrieval_target"],
        }
```

## 변수 표

| 변수명 | 의미 | 자료형/shape | 생성 단계 | KARINA 소비 모듈 |
|---|---|---|---|---|
| `sample` | 공통 포맷의 단일 QA sample | `dict`, schema `karina.movie.v1` | `MovieQANormalizer.iter_split` 또는 `MovieChatNormalizer.iter_samples` | `KARINAInputAdapter` |
| `Q` / `question` | 사용자 질문 | `str` | MovieQA/MovieChat annotation | QueryEncoder, Question Understanding Decoder |
| `X_t` | plot·subtitle·script text cache | `list[Asset]`, 각 path는 `.txt` | `MovieQANormalizer._text_assets` | DocumentEncoder, InputBridge |
| `X_i` | video에서 사전 추출한 sampled frame | `list[Asset]`, JPEG path | `FrameCache.get_or_create` | ImageEncoder, InputBridge |
| `X_v` | 질문 정렬 clip 또는 long video | `list[Asset]`, video path | MovieQA `video_clips`/MovieChat `info.video_path` 해석 | VideoEncoder, InputBridge |
| `F` / `file_paths` | 현재 sample에서 실제 활성화할 자산 경로 | `list[Path]`, ragged | `KARINAInputAdapter.to_router_payload` | `InputRouter.route` |
| `H` / `content_hints` | source별 의미 요약 | `dict[str,list[str]]` | normalizer + adapter | InputBridge, Question Understanding Decoder |
| `Z_q` | 질문 pooled embedding | `float32 [1,768]` | QueryEncoder + alignment projection | 연결 계측; Role-2는 원 질문도 함께 사용 |
| `Z_t` | text source pooled embedding | source별 `float32 [1,768]` | DocumentEncoder + text projection | InputBridge encoder signal |
| `Z_i` | frame pooled embedding | frame별 `float32 [1,768]` | ImageEncoder + image projection | InputBridge encoder signal |
| `Z_v` | clip pooled embedding | clip별 `float32 [1,768]` | VideoEncoder + video projection | InputBridge encoder signal |
| `M` | modality 존재 mask | `bool [B,3]` | `collate_karina_samples` | batch routing/실험 계측 |
| `Z_m^B` | modality별 padded embedding batch | `float32 [B,M_max,768]` | `collate_encoder_outputs` | batch형 후속 모듈 또는 분석 |
| `A_m^B` | padded item attention mask | `bool [B,M_max]` | `collate_encoder_outputs` | padding 무시 |
| `C_0` / `initial_context` | 경로·hint·embedding signal을 결합한 초기 문맥 | `InitialMultimodalContext` | `InputBridge.from_router_output` | Question Understanding Decoder |
| `C_Q` / `question_condition` | task, target, reasoning, required modality, external knowledge 필요성 | `QuestionCondition` dataclass | `QuestionUnderstandingDecoder.analyze` | Complexity Analyzer, Retrieval Gate |
| `E_target` | 정렬 plot/clip 및 선택적 MovieNet gold 근거 | `list[dict]` | normalizer/evidence pool 병합 | retrieval 평가 |
| `y` / `answer` | 정답 문자열 | `str`; MC index는 `int64 [B]` | official QA annotation | final decoder 평가/학습 target |

논문에서는 `X={X_t,X_i,X_v}`를 개념적 입력으로 사용하되, 구현상 각 원소는 고정 tensor가 아니라 lazy asset list이다. 고정 차원 표현은 인코더 이후 `Z_t`, `Z_i`, `Z_v`에서 형성되며, sample별 자산 수 차이는 `M_max` padding과 attention mask로 처리한다.

