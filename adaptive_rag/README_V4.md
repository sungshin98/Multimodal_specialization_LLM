# KARINA Adaptive Multimodal RAG V4

`sungshin` 담당 범위인 **Question Understanding Decoder + Retrieval Gate + Adaptive RAG**를 구현한다.

## 담당 파트별 전체 구조

```text
┌─────────────────────────────────────────────────────────────────────┐
│ [역할 1 · joyechan] Modular Multimodal Encoder                     │
│                                                                     │
│ Query / Document / Image / Video                                   │
│        -> InputRouter                                              │
│        -> modality-specific encoders                               │
│        -> projection + L2 normalization                            │
│        -> EncoderOutput(pooled_embedding, modality)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ EncoderOutput + original input path
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [역할 2 · sungshin] Question Understanding + Adaptive RAG          │
│                                                                     │
│ InputBridge                                                        │
│   -> RoutedInput(source_id, path, modality, EncoderOutput, hint)    │
│ Question Understanding Decoder                                     │
│   -> Question Condition                                            │
│ Retrieval Gate                                                     │
│   -> skip / retrieve / retrieve_and_verify                         │
│ Adaptive RAG                                                       │
│   -> modality selection / candidate_k / final_k / uncertainty      │
│   -> AdaptiveRAGOutput full packet                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ output.to_dict()
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [역할 3 · haneul] RAG Evidence Decoders + Final Decoder            │
│                                                                     │
│ PacketAdapter                                                      │
│   -> modality-specific evidence decoders                           │
│   -> evidence integration                                          │
│   -> final answer decoder                                          │
│   -> answer / citations / trace                                    │
└─────────────────────────────────────────────────────────────────────┘
```

## V3에서 변경한 핵심

1. **Question Condition과 Retrieval Gate를 분리했다.**
   - Question Understanding Decoder는 질문 조건을 생성한다.
   - Retrieval Gate가 질문 조건을 입력받아 검색 실행 여부와 검색 모달리티를 결정한다.
2. **역할 1의 `EncoderOutput`을 원본 입력과 함께 추적한다.**
   - `InputBridge`가 `EncoderOutput`, 원본 경로, `source_id`, 의미 보조 정보인 `content_hint`를 `RoutedInput`으로 결합한다.
   - 같은 모달리티 파일이 여러 개일 때도 파일별 출처를 유지할 수 있다.
   - query embedding은 원본 질문 문자열과 중복되므로 초기 멀티모달 입력 목록에서는 제외한다.
3. **RAG 생략 경로를 명시한다.**
   - 외부 지식이 필요하지 않으면 `retrieval_decision.action == "skip"`이고 검색 결과는 비어 있다.
   - 초기 입력 해석과 Question Condition은 full packet에 그대로 남는다.
4. **역할 3에는 full packet을 전달한다.**
   - `output.to_dict()` 전체를 haneul의 `PacketAdapter`에 넘기는 것이 권장 경로다.
   - `build_decoder_inputs()`는 구형 호환용이다.

## Encoder 출력과 의미 정보의 구분

역할 1의 `EncoderOutput`은 현재 다음 두 필드만 제공한다.

```python
EncoderOutput(
    pooled_embedding=...,
    modality=...,
)
```

공통 768차원 embedding은 입력 존재 여부, 벡터 연산, 연결 검증에는 사용할 수 있지만,
프롬프트 기반 Question Understanding LLM이 벡터만 보고 이미지나 영상의 구체적 내용을
언어로 복원할 수는 없다.

따라서 역할 2는 두 정보를 분리한다.

```text
EncoderOutput   -> modality / embedding dimension / vector signal
content_hint    -> 문서 내용, image caption, video summary 등 의미 정보
```

이 결합은 역할 1의 `EncoderOutput` 규격을 바꾸지 않고 역할 2의 `InputBridge`에서 수행한다.
`SourceAwarePromptQuestionUnderstandingDecoder`는 `RoutedInput`의 `content_hint`만을 의미 정보로 사용하고,
벡터 차원은 연결 검증 정보로만 취급한다.

## 주요 파일

```text
adaptive_rag/
├─ v4_contracts.py                 # RoutedInput, InputBridge, Question Condition, packet 계약
├─ adaptive_multimodal_rag_V4.py   # Question Understanding / Gate / Adaptive RAG 핵심
├─ role2_pipeline.py               # 역할 1 출력과 역할 2를 연결하는 권장 진입점
├─ test_pipeline_V4.py             # 기존 V4 결정적 테스트
├─ test_role2_pipeline.py          # Encoder -> 역할 2 -> 역할 3 packet 경계 테스트
└─ README_V4.md
```

## 권장 실행 예시

### 1. 예찬 `InputRouter.route()` 결과를 사용하는 경우

```python
from adaptive_rag.role2_pipeline import KARINARole2Pipeline

encoder_outputs = input_router.route(
    query=question,
    file_paths=["poster.jpg", "trailer.mp4"],
)

output = role2_pipeline.run(
    question,
    encoder_outputs=encoder_outputs,
    file_paths=["poster.jpg", "trailer.mp4"],
    content_hints={
        "image_0": "저채도 청색 배경과 단독 인물 구도",
        "video_0": "후반부로 갈수록 컷 전환과 움직임이 빨라짐",
    },
)
```

### 2. Router 호출까지 역할 2 진입점에 맡기는 경우

```python
output = role2_pipeline.run_from_router(
    router=input_router,
    question=question,
    file_paths=["poster.jpg", "trailer.mp4"],
    content_hints={
        "poster.jpg": "저채도 청색 배경과 단독 인물 구도",
        "trailer.mp4": "후반부로 갈수록 컷 전환과 움직임이 빨라짐",
    },
)
```

`run_from_router()`는 예찬의 기존 `route()` 계약을 그대로 사용하고, `InputBridge`가
모달리티별 출력 순서와 원본 파일 순서를 매핑하여 출처 관계를 유지한다.

## 역할 3 전달

```python
final_output = evidence_decoder_pipeline.run(output.to_dict())
```

full packet에는 다음이 포함된다.

- `initial_context`: `RoutedInput` 목록과 Encoder 연결 정보
- `question_condition`: T/O/R/L/M_Q/K에 대응하는 구조화 조건
- `retrieval_decision`: skip/retrieve/retrieve_and_verify 및 검색 모달리티
- `query_context`: haneul `PacketAdapter` 호환 필드
- `complexity`: 질문 복잡도와 모달리티별 검색 수요
- `retrieval_results`: 검색 근거, score, 불확실성, 최종 검색량

## 테스트

프로젝트 루트에서:

```bash
python -m adaptive_rag.test_pipeline_V4
python -m adaptive_rag.test_role2_pipeline
```
