# KARINA Adaptive Multimodal RAG V4

`sungshin` 담당 범위인 **Question Understanding Decoder + Retrieval Gate + Adaptive RAG**를 구현한다.

## 역할 분담

```text
[역할 1] Modular Encoders
  Query / Document / Image / Video -> EncoderOutput
                                      |
                                      v
[역할 2] Question Understanding + Adaptive RAG   (이 디렉터리)
  EncoderOutput + input summaries + original query
  -> Question Condition
  -> Retrieval Gate
  -> modality-specific adaptive retrieval
  -> AdaptiveRAGOutput full packet
                                      |
                                      v
[역할 3] RAG Evidence Decoders + Final Decoder
  full packet -> evidence analysis -> final answer
```

## V3에서 변경한 핵심

1. **Question Condition과 Retrieval Gate를 분리했다.**
   - Question Understanding Decoder는 질문 조건 `C_Q`를 생성한다.
   - Retrieval Gate가 `C_Q`를 입력받아 검색 실행 여부와 검색 모달리티를 결정한다.
2. **역할 1의 `EncoderOutput`을 직접 받을 수 있다.**
   - `InputRouter.route()`의 `dict[str, list[EncoderOutput]]`를 그대로 전달한다.
   - query embedding은 원본 질문 문자열과 중복되므로 초기 멀티모달 컨텍스트에서 제외한다.
3. **RAG 생략 경로를 명시한다.**
   - 외부 지식이 필요하지 않으면 `retrieval_decision.action == "skip"`이고 검색 결과는 비어 있다.
   - 그래도 초기 입력 해석과 Question Condition은 full packet에 남는다.
4. **뒤쪽 디코더에는 full packet을 전달한다.**
   - `output.to_dict()`를 그대로 haneul의 `PacketAdapter`에 넘기는 것이 권장 경로다.
   - `build_decoder_inputs()`는 구형 호환용이며 답변 제약·복잡도 등의 일부 문맥이 손실될 수 있다.

## 중요한 인터페이스 제한

현재 역할 1의 `EncoderOutput`은 `pooled_embedding`과 `modality`만 제공한다. 이 768차원 벡터는
검색·정렬에는 사용할 수 있지만, 프롬프트 기반 Question Understanding LLM이 벡터만 보고 이미지나
영상의 구체적인 내용을 언어로 해석할 수는 없다.

따라서 실제 질문 이해에는 다음 중 하나가 추가로 필요하다.

- 권장: `modality_summaries={"image": "...", "video": "..."}` 전달
- 또는 역할 1/통합 계층이 원본 경로·캡션·OCR·자막을 metadata로 함께 전달
- 장기적으로는 encoder representation을 LLM token space에 연결하는 별도 adapter 학습

V4는 이 한계를 숨기지 않고 embedding은 **입력 존재 여부·차원·연결 검증 신호**, summary는
**의미 정보**로 구분한다.

## 주요 파일

```text
adaptive_rag/
├─ v4_contracts.py                 # Encoder/Question Condition/RAG 패킷 공통 계약
├─ adaptive_multimodal_rag_V4.py   # 역할 2 전체 파이프라인
├─ test_pipeline_V4.py             # skip/retrieve/full packet 결정적 테스트
└─ README_V4.md
```

## 실행

프로젝트 루트에서:

```bash
python -m adaptive_rag.test_pipeline_V4
```

## 역할 1 출력 연결 예시

```python
encoder_outputs = input_router.route(
    query=question,
    file_paths=["poster.jpg", "trailer.mp4"],
)

output = rag_pipeline.run(
    question=question,
    encoder_outputs=encoder_outputs,
    modality_summaries={
        "image": "저채도 청색 배경과 단독 인물 구도",
        "video": "후반부로 갈수록 컷 전환과 움직임이 빨라짐",
    },
)
```

## 역할 3 전달

```python
# 권장: 전체 패킷 전달
final_output = evidence_decoder_pipeline.run(output.to_dict())
```

full packet에는 다음이 포함된다.

- `initial_context`: 초기 인코더 연결 정보와 입력 요약
- `question_condition`: T/O/R/L/M_Q/K에 대응하는 구조화 조건
- `retrieval_decision`: skip/retrieve/retrieve_and_verify와 검색 모달리티
- `query_context`: haneul PacketAdapter 호환 필드
- `complexity`: 질문 복잡도와 모달리티별 검색 수요
- `retrieval_results`: 검색 근거, score, 불확실성, 최종 검색량
