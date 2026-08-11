# Adaptive Multimodal RAG V3

## 포함 범위

이 코드는 사용자의 담당 범위인 다음 흐름만 구현합니다.

```text
질문 + 초기 모달리티 요약
→ 질문 이해 디코더
→ 질문 복잡도 분석 모듈
→ 모달리티별 Adaptive RAG
→ 질문 조건 + RAG 결과 표준 패킷 반환
```

모달리티별 Evidence Decoder와 최종 답변 Decoder는 포함하지 않습니다. 대신 `build_decoder_inputs()`가 다층 디코더 담당자에게 전달할 입력을 생성합니다.

## 핵심 변경점

1. 질문 이해 디코더가 `normalized_query`, `required_operations`, `modality_focus`, `sub_queries`를 반환합니다.
2. 별도 복잡도 분석 모듈이 Low/Medium/High를 계산합니다.
3. 복잡도 신호로 검색 전 `candidate_k`, `final_k`, reranker 사용 여부를 정합니다.
4. ANN 검색 후 Top-1, Gap, Variance, Shannon Entropy로 검색량을 다시 보정합니다.
5. 최종 반환 패킷에 질문 조건과 모달리티별 근거를 함께 포함합니다.

## 실행

```bash
pip install numpy
python example_usage_V3.py
```

FAISS HNSW를 사용할 때:

```bash
pip install faiss-cpu sentence-transformers
```

## 실제 LLM 연결

`PromptQueryUnderstandingDecoder`에 `StructuredLLMClient` 인터페이스를 구현한 객체를 전달합니다.

```python
class MyLLMClient:
    def generate_json(self, system_prompt: str, user_prompt: str):
        # OpenAI API, 로컬 Transformers, vLLM 등으로 JSON 생성
        return parsed_json
```

```python
query_decoder = PromptQueryUnderstandingDecoder(MyLLMClient())
```

## 다층 디코더 담당자에게 전달할 값

`build_decoder_inputs(output)`의 모달리티별 결과는 다음 필드를 포함합니다.

- `original_query`
- `normalized_query`
- `focus_features`
- `required_operations`
- `constraints`
- `evidence`

모달리티별 디코더는 `evidence`만 분석하지 말고 나머지 질문 조건을 함께 입력받아야 합니다.
