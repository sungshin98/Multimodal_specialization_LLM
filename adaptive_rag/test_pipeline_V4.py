"""KARINA V4 역할 2의 결정적 통합 테스트.

실행:
    python -m adaptive_rag.test_pipeline_V4
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from adaptive_rag.adaptive_multimodal_rag_V4 import (
    AdaptiveMultimodalRAGPipelineV4,
    CallableVectorEncoder,
    InMemoryRetriever,
    Modality,
    RetrievalAction,
    RuleBasedComplexityAnalyzer,
    RuleBasedQuestionUnderstandingDecoder,
    build_decoder_inputs,
)


@dataclass
class FakeEncoderOutput:
    pooled_embedding: np.ndarray
    modality: str


def deterministic_embedding(text: str, dim: int = 64) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed).normal(size=dim).astype(np.float32)


encoder = CallableVectorEncoder(deterministic_embedding)
text_docs = [
    "감독 A의 이전 작품은 느린 전개와 인물 중심 구도가 특징이다.",
    "현재 영화는 빠른 편집과 강한 장르적 긴장감을 강조한다.",
]
image_docs = ["poster_001.jpg", "poster_002.jpg"]

retrievers = {
    Modality.TEXT: InMemoryRetriever(
        Modality.TEXT,
        encoder,
        text_docs,
        np.vstack([deterministic_embedding(item) for item in text_docs]),
    ),
    Modality.IMAGE: InMemoryRetriever(
        Modality.IMAGE,
        encoder,
        image_docs,
        np.vstack([deterministic_embedding(item) for item in image_docs]),
    ),
}

pipeline = AdaptiveMultimodalRAGPipelineV4(
    query_decoder=RuleBasedQuestionUnderstandingDecoder(),
    complexity_analyzer=RuleBasedComplexityAnalyzer(),
    retrievers=retrievers,
)

# joyechan의 InputRouter.route() 반환 형식과 동일한 dict[str, list[EncoderOutput]] 모사
encoder_outputs = {
    "query": [FakeEncoderOutput(np.ones((1, 768), dtype=np.float32), "query")],
    "image": [FakeEncoderOutput(np.ones((1, 768), dtype=np.float32), "image")],
}


def test_skip_path() -> None:
    output = pipeline.run(
        "이 포스터의 색조와 구도를 설명해줘.",
        encoder_outputs=encoder_outputs,
        modality_summaries={"image": "저채도 청색 배경과 단독 인물 구도"},
    )
    assert output.retrieval_decision.action == RetrievalAction.SKIP
    assert output.retrieval_results == {}
    assert output.initial_context["signals"][0]["modality"] == "image"
    assert "저채도" in output.query_context["input_context"]


def test_retrieval_packet() -> None:
    output = pipeline.run(
        "이 포스터와 감독의 이전 작품 스타일을 비교하고 출처로 검증해줘.",
        encoder_outputs=encoder_outputs,
        modality_summaries={"image": "저채도 청색 배경과 단독 인물 구도"},
    )
    assert output.retrieval_decision.action == RetrievalAction.RETRIEVE_AND_VERIFY
    assert "text" in output.retrieval_results
    assert "image" in output.retrieval_results
    assert output.query_context["required_operations"] == ["compare", "verify"]

    # haneul PacketAdapter가 권장 경로에서 읽는 full packet 필드
    packet = output.to_dict()
    for key in (
        "original_query",
        "normalized_query",
        "query_context",
        "complexity",
        "retrieval_results",
    ):
        assert key in packet

    # 구형 build_decoder_inputs 경로도 최소 호환을 유지한다.
    legacy = build_decoder_inputs(output)
    assert "answer_constraints" in legacy["text"]
    assert "input_context" in legacy["image"]


if __name__ == "__main__":
    test_skip_path()
    test_retrieval_packet()
    print("PASS")
