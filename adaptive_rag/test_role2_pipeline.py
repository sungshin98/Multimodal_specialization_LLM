"""Encoder -> 역할 2 -> 역할 3 패킷 경계를 검증하는 결정적 테스트.

실행:
    python -m adaptive_rag.test_role2_pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .adaptive_multimodal_rag_V4 import (
    CallableVectorEncoder,
    InMemoryRetriever,
    Modality,
    RetrievalAction,
    RuleBasedComplexityAnalyzer,
    RuleBasedQuestionUnderstandingDecoder,
)
from .role2_pipeline import KARINARole2Pipeline


@dataclass
class FakeEncoderOutput:
    pooled_embedding: np.ndarray
    modality: str


def deterministic_embedding(text: str, dim: int = 64) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed).normal(size=dim).astype(np.float32)


retrieval_encoder = CallableVectorEncoder(deterministic_embedding)
text_docs = [
    "감독 A의 이전 작품은 느린 전개와 인물 중심 구도가 특징이다.",
    "현재 영화는 빠른 편집과 강한 장르적 긴장감을 강조한다.",
]
image_docs = ["poster_reference_001.jpg", "poster_reference_002.jpg"]

retrievers = {
    Modality.TEXT: InMemoryRetriever(
        Modality.TEXT,
        retrieval_encoder,
        text_docs,
        np.vstack([deterministic_embedding(item) for item in text_docs]),
    ),
    Modality.IMAGE: InMemoryRetriever(
        Modality.IMAGE,
        retrieval_encoder,
        image_docs,
        np.vstack([deterministic_embedding(item) for item in image_docs]),
    ),
}

pipeline = KARINARole2Pipeline(
    query_decoder=RuleBasedQuestionUnderstandingDecoder(),
    complexity_analyzer=RuleBasedComplexityAnalyzer(),
    retrievers=retrievers,
)

# joyechan InputRouter.route()의 반환 형식을 모사한다.
encoder_outputs = {
    "query": [FakeEncoderOutput(np.ones((1, 768), dtype=np.float32), "query")],
    "image": [FakeEncoderOutput(np.ones((1, 768), dtype=np.float32), "image")],
}
input_paths = ["data/poster_input.jpg"]
content_hints = {
    "image_0": "저채도 청색 배경과 단독 인물 구도가 나타나는 영화 포스터",
}


def test_input_bridge_and_skip() -> None:
    output = pipeline.run(
        "이 포스터의 색조와 구도를 설명해줘.",
        encoder_outputs=encoder_outputs,
        file_paths=input_paths,
        content_hints=content_hints,
    )

    assert output.retrieval_decision.action == RetrievalAction.SKIP
    assert output.retrieval_results == {}

    routed = output.initial_context["routed_inputs"]
    assert len(routed) == 1  # query embedding은 원본 질문과 중복되어 제외
    assert routed[0]["source_id"] == "image_0"
    assert routed[0]["source_path"].endswith("poster_input.jpg")
    assert routed[0]["encoder_signal"]["embedding_dim"] == 768
    assert "저채도" in routed[0]["content_hint"]
    assert "저채도" in output.query_context["input_context"]


def test_retrieval_and_role3_packet() -> None:
    output = pipeline.run(
        "이 포스터와 감독의 이전 작품 스타일을 비교하고 출처로 검증해줘.",
        encoder_outputs=encoder_outputs,
        file_paths=input_paths,
        content_hints=content_hints,
    )

    assert output.retrieval_decision.action == RetrievalAction.RETRIEVE_AND_VERIFY
    assert "text" in output.retrieval_results
    assert "image" in output.retrieval_results
    assert output.query_context["required_operations"] == ["compare", "verify"]

    # haneul PacketAdapter에 전달할 권장 full packet 계약
    packet = output.to_dict()
    for key in (
        "original_query",
        "normalized_query",
        "initial_context",
        "question_condition",
        "query_context",
        "retrieval_decision",
        "complexity",
        "retrieval_results",
    ):
        assert key in packet

    assert packet["query_context"]["modality_focus"]
    assert packet["retrieval_results"]["text"]["evidence"]


if __name__ == "__main__":
    test_input_bridge_and_skip()
    test_retrieval_and_role3_packet()
    print("PASS")
