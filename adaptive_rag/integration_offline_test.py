"""역할 1(가짜 EncoderOutput) -> 역할 2 -> 역할 3 최종 답변까지의 오프라인 통합 테스트.

주의: evidence_decoder 디렉터리가 함께 존재하는 통합 작업공간에서 실행해야 한다.
실행:
    python -m adaptive_rag.integration_offline_test
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from adaptive_rag.adaptive_multimodal_rag_V4 import (
    CallableVectorEncoder,
    InMemoryRetriever,
    Modality,
    RuleBasedComplexityAnalyzer,
    RuleBasedQuestionUnderstandingDecoder,
)
from adaptive_rag.role2_pipeline import KARINARole2Pipeline
from evidence_decoder.clients import ScriptedClient
from evidence_decoder.pipeline import PipelineConfig, build_pipeline


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

role2 = KARINARole2Pipeline(
    query_decoder=RuleBasedQuestionUnderstandingDecoder(),
    complexity_analyzer=RuleBasedComplexityAnalyzer(),
    retrievers={
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
    },
)

encoder_outputs = {
    "query": [FakeEncoderOutput(np.ones((1, 768), dtype=np.float32), "query")],
    "image": [FakeEncoderOutput(np.ones((1, 768), dtype=np.float32), "image")],
}

TEXT_RESPONSE = {
    "evidence_verdicts": [
        {"evidence_id": "text_0", "same_subject": True, "on_focus": True, "reason": "관련 근거"},
        {"evidence_id": "text_1", "same_subject": True, "on_focus": True, "reason": "관련 근거"},
    ],
    "cards": [
        {
            "source_evidence_id": "text_0",
            "claim": "감독 A의 이전 작품은 느린 전개와 인물 중심 구도가 특징이다.",
            "detail": "이전 작품은 느린 호흡과 인물 중심 연출을 사용한다.",
            "supports": ["compare", "verify"],
            "relevance": 0.95,
            "confidence": 0.90,
        },
        {
            "source_evidence_id": "text_1",
            "claim": "현재 영화는 빠른 편집과 장르적 긴장감을 강조한다.",
            "detail": "현재 영화는 편집 속도와 긴장감을 강화한다.",
            "supports": ["compare", "verify"],
            "relevance": 0.90,
            "confidence": 0.85,
        },
    ],
    "modality_summary": "텍스트 근거는 이전 작품과 현재 작품의 편집·구도 차이를 보여준다.",
    "is_sufficient": True,
    "insufficient_reason": "",
}

IMAGE_RESPONSE = {
    "evidence_verdicts": [
        {"evidence_id": "image_0", "same_subject": True, "on_focus": True, "reason": "관련 포스터"},
        {"evidence_id": "image_1", "same_subject": True, "on_focus": True, "reason": "관련 포스터"},
    ],
    "cards": [
        {
            "source_evidence_id": "image_0",
            "claim": "현재 포스터는 저채도 청색 배경과 단독 인물 구도를 사용한다.",
            "detail": "차가운 색조와 고립된 구도가 무거운 분위기를 만든다.",
            "supports": ["compare", "verify"],
            "relevance": 0.92,
            "confidence": 0.70,
        }
    ],
    "modality_summary": "이미지 근거는 차갑고 고립된 시각적 분위기를 보여준다.",
    "is_sufficient": True,
    "insufficient_reason": "",
}

INTEGRATION_RESPONSE = {
    "kept_card_ids": ["text_card_0", "text_card_1", "image_card_0"],
    "dropped_card_ids": [],
    "pair_verdicts": [],
    "conflicts": [],
    "coverage": [
        {"operation": "compare", "covered": True},
        {"operation": "verify", "covered": True},
    ],
}

ANSWER_RESPONSE = {
    "answer": "현재 작품은 이전 작품보다 편집이 빠르고 긴장감이 강하며, 포스터는 차갑고 고립된 분위기를 강조합니다.",
    "citations": ["text_card_0", "text_card_1", "image_card_0"],
    "unsupported_claims": [],
    "confidence": 0.88,
}


def main() -> None:
    rag_output = role2.run(
        "이 포스터와 감독의 이전 작품 스타일을 비교하고 출처로 검증해줘.",
        encoder_outputs=encoder_outputs,
        file_paths=["data/poster_input.jpg"],
        content_hints={
            "image_0": "저채도 청색 배경과 단독 인물 구도가 나타나는 영화 포스터"
        },
    )

    text_client = ScriptedClient(
        responses=[TEXT_RESPONSE, INTEGRATION_RESPONSE, ANSWER_RESPONSE]
    )
    vision_client = ScriptedClient(responses=[IMAGE_RESPONSE])

    role3 = build_pipeline(
        asset_root=None,
        config=PipelineConfig(enable_modality_bypass=False),
        text_client=text_client,
        vision_client=vision_client,
    )
    output = role3.run(rag_output.to_dict())

    assert output.final_answer.answer
    assert len(output.final_answer.citations) == 3
    assert output.trace.cards_before_integration == 3
    assert output.trace.cards_after_integration == 3
    print("PASS")
    print(output.final_answer.answer)


if __name__ == "__main__":
    main()
