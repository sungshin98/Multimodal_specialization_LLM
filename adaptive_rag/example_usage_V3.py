"""학습 없이 전체 연결을 확인하는 최소 실행 예제."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from adaptive_multimodal_rag_V3 import (
    AdaptiveMultimodalRAGPipeline,
    CallableVectorEncoder,
    InMemoryRetriever,
    Modality,
    RuleBasedComplexityAnalyzer,
    RuleBasedQueryUnderstandingDecoder,
    build_decoder_inputs,
)


def deterministic_embedding(text: str, dim: int = 64) -> np.ndarray:
    """예제용 결정적 임베딩. 실제 실험에서는 SentenceTransformer/CLIP 등으로 교체."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)


encoder = CallableVectorEncoder(deterministic_embedding)

text_docs = [
    "감독 A의 이전 작품은 느린 전개와 인물 중심 서사가 특징이다.",
    "현재 작품은 빠른 편집과 장르적 긴장감을 강조한다.",
    "영화 포스터는 어두운 색조와 단독 인물 구도를 사용한다.",
]
image_docs = [
    "poster_001.jpg",
    "poster_002.jpg",
    "poster_003.jpg",
]

text_embeddings = np.vstack([deterministic_embedding(doc) for doc in text_docs])
image_embeddings = np.vstack([deterministic_embedding(doc) for doc in image_docs])

retrievers = {
    Modality.TEXT: InMemoryRetriever(
        modality=Modality.TEXT,
        encoder=encoder,
        documents=text_docs,
        embeddings=text_embeddings,
    ),
    Modality.IMAGE: InMemoryRetriever(
        modality=Modality.IMAGE,
        encoder=encoder,
        documents=image_docs,
        embeddings=image_embeddings,
    ),
}

pipeline = AdaptiveMultimodalRAGPipeline(
    query_decoder=RuleBasedQueryUnderstandingDecoder(),
    complexity_analyzer=RuleBasedComplexityAnalyzer(),
    retrievers=retrievers,
)

output = pipeline.run(
    question="입력 포스터를 바탕으로 현재 영화와 감독의 이전 작품의 시각적 스타일 차이를 비교해줘.",
    modality_summaries={
        "image": "어두운 색조, 단독 인물 구도, 강한 명암 대비가 나타나는 영화 포스터"
    },
)

print("=== Adaptive RAG Output ===")
print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2))

print("\n=== Inputs for Modality Decoders ===")
print(json.dumps(build_decoder_inputs(output), ensure_ascii=False, indent=2))
