"""다층 디코더 - 모달리티별 근거 디코더 / 근거 통합 계층 / 최종 답변 디코더.

adaptive_rag 가 반환한 표준 패킷을 받아 최종 답변까지 만든다.

    from evidence_decoder import build_pipeline
    pipeline = build_pipeline(asset_root="./data")
    output = pipeline.run(rag_output)      # AdaptiveRAGOutput 또는 dict
    print(output.final_answer.answer)
"""

from .assets import AssetLoader
from .clients import (
    CaptionFallbackVisionClient,
    GeminiVisionClient,
    LLMError,
    MediaAsset,
    OpenAIVisionClient,
    ResilientVisionClient,
    ScriptedClient,
    SolarStructuredClient,
    build_default_clients,
)
from .final_decoder import FinalAnswerDecoder
from .integration import EvidenceIntegrationLayer
from .modality import (
    ModalityEvidenceDecoder,
    TextEvidenceDecoder,
    VisionEvidenceDecoder,
    build_modality_decoders,
)
from .packet import PacketAdapter
from .pipeline import MultiLayerDecoderPipeline, PipelineConfig, build_pipeline
from .schemas import (
    ConflictNote,
    DecoderOutput,
    DecoderTask,
    DecoderTrace,
    EvidenceCard,
    EvidenceItem,
    FinalAnswer,
    IntegratedEvidence,
    Level,
    Modality,
    ModalityEvidenceResult,
    ModalityTask,
)

__all__ = [
    "AssetLoader",
    "CaptionFallbackVisionClient",
    "ConflictNote",
    "DecoderOutput",
    "DecoderTask",
    "DecoderTrace",
    "EvidenceCard",
    "EvidenceIntegrationLayer",
    "EvidenceItem",
    "FinalAnswer",
    "FinalAnswerDecoder",
    "GeminiVisionClient",
    "IntegratedEvidence",
    "LLMError",
    "Level",
    "MediaAsset",
    "Modality",
    "ModalityEvidenceDecoder",
    "ModalityEvidenceResult",
    "ModalityTask",
    "MultiLayerDecoderPipeline",
    "OpenAIVisionClient",
    "PacketAdapter",
    "PipelineConfig",
    "ResilientVisionClient",
    "ScriptedClient",
    "SolarStructuredClient",
    "TextEvidenceDecoder",
    "VisionEvidenceDecoder",
    "build_default_clients",
    "build_modality_decoders",
    "build_pipeline",
]

__version__ = "1.0.0"
