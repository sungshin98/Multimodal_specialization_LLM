"""KARINA V4 공통 계약: 초기 인코더 입력, Question Condition, RAG 출력 스키마."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# 1. 공통 열거형 및 초기 입력 계약
# ============================================================


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    TABLE = "table"


class Level(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RetrievalAction(str, Enum):
    SKIP = "skip"
    RETRIEVE = "retrieve"
    RETRIEVE_AND_VERIFY = "retrieve_and_verify"


_ENCODER_MODALITY_MAP: Dict[str, Optional[Modality]] = {
    "query": None,  # 원본 질문 문자열을 별도로 사용하므로 query embedding은 컨텍스트에서 제외
    "document": Modality.TEXT,
    "text": Modality.TEXT,
    "image": Modality.IMAGE,
    "video": Modality.VIDEO,
    "audio": Modality.AUDIO,
    "table": Modality.TABLE,
}


@dataclass
class EncoderSignal:
    """앞단 모듈형 인코더의 출력을 질문/RAG 모듈에서 사용할 최소 계약으로 변환한 값.

    raw embedding은 패킷에 직렬화하지 않는다. 현재 프롬프트 기반 질문 이해기는
    summary를 의미 정보로 사용하고, embedding_dim/norm은 연결 검증과 계측에만 쓴다.
    """

    modality: Modality
    source_modality: str
    summary: str = ""
    embedding_dim: Optional[int] = None
    embedding_norm: Optional[float] = None
    source_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialMultimodalContext:
    signals: List[EncoderSignal] = field(default_factory=list)

    @classmethod
    def from_encoder_outputs(
        cls,
        encoder_outputs: Optional[Any] = None,
        modality_summaries: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> "InitialMultimodalContext":
        summaries = {str(k).lower(): v for k, v in (modality_summaries or {}).items()}
        metadata = {str(k).lower(): dict(v) for k, v in (metadata or {}).items()}

        flat_outputs: List[Any] = []
        if isinstance(encoder_outputs, Mapping):
            for key, values in encoder_outputs.items():
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    flat_outputs.extend(values)
                elif values is not None:
                    flat_outputs.append(values)
        elif isinstance(encoder_outputs, Sequence) and not isinstance(encoder_outputs, (str, bytes)):
            flat_outputs.extend(encoder_outputs)
        elif encoder_outputs is not None:
            flat_outputs.append(encoder_outputs)

        signals: List[EncoderSignal] = []
        seen_summary_keys: set[str] = set()

        for index, output in enumerate(flat_outputs):
            raw_modality = str(getattr(output, "modality", "") or "").strip().lower()
            modality = _ENCODER_MODALITY_MAP.get(raw_modality)
            if modality is None:
                continue

            summary_value = summaries.get(raw_modality, summaries.get(modality.value, ""))
            summary = _summary_text(summary_value)
            if summary:
                seen_summary_keys.add(raw_modality)
                seen_summary_keys.add(modality.value)

            dim, norm = _embedding_stats(getattr(output, "pooled_embedding", None))
            signals.append(
                EncoderSignal(
                    modality=modality,
                    source_modality=raw_modality or modality.value,
                    summary=summary,
                    embedding_dim=dim,
                    embedding_norm=norm,
                    source_id=f"encoder_{raw_modality or modality.value}_{index}",
                    metadata=dict(metadata.get(raw_modality, metadata.get(modality.value, {}))),
                )
            )

        # EncoderOutput 없이 요약만 전달된 경우도 지원한다.
        for key, value in summaries.items():
            if key in seen_summary_keys:
                continue
            modality = _ENCODER_MODALITY_MAP.get(key)
            if modality is None:
                continue
            signals.append(
                EncoderSignal(
                    modality=modality,
                    source_modality=key,
                    summary=_summary_text(value),
                    source_id=f"summary_{key}_{len(signals)}",
                    metadata=dict(metadata.get(key, {})),
                )
            )

        return cls(signals=signals)

    def available_modalities(self) -> List[Modality]:
        order = {modality: index for index, modality in enumerate(Modality)}
        return sorted({signal.modality for signal in self.signals}, key=lambda m: order[m])

    def summary_map(self) -> Dict[str, str]:
        buckets: Dict[str, List[str]] = {}
        for signal in self.signals:
            if signal.summary.strip():
                buckets.setdefault(signal.modality.value, []).append(signal.summary.strip())
        return {
            modality: " | ".join(dict.fromkeys(parts))
            for modality, parts in buckets.items()
        }

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "available_modalities": [m.value for m in self.available_modalities()],
            "modality_summaries": self.summary_map(),
            "encoder_signals": [
                {
                    "modality": signal.modality.value,
                    "source_modality": signal.source_modality,
                    "embedding_dim": signal.embedding_dim,
                    "summary_available": bool(signal.summary.strip()),
                }
                for signal in self.signals
            ],
        }

    def to_dict(self) -> Dict[str, Any]:
        return _enum_safe(asdict(self))


# ============================================================
# 2. 질문 조건 및 검색 출력 스키마
# ============================================================


@dataclass
class SubQuery:
    query: str
    modality: Modality
    priority: Level = Level.MEDIUM


@dataclass
class QuestionCondition:
    """논문의 C_Q에 대응하는 구조화된 질문 이해 결과.

    task/targets/reasoning/constraints/required_modalities/external_knowledge_required는
    각각 T/O/R/L/M_Q/K의 구현 필드다.
    """

    original_query: str
    normalized_query: str
    input_interpretation: str = ""
    task: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    required_modalities: List[Modality] = field(default_factory=list)
    external_knowledge_required: bool = False
    modality_focus: Dict[str, List[str]] = field(default_factory=dict)
    sub_queries: List[SubQuery] = field(default_factory=list)
    answer_constraints: List[str] = field(default_factory=list)

    def focus_for(self, modality: Modality) -> List[str]:
        return list(self.modality_focus.get(modality.value, []))


@dataclass
class ComplexityFeatures:
    query_length: int
    entity_count: int
    operation_count: int
    required_modality_count: int
    sub_query_count: int
    comparison_required: bool
    verification_required: bool
    temporal_reasoning: bool
    ambiguity_detected: bool
    external_knowledge_required: bool
    multimodal_reasoning_required: bool


@dataclass
class ComplexityResult:
    level: Level
    score: float
    retrieval_demand: Dict[str, Level]
    features: ComplexityFeatures
    reasons: List[str] = field(default_factory=list)


@dataclass
class RetrievalDecision:
    action: RetrievalAction
    modalities: List[Modality] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    unavailable_modalities: List[Modality] = field(default_factory=list)


@dataclass
class RetrievedEvidence:
    evidence_id: str
    modality: Modality
    score: float
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalUncertainty:
    top1_score: float
    top1_top2_gap: float
    score_variance: float
    shannon_entropy: float
    normalized_entropy: float
    level: Level


@dataclass
class ModalityRetrievalResult:
    modality: Modality
    query: str
    candidate_k: int
    final_k: int
    use_reranker: bool
    uncertainty: RetrievalUncertainty
    evidence: List[RetrievedEvidence]


@dataclass
class AdaptiveRAGOutput:
    """뒤쪽 RAG Evidence Decoder + Final Decoder에 전달하는 full packet."""

    schema_version: str
    original_query: str
    normalized_query: str
    initial_context: Dict[str, Any]
    question_condition: QuestionCondition
    query_context: Dict[str, Any]
    retrieval_decision: RetrievalDecision
    complexity: ComplexityResult
    retrieval_results: Dict[str, ModalityRetrievalResult]

    def to_dict(self) -> Dict[str, Any]:
        return _enum_safe(asdict(self))


# ============================================================


def _embedding_stats(value: Any) -> Tuple[Optional[int], Optional[float]]:
    if value is None:
        return None, None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            array = np.asarray(value.numpy(), dtype=np.float32)
        else:
            array = np.asarray(value, dtype=np.float32)
        if array.size == 0:
            return None, None
        return int(array.shape[-1]), round(float(np.linalg.norm(array.reshape(-1))), 6)
    except (TypeError, ValueError, AttributeError):
        return None, None


def _summary_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " | ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _parse_modality(value: Any) -> Optional[Modality]:
    key = str(value or "").strip().lower()
    mapped = _ENCODER_MODALITY_MAP.get(key)
    if mapped is not None:
        return mapped
    try:
        return Modality(key)
    except ValueError:
        return None


def _enum_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_enum_safe(key)): _enum_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_safe(item) for item in value]
    return value
