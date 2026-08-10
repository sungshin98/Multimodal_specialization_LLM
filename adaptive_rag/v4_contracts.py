"""KARINA V4 공통 계약: 입력 추적, Question Condition, Adaptive RAG 출력."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 1. 공통 열거형
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
    "query": None,
    "document": Modality.TEXT,
    "text": Modality.TEXT,
    "image": Modality.IMAGE,
    "video": Modality.VIDEO,
    "audio": Modality.AUDIO,
    "table": Modality.TABLE,
}


# ============================================================
# 2. 역할 1 Encoder -> 역할 2 입력 계약
# ============================================================


@dataclass
class EncoderSignal:
    """역할 1의 EncoderOutput을 역할 2에서 사용하는 최소 신호로 정규화한 값."""

    modality: Modality
    source_modality: str
    embedding_dim: Optional[int] = None
    embedding_norm: Optional[float] = None


@dataclass
class RoutedInput:
    """원본 입력, 출처 정보, 인코더 출력을 하나로 묶는 연결 단위.

    EncoderOutput 자체는 변경하지 않는다. source_path/content_hint와의 결합은
    역할 2의 InputBridge가 담당한다.
    """

    source_id: str
    modality: Modality
    source_modality: str
    source_path: str = ""
    content_hint: str = ""
    encoder_signal: Optional[EncoderSignal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "modality": self.modality.value,
            "source_path": self.source_path,
            "content_hint": self.content_hint,
            "embedding_dim": (
                self.encoder_signal.embedding_dim if self.encoder_signal else None
            ),
        }


@dataclass
class InitialMultimodalContext:
    routed_inputs: List[RoutedInput] = field(default_factory=list)

    def available_modalities(self) -> List[Modality]:
        order = {modality: index for index, modality in enumerate(Modality)}
        return sorted(
            {item.modality for item in self.routed_inputs},
            key=lambda modality: order[modality],
        )

    def summary_map(self) -> Dict[str, str]:
        buckets: Dict[str, List[str]] = {}
        for item in self.routed_inputs:
            hint = item.content_hint.strip()
            if hint:
                buckets.setdefault(item.modality.value, []).append(hint)
        return {
            modality: " | ".join(dict.fromkeys(values))
            for modality, values in buckets.items()
        }

    def prompt_payload(self) -> Dict[str, Any]:
        return {
            "available_modalities": [
                modality.value for modality in self.available_modalities()
            ],
            "initial_inputs": [item.to_prompt_dict() for item in self.routed_inputs],
        }

    def to_dict(self) -> Dict[str, Any]:
        return _enum_safe(asdict(self))


class InputBridge:
    """예찬 파트의 InputRouter 결과와 원본 파일을 역할 2 입력으로 결합한다.

    지원 입력
    - encoder_outputs: InputRouter.route()의 dict[str, list[EncoderOutput]]
    - file_paths: route()에 넘겼던 원본 파일 순서
    - modality_summaries: 모달리티 단위 또는 source_id 단위 의미 요약

    같은 모달리티 파일이 여러 개인 경우 출처 추적을 보장하려면
    ``from_router_calls``를 사용하는 것이 권장된다.
    """

    @classmethod
    def from_router_output(
        cls,
        encoder_outputs: Optional[Any] = None,
        file_paths: Optional[Sequence[str | Path]] = None,
        modality_summaries: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> InitialMultimodalContext:
        summaries = {
            str(key).strip().lower(): value
            for key, value in (modality_summaries or {}).items()
        }
        source_metadata = {
            str(key).strip().lower(): dict(value)
            for key, value in (source_metadata or {}).items()
        }
        paths_by_modality = cls._paths_by_modality(file_paths or ())
        counters: Dict[str, int] = {}
        routed: List[RoutedInput] = []

        for output in cls._flatten_outputs(encoder_outputs):
            source_modality = str(getattr(output, "modality", "") or "").lower()
            modality = _parse_encoder_modality(source_modality)
            if modality is None:
                # query embedding은 원본 질문 문자열로 대체하므로 입력 컨텍스트에서 제외한다.
                continue

            index = counters.get(source_modality, 0)
            counters[source_modality] = index + 1
            source_id = f"{source_modality}_{index}"

            path_pool = paths_by_modality.get(source_modality) or paths_by_modality.get(
                modality.value, []
            )
            source_path = str(path_pool[index]) if index < len(path_pool) else ""
            hint = cls._resolve_hint(
                summaries=summaries,
                source_id=source_id,
                source_modality=source_modality,
                modality=modality,
                index=index,
            )
            dim, norm = _embedding_stats(getattr(output, "pooled_embedding", None))
            routed.append(
                RoutedInput(
                    source_id=source_id,
                    modality=modality,
                    source_modality=source_modality,
                    source_path=source_path,
                    content_hint=hint,
                    encoder_signal=EncoderSignal(
                        modality=modality,
                        source_modality=source_modality,
                        embedding_dim=dim,
                        embedding_norm=norm,
                    ),
                    metadata=dict(
                        source_metadata.get(
                            source_id,
                            source_metadata.get(source_modality, source_metadata.get(modality.value, {})),
                        )
                    ),
                )
            )

        # EncoderOutput 없이 요약만 전달되는 통합 테스트도 허용한다.
        if not routed:
            for key, value in summaries.items():
                modality = _parse_encoder_modality(key)
                if modality is None:
                    continue
                source_id = f"{key}_0"
                routed.append(
                    RoutedInput(
                        source_id=source_id,
                        modality=modality,
                        source_modality=key,
                        content_hint=_summary_text(value),
                        metadata=dict(source_metadata.get(key, {})),
                    )
                )

        return InitialMultimodalContext(routed_inputs=routed)

    @classmethod
    def from_router_calls(
        cls,
        router: Any,
        query: Optional[str] = None,
        file_paths: Optional[Sequence[str | Path]] = None,
        content_hints: Optional[Mapping[str, Any]] = None,
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Tuple[InitialMultimodalContext, Optional[Any]]:
        """파일별 route_file 호출로 출처와 EncoderOutput의 1:1 관계를 보존한다.

        반환값 두 번째 항목은 QueryEncoder 출력이다. 역할 2는 원본 질문 문자열을
        사용하므로 기본 파이프라인에서는 소비하지 않지만 통합 계측용으로 보존한다.
        """

        content_hints = {
            str(key).strip().lower(): value for key, value in (content_hints or {}).items()
        }
        source_metadata = {
            str(key).strip().lower(): dict(value)
            for key, value in (source_metadata or {}).items()
        }
        query_output = router.route_query(query) if query else None
        routed: List[RoutedInput] = []

        for index, path_value in enumerate(file_paths or ()): 
            path = Path(path_value)
            output = router.route_file(path)
            source_modality = str(getattr(output, "modality", "") or "").lower()
            modality = _parse_encoder_modality(source_modality)
            if modality is None:
                continue
            source_id = f"{source_modality}_{index}"
            hint = cls._resolve_hint(
                summaries=content_hints,
                source_id=source_id,
                source_modality=source_modality,
                modality=modality,
                index=index,
                source_path=str(path),
            )
            dim, norm = _embedding_stats(getattr(output, "pooled_embedding", None))
            routed.append(
                RoutedInput(
                    source_id=source_id,
                    modality=modality,
                    source_modality=source_modality,
                    source_path=str(path),
                    content_hint=hint,
                    encoder_signal=EncoderSignal(
                        modality=modality,
                        source_modality=source_modality,
                        embedding_dim=dim,
                        embedding_norm=norm,
                    ),
                    metadata=dict(source_metadata.get(source_id, {})),
                )
            )

        return InitialMultimodalContext(routed_inputs=routed), query_output

    @staticmethod
    def _flatten_outputs(encoder_outputs: Optional[Any]) -> List[Any]:
        if encoder_outputs is None:
            return []
        if isinstance(encoder_outputs, Mapping):
            flat: List[Any] = []
            for values in encoder_outputs.values():
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    flat.extend(values)
                elif values is not None:
                    flat.append(values)
            return flat
        if isinstance(encoder_outputs, Sequence) and not isinstance(
            encoder_outputs, (str, bytes)
        ):
            return list(encoder_outputs)
        return [encoder_outputs]

    @staticmethod
    def _paths_by_modality(
        file_paths: Sequence[str | Path],
    ) -> Dict[str, List[Path]]:
        grouped: Dict[str, List[Path]] = {}
        for value in file_paths:
            path = Path(value)
            modality = _modality_from_path(path)
            if modality is None:
                continue
            key = "document" if modality == Modality.TEXT else modality.value
            grouped.setdefault(key, []).append(path)
            grouped.setdefault(modality.value, []).append(path)
        return grouped

    @staticmethod
    def _resolve_hint(
        summaries: Mapping[str, Any],
        source_id: str,
        source_modality: str,
        modality: Modality,
        index: int,
        source_path: str = "",
    ) -> str:
        candidates = (
            source_id,
            source_path.lower(),
            Path(source_path).name.lower() if source_path else "",
            f"{source_modality}_{index}",
            source_modality,
            f"{modality.value}_{index}",
            modality.value,
        )
        for candidate in candidates:
            if candidate and candidate in summaries:
                return _summary_text(summaries[candidate])
        return ""


# ============================================================
# 3. 질문 조건 및 검색 출력 스키마
# ============================================================


@dataclass
class SubQuery:
    query: str
    modality: Modality
    priority: Level = Level.MEDIUM


@dataclass
class QuestionCondition:
    """논문의 C_Q={T,O,R,L,M_Q,K}에 대응하는 구조화 출력."""

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
    """역할 3의 RAG Evidence Decoder + Final Decoder로 전달하는 full packet."""

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
# 4. 유틸리티
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


def _parse_encoder_modality(value: Any) -> Optional[Modality]:
    key = str(value or "").strip().lower()
    if key in _ENCODER_MODALITY_MAP:
        return _ENCODER_MODALITY_MAP[key]
    try:
        return Modality(key)
    except ValueError:
        return None


def _modality_from_path(path: Path) -> Optional[Modality]:
    extension = path.suffix.lower()
    if extension in {".txt", ".pdf", ".docx"}:
        return Modality.TEXT
    if extension in {".jpg", ".jpeg", ".png"}:
        return Modality.IMAGE
    if extension in {".mp4", ".webm", ".mkv", ".avi", ".mov", ".ogv"}:
        return Modality.VIDEO
    return None


def _enum_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(_enum_safe(key)): _enum_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_safe(item) for item in value]
    return value
