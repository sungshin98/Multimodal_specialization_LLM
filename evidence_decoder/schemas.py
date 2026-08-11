"""다층 디코더 공통 스키마.

모달리티별 디코더는 서로 다른 원본(텍스트/이미지/영상)을 보지만
출력은 전부 EvidenceCard 하나로 통일한다. 근거 통합 계층이
모달리티를 몰라도 동작하게 만드는 것이 이 파일의 목적이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


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


# ============================================================
# 1. 입력측 - Adaptive RAG 패킷을 디코더 작업 단위로 옮긴 형태
# ============================================================


@dataclass
class EvidenceItem:
    """Adaptive RAG가 반환한 근거 1건."""

    evidence_id: str
    modality: Modality
    score: float
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def caption_hint(self) -> str:
        """비전 모델 없이도 쓸 수 있는 텍스트 단서를 metadata에서 모은다."""
        keys = ("caption", "description", "summary", "ocr", "ocr_text", "transcript", "alt_text")
        parts = [str(self.metadata[k]).strip() for k in keys if self.metadata.get(k)]
        return "\n".join(parts)


@dataclass
class ModalityTask:
    """모달리티별 디코더 1회 호출에 필요한 모든 것."""

    modality: Modality
    query: str
    focus_features: List[str] = field(default_factory=list)
    required_operations: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    evidence: List[EvidenceItem] = field(default_factory=list)
    uncertainty_level: Level = Level.MEDIUM
    retrieval_query: str = ""


@dataclass
class DecoderTask:
    """PacketAdapter가 만드는 다층 디코더 전체 입력.

    build_decoder_inputs()가 떨어뜨리는 answer_constraints / complexity /
    uncertainty 까지 포함한다. 최종 답변 디코더가 답변 제약을 봐야 하기 때문이다.
    """

    original_query: str
    normalized_query: str
    input_context: str = ""
    identified_entities: List[str] = field(default_factory=list)
    required_operations: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    answer_constraints: List[str] = field(default_factory=list)
    complexity_level: Level = Level.MEDIUM
    complexity_score: float = 0.0
    modality_tasks: List[ModalityTask] = field(default_factory=list)
    schema_version: str = ""

    def total_evidence_count(self) -> int:
        return sum(len(task.evidence) for task in self.modality_tasks)


# ============================================================
# 2. 1층 출력 - 모달리티별 근거 카드
# ============================================================


@dataclass
class EvidenceCard:
    """모달리티와 무관한 근거 최소 단위.

    claim  : 이 근거가 말하는 한 문장 (중복제거/충돌탐지의 비교 기준)
    detail : 최종 답변에 실제로 인용될 구체 서술
    """

    card_id: str
    source_evidence_id: str
    modality: Modality
    claim: str
    detail: str = ""
    supports: List[str] = field(default_factory=list)
    relevance: float = 0.0
    confidence: float = 0.0
    retrieval_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def priority(self) -> float:
        """통합 계층의 정렬 기준. 관련도를 확신도보다 무겁게 본다."""
        return 0.6 * self.relevance + 0.4 * self.confidence

    def char_cost(self) -> int:
        return len(self.claim) + len(self.detail)


@dataclass
class ModalityEvidenceResult:
    modality: Modality
    cards: List[EvidenceCard] = field(default_factory=list)
    modality_summary: str = ""
    is_sufficient: bool = True
    insufficient_reason: str = ""
    bypassed: bool = False
    failed: bool = False
    error: str = ""
    latency_ms: float = 0.0


# ============================================================
# 3. 2층 출력 - 통합된 근거
# ============================================================


@dataclass
class ConflictNote:
    card_ids: List[str]
    description: str
    resolution: str = ""


@dataclass
class IntegratedEvidence:
    cards: List[EvidenceCard] = field(default_factory=list)
    dropped_card_ids: List[str] = field(default_factory=list)
    duplicate_groups: List[List[str]] = field(default_factory=list)
    conflicts: List[ConflictNote] = field(default_factory=list)
    coverage: Dict[str, bool] = field(default_factory=dict)
    missing_operations: List[str] = field(default_factory=list)
    char_budget: int = 0
    char_used: int = 0
    bypassed: bool = False
    latency_ms: float = 0.0


# ============================================================
# 4. 최종 출력
# ============================================================


@dataclass
class FinalAnswer:
    answer: str
    citations: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0


@dataclass
class DecoderTrace:
    """실험용 계측값. 논문 실험표가 여기서 바로 나온다."""

    modality_latency_ms: Dict[str, float] = field(default_factory=dict)
    modality_stage_ms: float = 0.0
    integration_ms: float = 0.0
    final_ms: float = 0.0
    total_ms: float = 0.0
    llm_calls: int = 0
    cards_before_integration: int = 0
    cards_after_integration: int = 0
    bypassed_modality_stage: bool = False
    bypassed_integration: bool = False
    # 실행 중 백엔드가 폴백으로 내려갔는지. 비어 있지 않으면 그 실행의
    # 지연/품질 수치는 정상 경로의 값이 아니므로 실험에서 제외해야 한다.
    degraded_backends: List[str] = field(default_factory=list)
    # 모달리티 디코더가 호출 실패로 카드를 만들지 못한 경우. 폴백과 달리
    # 조용히 빈 결과가 되므로 별도로 기록하지 않으면 실패한 실행이 정상처럼
    # 집계된다(실측에서 40% 실패가 "오류 0"으로 보고된 적이 있다).
    failed_modalities: List[str] = field(default_factory=list)

    @property
    def is_valid_sample(self) -> bool:
        return not self.degraded_backends and not self.failed_modalities


@dataclass
class DecoderOutput:
    original_query: str
    final_answer: FinalAnswer
    integrated: IntegratedEvidence
    modality_results: List[ModalityEvidenceResult] = field(default_factory=list)
    trace: DecoderTrace = field(default_factory=DecoderTrace)
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return _enum_safe(asdict(self))


def _enum_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {_enum_safe(k): _enum_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_safe(v) for v in value]
    return value


def parse_modality(value: Any, default: Optional[Modality] = None) -> Optional[Modality]:
    if isinstance(value, Modality):
        return value
    try:
        return Modality(str(value).strip().lower())
    except (ValueError, AttributeError):
        return default


def parse_level(value: Any, default: Level = Level.MEDIUM) -> Level:
    if isinstance(value, Level):
        return value
    try:
        return Level(str(value).strip().lower())
    except (ValueError, AttributeError):
        return default
