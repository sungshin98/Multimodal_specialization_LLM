from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


# ============================================================
# 1. 공통 스키마
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


@dataclass
class SubQuery:
    query: str
    modality: Modality
    priority: Level = Level.MEDIUM


@dataclass
class QueryUnderstandingResult:
    """질문 이해 디코더가 반환하는 표준 출력.

    파일 수나 최종 top-k는 여기서 직접 결정하지 않는다.
    질문 의미, 필요한 작업, 모달리티별 분석 초점만 반환한다.
    """

    original_query: str
    normalized_query: str
    input_context: str = ""
    identified_entities: List[str] = field(default_factory=list)
    required_operations: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    modality_focus: Dict[str, List[str]] = field(default_factory=dict)
    sub_queries: List[SubQuery] = field(default_factory=list)
    retrieval_action: RetrievalAction = RetrievalAction.RETRIEVE
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
    """다층 디코더 담당자에게 전달할 최종 패킷."""

    schema_version: str
    original_query: str
    normalized_query: str
    query_context: Dict[str, Any]
    complexity: ComplexityResult
    retrieval_results: Dict[str, ModalityRetrievalResult]

    def to_dict(self) -> Dict[str, Any]:
        return _enum_safe(asdict(self))


# ============================================================
# 2. 질문 이해 디코더
# ============================================================


class QueryUnderstandingDecoder(ABC):
    @abstractmethod
    def analyze(
        self,
        question: str,
        modality_summaries: Optional[Mapping[str, str]] = None,
    ) -> QueryUnderstandingResult:
        raise NotImplementedError


class StructuredLLMClient(Protocol):
    """어떤 LLM/API/로컬 모델이든 이 인터페이스에 맞춰 연결한다."""

    def generate_json(self, system_prompt: str, user_prompt: str) -> Mapping[str, Any]:
        ...


class PromptQueryUnderstandingDecoder(QueryUnderstandingDecoder):
    """추가 학습 없이 프롬프트 기반 구조화 출력을 생성하는 디코더.

    modality_summaries는 앞단 모달리티 인코더/분석기에서 만든 텍스트 요약이다.
    실제 이미지·영상 입력 방식은 사용하는 멀티모달 LLM 어댑터에서 처리해도 된다.
    """

    SYSTEM_PROMPT = """
당신은 멀티모달 RAG 시스템의 질문 이해 디코더다.
답변을 생성하지 말고 질문을 검색 및 근거 분석에 적합한 형태로 구조화하라.
파일 수와 top-k는 결정하지 않는다.

반드시 JSON 객체만 반환한다.
필수 필드:
- normalized_query: string
- input_context: string
- identified_entities: string[]
- required_operations: string[]
- constraints: string[]
- modality_focus: object. 키는 text/image/video/audio/table 중 필요한 것만 사용하고 값은 분석 초점 string[]
- sub_queries: [{query: string, modality: string, priority: low|medium|high}]
- retrieval_action: skip|retrieve|retrieve_and_verify
- answer_constraints: string[]
""".strip()

    def __init__(self, client: StructuredLLMClient) -> None:
        self.client = client

    def analyze(
        self,
        question: str,
        modality_summaries: Optional[Mapping[str, str]] = None,
    ) -> QueryUnderstandingResult:
        summaries = dict(modality_summaries or {})
        user_prompt = json.dumps(
            {
                "question": question,
                "input_modality_summaries": summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
        raw = self.client.generate_json(self.SYSTEM_PROMPT, user_prompt)
        return self._parse(question, raw)

    @staticmethod
    def _parse(question: str, raw: Mapping[str, Any]) -> QueryUnderstandingResult:
        sub_queries: List[SubQuery] = []
        for item in raw.get("sub_queries", []) or []:
            try:
                modality = Modality(str(item.get("modality", "text")).lower())
                priority = Level(str(item.get("priority", "medium")).lower())
            except ValueError:
                continue
            query = str(item.get("query", "")).strip()
            if query:
                sub_queries.append(SubQuery(query=query, modality=modality, priority=priority))

        try:
            action = RetrievalAction(str(raw.get("retrieval_action", "retrieve")).lower())
        except ValueError:
            action = RetrievalAction.RETRIEVE

        modality_focus: Dict[str, List[str]] = {}
        raw_focus = raw.get("modality_focus", {}) or {}
        if isinstance(raw_focus, Mapping):
            for key, value in raw_focus.items():
                try:
                    modality = Modality(str(key).lower())
                except ValueError:
                    continue
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    modality_focus[modality.value] = [str(v) for v in value if str(v).strip()]

        normalized = str(raw.get("normalized_query", question)).strip() or question
        return QueryUnderstandingResult(
            original_query=question,
            normalized_query=normalized,
            input_context=str(raw.get("input_context", "")).strip(),
            identified_entities=_string_list(raw.get("identified_entities")),
            required_operations=_string_list(raw.get("required_operations")),
            constraints=_string_list(raw.get("constraints")),
            modality_focus=modality_focus,
            sub_queries=sub_queries,
            retrieval_action=action,
            answer_constraints=_string_list(raw.get("answer_constraints")),
        )


class RuleBasedQueryUnderstandingDecoder(QueryUnderstandingDecoder):
    """LLM 연결 전 통합 테스트에 사용하는 무학습 fallback."""

    MODALITY_KEYWORDS: Dict[Modality, Tuple[str, ...]] = {
        Modality.TEXT: ("글", "문서", "리뷰", "시놉시스", "설명", "논문", "텍스트", "감독", "이전 작품", "정보", "줄거리"),
        Modality.IMAGE: ("이미지", "사진", "포스터", "그림", "색조", "구도"),
        Modality.VIDEO: ("영상", "예고편", "장면", "편집", "카메라", "촬영"),
        Modality.AUDIO: ("음성", "소리", "음악", "오디오", "대사"),
        Modality.TABLE: ("표", "그래프", "차트", "수치"),
    }

    def analyze(
        self,
        question: str,
        modality_summaries: Optional[Mapping[str, str]] = None,
    ) -> QueryUnderstandingResult:
        joined = " ".join([question, *(modality_summaries or {}).values()]).lower()
        required_ops: List[str] = []
        if any(k in joined for k in ("비교", "차이", "공통점")):
            required_ops.append("compare")
        if any(k in joined for k in ("검증", "사실", "맞는지", "근거")):
            required_ops.append("verify")
        if any(k in joined for k in ("분석", "이유", "왜")):
            required_ops.append("analyze")
        if not required_ops:
            required_ops.append("identify_and_explain")

        focus: Dict[str, List[str]] = {}
        sub_queries: List[SubQuery] = []
        for modality, keywords in self.MODALITY_KEYWORDS.items():
            matched = [kw for kw in keywords if kw in joined]
            if matched:
                focus[modality.value] = matched
                sub_queries.append(SubQuery(question, modality, Level.MEDIUM))

        if not sub_queries:
            focus[Modality.TEXT.value] = ["질문 관련 핵심 사실"]
            sub_queries.append(SubQuery(question, Modality.TEXT, Level.MEDIUM))

        action = (
            RetrievalAction.RETRIEVE_AND_VERIFY
            if "verify" in required_ops
            else RetrievalAction.RETRIEVE
        )

        return QueryUnderstandingResult(
            original_query=question,
            normalized_query=question.strip(),
            input_context=" | ".join((modality_summaries or {}).values()),
            required_operations=required_ops,
            modality_focus=focus,
            sub_queries=sub_queries,
            retrieval_action=action,
        )


# ============================================================
# 3. 질문 복잡도 분석 모듈
# ============================================================


class RuleBasedComplexityAnalyzer:
    """설명 가능한 검색 전 복잡도 게이트.

    초기에는 학습 없이 사용하고, 이후 동일 feature를 MLP 입력으로 교체할 수 있다.
    """

    COMPARISON_MARKERS = ("비교", "차이", "공통점", "대조", "versus", " vs ")
    VERIFICATION_MARKERS = ("검증", "사실", "맞는지", "진위", "근거", "출처")
    TEMPORAL_MARKERS = ("변화", "이전", "이후", "과거", "현재", "추세", "시간", "연도")
    AMBIGUITY_MARKERS = ("이것", "저것", "그거", "어떤", "무엇인지", "추정", "아마")
    EXTERNAL_MARKERS = ("최신", "감독", "이전 작품", "논문", "통계", "자료", "검색", "외부")

    def analyze(self, result: QueryUnderstandingResult) -> ComplexityResult:
        text = " ".join(
            [
                result.original_query,
                result.normalized_query,
                result.input_context,
                *result.required_operations,
                *result.constraints,
            ]
        ).lower()

        modalities = {
            sq.modality.value for sq in result.sub_queries
        } | set(result.modality_focus.keys())

        features = ComplexityFeatures(
            query_length=len(result.original_query),
            entity_count=len(result.identified_entities),
            operation_count=len(result.required_operations),
            required_modality_count=len(modalities),
            sub_query_count=len(result.sub_queries),
            comparison_required=_contains_any(text, self.COMPARISON_MARKERS)
            or "compare" in result.required_operations,
            verification_required=_contains_any(text, self.VERIFICATION_MARKERS)
            or "verify" in result.required_operations
            or result.retrieval_action == RetrievalAction.RETRIEVE_AND_VERIFY,
            temporal_reasoning=_contains_any(text, self.TEMPORAL_MARKERS),
            ambiguity_detected=_contains_any(text, self.AMBIGUITY_MARKERS),
            external_knowledge_required=_contains_any(text, self.EXTERNAL_MARKERS)
            or result.retrieval_action != RetrievalAction.SKIP,
            multimodal_reasoning_required=len(modalities) >= 2,
        )

        score, reasons = self._score(features)
        level = Level.LOW if score < 0.34 else Level.MEDIUM if score < 0.67 else Level.HIGH
        demand = self._modality_demand(result, level)
        return ComplexityResult(
            level=level,
            score=round(score, 4),
            retrieval_demand=demand,
            features=features,
            reasons=reasons,
        )

    @staticmethod
    def _score(f: ComplexityFeatures) -> Tuple[float, List[str]]:
        points = 0.0
        max_points = 10.0
        reasons: List[str] = []

        if f.query_length >= 80:
            points += 0.5
            reasons.append("질문 길이가 김")
        if f.entity_count >= 2:
            points += 1.0
            reasons.append("복수 대상 식별 필요")
        if f.operation_count >= 3:
            points += 1.5
            reasons.append("다단계 작업 필요")
        elif f.operation_count >= 2:
            points += 0.8
        if f.required_modality_count >= 3:
            points += 1.5
            reasons.append("3개 이상 모달리티 필요")
        elif f.required_modality_count == 2:
            points += 0.8
            reasons.append("멀티모달 결합 필요")
        if f.sub_query_count >= 4:
            points += 1.0
            reasons.append("다수 하위 질의 필요")
        elif f.sub_query_count >= 2:
            points += 0.5
        if f.comparison_required:
            points += 1.0
            reasons.append("비교 추론 필요")
        if f.verification_required:
            points += 1.5
            reasons.append("교차 검증 필요")
        if f.temporal_reasoning:
            points += 0.8
            reasons.append("시간적 관계 분석 필요")
        if f.ambiguity_detected:
            points += 0.7
            reasons.append("질문 모호성 존재")
        if f.external_knowledge_required:
            points += 0.5
        if f.multimodal_reasoning_required:
            points += 0.5

        return min(points / max_points, 1.0), reasons

    @staticmethod
    def _modality_demand(
        result: QueryUnderstandingResult,
        overall: Level,
    ) -> Dict[str, Level]:
        demand: Dict[str, Level] = {}
        for modality in Modality:
            focus = result.focus_for(modality)
            priorities = [
                sq.priority for sq in result.sub_queries if sq.modality == modality
            ]
            if not focus and not priorities:
                continue
            if Level.HIGH in priorities:
                demand[modality.value] = Level.HIGH
            elif Level.MEDIUM in priorities:
                demand[modality.value] = overall if overall != Level.LOW else Level.MEDIUM
            else:
                demand[modality.value] = Level.LOW
        return demand


# ============================================================
# 4. 검색기 인터페이스 및 FAISS 구현
# ============================================================


class VectorEncoder(Protocol):
    def encode_query(self, text: str) -> np.ndarray:
        ...


class ModalityRetriever(Protocol):
    modality: Modality

    def search(self, query: str, top_k: int) -> List[RetrievedEvidence]:
        ...


class CallableVectorEncoder:
    def __init__(self, fn: Callable[[str], np.ndarray]) -> None:
        self.fn = fn

    def encode_query(self, text: str) -> np.ndarray:
        vector = np.asarray(self.fn(text), dtype=np.float32).reshape(-1)
        return _l2_normalize(vector)


class SentenceTransformerTextEncoder:
    """선택 의존성: sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode_query(self, text: str) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            vec = self.model.encode_query(text, convert_to_numpy=True)
        else:
            vec = self.model.encode(text, convert_to_numpy=True)
        return _l2_normalize(np.asarray(vec, dtype=np.float32).reshape(-1))


class FaissHNSWRetriever:
    """모달리티별 별도 인덱스를 구성하는 HNSW 검색기.

    embeddings는 반드시 문서 순서와 documents 순서가 같아야 한다.
    cosine similarity를 위해 임베딩을 L2 정규화하고 inner product를 사용한다.
    """

    def __init__(
        self,
        modality: Modality,
        encoder: VectorEncoder,
        documents: Sequence[Any],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        hnsw_m: int = 32,
        ef_search: int = 128,
    ) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu 또는 faiss-gpu 설치가 필요합니다.") from exc

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings는 [문서 수, 차원] 형태여야 합니다.")
        if len(documents) != vectors.shape[0]:
            raise ValueError("documents와 embeddings의 개수가 다릅니다.")

        vectors = _l2_normalize_rows(vectors)
        self.modality = modality
        self.encoder = encoder
        self.documents = list(documents)
        self.metadata = [dict(m) for m in metadata] if metadata else [{} for _ in documents]

        dim = vectors.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.index.add(vectors)

    def search(self, query: str, top_k: int) -> List[RetrievedEvidence]:
        if top_k <= 0 or not self.documents:
            return []
        query_vector = self.encoder.encode_query(query).astype(np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query_vector, top_k)

        results: List[RetrievedEvidence] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            meta = dict(self.metadata[idx])
            meta.setdefault("rank", rank + 1)
            results.append(
                RetrievedEvidence(
                    evidence_id=str(meta.get("id", f"{self.modality.value}_{idx}")),
                    modality=self.modality,
                    score=float(score),
                    content=self.documents[idx],
                    metadata=meta,
                )
            )
        return results


class InMemoryRetriever:
    """FAISS 없이 단위 테스트할 수 있는 exact cosine 검색기."""

    def __init__(
        self,
        modality: Modality,
        encoder: VectorEncoder,
        documents: Sequence[Any],
        embeddings: np.ndarray,
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise ValueError("documents/embeddings shape가 올바르지 않습니다.")
        self.modality = modality
        self.encoder = encoder
        self.documents = list(documents)
        self.embeddings = _l2_normalize_rows(vectors)
        self.metadata = [dict(m) for m in metadata] if metadata else [{} for _ in documents]

    def search(self, query: str, top_k: int) -> List[RetrievedEvidence]:
        q = self.encoder.encode_query(query)
        scores = self.embeddings @ q
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        return [
            RetrievedEvidence(
                evidence_id=str(self.metadata[i].get("id", f"{self.modality.value}_{i}")),
                modality=self.modality,
                score=float(scores[i]),
                content=self.documents[i],
                metadata={**self.metadata[i], "rank": rank + 1},
            )
            for rank, i in enumerate(order)
        ]


# ============================================================
# 5. Adaptive RAG 게이트
# ============================================================


@dataclass(frozen=True)
class Budget:
    candidate_k: int
    final_k: int
    use_reranker: bool


class RetrievalBudgetPolicy:
    DEFAULTS: Dict[Level, Budget] = {
        Level.LOW: Budget(candidate_k=20, final_k=3, use_reranker=False),
        Level.MEDIUM: Budget(candidate_k=50, final_k=6, use_reranker=True),
        Level.HIGH: Budget(candidate_k=100, final_k=10, use_reranker=True),
    }

    MODALITY_FINAL_CAP: Dict[Modality, int] = {
        Modality.TEXT: 12,
        Modality.IMAGE: 6,
        Modality.VIDEO: 4,
        Modality.AUDIO: 4,
        Modality.TABLE: 6,
    }

    def initial_budget(self, modality: Modality, demand: Level) -> Budget:
        base = self.DEFAULTS[demand]
        cap = self.MODALITY_FINAL_CAP[modality]
        final_k = min(base.final_k, cap)
        candidate_k = max(final_k, base.candidate_k)
        return Budget(candidate_k, final_k, base.use_reranker)


class PostRetrievalUncertaintyGate:
    """기존 Adaptive RAG의 검색 후 score 기반 게이트."""

    def analyze(self, scores: Sequence[float]) -> RetrievalUncertainty:
        arr = np.asarray(scores, dtype=np.float64)
        if arr.size == 0:
            return RetrievalUncertainty(0.0, 0.0, 0.0, 0.0, 1.0, Level.HIGH)

        arr = np.sort(arr)[::-1]
        top1 = float(arr[0])
        gap = float(arr[0] - arr[1]) if arr.size >= 2 else float(abs(arr[0]))
        variance = float(np.var(arr))

        probabilities = _softmax(arr)
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        max_entropy = math.log(arr.size) if arr.size > 1 else 1.0
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        # 높은 entropy, 작은 gap, 작은 variance일수록 결과가 서로 비슷하여 불확실함.
        gap_signal = 1.0 - min(max(gap, 0.0) / 0.25, 1.0)
        variance_signal = 1.0 - min(max(variance, 0.0) / 0.05, 1.0)
        uncertainty_score = 0.55 * normalized_entropy + 0.30 * gap_signal + 0.15 * variance_signal
        level = (
            Level.LOW
            if uncertainty_score < 0.38
            else Level.MEDIUM
            if uncertainty_score < 0.68
            else Level.HIGH
        )
        return RetrievalUncertainty(
            top1_score=round(top1, 6),
            top1_top2_gap=round(gap, 6),
            score_variance=round(variance, 6),
            shannon_entropy=round(entropy, 6),
            normalized_entropy=round(normalized_entropy, 6),
            level=level,
        )

    def adjust_budget(
        self,
        initial: Budget,
        uncertainty: RetrievalUncertainty,
        modality: Modality,
        max_available: int,
    ) -> Budget:
        cap = min(RetrievalBudgetPolicy.MODALITY_FINAL_CAP[modality], max_available)
        if uncertainty.level == Level.HIGH:
            final_k = min(cap, max(initial.final_k + 3, math.ceil(initial.final_k * 1.5)))
            use_reranker = True
        elif uncertainty.level == Level.LOW:
            final_k = max(1, math.ceil(initial.final_k * 0.7))
            use_reranker = initial.use_reranker and modality == Modality.TEXT
        else:
            final_k = min(cap, initial.final_k)
            use_reranker = initial.use_reranker
        return Budget(initial.candidate_k, final_k, use_reranker)


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        evidence: Sequence[RetrievedEvidence],
        top_k: int,
    ) -> List[RetrievedEvidence]:
        ...


# ============================================================
# 6. 전체 파이프라인
# ============================================================


class AdaptiveMultimodalRAGPipeline:
    def __init__(
        self,
        query_decoder: QueryUnderstandingDecoder,
        complexity_analyzer: RuleBasedComplexityAnalyzer,
        retrievers: Mapping[Modality, ModalityRetriever],
        budget_policy: Optional[RetrievalBudgetPolicy] = None,
        uncertainty_gate: Optional[PostRetrievalUncertaintyGate] = None,
        rerankers: Optional[Mapping[Modality, Reranker]] = None,
    ) -> None:
        self.query_decoder = query_decoder
        self.complexity_analyzer = complexity_analyzer
        self.retrievers = dict(retrievers)
        self.budget_policy = budget_policy or RetrievalBudgetPolicy()
        self.uncertainty_gate = uncertainty_gate or PostRetrievalUncertaintyGate()
        self.rerankers = dict(rerankers or {})

    def run(
        self,
        question: str,
        modality_summaries: Optional[Mapping[str, str]] = None,
    ) -> AdaptiveRAGOutput:
        query_result = self.query_decoder.analyze(question, modality_summaries)
        complexity = self.complexity_analyzer.analyze(query_result)

        retrieval_results: Dict[str, ModalityRetrievalResult] = {}
        if query_result.retrieval_action != RetrievalAction.SKIP:
            grouped_queries = self._group_queries(query_result)
            for modality, search_query in grouped_queries.items():
                retriever = self.retrievers.get(modality)
                if retriever is None:
                    continue

                demand = complexity.retrieval_demand.get(modality.value, complexity.level)
                initial_budget = self.budget_policy.initial_budget(modality, demand)
                candidates = retriever.search(search_query, initial_budget.candidate_k)
                uncertainty = self.uncertainty_gate.analyze([item.score for item in candidates])
                adjusted = self.uncertainty_gate.adjust_budget(
                    initial_budget,
                    uncertainty,
                    modality,
                    max_available=len(candidates),
                )

                selected = candidates
                reranker = self.rerankers.get(modality)
                if adjusted.use_reranker and reranker is not None:
                    selected = reranker.rerank(search_query, candidates, adjusted.final_k)
                else:
                    selected = candidates[: adjusted.final_k]

                retrieval_results[modality.value] = ModalityRetrievalResult(
                    modality=modality,
                    query=search_query,
                    candidate_k=initial_budget.candidate_k,
                    final_k=len(selected),
                    use_reranker=adjusted.use_reranker and reranker is not None,
                    uncertainty=uncertainty,
                    evidence=selected,
                )

        context = {
            "input_context": query_result.input_context,
            "identified_entities": query_result.identified_entities,
            "required_operations": query_result.required_operations,
            "constraints": query_result.constraints,
            "modality_focus": query_result.modality_focus,
            "answer_constraints": query_result.answer_constraints,
            "retrieval_action": query_result.retrieval_action.value,
            "sub_queries": [
                {
                    "query": sq.query,
                    "modality": sq.modality.value,
                    "priority": sq.priority.value,
                }
                for sq in query_result.sub_queries
            ],
        }
        return AdaptiveRAGOutput(
            schema_version="3.0",
            original_query=query_result.original_query,
            normalized_query=query_result.normalized_query,
            query_context=context,
            complexity=complexity,
            retrieval_results=retrieval_results,
        )

    @staticmethod
    def _group_queries(result: QueryUnderstandingResult) -> Dict[Modality, str]:
        grouped: Dict[Modality, List[SubQuery]] = {}
        for sub_query in result.sub_queries:
            grouped.setdefault(sub_query.modality, []).append(sub_query)

        if not grouped:
            for modality_name in result.modality_focus:
                try:
                    grouped[Modality(modality_name)] = []
                except ValueError:
                    continue

        output: Dict[Modality, str] = {}
        for modality, items in grouped.items():
            if items:
                items = sorted(
                    items,
                    key=lambda item: {Level.HIGH: 0, Level.MEDIUM: 1, Level.LOW: 2}[item.priority],
                )
                # 한 모달리티에 여러 검색 의도가 있으면 한 검색문으로 합친다.
                output[modality] = " ; ".join(item.query for item in items)
            else:
                output[modality] = result.normalized_query
        return output


# ============================================================
# 7. 다층 디코더 연동용 헬퍼
# ============================================================


def build_decoder_inputs(rag_output: AdaptiveRAGOutput) -> Dict[str, Dict[str, Any]]:
    """다층 디코더 담당자가 바로 사용할 모달리티별 입력으로 변환한다."""

    inputs: Dict[str, Dict[str, Any]] = {}
    modality_focus = rag_output.query_context.get("modality_focus", {})

    for modality_name, result in rag_output.retrieval_results.items():
        inputs[modality_name] = {
            "original_query": rag_output.original_query,
            "normalized_query": rag_output.normalized_query,
            "focus_features": modality_focus.get(modality_name, []),
            "required_operations": rag_output.query_context.get("required_operations", []),
            "constraints": rag_output.query_context.get("constraints", []),
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "score": item.score,
                    "content": item.content,
                    "metadata": item.metadata,
                }
                for item in result.evidence
            ],
        }
    return inputs


# ============================================================
# 8. 유틸리티
# ============================================================


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(marker in text for marker in markers)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


def _l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _enum_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _enum_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_enum_safe(v) for v in value]
    return value
