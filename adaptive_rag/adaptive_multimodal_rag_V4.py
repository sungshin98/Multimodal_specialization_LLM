"""KARINA 역할 2: Question Understanding Decoder + Retrieval Gate + Adaptive RAG.

역할 경계
- 입력: 원본 질문, 역할 1의 EncoderOutput 묶음, 모달리티별 의미 요약(선택)
- 출력: Question Condition, 검색 결정, 모달리티별 RAG 근거를 포함한 full packet
- 뒤쪽 디코더에는 ``AdaptiveRAGOutput.to_dict()`` 전체를 전달한다.

주의: 768차원 pooled embedding 자체는 프롬프트 기반 LLM이 의미적으로 해석하지
않는다. 입력 내용의 의미 판단에는 ``modality_summaries``가 필요하다. embedding은
입력 존재 여부·차원·연결 검증 신호로만 사용한다.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from .v4_contracts import (
    AdaptiveRAGOutput,
    ComplexityFeatures,
    ComplexityResult,
    InitialMultimodalContext,
    Level,
    Modality,
    ModalityRetrievalResult,
    QuestionCondition,
    RetrievalAction,
    RetrievalDecision,
    RetrievalUncertainty,
    RetrievedEvidence,
    SubQuery,
)


class StructuredLLMClient(Protocol):
    def generate_json(self, system_prompt: str, user_prompt: str) -> Mapping[str, Any]:
        ...


class QuestionUnderstandingDecoder(ABC):
    @abstractmethod
    def analyze(
        self,
        question: str,
        initial_context: InitialMultimodalContext,
    ) -> QuestionCondition:
        raise NotImplementedError


class PromptQuestionUnderstandingDecoder(QuestionUnderstandingDecoder):
    """질문과 초기 입력 요약을 C_Q 구조로 변환하는 프롬프트 기반 디코더."""

    SYSTEM_PROMPT = """
너는 KARINA의 질문 이해 디코더다. 답변을 생성하지 말고 질문과 초기 입력의
관계를 분석해 구조화된 Question Condition을 만들어라.

주의
- encoder_signals의 embedding_dim은 입력 연결 정보일 뿐 의미가 아니다.
- 초기 입력의 실제 내용은 modality_summaries에 적힌 범위에서만 판단하라.
- 검색 실행 여부를 직접 집행하지 않는다. external_knowledge_required와 검색용
  sub_queries만 제시하고, 최종 결정은 뒤의 Retrieval Gate가 한다.

반드시 JSON 객체만 반환한다.
필수 필드
- normalized_query: string
- input_interpretation: string
- task: string[]
- targets: string[]
- reasoning: string[]
- constraints: string[]
- required_modalities: string[]
- external_knowledge_required: boolean
- modality_focus: object
- sub_queries: [{query:string, modality:string, priority:low|medium|high}]
- answer_constraints: string[]
""".strip()

    def __init__(self, client: StructuredLLMClient) -> None:
        self.client = client

    def analyze(
        self,
        question: str,
        initial_context: InitialMultimodalContext,
    ) -> QuestionCondition:
        payload = {
            "question": question,
            "initial_multimodal_context": initial_context.prompt_payload(),
        }
        raw = self.client.generate_json(
            self.SYSTEM_PROMPT,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        return self._parse(question, initial_context, raw)

    @staticmethod
    def _parse(
        question: str,
        initial_context: InitialMultimodalContext,
        raw: Mapping[str, Any],
    ) -> QuestionCondition:
        normalized = str(raw.get("normalized_query", question) or question).strip() or question
        interpretation = str(raw.get("input_interpretation", "") or "").strip()
        if not interpretation:
            interpretation = _fallback_input_interpretation(initial_context)
        return QuestionCondition(
            original_query=question,
            normalized_query=normalized,
            input_interpretation=interpretation,
            task=_string_list(raw.get("task")),
            targets=_string_list(raw.get("targets")),
            reasoning=_string_list(raw.get("reasoning")),
            constraints=_string_list(raw.get("constraints")),
            required_modalities=_parse_modalities(raw.get("required_modalities")),
            external_knowledge_required=_as_bool(raw.get("external_knowledge_required")),
            modality_focus=_parse_modality_focus(raw.get("modality_focus")),
            sub_queries=_parse_sub_queries(raw.get("sub_queries")),
            answer_constraints=_string_list(raw.get("answer_constraints")),
        )


class RuleBasedQuestionUnderstandingDecoder(QuestionUnderstandingDecoder):
    """실제 LLM 연결 전 통합 테스트를 위한 결정적 fallback."""

    MODALITY_KEYWORDS: Dict[Modality, Tuple[str, ...]] = {
        Modality.TEXT: (
            "텍스트", "문서", "글", "리뷰", "시놉시스", "설명", "논문", "출처",
            "감독", "이전 작품", "정보", "줄거리", "사실",
        ),
        Modality.IMAGE: ("이미지", "사진", "포스터", "그림", "색조", "구도", "시각"),
        Modality.VIDEO: ("영상", "예고편", "장면", "편집", "카메라", "촬영", "시간적"),
        Modality.AUDIO: ("음성", "소리", "음악", "오디오", "대사"),
        Modality.TABLE: ("표", "그래프", "차트", "수치"),
    }
    EXTERNAL_MARKERS = (
        "최신", "검색", "출처", "논문", "통계", "자료", "외부", "이전 작품",
        "사실 여부", "검증", "맞는지", "근거", "누구", "언제", "어디서",
    )

    def analyze(
        self,
        question: str,
        initial_context: InitialMultimodalContext,
    ) -> QuestionCondition:
        summaries = initial_context.summary_map()
        joined = " ".join([question, *summaries.values()]).lower()

        task: List[str] = []
        reasoning: List[str] = []
        if _contains_any(joined, ("비교", "차이", "공통점", "대조")):
            task.append("compare")
            reasoning.append("comparative")
        if _contains_any(joined, ("검증", "사실", "맞는지", "근거", "출처")):
            task.append("verify")
            reasoning.append("verification")
        if _contains_any(joined, ("분석", "이유", "왜", "설명")):
            task.append("analyze")
        if _contains_any(joined, ("변화", "이전", "이후", "시간", "순서")):
            reasoning.append("temporal")
        if not task:
            task.append("identify_and_explain")

        focus: Dict[str, List[str]] = {}
        required_modalities: List[Modality] = []
        sub_queries: List[SubQuery] = []
        for modality, keywords in self.MODALITY_KEYWORDS.items():
            matched = [keyword for keyword in keywords if keyword in joined]
            if matched:
                required_modalities.append(modality)
                focus[modality.value] = matched

        if not required_modalities:
            available = initial_context.available_modalities()
            required_modalities = available[:1] if available else [Modality.TEXT]
            for modality in required_modalities:
                focus[modality.value] = ["질문과 직접 관련된 정보"]

        external_required = _contains_any(joined, self.EXTERNAL_MARKERS)
        if external_required:
            for modality in required_modalities:
                sub_queries.append(
                    SubQuery(
                        query=question.strip(),
                        modality=modality,
                        priority=Level.HIGH if "verify" in task else Level.MEDIUM,
                    )
                )

        return QuestionCondition(
            original_query=question,
            normalized_query=question.strip(),
            input_interpretation=_fallback_input_interpretation(initial_context),
            task=task,
            targets=[],
            reasoning=list(dict.fromkeys(reasoning)),
            constraints=[],
            required_modalities=required_modalities,
            external_knowledge_required=external_required,
            modality_focus=focus,
            sub_queries=sub_queries,
            answer_constraints=[],
        )


class RuleBasedComplexityAnalyzer:
    """Question Condition을 이용해 검색 전 복잡도를 계산한다."""

    def analyze(self, condition: QuestionCondition) -> ComplexityResult:
        text = " ".join(
            [
                condition.original_query,
                condition.normalized_query,
                condition.input_interpretation,
                *condition.task,
                *condition.reasoning,
                *condition.constraints,
            ]
        ).lower()
        features = ComplexityFeatures(
            query_length=len(condition.original_query),
            entity_count=len(condition.targets),
            operation_count=len(condition.task),
            required_modality_count=len(condition.required_modalities),
            sub_query_count=len(condition.sub_queries),
            comparison_required="compare" in condition.task or "comparative" in condition.reasoning,
            verification_required="verify" in condition.task or "verification" in condition.reasoning,
            temporal_reasoning="temporal" in condition.reasoning,
            ambiguity_detected=_contains_any(text, ("이것", "저것", "그거", "어떤", "추정")),
            external_knowledge_required=condition.external_knowledge_required,
            multimodal_reasoning_required=len(condition.required_modalities) >= 2,
        )
        score, reasons = self._score(features)
        level = Level.LOW if score < 0.34 else Level.MEDIUM if score < 0.67 else Level.HIGH
        return ComplexityResult(
            level=level,
            score=round(score, 4),
            retrieval_demand=self._modality_demand(condition, level),
            features=features,
            reasons=reasons,
        )

    @staticmethod
    def _score(features: ComplexityFeatures) -> Tuple[float, List[str]]:
        points = 0.0
        reasons: List[str] = []
        if features.query_length >= 80:
            points += 0.5
            reasons.append("long_query")
        if features.entity_count >= 2:
            points += 1.0
            reasons.append("multiple_targets")
        if features.operation_count >= 3:
            points += 1.5
            reasons.append("multi_step")
        elif features.operation_count >= 2:
            points += 0.8
        if features.required_modality_count >= 3:
            points += 1.5
            reasons.append("three_or_more_modalities")
        elif features.required_modality_count == 2:
            points += 0.8
            reasons.append("multimodal")
        if features.sub_query_count >= 4:
            points += 1.0
        elif features.sub_query_count >= 2:
            points += 0.5
        if features.comparison_required:
            points += 1.0
            reasons.append("comparison")
        if features.verification_required:
            points += 1.5
            reasons.append("verification")
        if features.temporal_reasoning:
            points += 0.8
            reasons.append("temporal")
        if features.ambiguity_detected:
            points += 0.7
            reasons.append("ambiguity")
        if features.external_knowledge_required:
            points += 0.5
        if features.multimodal_reasoning_required:
            points += 0.5
        return min(points / 10.0, 1.0), reasons

    @staticmethod
    def _modality_demand(
        condition: QuestionCondition,
        overall: Level,
    ) -> Dict[str, Level]:
        result: Dict[str, Level] = {}
        for modality in condition.required_modalities:
            priorities = [q.priority for q in condition.sub_queries if q.modality == modality]
            if Level.HIGH in priorities:
                result[modality.value] = Level.HIGH
            elif Level.MEDIUM in priorities:
                result[modality.value] = overall if overall != Level.LOW else Level.MEDIUM
            else:
                result[modality.value] = Level.LOW if overall == Level.LOW else overall
        return result


class RetrievalGate:
    """Question Condition을 받아 검색 실행 여부와 검색 모달리티를 결정한다."""

    def decide(
        self,
        condition: QuestionCondition,
        complexity: ComplexityResult,
        available_retrievers: Iterable[Modality],
    ) -> RetrievalDecision:
        available = set(available_retrievers)
        if not condition.external_knowledge_required:
            return RetrievalDecision(
                action=RetrievalAction.SKIP,
                reasons=["question_condition.external_knowledge_required=false"],
            )

        requested: List[Modality] = []
        for sub_query in condition.sub_queries:
            if sub_query.modality not in requested:
                requested.append(sub_query.modality)
        for modality in condition.required_modalities:
            if modality not in requested:
                requested.append(modality)
        if not requested:
            requested = [Modality.TEXT]

        selected = [modality for modality in requested if modality in available]
        unavailable = [modality for modality in requested if modality not in available]
        if not selected and Modality.TEXT in available:
            selected = [Modality.TEXT]

        action = (
            RetrievalAction.RETRIEVE_AND_VERIFY
            if "verify" in condition.task or complexity.features.verification_required
            else RetrievalAction.RETRIEVE
        )
        return RetrievalDecision(
            action=action,
            modalities=selected,
            reasons=[
                "external_knowledge_required=true",
                f"complexity={complexity.level.value}",
                f"selected_modalities={','.join(m.value for m in selected) or 'none'}",
            ],
            unavailable_modalities=unavailable,
        )


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
        return _l2_normalize(np.asarray(self.fn(text), dtype=np.float32).reshape(-1))


class SentenceTransformerTextEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode_query(self, text: str) -> np.ndarray:
        if hasattr(self.model, "encode_query"):
            value = self.model.encode_query(text, convert_to_numpy=True)
        else:
            value = self.model.encode(text, convert_to_numpy=True)
        return _l2_normalize(np.asarray(value, dtype=np.float32).reshape(-1))


class InMemoryRetriever:
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
        self.metadata = [dict(item) for item in metadata] if metadata else [{} for _ in documents]

    def search(self, query: str, top_k: int) -> List[RetrievedEvidence]:
        if top_k <= 0 or not self.documents:
            return []
        vector = self.encoder.encode_query(query)
        scores = self.embeddings @ vector
        order = np.argsort(-scores)[: min(top_k, len(scores))]
        return [
            RetrievedEvidence(
                evidence_id=str(self.metadata[index].get("id", f"{self.modality.value}_{index}")),
                modality=self.modality,
                score=float(scores[index]),
                content=self.documents[index],
                metadata={**self.metadata[index], "rank": rank + 1},
            )
            for rank, index in enumerate(order)
        ]


class FaissHNSWRetriever:
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
        if vectors.ndim != 2 or vectors.shape[0] != len(documents):
            raise ValueError("documents/embeddings shape가 올바르지 않습니다.")
        vectors = _l2_normalize_rows(vectors)
        self.modality = modality
        self.encoder = encoder
        self.documents = list(documents)
        self.metadata = [dict(item) for item in metadata] if metadata else [{} for _ in documents]
        self.index = faiss.IndexHNSWFlat(vectors.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.index.add(vectors)

    def search(self, query: str, top_k: int) -> List[RetrievedEvidence]:
        if top_k <= 0 or not self.documents:
            return []
        vector = self.encoder.encode_query(query).astype(np.float32).reshape(1, -1)
        top_k = min(top_k, len(self.documents))
        scores, indices = self.index.search(vector, top_k)
        results: List[RetrievedEvidence] = []
        for rank, (score, index) in enumerate(zip(scores[0], indices[0])):
            if index < 0:
                continue
            meta = dict(self.metadata[index])
            meta.setdefault("rank", rank + 1)
            results.append(
                RetrievedEvidence(
                    evidence_id=str(meta.get("id", f"{self.modality.value}_{index}")),
                    modality=self.modality,
                    score=float(score),
                    content=self.documents[index],
                    metadata=meta,
                )
            )
        return results


@dataclass(frozen=True)
class Budget:
    candidate_k: int
    final_k: int
    use_reranker: bool


class RetrievalBudgetPolicy:
    DEFAULTS: Dict[Level, Budget] = {
        Level.LOW: Budget(20, 3, False),
        Level.MEDIUM: Budget(50, 6, True),
        Level.HIGH: Budget(100, 10, True),
    }
    FINAL_CAP: Dict[Modality, int] = {
        Modality.TEXT: 12,
        Modality.IMAGE: 6,
        Modality.VIDEO: 4,
        Modality.AUDIO: 4,
        Modality.TABLE: 6,
    }

    def initial_budget(self, modality: Modality, demand: Level) -> Budget:
        base = self.DEFAULTS[demand]
        final_k = min(base.final_k, self.FINAL_CAP[modality])
        return Budget(max(final_k, base.candidate_k), final_k, base.use_reranker)


class PostRetrievalUncertaintyGate:
    def analyze(self, scores: Sequence[float]) -> RetrievalUncertainty:
        values = np.asarray(scores, dtype=np.float64)
        if values.size == 0:
            return RetrievalUncertainty(0.0, 0.0, 0.0, 0.0, 1.0, Level.HIGH)
        values = np.sort(values)[::-1]
        top1 = float(values[0])
        gap = float(values[0] - values[1]) if values.size >= 2 else float(abs(values[0]))
        variance = float(np.var(values))
        probabilities = _softmax(values)
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        max_entropy = math.log(values.size) if values.size > 1 else 1.0
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        gap_signal = 1.0 - min(max(gap, 0.0) / 0.25, 1.0)
        variance_signal = 1.0 - min(max(variance, 0.0) / 0.05, 1.0)
        uncertainty_score = 0.55 * normalized_entropy + 0.30 * gap_signal + 0.15 * variance_signal
        level = Level.LOW if uncertainty_score < 0.38 else Level.MEDIUM if uncertainty_score < 0.68 else Level.HIGH
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
        cap = min(RetrievalBudgetPolicy.FINAL_CAP[modality], max_available)
        if uncertainty.level == Level.HIGH:
            final_k = min(cap, max(initial.final_k + 3, math.ceil(initial.final_k * 1.5)))
            use_reranker = True
        elif uncertainty.level == Level.LOW:
            final_k = max(1, min(cap, math.ceil(initial.final_k * 0.7)))
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


class AdaptiveMultimodalRAGPipelineV4:
    def __init__(
        self,
        query_decoder: QuestionUnderstandingDecoder,
        complexity_analyzer: RuleBasedComplexityAnalyzer,
        retrievers: Mapping[Modality, ModalityRetriever],
        retrieval_gate: Optional[RetrievalGate] = None,
        budget_policy: Optional[RetrievalBudgetPolicy] = None,
        uncertainty_gate: Optional[PostRetrievalUncertaintyGate] = None,
        rerankers: Optional[Mapping[Modality, Reranker]] = None,
    ) -> None:
        self.query_decoder = query_decoder
        self.complexity_analyzer = complexity_analyzer
        self.retrievers = dict(retrievers)
        self.retrieval_gate = retrieval_gate or RetrievalGate()
        self.budget_policy = budget_policy or RetrievalBudgetPolicy()
        self.uncertainty_gate = uncertainty_gate or PostRetrievalUncertaintyGate()
        self.rerankers = dict(rerankers or {})

    def run(
        self,
        question: str,
        encoder_outputs: Optional[Any] = None,
        modality_summaries: Optional[Mapping[str, Any]] = None,
        encoder_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> AdaptiveRAGOutput:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        initial_context = InitialMultimodalContext.from_encoder_outputs(
            encoder_outputs=encoder_outputs,
            modality_summaries=modality_summaries,
            metadata=encoder_metadata,
        )
        condition = self.query_decoder.analyze(question.strip(), initial_context)
        complexity = self.complexity_analyzer.analyze(condition)
        decision = self.retrieval_gate.decide(condition, complexity, self.retrievers.keys())

        retrieval_results: Dict[str, ModalityRetrievalResult] = {}
        if decision.action != RetrievalAction.SKIP:
            grouped_queries = self._group_queries(condition, decision.modalities)
            for modality in decision.modalities:
                retriever = self.retrievers.get(modality)
                if retriever is None:
                    continue
                search_query = grouped_queries.get(modality, condition.normalized_query)
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

        query_context = {
            "input_context": condition.input_interpretation,
            "identified_entities": list(condition.targets),
            "required_operations": list(condition.task),
            "constraints": list(condition.constraints),
            "modality_focus": dict(condition.modality_focus),
            "answer_constraints": list(condition.answer_constraints),
            "retrieval_action": decision.action.value,
            "required_modalities": [m.value for m in condition.required_modalities],
            "reasoning": list(condition.reasoning),
            "initial_modalities": [m.value for m in initial_context.available_modalities()],
            "sub_queries": [
                {
                    "query": query.query,
                    "modality": query.modality.value,
                    "priority": query.priority.value,
                }
                for query in condition.sub_queries
            ],
        }
        return AdaptiveRAGOutput(
            schema_version="4.0",
            original_query=condition.original_query,
            normalized_query=condition.normalized_query,
            initial_context=initial_context.to_dict(),
            question_condition=condition,
            query_context=query_context,
            retrieval_decision=decision,
            complexity=complexity,
            retrieval_results=retrieval_results,
        )

    @staticmethod
    def _group_queries(
        condition: QuestionCondition,
        selected_modalities: Sequence[Modality],
    ) -> Dict[Modality, str]:
        grouped: Dict[Modality, List[SubQuery]] = {}
        for sub_query in condition.sub_queries:
            if sub_query.modality in selected_modalities:
                grouped.setdefault(sub_query.modality, []).append(sub_query)
        priority_order = {Level.HIGH: 0, Level.MEDIUM: 1, Level.LOW: 2}
        result: Dict[Modality, str] = {}
        for modality in selected_modalities:
            items = sorted(grouped.get(modality, []), key=lambda item: priority_order[item.priority])
            result[modality] = " ; ".join(item.query for item in items) if items else condition.normalized_query
        return result


def build_decoder_inputs(output: AdaptiveRAGOutput) -> Dict[str, Dict[str, Any]]:
    """구형 연동용. 권장 경로는 output.to_dict() 전체 전달이다."""
    inputs: Dict[str, Dict[str, Any]] = {}
    for modality_name, result in output.retrieval_results.items():
        inputs[modality_name] = {
            "original_query": output.original_query,
            "normalized_query": output.normalized_query,
            "input_context": output.query_context.get("input_context", ""),
            "focus_features": output.query_context.get("modality_focus", {}).get(modality_name, []),
            "required_operations": output.query_context.get("required_operations", []),
            "constraints": output.query_context.get("constraints", []),
            "answer_constraints": output.query_context.get("answer_constraints", []),
            "complexity_level": output.complexity.level.value,
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "modality": item.modality.value,
                    "score": item.score,
                    "content": item.content,
                    "metadata": item.metadata,
                }
                for item in result.evidence
            ],
        }
    return inputs


def _parse_modalities(value: Any) -> List[Modality]:
    result: List[Modality] = []
    for item in value or []:
        try:
            modality = Modality(str(item).strip().lower())
        except ValueError:
            continue
        if modality not in result:
            result.append(modality)
    return result


def _parse_modality_focus(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, Mapping):
        return {}
    result: Dict[str, List[str]] = {}
    for key, items in value.items():
        try:
            modality = Modality(str(key).strip().lower())
        except ValueError:
            continue
        result[modality.value] = _string_list(items)
    return result


def _parse_sub_queries(value: Any) -> List[SubQuery]:
    result: List[SubQuery] = []
    for item in value or []:
        if not isinstance(item, Mapping):
            continue
        try:
            modality = Modality(str(item.get("modality", "text")).strip().lower())
            priority = Level(str(item.get("priority", "medium")).strip().lower())
        except ValueError:
            continue
        query = str(item.get("query", "") or "").strip()
        if query:
            result.append(SubQuery(query=query, modality=modality, priority=priority))
    return result


def _fallback_input_interpretation(context: InitialMultimodalContext) -> str:
    summaries = context.summary_map()
    if summaries:
        return " | ".join(f"{modality}: {summary}" for modality, summary in summaries.items())
    available = [modality.value for modality in context.available_modalities()]
    if available:
        return "Available initial modalities: " + ", ".join(available)
    return "No initial multimodal input"


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    try:
        return [str(item).strip() for item in value if str(item).strip()]
    except TypeError:
        return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


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
