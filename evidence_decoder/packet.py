"""Adaptive RAG 패킷 -> DecoderTask 어댑터.

adaptive_rag 의 build_decoder_inputs() 는 answer_constraints, complexity,
uncertainty 를 넘겨주지 않는다. 최종 답변 디코더는 answer_constraints 없이는
답변 형식 제약을 지킬 수 없으므로, 여기서는 전체 AdaptiveRAGOutput 을 직접 받는다.

세 가지 입력을 모두 허용한다.
1. AdaptiveRAGOutput 객체
2. AdaptiveRAGOutput.to_dict() 결과 (JSON 파일/네트워크 경유)
3. build_decoder_inputs() 결과 (구형 경로, 질문 조건 일부 손실)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .schemas import (
    DecoderTask,
    EvidenceItem,
    Level,
    Modality,
    ModalityTask,
    parse_level,
    parse_modality,
)


class PacketAdapter:
    """상대 저장소 코드를 수정하지 않아도 동작하도록 만든 방어적 어댑터."""

    def adapt(self, packet: Any) -> DecoderTask:
        data = self._to_dict(packet)

        if self._looks_like_decoder_inputs(data):
            return self._from_decoder_inputs(data)
        return self._from_full_packet(data)

    # ------------------------------------------------------------------
    # 정규화
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dict(packet: Any) -> Dict[str, Any]:
        if isinstance(packet, Mapping):
            return dict(packet)
        to_dict = getattr(packet, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        raise TypeError(
            "패킷은 Mapping 이거나 to_dict() 를 가진 객체여야 한다: " f"{type(packet)!r}"
        )

    @staticmethod
    def _looks_like_decoder_inputs(data: Mapping[str, Any]) -> bool:
        """build_decoder_inputs() 결과는 {모달리티명: {...evidence}} 형태다."""
        if "retrieval_results" in data or "query_context" in data:
            return False
        for key, value in data.items():
            if parse_modality(key) is not None and isinstance(value, Mapping) and "evidence" in value:
                return True
        return False

    # ------------------------------------------------------------------
    # 경로 1 - 전체 패킷 (권장)
    # ------------------------------------------------------------------

    def _from_full_packet(self, data: Mapping[str, Any]) -> DecoderTask:
        context: Mapping[str, Any] = data.get("query_context") or {}
        complexity: Mapping[str, Any] = data.get("complexity") or {}
        modality_focus: Mapping[str, Any] = context.get("modality_focus") or {}
        retrieval_results: Mapping[str, Any] = data.get("retrieval_results") or {}

        original = str(data.get("original_query", "") or "")
        normalized = str(data.get("normalized_query", "") or original)

        required_operations = _str_list(context.get("required_operations"))
        constraints = _str_list(context.get("constraints"))

        sub_query_by_modality = self._sub_queries_by_modality(context.get("sub_queries"))

        modality_tasks: List[ModalityTask] = []
        for name, result in retrieval_results.items():
            modality = parse_modality(name)
            if modality is None or not isinstance(result, Mapping):
                continue

            uncertainty = result.get("uncertainty") or {}
            retrieval_query = str(result.get("query", "") or "")
            modality_tasks.append(
                ModalityTask(
                    modality=modality,
                    query=sub_query_by_modality.get(modality) or normalized,
                    focus_features=_str_list(modality_focus.get(name)),
                    required_operations=required_operations,
                    constraints=constraints,
                    evidence=self._evidence_items(modality, result.get("evidence")),
                    uncertainty_level=parse_level(uncertainty.get("level"), Level.MEDIUM),
                    retrieval_query=retrieval_query,
                )
            )

        return DecoderTask(
            original_query=original,
            normalized_query=normalized,
            input_context=str(context.get("input_context", "") or ""),
            identified_entities=_str_list(context.get("identified_entities")),
            required_operations=required_operations,
            constraints=constraints,
            answer_constraints=_str_list(context.get("answer_constraints")),
            complexity_level=parse_level(complexity.get("level"), Level.MEDIUM),
            complexity_score=_as_float(complexity.get("score")),
            modality_tasks=modality_tasks,
            schema_version=str(data.get("schema_version", "") or ""),
        )

    # ------------------------------------------------------------------
    # 경로 2 - build_decoder_inputs() 결과 (질문 조건 일부 손실)
    # ------------------------------------------------------------------

    def _from_decoder_inputs(self, data: Mapping[str, Any]) -> DecoderTask:
        original = ""
        normalized = ""
        required_operations: List[str] = []
        constraints: List[str] = []
        modality_tasks: List[ModalityTask] = []

        for name, value in data.items():
            modality = parse_modality(name)
            if modality is None or not isinstance(value, Mapping):
                continue

            original = original or str(value.get("original_query", "") or "")
            normalized = normalized or str(value.get("normalized_query", "") or "")
            required_operations = required_operations or _str_list(value.get("required_operations"))
            constraints = constraints or _str_list(value.get("constraints"))

            modality_tasks.append(
                ModalityTask(
                    modality=modality,
                    query=str(value.get("normalized_query", "") or original),
                    focus_features=_str_list(value.get("focus_features")),
                    required_operations=_str_list(value.get("required_operations")),
                    constraints=_str_list(value.get("constraints")),
                    evidence=self._evidence_items(modality, value.get("evidence")),
                )
            )

        return DecoderTask(
            original_query=original,
            normalized_query=normalized or original,
            required_operations=required_operations,
            constraints=constraints,
            # answer_constraints / complexity 는 이 경로로 오면 복구 불가.
            # 최종 디코더는 기본 정책으로 동작한다.
            modality_tasks=modality_tasks,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _evidence_items(modality: Modality, raw: Any) -> List[EvidenceItem]:
        items: List[EvidenceItem] = []
        for entry in raw or []:
            if not isinstance(entry, Mapping):
                continue
            metadata = entry.get("metadata")
            items.append(
                EvidenceItem(
                    evidence_id=str(entry.get("evidence_id", f"{modality.value}_{len(items)}")),
                    modality=parse_modality(entry.get("modality"), modality) or modality,
                    score=_as_float(entry.get("score")),
                    content=entry.get("content"),
                    metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                )
            )
        return items

    @staticmethod
    def _sub_queries_by_modality(raw: Any) -> Dict[Modality, str]:
        """모달리티당 서브질의가 여러 개면 우선순위 순으로 합친다."""
        order = {Level.HIGH: 0, Level.MEDIUM: 1, Level.LOW: 2}
        buckets: Dict[Modality, List[tuple]] = {}
        for entry in raw or []:
            if not isinstance(entry, Mapping):
                continue
            modality = parse_modality(entry.get("modality"))
            query = str(entry.get("query", "") or "").strip()
            if modality is None or not query:
                continue
            priority = parse_level(entry.get("priority"), Level.MEDIUM)
            buckets.setdefault(modality, []).append((order[priority], query))

        merged: Dict[Modality, str] = {}
        for modality, entries in buckets.items():
            entries.sort(key=lambda item: item[0])
            merged[modality] = " ; ".join(query for _, query in entries)
        return merged


def _str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Mapping):
        return [str(v) for v in value.values() if str(v).strip()]
    try:
        return [str(v).strip() for v in value if str(v).strip()]
    except TypeError:
        return []


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
