"""1층 - 모달리티별 근거 디코더.

각 디코더는 자기 모달리티의 원본만 본다. 하지만 출력은 전부
EvidenceCard 리스트로 동일하다. 통합 계층이 모달리티를 몰라도
동작하게 만드는 것이 이 계약의 핵심이다.

디코더는 evidence 만 보지 않는다. 질문 이해 디코더가 준 focus_features,
required_operations, constraints 를 함께 받아 "질문에 필요한 것만" 추출한다.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .assets import AssetLoader
from .clients import LLMError, MediaAsset, StructuredLLMClient, VisionStructuredClient
from .schemas import (
    DecoderTask,
    EvidenceCard,
    EvidenceItem,
    Modality,
    ModalityEvidenceResult,
    ModalityTask,
)

# ============================================================
# 공통 출력 스키마
# ============================================================

CARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_evidence_id": {"type": "string"},
                    "claim": {"type": "string"},
                    "detail": {"type": "string"},
                    "supports": {"type": "array", "items": {"type": "string"}},
                    "relevance": {"type": "number"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "source_evidence_id",
                    "claim",
                    "detail",
                    "supports",
                    "relevance",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
        # 근거마다 대상·초점 일치 여부를 명시적으로 판정하게 한다.
        # "무관한 근거는 카드를 만들지 마라"를 규칙으로 넣어 봤으나 실패했다
        # (무관거부 0.00 유지, gold채택만 1.00 -> 0.93 하락).
        # verify 질문에서 모델이 다른 대상의 근거를 "대조용 재료"로 해석하며
        # 그 해석이 프롬프트 규칙을 이기기 때문이다. 판정을 카드 생성과 분리하고
        # 강제 출력시킨 뒤, 배제는 코드가 결정한다.
        "evidence_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "evidence_id": {"type": "string"},
                    "same_subject": {"type": "boolean"},
                    "on_focus": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["evidence_id", "same_subject", "on_focus", "reason"],
                "additionalProperties": False,
            },
        },
        "modality_summary": {"type": "string"},
        "is_sufficient": {"type": "boolean"},
        "insufficient_reason": {"type": "string"},
    },
    "required": [
        "evidence_verdicts", "cards", "modality_summary", "is_sufficient", "insufficient_reason",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """너는 멀티모달 RAG 시스템의 {modality} 근거 해석 디코더다.

역할: 검색된 {modality} 근거를 읽고, 질문에 답하는 데 실제로 쓰일 근거 카드만 뽑는다.
너는 최종 답변을 작성하지 않는다. 답변은 뒤쪽 디코더가 한다.

규칙
0. 먼저 evidence_verdicts 를 채운다. 주어진 근거 **전부**에 대해 두 가지를 판정한다.
   빠뜨리지 마라. 이것은 카드 생성과 별개의 작업이며 먼저 수행한다.
   - same_subject: 이 근거가 질문이 지목한 바로 그 대상을 다루는가.
     질문이 감독 A 를 묻는데 근거가 감독 B 를 설명하면 false 다. 주제어가
     겹쳐도, 대조에 쓸 만해 보여도, 대상이 다르면 false 다.
     질문이 "따져 보라"고 요구하더라도 이 판정은 바뀌지 않는다.
   - on_focus: 질문의 초점과 같은 측면을 다루는가.
     연출 스타일을 묻는데 학력·제작비를 설명하면 false 다.
     같은 대상의 다른 매체·시기를 다루어도 false 다.
1. 근거 1건당 카드는 최대 {max_cards}개다.
2. claim 은 이 근거가 말하는 핵심을 담은 한 문장이다. 뒤 단계에서 중복/충돌 판정의 기준이 되므로 구체적으로 써라.
3. detail 은 답변에 실제 인용될 서술이다. 근거에 실제로 있는 내용만 써라.
4. supports 에는 이 카드가 기여하는 required_operations 항목을 적어라. 해당 없으면 빈 배열.
5. relevance 는 질문 초점 대비 관련도, confidence 는 근거 자체의 명확성이다. 둘 다 0.0~1.0.
6. 근거에 없는 내용을 추론으로 채우지 마라. 부족하면 is_sufficient=false 로 신고하라.
7. 모든 문자열은 한국어로 쓴다.{brevity}"""

# 출력 토큰이 지연에 직접 반영된다. 실측(gemini-3.5-flash, 이미지 1건):
#   기본 421토큰 6.31s -> 간결 291토큰 5.22s (-17%)
#   solar-pro3 텍스트: 197토큰 1.84s -> 164토큰 1.77s (-4%)
# 비전 쪽 이득이 크므로 기본값으로 켠다.
BREVITY_CLAUSE = """
8. 분량을 압축하라. claim 은 한 문장, detail 은 60자 이내, modality_summary 는
   40자 이내로 쓴다. 뒤 단계가 다시 요약하므로 여기서 길게 쓸 이유가 없다."""


def _build_user_prompt(task: ModalityTask, context: DecoderTask, evidence_block: str) -> str:
    lines = [
        f"[원본 질문]\n{context.original_query}",
        f"[정규화 질문]\n{task.query or context.normalized_query}",
    ]
    if context.input_context:
        lines.append(f"[입력 모달리티 상황]\n{context.input_context}")
    if task.focus_features:
        lines.append(f"[이 모달리티에서 볼 초점]\n{', '.join(task.focus_features)}")
    if task.required_operations:
        lines.append(f"[필요한 작업]\n{', '.join(task.required_operations)}")
    if context.identified_entities:
        lines.append(f"[식별된 대상]\n{', '.join(context.identified_entities)}")
    if task.constraints:
        lines.append(f"[제약]\n{', '.join(task.constraints)}")
    lines.append(f"[검색된 {task.modality.value} 근거 {len(task.evidence)}건]\n{evidence_block}")
    lines.append(
        "위 근거를 분석해 근거 카드를 만들어라. "
        "source_evidence_id 는 반드시 위에 제시된 evidence_id 중 하나를 그대로 써라."
    )
    return "\n\n".join(lines)


class ModalityEvidenceDecoder(ABC):
    """모달리티별 디코더 공통 인터페이스."""

    modality: Modality
    max_cards_per_evidence: int = 2
    concise: bool = True
    # 근거별 대상·초점 판정을 코드가 강제할지. 끄면 판정만 받고 배제하지 않는다.
    enforce_subject_filter: bool = True

    def _system_prompt(self, modality: Modality) -> str:
        return SYSTEM_PROMPT.format(
            modality=modality.value,
            max_cards=self.max_cards_per_evidence,
            brevity=BREVITY_CLAUSE if self.concise else "",
        )

    @abstractmethod
    def decode(self, task: ModalityTask, context: DecoderTask) -> ModalityEvidenceResult:
        raise NotImplementedError

    # ------------------------------------------------------------------

    def _empty(self, reason: str, started: float) -> ModalityEvidenceResult:
        return ModalityEvidenceResult(
            modality=self.modality,
            cards=[],
            is_sufficient=False,
            insufficient_reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _parse(
        self,
        raw: Mapping[str, Any],
        task: ModalityTask,
        started: float,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> ModalityEvidenceResult:
        score_by_id = {item.evidence_id: item.score for item in task.evidence}
        valid_ids = set(score_by_id)

        # 근거별 판정에서 대상이 다르다고 나온 것은 코드가 배제한다.
        # 모델에게 "만들지 마라"라고 지시하는 방식은 실패했으므로(질문
        # 프레이밍이 규칙을 이긴다) 판정만 받고 배제는 여기서 결정한다.
        rejected: Dict[str, str] = {}
        if self.enforce_subject_filter:
            for verdict in raw.get("evidence_verdicts") or []:
                if not isinstance(verdict, Mapping):
                    continue
                eid = str(verdict.get("evidence_id", "") or "").strip()
                if eid not in valid_ids:
                    continue
                if not verdict.get("same_subject", True):
                    rejected[eid] = str(verdict.get("reason", "") or "대상 불일치")
                elif not verdict.get("on_focus", True):
                    rejected[eid] = str(verdict.get("reason", "") or "초점 불일치")

        cards: List[EvidenceCard] = []
        for index, entry in enumerate(raw.get("cards") or []):
            if not isinstance(entry, Mapping):
                continue
            claim = str(entry.get("claim", "") or "").strip()
            if not claim:
                continue

            source_id = str(entry.get("source_evidence_id", "") or "").strip()
            if source_id in rejected:
                continue  # 대상·초점 불일치로 판정된 근거는 카드로 만들지 않는다
            if source_id not in valid_ids:
                # 모델이 없는 근거를 지어낸 경우. 버리지 않고 표시만 해서
                # 환각률을 실험에서 셀 수 있게 한다.
                metadata: Dict[str, Any] = {"unmapped_source": source_id}
                source_id = task.evidence[0].evidence_id if task.evidence else "unknown"
            else:
                metadata = {}
            if extra_metadata:
                metadata.update(extra_metadata)

            cards.append(
                EvidenceCard(
                    card_id=f"{task.modality.value}_card_{index}",
                    source_evidence_id=source_id,
                    modality=task.modality,
                    claim=claim,
                    detail=str(entry.get("detail", "") or "").strip(),
                    supports=[str(s) for s in (entry.get("supports") or []) if str(s).strip()],
                    relevance=_clamp(entry.get("relevance")),
                    confidence=_clamp(entry.get("confidence")),
                    retrieval_score=score_by_id.get(source_id, 0.0),
                    metadata=metadata,
                )
            )

        result = ModalityEvidenceResult(
            modality=task.modality,
            cards=cards,
            modality_summary=str(raw.get("modality_summary", "") or "").strip(),
            is_sufficient=bool(raw.get("is_sufficient", True)) and bool(cards),
            insufficient_reason=str(raw.get("insufficient_reason", "") or "").strip(),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if rejected:
            note = "대상·초점 불일치로 배제: " + ", ".join(rejected)
            result.insufficient_reason = (
                f"{result.insufficient_reason} / {note}" if result.insufficient_reason else note
            )
        return result


# ============================================================
# 텍스트
# ============================================================


class TextEvidenceDecoder(ModalityEvidenceDecoder):
    modality = Modality.TEXT

    def __init__(
        self,
        client: StructuredLLMClient,
        max_cards_per_evidence: int = 2,
        max_chars_per_evidence: int = 2000,
        modality: Modality = Modality.TEXT,
        concise: bool = True,
        enforce_subject_filter: bool = True,
    ) -> None:
        self.client = client
        self.max_cards_per_evidence = max_cards_per_evidence
        self.max_chars_per_evidence = max_chars_per_evidence
        self.modality = modality
        self.concise = concise
        self.enforce_subject_filter = enforce_subject_filter

    def decode(self, task: ModalityTask, context: DecoderTask) -> ModalityEvidenceResult:
        started = time.perf_counter()
        if not task.evidence:
            return self._empty("검색된 근거가 없음", started)

        block = "\n\n".join(self._render(item) for item in task.evidence)
        system = self._system_prompt(task.modality)
        try:
            raw = self.client.generate_json(
                system, _build_user_prompt(task, context, block), CARD_SCHEMA
            )
        except LLMError as error:
            result = self._empty(f"디코더 호출 실패: {error}", started)
            result.failed = True
            result.error = str(error)
            return result

        return self._parse(raw, task, started)

    def _render(self, item: EvidenceItem) -> str:
        body = item.content if isinstance(item.content, str) else repr(item.content)
        if len(body) > self.max_chars_per_evidence:
            body = body[: self.max_chars_per_evidence] + " ...(생략)"
        header = f"- evidence_id: {item.evidence_id} (검색점수 {item.score:.4f})"
        hint = item.caption_hint()
        if hint:
            header += f"\n  보조설명: {hint}"
        return f"{header}\n  내용: {body}"


# ============================================================
# 이미지 / 영상
# ============================================================


class VisionEvidenceDecoder(ModalityEvidenceDecoder):
    """이미지와 영상을 같은 코드로 처리한다.

    차이는 AssetLoader 가 흡수한다 (영상 = 네이티브 입력 또는 프레임 샘플).
    """

    def __init__(
        self,
        client: VisionStructuredClient,
        asset_loader: AssetLoader,
        modality: Modality = Modality.IMAGE,
        max_cards_per_evidence: int = 2,
        max_assets: int = 8,
        concise: bool = True,
        enforce_subject_filter: bool = True,
    ) -> None:
        self.client = client
        self.asset_loader = asset_loader
        self.modality = modality
        self.max_cards_per_evidence = max_cards_per_evidence
        self.max_assets = max_assets
        self.concise = concise
        self.enforce_subject_filter = enforce_subject_filter

    def decode(self, task: ModalityTask, context: DecoderTask) -> ModalityEvidenceResult:
        started = time.perf_counter()
        if not task.evidence:
            return self._empty("검색된 근거가 없음", started)

        assets: List[MediaAsset] = []
        lines: List[str] = []
        degraded: List[str] = []

        for item in task.evidence:
            loaded = self.asset_loader.load(item)
            line = f"- evidence_id: {item.evidence_id} (검색점수 {item.score:.4f})"
            name = item.content if isinstance(item.content, str) else str(item.content)
            line += f"\n  파일: {name}"

            if loaded.ok and len(assets) < self.max_assets:
                room = self.max_assets - len(assets)
                chosen = loaded.assets[:room]
                for asset in chosen:
                    asset.label = f"{item.evidence_id} :: {asset.label}"
                    if not asset.text_hint:
                        asset.text_hint = item.caption_hint()
                assets.extend(chosen)
                line += f"\n  첨부: {len(chosen)}건 (아래 자료 참조)"
            else:
                reason = loaded.degraded_reason or "첨부 한도 초과"
                degraded.append(f"{item.evidence_id}: {reason}")
                hint = item.caption_hint()
                line += f"\n  원본 미첨부 ({reason})"
                if hint:
                    line += f"\n  보조설명: {hint}"

            if loaded.degraded_reason:
                line += f"\n  비고: {loaded.degraded_reason}"
            lines.append(line)

        system = self._system_prompt(task.modality)
        user = _build_user_prompt(task, context, "\n\n".join(lines))
        if degraded:
            user += (
                "\n\n[주의] 다음 근거는 원본을 직접 확인하지 못했다. "
                "보조설명만으로 판단하고 confidence 를 낮춰라.\n" + "\n".join(degraded)
            )

        try:
            raw = self.client.generate_json(system, user, assets, CARD_SCHEMA)
        except LLMError as error:
            result = self._empty(f"디코더 호출 실패: {error}", started)
            result.failed = True
            result.error = str(error)
            return result

        extra = {"degraded": True} if degraded else None
        result = self._parse(raw, task, started, extra_metadata=extra)
        if degraded and not result.insufficient_reason:
            result.insufficient_reason = "; ".join(degraded)
        return result


def build_modality_decoders(
    text_client: StructuredLLMClient,
    vision_client: VisionStructuredClient,
    asset_loader: AssetLoader,
    modalities: Sequence[Modality] = (Modality.TEXT, Modality.IMAGE, Modality.VIDEO),
    concise: bool = True,
    enforce_subject_filter: bool = True,
) -> Dict[Modality, ModalityEvidenceDecoder]:
    """모달리티 -> 디코더 매핑. 새 모달리티는 여기 한 줄로 추가된다."""
    decoders: Dict[Modality, ModalityEvidenceDecoder] = {}
    for modality in modalities:
        if modality in (Modality.IMAGE, Modality.VIDEO):
            decoders[modality] = VisionEvidenceDecoder(
                vision_client, asset_loader, modality, concise=concise,
                enforce_subject_filter=enforce_subject_filter,
            )
        else:
            decoders[modality] = TextEvidenceDecoder(
                text_client, modality=modality, concise=concise,
                enforce_subject_filter=enforce_subject_filter,
            )
    return decoders


def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.5) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default
