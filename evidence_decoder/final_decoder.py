"""3층 - 최종 답변 디코더.

통합된 근거 + 원본 질문 + 답변 제약을 받아 사용자 답변을 만든다.

원본 질문을 정규화 질문이 아니라 그대로 넣는 것이 중요하다.
정규화 과정에서 사용자의 말투/의도/세부 요구가 깎이기 때문이다.
(다이어그램의 '원본 질문 연결' 선)
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .clients import LLMError, StructuredLLMClient
from .schemas import (
    DecoderTask,
    FinalAnswer,
    IntegratedEvidence,
    ModalityEvidenceResult,
)

ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
    },
    "required": ["answer", "citations", "unsupported_claims", "confidence"],
    "additionalProperties": False,
}

FINAL_SYSTEM = """너는 멀티모달 RAG 시스템의 최종 답변 디코더다.

통합된 근거 카드만을 사용해 사용자의 원본 질문에 답한다.

규칙
1. answer 는 사용자에게 그대로 보여줄 최종 답변이다. card_id 를 절대 쓰지 마라.
   괄호 안에도, 문장 속에도 쓰지 마라. 출처는 citations 필드로만 표시한다.
   금지: "긴 컷을 사용했다(text_card_0)."
   금지: "text_card_1과 text_card_4는 서로 모순된다."
   허용: "긴 컷을 사용했다."
   허용: "편집 속도에 대해서는 상반된 분석이 있다."
   근거끼리 대조할 때는 card_id 대신 내용으로 지칭하라.
2. 근거 카드에 없는 사실을 만들어내지 마라. 근거가 부족하면 부족하다고 말하라.
3. citations 에는 답변에 실제로 사용한 card_id 를 적어라.
4. 근거 없이 쓸 수밖에 없었던 문장이 있으면 unsupported_claims 에 그대로 옮겨라. 없으면 빈 배열.
5. 근거 간 충돌이 보고되면 한쪽을 숨기지 말고 답변에서 차이를 밝혀라.
6. confidence 는 답변 전체의 신뢰도다 (0.0~1.0).
7. 답변은 한국어로 쓴다."""

DEFAULT_ANSWER_CONSTRAINTS = ["질문에 직접 답하는 문장으로 시작할 것", "불필요한 서론을 붙이지 말 것"]


class FinalAnswerDecoder:
    def __init__(
        self,
        client: StructuredLLMClient,
        default_constraints: Optional[List[str]] = None,
    ) -> None:
        self.client = client
        self.default_constraints = (
            list(default_constraints) if default_constraints is not None
            else list(DEFAULT_ANSWER_CONSTRAINTS)
        )

    def decode(
        self,
        integrated: IntegratedEvidence,
        context: DecoderTask,
        modality_results: Optional[Sequence[ModalityEvidenceResult]] = None,
    ) -> FinalAnswer:
        started = time.perf_counter()

        if not integrated.cards:
            return FinalAnswer(
                answer=(
                    "검색된 근거에서 질문에 답할 만한 내용을 찾지 못했습니다. "
                    "질문을 더 구체화하거나 다른 자료를 제공해 주세요."
                ),
                confidence=0.0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            raw = self.client.generate_json(
                FINAL_SYSTEM,
                self._user_prompt(integrated, context, modality_results or ()),
                ANSWER_SCHEMA,
            )
        except LLMError as error:
            return FinalAnswer(
                answer=f"답변 생성에 실패했습니다: {error}",
                confidence=0.0,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return self._parse(raw, integrated, started)

    # ------------------------------------------------------------------

    def _user_prompt(
        self,
        integrated: IntegratedEvidence,
        context: DecoderTask,
        modality_results: Sequence[ModalityEvidenceResult] = (),
    ) -> str:
        lines = [f"[원본 질문]\n{context.original_query}"]

        if context.input_context:
            lines.append(f"[사용자가 함께 준 입력]\n{context.input_context}")
        if context.identified_entities:
            lines.append(f"[질문에서 식별된 대상]\n{', '.join(context.identified_entities)}")
        if context.required_operations:
            lines.append(f"[수행할 작업]\n{', '.join(context.required_operations)}")

        constraints = context.answer_constraints or self.default_constraints
        lines.append(f"[답변 제약]\n" + "\n".join(f"- {c}" for c in constraints))

        if context.constraints:
            lines.append(f"[질문 제약]\n{', '.join(context.constraints)}")

        # 모달리티별 디코더의 전체 소견. 개별 카드로 쪼개지면 사라지는
        # "이 모달리티가 전반적으로 무엇을 말하는가" 를 최종 답변에 전달한다.
        summaries = [
            f"- {result.modality.value}: {result.modality_summary}"
            for result in modality_results
            if result.modality_summary
        ]
        if summaries:
            lines.append("[모달리티별 종합 소견]\n" + "\n".join(summaries))

        card_lines = []
        for card in integrated.cards:
            flags = []
            if card.metadata.get("degraded"):
                flags.append("원본 미확인")
            if card.metadata.get("passthrough"):
                flags.append("해석 생략")
            suffix = f", {'/'.join(flags)}" if flags else ""
            card_lines.append(
                f"- card_id: {card.card_id} (출처 {card.modality.value}/"
                f"{card.source_evidence_id}, 확신도 {card.confidence:.2f}{suffix})\n"
                f"  주장: {card.claim}\n"
                f"  상세: {card.detail}"
            )
        lines.append(f"[통합 근거 {len(integrated.cards)}개]\n" + "\n".join(card_lines))

        if any(card.metadata.get("degraded") for card in integrated.cards):
            lines.append(
                "[주의] '원본 미확인' 으로 표시된 근거는 이미지/영상 원본을 직접 보지 못하고 "
                "설명 텍스트만으로 해석한 것이다. 그 근거에 의존한 서술은 단정하지 말고 "
                "답변에서 한계를 밝혀라."
            )

        insufficient = [
            f"- {result.modality.value}: {result.insufficient_reason}"
            for result in modality_results
            if not result.is_sufficient and result.insufficient_reason
        ]
        if insufficient:
            lines.append("[모달리티별 근거 부족 신고]\n" + "\n".join(insufficient))

        if integrated.conflicts:
            conflict_lines = [
                f"- {note.description} (관련 카드: {', '.join(note.card_ids)})"
                + (f" / 처리: {note.resolution}" if note.resolution else "")
                for note in integrated.conflicts
            ]
            lines.append("[근거 충돌]\n" + "\n".join(conflict_lines))

        if integrated.missing_operations:
            lines.append(
                "[근거가 부족한 작업]\n"
                + ", ".join(integrated.missing_operations)
                + "\n이 부분은 단정하지 말고 한계를 밝혀라."
            )

        return "\n\n".join(lines)

    def _parse(
        self, raw: Mapping[str, Any], integrated: IntegratedEvidence, started: float
    ) -> FinalAnswer:
        valid_ids = {card.card_id for card in integrated.cards}
        citations = [
            str(cid) for cid in (raw.get("citations") or []) if str(cid) in valid_ids
        ]
        return FinalAnswer(
            answer=_strip_card_ids(str(raw.get("answer", "") or "").strip()),
            citations=citations,
            unsupported_claims=[
                str(claim) for claim in (raw.get("unsupported_claims") or []) if str(claim).strip()
            ],
            confidence=_clamp(raw.get("confidence")),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


_CARD_ID = r"(?:text|image|video|audio|table)_card_\d+"
# "(text_card_0)", "[text_card_0, text_card_1]" 같은 괄호 인용
_PAREN_CITE = re.compile(r"\s*[\(\[]\s*" + _CARD_ID + r"(?:\s*[,、]\s*" + _CARD_ID + r")*\s*[\)\]]")
# "text_card_1과 text_card_4는" 처럼 문장 속에 박힌 경우.
# \b 경계를 쓰면 안 된다 - 한글 조사가 단어 문자라 "text_card_1과" 에서
# 경계가 성립하지 않아 매칭에 실패한다. 패턴 자체가 충분히 특정적이다.
_BARE_CITE = re.compile(_CARD_ID)

# 치환어 "한 근거"는 받침이 없으므로 뒤따르는 조사를 바꿔야 한다.
_PARTICLE_MAP = {"과": "와", "은": "는", "이": "가", "을": "를", "으로": "로"}
_PARTICLE = re.compile("한 근거(" + "|".join(sorted(_PARTICLE_MAP, key=len, reverse=True)) + ")")


def _strip_card_ids(answer: str) -> str:
    """답변 본문에서 내부 식별자를 제거한다.

    프롬프트로 금지해도 지켜지지 않는다(실측 ID누출 0.33건/실행). 사용자에게
    보여줄 문자열이므로 규칙으로 확실히 막는다. 출처는 citations 필드에 남는다.

    괄호 인용은 통째로 지우고, 문장 속에 박힌 것은 "한 근거"로 바꿔
    문법이 깨지지 않게 한다.
    """
    if not answer:
        return answer
    cleaned = _PAREN_CITE.sub("", answer)
    cleaned = _BARE_CITE.sub("한 근거", cleaned)
    # 치환 결과 뒤에 오는 조사를 받침 없는 "거"에 맞춘다.
    # (text_card_1'과' -> 한 근거'와')
    cleaned = _PARTICLE.sub(lambda m: "한 근거" + _PARTICLE_MAP[m.group(1)], cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return cleaned.strip()


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
