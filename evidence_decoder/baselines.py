"""외부 베이스라인.

제안 구조의 ablation(raw / no-integ / full)은 모두 같은 파이프라인 내부의
구성이므로, "무엇 대비 개선인가"에 답하려면 파이프라인 밖의 기준이 필요하다.

PlainRAGPipeline 은 검색 결과를 별도의 해석 없이 프롬프트에 이어 붙여
한 번의 호출로 답변을 생성하는 통상적인 RAG 구성이다. 근거 카드, 인용,
충돌 처리, 분량 예산이 모두 없다는 점에서 제안 구조와 대비된다.

주의: 이 구성에는 명시적 근거 집합이 존재하지 않으므로 근거 정밀도와
인용 관련 지표는 정의되지 않는다(채점기에서 None 으로 남는다). 비교는
답변 수준 지표(요지충족, 오염서술)와 지연에서 이루어진다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .clients import LLMError, StructuredLLMClient
from .packet import PacketAdapter
from .schemas import (
    DecoderOutput,
    DecoderTrace,
    FinalAnswer,
    IntegratedEvidence,
)

PLAIN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

PLAIN_SYSTEM = """다음 문서들을 참고하여 사용자의 질문에 답하시오.
문서에 없는 내용을 지어내지 말고, 한국어로 답한다."""


@dataclass
class PlainRAGPipeline:
    """검색 결과를 그대로 이어 붙여 단일 호출로 답변하는 통상적 RAG."""

    client: StructuredLLMClient
    adapter: PacketAdapter = field(default_factory=PacketAdapter)
    max_chars_per_doc: int = 2000

    def run(self, packet: Any) -> DecoderOutput:
        started = time.perf_counter()
        task = self.adapter.adapt(packet)

        docs: List[str] = []
        for modality_task in task.modality_tasks:
            for item in modality_task.evidence:
                body = item.content if isinstance(item.content, str) else str(item.content)
                docs.append(body[: self.max_chars_per_doc])

        user = (
            f"[질문]\n{task.original_query}\n\n"
            f"[문서 {len(docs)}건]\n"
            + "\n\n".join(f"({i + 1}) {d}" for i, d in enumerate(docs))
        )

        answer_text = ""
        try:
            raw = self.client.generate_json(PLAIN_SYSTEM, user, PLAIN_SCHEMA)
            answer_text = str(raw.get("answer", "") or "").strip()
        except LLMError as error:
            answer_text = f"답변 생성 실패: {error}"

        elapsed = (time.perf_counter() - started) * 1000
        return DecoderOutput(
            original_query=task.original_query,
            final_answer=FinalAnswer(answer=answer_text, latency_ms=elapsed),
            # 명시적 근거 집합이 없다. 빈 집합으로 두어 근거 수준 지표가
            # 계산되지 않도록 한다(0 으로 채우면 결과를 왜곡한다).
            integrated=IntegratedEvidence(),
            modality_results=[],
            trace=DecoderTrace(
                final_ms=elapsed,
                total_ms=elapsed,
                llm_calls=1,
                cards_before_integration=len(docs),
                cards_after_integration=len(docs),
            ),
        )
