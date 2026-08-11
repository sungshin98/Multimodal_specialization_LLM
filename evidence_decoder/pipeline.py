"""다층 디코더 파이프라인.

  Adaptive RAG 패킷
      -> PacketAdapter
      -> [모달리티별 근거 디코더]  (병렬)
      -> [근거 통합 계층]
      -> [최종 답변 디코더]        (+ 원본 질문)

속도가 이 연구의 지표이므로 두 개의 우회 경로를 둔다.
1. 모달 디코더 바이패스 - 복잡도 low + 근거 소수면 1층을 건너뛰고
   검색 결과를 그대로 카드로 만든다. LLM 호출 N번이 0번이 된다.
2. 통합 계층 바이패스 - integration.py 내부에서 규칙만으로 끝낸다.

두 경로 모두 config 로 끌 수 있다. 논문 실험의 비교군이 된다.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .assets import AssetLoader
from .clients import StructuredLLMClient, VisionStructuredClient, build_default_clients
from .final_decoder import FinalAnswerDecoder
from .integration import EvidenceIntegrationLayer
from .modality import ModalityEvidenceDecoder, build_modality_decoders
from .packet import PacketAdapter
from .schemas import (
    DecoderOutput,
    DecoderTask,
    DecoderTrace,
    EvidenceCard,
    IntegratedEvidence,
    Level,
    Modality,
    ModalityEvidenceResult,
    ModalityTask,
)


@dataclass
class PipelineConfig:
    max_workers: int = 4
    # 1층 바이패스 조건
    enable_modality_bypass: bool = True
    bypass_max_evidence: int = 3
    bypass_levels: Sequence[Level] = (Level.LOW,)
    # 실험 전용. 이미지/영상까지 강제로 바이패스해 "디코더 계층 없음" 베이스라인을
    # 만든다. 비전 근거는 파일명만 최종 디코더로 가므로 운영에서 켜면 안 된다.
    force_modality_bypass: bool = False
    # 2층
    enable_integration: bool = True
    integration_max_cards: int = 12
    integration_char_budget: int = 4000
    # 기타
    max_cards_per_evidence: int = 2
    # 출력 토큰을 압축해 생성 지연을 줄인다. 실측 비전 -17%.
    concise: bool = True


@dataclass
class MultiLayerDecoderPipeline:
    """모달별 디코더 + 통합 계층 + 최종 답변 디코더를 묶은 실행기."""

    modality_decoders: Dict[Modality, ModalityEvidenceDecoder]
    integration_layer: EvidenceIntegrationLayer
    final_decoder: FinalAnswerDecoder
    config: PipelineConfig = field(default_factory=PipelineConfig)
    adapter: PacketAdapter = field(default_factory=PacketAdapter)

    # ------------------------------------------------------------------

    def run(self, packet: Any) -> DecoderOutput:
        started = time.perf_counter()
        task = self.adapter.adapt(packet)
        trace = DecoderTrace()

        # ---- 1층 --------------------------------------------------
        stage_started = time.perf_counter()
        if self._should_bypass_modality(task):
            results = [self._passthrough(mt) for mt in task.modality_tasks]
            trace.bypassed_modality_stage = True
        else:
            results = self._decode_modalities(task)
            trace.llm_calls += len(results)
        trace.modality_stage_ms = (time.perf_counter() - stage_started) * 1000
        trace.modality_latency_ms = {
            result.modality.value: round(result.latency_ms, 2) for result in results
        }
        trace.cards_before_integration = sum(len(result.cards) for result in results)
        trace.degraded_backends = self._degraded_backends()
        trace.failed_modalities = [
            f"{r.modality.value}:{(r.error or r.insufficient_reason)[:60]}"
            for r in results if r.failed
        ]

        # ---- 2층 --------------------------------------------------
        if self.config.enable_integration:
            integrated = self.integration_layer.integrate(results, task)
            if not integrated.bypassed:
                trace.llm_calls += 1
        else:
            integrated = self._flatten(results)
        trace.integration_ms = integrated.latency_ms
        trace.bypassed_integration = integrated.bypassed
        trace.cards_after_integration = len(integrated.cards)

        # ---- 3층 --------------------------------------------------
        answer = self.final_decoder.decode(integrated, task, results)
        trace.final_ms = answer.latency_ms
        trace.llm_calls += 1
        trace.total_ms = (time.perf_counter() - started) * 1000

        return DecoderOutput(
            original_query=task.original_query,
            final_answer=answer,
            integrated=integrated,
            modality_results=results,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # 1층
    # ------------------------------------------------------------------

    def _decode_modalities(self, task: DecoderTask) -> List[ModalityEvidenceResult]:
        """모달리티별 디코더를 병렬 실행한다.

        모달리티 수만큼 순차 호출하면 latency 가 그대로 누적된다.
        네트워크 대기가 대부분이라 스레드로 충분하다.
        """
        pending = [
            (modality_task, self.modality_decoders.get(modality_task.modality))
            for modality_task in task.modality_tasks
        ]
        runnable = [(mt, decoder) for mt, decoder in pending if decoder is not None]

        results: List[ModalityEvidenceResult] = [
            ModalityEvidenceResult(
                modality=mt.modality,
                is_sufficient=False,
                insufficient_reason=f"{mt.modality.value} 디코더가 등록되지 않음",
            )
            for mt, decoder in pending
            if decoder is None
        ]

        if not runnable:
            return results
        if len(runnable) == 1:
            modality_task, decoder = runnable[0]
            results.append(self._safe_decode(decoder, modality_task, task))
            return results

        workers = min(self.config.max_workers, len(runnable))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._safe_decode, decoder, modality_task, task)
                for modality_task, decoder in runnable
            ]
            results.extend(future.result() for future in futures)
        return results

    @staticmethod
    def _safe_decode(
        decoder: ModalityEvidenceDecoder, modality_task: ModalityTask, task: DecoderTask
    ) -> ModalityEvidenceResult:
        """한 모달리티가 죽어도 나머지 답변 경로는 살린다."""
        try:
            return decoder.decode(modality_task, task)
        except Exception as error:  # noqa: BLE001 - 파이프라인 전체 중단 방지
            return ModalityEvidenceResult(
                modality=modality_task.modality,
                is_sufficient=False,
                insufficient_reason=f"디코더 예외: {error}",
                failed=True,
                error=repr(error),
            )

    def _degraded_backends(self) -> List[str]:
        """폴백으로 내려간 백엔드를 수집한다.

        실측에서 Gemini 무료 할당량이 실행 도중 소진되어(429) 비전 디코더가
        조용히 캡션 폴백으로 내려갔고, 그 사실을 모른 채 측정한 지연 비교가
        통째로 무효가 된 적이 있다. 폴백은 운영에서는 옳지만 실험에서는
        오염이므로 반드시 표면에 드러나야 한다.
        """
        degraded: List[str] = []
        for modality, decoder in self.modality_decoders.items():
            client = getattr(decoder, "client", None)
            if getattr(client, "degraded", False):
                reason = getattr(client, "failure_reason", "")
                degraded.append(f"{modality.value}:{reason[:80]}" if reason else modality.value)
        return degraded

    def _should_bypass_modality(self, task: DecoderTask) -> bool:
        if self.config.force_modality_bypass:
            return True
        if not self.config.enable_modality_bypass:
            return False
        if task.complexity_level not in tuple(self.config.bypass_levels):
            return False
        if task.total_evidence_count() > self.config.bypass_max_evidence:
            return False
        # 원본을 해석해야만 내용을 알 수 있는 모달리티는 건너뛸 수 없다.
        for modality_task in task.modality_tasks:
            if modality_task.modality in (Modality.IMAGE, Modality.VIDEO, Modality.AUDIO):
                return False
        return True

    @staticmethod
    def _passthrough(modality_task: ModalityTask) -> ModalityEvidenceResult:
        """LLM 호출 없이 검색 결과를 카드로 그대로 승격한다."""
        cards = []
        for index, item in enumerate(modality_task.evidence):
            body = item.content if isinstance(item.content, str) else str(item.content)
            cards.append(
                EvidenceCard(
                    card_id=f"{modality_task.modality.value}_card_{index}",
                    source_evidence_id=item.evidence_id,
                    modality=modality_task.modality,
                    claim=body[:120],
                    detail=body,
                    supports=list(modality_task.required_operations),
                    relevance=0.5,
                    confidence=0.5,
                    retrieval_score=item.score,
                    metadata={"passthrough": True},
                )
            )
        return ModalityEvidenceResult(
            modality=modality_task.modality,
            cards=cards,
            modality_summary="(바이패스: 검색 결과를 해석 없이 전달)",
            bypassed=True,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(results: Sequence[ModalityEvidenceResult]) -> IntegratedEvidence:
        """통합 계층을 끈 비교군. 카드를 그대로 이어붙인다."""
        cards = [card for result in results for card in result.cards]
        return IntegratedEvidence(
            cards=cards,
            char_used=sum(card.char_cost() for card in cards),
            bypassed=True,
        )


# ============================================================
# 조립 헬퍼
# ============================================================


def build_pipeline(
    asset_root: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    text_client: Optional[StructuredLLMClient] = None,
    vision_client: Optional[VisionStructuredClient] = None,
    modalities: Sequence[Modality] = (Modality.TEXT, Modality.IMAGE, Modality.VIDEO),
) -> MultiLayerDecoderPipeline:
    """환경변수를 보고 파이프라인을 조립한다.

    text_client / vision_client 를 주면 그것을 쓰고, 없으면
    build_default_clients() 가 사용 가능한 백엔드를 고른다.
    """
    config = config or PipelineConfig()

    if text_client is None or vision_client is None:
        defaults = build_default_clients()
        text_client = text_client or defaults["text"]
        vision_client = vision_client or defaults["vision"]

    loader = AssetLoader(asset_root=asset_root)
    decoders = build_modality_decoders(
        text_client, vision_client, loader, modalities, concise=config.concise
    )
    for decoder in decoders.values():
        decoder.max_cards_per_evidence = config.max_cards_per_evidence

    return MultiLayerDecoderPipeline(
        modality_decoders=decoders,
        integration_layer=EvidenceIntegrationLayer(
            client=text_client,
            max_cards=config.integration_max_cards,
            char_budget=config.integration_char_budget,
        ),
        final_decoder=FinalAnswerDecoder(text_client),
        config=config,
    )
