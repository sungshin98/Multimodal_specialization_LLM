"""네트워크 없이 파이프라인 구조를 검증한다.

    python -m evidence_decoder.test_offline

LLM 응답을 ScriptedClient 로 고정해 스키마/흐름/바이패스만 본다.
"""

from __future__ import annotations

import sys

from .assets import AssetLoader
from .clients import ScriptedClient
from .final_decoder import FinalAnswerDecoder
from .integration import EvidenceIntegrationLayer
from .modality import TextEvidenceDecoder, VisionEvidenceDecoder
from .packet import PacketAdapter
from .pipeline import MultiLayerDecoderPipeline, PipelineConfig
from .schemas import Level, Modality

FULL_PACKET = {
    "schema_version": "3.0",
    "original_query": "포스터와 리뷰를 근거로 이 영화의 분위기를 설명해줘.",
    "normalized_query": "포스터와 리뷰를 근거로 영화 분위기 설명",
    "query_context": {
        "input_context": "어두운 색조의 영화 포스터",
        "identified_entities": ["영화A"],
        "required_operations": ["describe", "compare"],
        "constraints": ["2020년 이후 자료만"],
        "modality_focus": {"text": ["리뷰", "분위기"], "image": ["색조", "구도"]},
        "answer_constraints": ["3문장 이내"],
        "retrieval_action": "retrieve",
        "sub_queries": [
            {"query": "영화A 리뷰 분위기", "modality": "text", "priority": "high"},
            {"query": "영화A 포스터 색조", "modality": "image", "priority": "medium"},
        ],
    },
    "complexity": {"level": "medium", "score": 0.55, "retrieval_demand": {}, "features": {}},
    "retrieval_results": {
        "text": {
            "modality": "text",
            "query": "영화A 리뷰 분위기",
            "candidate_k": 50,
            "final_k": 2,
            "use_reranker": False,
            "uncertainty": {"level": "low"},
            "evidence": [
                {
                    "evidence_id": "text_0",
                    "modality": "text",
                    "score": 0.91,
                    "content": "평론가들은 영화A의 침울한 정서를 반복해 언급했다.",
                    "metadata": {"rank": 1},
                },
                {
                    "evidence_id": "text_1",
                    "modality": "text",
                    "score": 0.72,
                    "content": "관객 리뷰는 후반부의 긴장감을 높이 평가했다.",
                    "metadata": {"rank": 2},
                },
            ],
        },
        "image": {
            "modality": "image",
            "query": "영화A 포스터 색조",
            "candidate_k": 50,
            "final_k": 1,
            "use_reranker": False,
            "uncertainty": {"level": "high"},
            "evidence": [
                {
                    "evidence_id": "image_0",
                    "modality": "image",
                    "score": -0.03,
                    "content": "poster_001.jpg",
                    "metadata": {"rank": 1, "caption": "저채도 청색 배경에 단독 인물"},
                }
            ],
        },
    },
}

DECODER_INPUTS_PACKET = {
    "text": {
        "original_query": "요약해줘",
        "normalized_query": "요약",
        "focus_features": ["핵심"],
        "required_operations": ["summarize"],
        "constraints": [],
        "evidence": [
            {"evidence_id": "text_0", "score": 0.8, "content": "본문", "metadata": {}}
        ],
    }
}

TEXT_RESPONSE = {
    "cards": [
        {
            "source_evidence_id": "text_0",
            "claim": "평론가들이 영화A의 침울한 정서를 반복 언급했다.",
            "detail": "다수 평론에서 침울한 정서가 공통적으로 지적된다.",
            "supports": ["describe"],
            "relevance": 0.9,
            "confidence": 0.85,
        },
        {
            "source_evidence_id": "text_1",
            "claim": "관객 리뷰는 후반부 긴장감을 높이 평가했다.",
            "detail": "후반부 긴장 고조가 호평 요인이다.",
            "supports": ["describe"],
            "relevance": 0.7,
            "confidence": 0.7,
        },
    ],
    "modality_summary": "리뷰는 침울함과 후반 긴장감을 함께 지적한다.",
    "is_sufficient": True,
    "insufficient_reason": "",
}

IMAGE_RESPONSE = {
    "cards": [
        {
            "source_evidence_id": "image_0",
            "claim": "포스터는 저채도 청색 배경에 단독 인물을 배치했다.",
            "detail": "차가운 색조와 고립된 구도가 무거운 분위기를 만든다.",
            "supports": ["describe"],
            "relevance": 0.8,
            "confidence": 0.4,
        }
    ],
    "modality_summary": "포스터는 차갑고 고립된 인상을 준다.",
    "is_sufficient": True,
    "insufficient_reason": "",
}

INTEGRATION_RESPONSE = {
    "kept_card_ids": ["text_card_0", "text_card_1", "image_card_0"],
    "dropped_card_ids": [],
    "duplicate_groups": [],
    "conflicts": [
        {
            "card_ids": ["text_card_1", "image_card_0"],
            "description": "리뷰는 긴장감을, 포스터는 침잠을 강조한다.",
            "resolution": "둘 다 유지하고 답변에서 차이를 밝힌다.",
        }
    ],
    "coverage": [
        {"operation": "describe", "covered": True},
        {"operation": "compare", "covered": False},
    ],
}

ANSWER_RESPONSE = {
    "answer": "영화A는 침울하고 차가운 분위기를 기본으로 합니다.",
    "citations": ["text_card_0", "image_card_0"],
    "unsupported_claims": [],
    "confidence": 0.8,
}


def _check(condition: bool, label: str) -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    return condition


def test_packet_adapter() -> bool:
    print("[1] PacketAdapter - 전체 패킷")
    task = PacketAdapter().adapt(FULL_PACKET)
    ok = True
    ok &= _check(task.answer_constraints == ["3문장 이내"], "answer_constraints 복원 (구형 경로에서 누락되던 값)")
    ok &= _check(task.complexity_level == Level.MEDIUM, "complexity_level 복원")
    ok &= _check(len(task.modality_tasks) == 2, "모달리티 태스크 2개")
    text_task = next(t for t in task.modality_tasks if t.modality == Modality.TEXT)
    ok &= _check(text_task.query == "영화A 리뷰 분위기", "서브질의가 모달리티별로 매핑됨")
    ok &= _check(text_task.focus_features == ["리뷰", "분위기"], "focus_features 매핑")
    image_task = next(t for t in task.modality_tasks if t.modality == Modality.IMAGE)
    ok &= _check(image_task.uncertainty_level == Level.HIGH, "uncertainty level 전달")
    ok &= _check(image_task.evidence[0].caption_hint() == "저채도 청색 배경에 단독 인물", "caption 힌트 추출")

    print("[2] PacketAdapter - build_decoder_inputs() 구형 경로")
    legacy = PacketAdapter().adapt(DECODER_INPUTS_PACKET)
    ok &= _check(len(legacy.modality_tasks) == 1, "구형 패킷도 인식")
    ok &= _check(legacy.original_query == "요약해줘", "원본 질문 복원")
    return bool(ok)


def test_pipeline() -> bool:
    print("[3] 파이프라인 - 정상 경로")
    text_client = ScriptedClient(responses=[TEXT_RESPONSE, INTEGRATION_RESPONSE, ANSWER_RESPONSE])
    vision_client = ScriptedClient(responses=[IMAGE_RESPONSE])
    loader = AssetLoader(asset_root=None)

    pipeline = MultiLayerDecoderPipeline(
        modality_decoders={
            Modality.TEXT: TextEvidenceDecoder(text_client),
            Modality.IMAGE: VisionEvidenceDecoder(vision_client, loader, Modality.IMAGE),
        },
        # FULL_PACKET 의 required_operations 에 compare 가 있어 LLM 통합이 호출된다.
        integration_layer=EvidenceIntegrationLayer(client=text_client),
        final_decoder=FinalAnswerDecoder(text_client),
        config=PipelineConfig(enable_modality_bypass=False),
    )
    output = pipeline.run(FULL_PACKET)

    ok = True
    ok &= _check(len(output.modality_results) == 2, "모달리티 결과 2개")
    ok &= _check(output.trace.cards_before_integration == 3, "통합 전 카드 3개")
    ok &= _check(len(output.integrated.conflicts) == 1, "충돌 1건 탐지")
    ok &= _check(output.integrated.missing_operations == ["compare"], "미충족 작업 식별")
    ok &= _check(output.final_answer.citations == ["text_card_0", "image_card_0"], "인용 card_id 검증 통과")
    ok &= _check(output.trace.llm_calls == 4, f"LLM 호출 4회 (실제 {output.trace.llm_calls})")
    ok &= _check(output.trace.total_ms > 0, "총 지연 계측")
    ok &= _check("image_card_0" in [c.card_id for c in output.integrated.cards], "이미지 카드 유지")
    degraded = output.modality_results
    image_result = next(r for r in degraded if r.modality == Modality.IMAGE)
    ok &= _check(
        image_result.cards[0].metadata.get("degraded") is True,
        "원본 파일 없음 -> degraded 표시 (실험에서 정상 경로와 구분)",
    )
    return bool(ok)


def test_bypass() -> bool:
    print("[4] 파이프라인 - 저복잡도 바이패스")
    packet = {
        "schema_version": "3.0",
        "original_query": "이 문서 요약해줘",
        "normalized_query": "문서 요약",
        "query_context": {"required_operations": ["summarize"], "answer_constraints": []},
        "complexity": {"level": "low", "score": 0.1},
        "retrieval_results": {
            "text": {
                "modality": "text",
                "query": "문서 요약",
                "uncertainty": {"level": "low"},
                "evidence": [
                    {"evidence_id": "text_0", "modality": "text", "score": 0.9,
                     "content": "본문 내용", "metadata": {}}
                ],
            }
        },
    }
    client = ScriptedClient(responses=[ANSWER_RESPONSE])
    pipeline = MultiLayerDecoderPipeline(
        modality_decoders={Modality.TEXT: TextEvidenceDecoder(client)},
        integration_layer=EvidenceIntegrationLayer(client=client),
        final_decoder=FinalAnswerDecoder(client),
        config=PipelineConfig(enable_modality_bypass=True),
    )
    output = pipeline.run(packet)
    ok = True
    ok &= _check(output.trace.bypassed_modality_stage, "1층 바이패스 작동")
    ok &= _check(output.trace.llm_calls == 1, f"LLM 호출 1회로 감소 (실제 {output.trace.llm_calls})")
    ok &= _check(output.integrated.cards[0].metadata.get("passthrough") is True, "패스스루 카드 표시")
    return bool(ok)


def test_integration_rules() -> bool:
    print("[5] 통합 계층 - 규칙 기반 중복제거/예산")
    from .modality import ModalityEvidenceResult  # noqa: F811
    from .schemas import DecoderTask, EvidenceCard

    def card(cid: str, claim: str, modality: Modality, rel: float) -> EvidenceCard:
        return EvidenceCard(
            card_id=cid, source_evidence_id="e", modality=modality,
            claim=claim, detail="d", relevance=rel, confidence=rel,
        )

    results = [
        ModalityEvidenceResult(
            modality=Modality.TEXT,
            cards=[
                card("a", "영화A는 침울한 분위기를 가진다", Modality.TEXT, 0.9),
                card("b", "영화A는 침울한 분위기를 가진다", Modality.TEXT, 0.6),
                card("c", "영화A는 침울한 분위기를 가진다", Modality.IMAGE, 0.5),
            ],
        )
    ]
    layer = EvidenceIntegrationLayer(client=None)
    integrated = layer.integrate(results, DecoderTask(original_query="q", normalized_query="q"))

    ok = True
    ok &= _check("b" in integrated.dropped_card_ids, "동일 모달리티 중복 제거")
    ok &= _check("c" in [c.card_id for c in integrated.cards], "타 모달리티 동일 주장은 보강으로 유지")
    ok &= _check(integrated.bypassed, "client 없으면 규칙만으로 통합")

    layer_small = EvidenceIntegrationLayer(client=None, max_cards=1)
    small = layer_small.integrate(results, DecoderTask(original_query="q", normalized_query="q"))
    ok &= _check(len(small.cards) == 1, "분량 예산으로 카드 절단")
    return bool(ok)


def test_scoring() -> bool:
    """채점기가 심어둔 라벨을 정확히 집계하는지 본다.

    LLM 없이 규칙 지표만 검증한다. 채점 기준 자체가 틀리면 이후 모든
    품질 실험이 무의미해지므로, 정답을 아는 상황을 만들어 대조한다.
    """
    print("[6] 품질 채점기 - 규칙 지표")
    from .datagen import PacketSpec, build_packet
    from .scoring import score_rules
    from .schemas import (
        ConflictNote,
        DecoderOutput,
        DecoderTrace,
        EvidenceCard,
        FinalAnswer,
        IntegratedEvidence,
        ModalityEvidenceResult,
    )

    packet = build_packet(
        PacketSpec(
            group="test",
            level=Level.MEDIUM,
            modalities=[Modality.TEXT],
            gold_per_modality=2,      # text_0, text_1
            duplicates=1,             # text_2
            contradictions=1,         # text_3
            irrelevant_per_modality=2,  # text_4, text_5
            packet_id="score_test",
        )
    )

    def card(cid, source, claim="c"):
        return EvidenceCard(
            card_id=cid, source_evidence_id=source, modality=Modality.TEXT,
            claim=claim, detail="d", relevance=0.8, confidence=0.8,
        )

    # 1층: gold 2건 모두 카드화, 중복·모순도 카드화, 무관 2건 중 1건만 잘못 카드화
    made = [card("c_g0", "text_0"), card("c_g1", "text_1"),
            card("c_dup", "text_2"), card("c_con", "text_3"),
            card("c_irr", "text_4")]
    # 2층: 중복은 제거, 모순은 충돌로 보고하고 유지, 무관 카드는 제거
    kept = [made[0], made[1], made[3]]

    output = DecoderOutput(
        original_query=packet["original_query"],
        final_answer=FinalAnswer(answer="답변", citations=["c_g0"], confidence=0.8),
        integrated=IntegratedEvidence(
            cards=kept,
            dropped_card_ids=["c_dup", "c_irr"],
            duplicate_groups=[["c_g0", "c_dup"]],
            conflicts=[ConflictNote(card_ids=["c_g1", "c_con"], description="모순")],
        ),
        modality_results=[ModalityEvidenceResult(modality=Modality.TEXT, cards=made)],
        trace=DecoderTrace(),
    )

    s = score_rules(output, packet)
    ok = True
    ok &= _check(s.gold_recall == 1.0, f"gold 채택률 1.00 (실제 {s.gold_recall})")
    ok &= _check(s.irrelevant_rejection == 0.5, f"무관 거부율 0.50 — 2건 중 1건 오채택 (실제 {s.irrelevant_rejection})")
    ok &= _check(s.duplicate_removal == 1.0, f"중복 제거율 1.00 (실제 {s.duplicate_removal})")
    ok &= _check(s.conflict_detection == 1.0, f"충돌 탐지율 1.00 (실제 {s.conflict_detection})")
    ok &= _check(s.final_gold_precision == round(2 / 3, 4), f"근거 정밀도 0.67 — 최종 3장 중 gold 2장 (실제 {s.final_gold_precision})")
    ok &= _check(s.citation_validity == 1.0, f"인용 유효율 1.00 (실제 {s.citation_validity})")
    ok &= _check(s.citation_coverage == 0.5, f"인용 범위 0.50 — gold 2장 중 1장만 인용 (실제 {s.citation_coverage})")
    ok &= _check(s.modality_loss == 0, f"모달 소실 0 (실제 {s.modality_loss})")
    ok &= _check(s.source_hallucination == 0.0, f"출처 환각 0.00 (실제 {s.source_hallucination})")

    # 분모가 0인 지표는 0.0 이 아니라 None 이어야 평균이 왜곡되지 않는다
    plain = build_packet(
        PacketSpec(group="t", level=Level.MEDIUM, modalities=[Modality.TEXT],
                   gold_per_modality=1, packet_id="plain")
    )
    s2 = score_rules(output, plain)
    ok &= _check(s2.duplicate_removal is None, "중복 근거가 없으면 중복제거율은 None (0.0 아님)")
    return bool(ok)


def main() -> int:
    print("=" * 60)
    print("다층 디코더 오프라인 테스트 (네트워크 없음)")
    print("=" * 60)
    results = [
        test_packet_adapter(),
        test_pipeline(),
        test_bypass(),
        test_integration_rules(),
        test_scoring(),
    ]
    print("=" * 60)
    if all(results):
        print("전체 통과")
        return 0
    print(f"실패 {results.count(False)}건")
    return 1


if __name__ == "__main__":
    sys.exit(main())
