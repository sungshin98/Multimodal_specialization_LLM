"""합성 패킷 생성기.

앞단 adaptive_rag 의 코퍼스와 인코더가 아직 더미라 실제 패킷을 받을 수 없다.
그래서 AdaptiveRAGOutput 과 동일한 형태(schema 3.0)의 패킷을 직접 만든다.

두 가지 용도를 겸한다.
1. 속도 실험 - 복잡도 / 모달조합 / 근거개수 축을 훑어 지연 구조를 본다.
2. 품질 실험 - 근거마다 역할 라벨을 심는다. 그 라벨이 채점 기준이 된다.

근거 역할 (EvidenceRole)
    gold          정답에 필요한 근거
    irrelevant    검색기가 섞어오는 잡음        -> 근거 희석 억제 측정
    duplicate     gold 와 같은 사실, 다른 표현  -> 중복 제거율 측정
    contradictory gold 와 반대되는 진술         -> 충돌 탐지 재현율 측정
    reinforcing   다른 모달리티의 같은 사실      -> 모달 간 오탐 방지 측정

소재는 scenarios.py 에 주제별로 분리되어 있다. 한 시나리오만 쓰면 결론이
그 주제의 특성에 좌우되므로, 스윕은 기본적으로 전 시나리오를 돈다.

실행
    python -m evidence_decoder.datagen --sweep latency --out packets_latency.json
    python -m evidence_decoder.datagen --sweep dilution --scenarios film fish battery
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from . import scenarios as SC
from .schemas import Level, Modality


class EvidenceRole(str, Enum):
    GOLD = "gold"
    IRRELEVANT = "irrelevant"
    DUPLICATE = "duplicate"
    CONTRADICTORY = "contradictory"
    REINFORCING = "reinforcing"


@dataclass
class PacketSpec:
    """패킷 1개의 구성 조건."""

    group: str
    level: Level
    modalities: Sequence[Modality]
    scenario: str = SC.DEFAULT_SCENARIO.id
    gold_per_modality: int = 2
    irrelevant_per_modality: int = 0
    # True 면 주제를 공유하는 hard negative 를 쓴다. 주제가 아예 다른 잡음은
    # 1층이 전부 걸러내 실험 변별력이 사라진다(무관거부율 모든 구간 1.00).
    hard_negatives: bool = True
    duplicates: int = 0
    contradictions: int = 0
    reinforcing: int = 0
    # True 면 이미지·영상 근거가 시나리오의 실제 파일을 가리킨다.
    # 비전 경로를 캡션 폴백이 아닌 정상 경로로 측정할 때 쓴다.
    use_real_assets: bool = False
    packet_id: str = ""


def _pick(pool: Sequence[str], index: int) -> str:
    return pool[index % len(pool)] if pool else ""


def _text_evidence(spec: PacketSpec, sc: SC.Scenario) -> List[Dict[str, Any]]:
    entries: List[tuple] = []
    for i in range(spec.gold_per_modality):
        entries.append((EvidenceRole.GOLD, _pick(sc.gold_texts, i)))
    for i in range(spec.duplicates):
        entries.append((EvidenceRole.DUPLICATE, _pick(sc.duplicate_texts, i)))
    for i in range(spec.contradictions):
        entries.append((EvidenceRole.CONTRADICTORY, _pick(sc.contradictory_texts, i)))
    pool = sc.hard_negatives if spec.hard_negatives else sc.easy_negatives
    for i in range(spec.irrelevant_per_modality):
        entries.append((EvidenceRole.IRRELEVANT, _pick(pool, i)))

    return [
        {
            "evidence_id": f"text_{index}",
            "modality": "text",
            # 검색 점수를 역할 순서대로 단조 감소시켜, 실제 검색기가 잡음을
            # 하위에 두는 상황을 모사한다.
            "score": round(0.95 - 0.07 * index, 4),
            "content": content,
            "metadata": {"rank": index + 1, "_role": role.value},
        }
        for index, (role, content) in enumerate(entries)
        if content
    ]


def _media_evidence(
    spec: PacketSpec, modality: Modality, sc: SC.Scenario
) -> List[Dict[str, Any]]:
    captions = sc.image_captions if modality == Modality.IMAGE else sc.video_captions
    gold_pool = captions.get(EvidenceRole.GOLD.value, [])

    entries: List[tuple] = []
    for i in range(spec.gold_per_modality):
        entries.append((EvidenceRole.GOLD, _pick(gold_pool, i)))
    for i in range(spec.reinforcing):
        entries.append(
            (EvidenceRole.REINFORCING, _pick(captions.get(EvidenceRole.REINFORCING.value) or gold_pool, i))
        )
    irrelevant_pool = captions.get(EvidenceRole.IRRELEVANT.value, [])
    for i in range(spec.irrelevant_per_modality):
        if irrelevant_pool:
            entries.append((EvidenceRole.IRRELEVANT, _pick(irrelevant_pool, i)))

    real_files = sc.asset_files.get(modality.value, []) if spec.use_real_assets else []
    ext = "jpg" if modality == Modality.IMAGE else "mp4"

    evidence = []
    for index, (role, caption) in enumerate(entries):
        if not caption:
            continue
        if real_files:
            content = real_files[index % len(real_files)]
            # 실제 파일이 있으면 캡션을 넣지 않는다. 캡션이 있으면 비전 모델이
            # 원본 대신 캡션에 기대게 되어, 비전 경로를 측정하는 의미가 사라진다.
            metadata: Dict[str, Any] = {"rank": index + 1, "_role": role.value}
        else:
            content = f"{modality.value}_{index:03d}.{ext}"
            # 원본이 없으므로 caption 이 폴백 경로의 입력이 된다.
            metadata = {"rank": index + 1, "caption": caption, "_role": role.value}
        evidence.append(
            {
                "evidence_id": f"{modality.value}_{index}",
                "modality": modality.value,
                "score": round(-0.03 - 0.05 * index, 4),
                "content": content,
                "metadata": metadata,
            }
        )
    return evidence


def build_packet(spec: PacketSpec) -> Dict[str, Any]:
    sc = SC.get(spec.scenario)
    question, operations, answer_constraints = sc.questions[spec.level]

    retrieval_results: Dict[str, Any] = {}
    sub_queries: List[Dict[str, Any]] = []
    modality_focus: Dict[str, List[str]] = {}

    for modality in spec.modalities:
        if modality == Modality.TEXT:
            evidence = _text_evidence(spec, sc)
        elif modality in (Modality.IMAGE, Modality.VIDEO):
            evidence = _media_evidence(spec, modality, sc)
        else:
            continue
        if not evidence:
            continue

        focus = sc.focus.get(modality.value, [])
        modality_focus[modality.value] = focus
        search_query = f"{sc.topic} {' '.join(focus[:2])}"
        sub_queries.append(
            {"query": search_query, "modality": modality.value, "priority": "high"}
        )
        retrieval_results[modality.value] = {
            "modality": modality.value,
            "query": search_query,
            "candidate_k": 50,
            "final_k": len(evidence),
            "use_reranker": False,
            "uncertainty": {
                "top1_score": evidence[0]["score"],
                "top1_top2_gap": 0.05,
                "score_variance": 0.01,
                "shannon_entropy": 1.0,
                "normalized_entropy": 0.9,
                "level": "medium",
            },
            "evidence": evidence,
        }

    def count(role: EvidenceRole) -> int:
        return sum(
            1
            for result in retrieval_results.values()
            for item in result["evidence"]
            if item["metadata"]["_role"] == role.value
        )

    negatives = sc.hard_negatives if spec.hard_negatives else sc.easy_negatives
    return {
        "schema_version": "3.0",
        "original_query": question,
        "normalized_query": question,
        "query_context": {
            "input_context": "",
            "identified_entities": list(sc.entities),
            "required_operations": operations,
            "constraints": [],
            "modality_focus": modality_focus,
            "answer_constraints": answer_constraints,
            "retrieval_action": "retrieve",
            "sub_queries": sub_queries,
        },
        "complexity": {
            "level": spec.level.value,
            "score": {"low": 0.2, "medium": 0.55, "high": 0.85}[spec.level.value],
            "retrieval_demand": {m: spec.level.value for m in retrieval_results},
            "features": {},
            "reasons": [],
        },
        "retrieval_results": retrieval_results,
        # 파이프라인은 무시하고 실험 스크립트와 채점기만 읽는 필드.
        "_meta": {
            "packet_id": spec.packet_id or f"{spec.scenario}_{spec.group}",
            "group": spec.group,
            "scenario": spec.scenario,
            "level": spec.level.value,
            "modalities": [m.value for m in spec.modalities],
            "evidence_total": sum(len(r["evidence"]) for r in retrieval_results.values()),
            "gold_total": count(EvidenceRole.GOLD),
            "irrelevant_total": count(EvidenceRole.IRRELEVANT),
            "duplicate_total": count(EvidenceRole.DUPLICATE),
            "contradiction_total": count(EvidenceRole.CONTRADICTORY),
            "hard_negatives": spec.hard_negatives,
            "use_real_assets": spec.use_real_assets,
            # 채점 기준
            "key_points": sc.gold_key_points[: spec.gold_per_modality]
            if Modality.TEXT in spec.modalities
            else [],
            "irrelevant_topics": [
                t.split(".")[0][:60] for t in negatives[: spec.irrelevant_per_modality]
            ],
            "has_conflict": spec.contradictions > 0,
        },
    }


# ============================================================
# 스윕 - 기본적으로 전 시나리오를 돈다
# ============================================================


def _scenarios(ids: Optional[Sequence[str]]) -> Sequence[str]:
    return tuple(ids) if ids else SC.all_ids()


def sweep_latency(ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """지연 구조 파악용. 복잡도 x 모달조합 x 근거개수."""
    combos = [
        ("T", [Modality.TEXT]),
        ("TI", [Modality.TEXT, Modality.IMAGE]),
        ("TIV", [Modality.TEXT, Modality.IMAGE, Modality.VIDEO]),
    ]
    packets = []
    for scenario in _scenarios(ids):
        for level in (Level.LOW, Level.MEDIUM, Level.HIGH):
            for tag, modalities in combos:
                for gold in (1, 3):
                    packets.append(
                        build_packet(
                            PacketSpec(
                                group=f"{level.value}/{tag}/g{gold}",
                                level=level,
                                modalities=modalities,
                                scenario=scenario,
                                gold_per_modality=gold,
                                packet_id=f"lat_{scenario}_{level.value}_{tag}_g{gold}",
                            )
                        )
                    )
    return packets


def sweep_dilution(ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """근거 희석 실험. 무관 근거 비율을 0 -> 66% 로 올린다. 텍스트 전용."""
    return [
        build_packet(
            PacketSpec(
                group=f"dilution/{int(n / (3 + n) * 100)}%",
                level=Level.MEDIUM,
                modalities=[Modality.TEXT],
                scenario=scenario,
                gold_per_modality=3,
                irrelevant_per_modality=n,
                packet_id=f"dil_{scenario}_{n}",
            )
        )
        for scenario in _scenarios(ids)
        for n in (0, 2, 4, 6)
    ]


def sweep_dilution_easy(ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """주제가 아예 다른 쉬운 잡음. hard negative 와의 대조군."""
    return [
        build_packet(
            PacketSpec(
                group=f"dilution-easy/{int(n / (3 + n) * 100)}%",
                level=Level.MEDIUM,
                modalities=[Modality.TEXT],
                scenario=scenario,
                gold_per_modality=3,
                irrelevant_per_modality=n,
                hard_negatives=False,
                packet_id=f"dileasy_{scenario}_{n}",
            )
        )
        for scenario in _scenarios(ids)
        for n in (0, 2, 4, 6)
    ]


def sweep_integration(ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """통합 계층 검증. 중복/충돌/잡음을 명시적으로 심는다. 텍스트 전용."""
    packets = []
    for scenario in _scenarios(ids):
        packets.append(
            build_packet(
                PacketSpec(group="integ/duplicate", level=Level.MEDIUM, modalities=[Modality.TEXT],
                           scenario=scenario, gold_per_modality=3, duplicates=1,
                           packet_id=f"integ_{scenario}_dup")
            )
        )
        packets.append(
            build_packet(
                PacketSpec(group="integ/conflict", level=Level.HIGH, modalities=[Modality.TEXT],
                           scenario=scenario, gold_per_modality=3, contradictions=1,
                           packet_id=f"integ_{scenario}_conf")
            )
        )
        packets.append(
            build_packet(
                PacketSpec(group="integ/both", level=Level.HIGH, modalities=[Modality.TEXT],
                           scenario=scenario, gold_per_modality=3, duplicates=1,
                           contradictions=1, irrelevant_per_modality=2,
                           packet_id=f"integ_{scenario}_both")
            )
        )
    return packets


def sweep_vision(ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """비전 경로 측정용. 실제 이미지·영상 파일을 가리키는 패킷만 만든다.

    asset_files 를 가진 시나리오에서만 생성된다. --asset-root 로 파일 위치를 준다.
    """
    packets = []
    for scenario in _scenarios(ids):
        sc = SC.get(scenario)
        if not sc.asset_files:
            continue
        for tag, modalities in [
            ("T", [Modality.TEXT]),
            ("TI", [Modality.TEXT, Modality.IMAGE]),
            ("TIV", [Modality.TEXT, Modality.IMAGE, Modality.VIDEO]),
        ]:
            packets.append(
                build_packet(
                    PacketSpec(
                        group=f"vision/{tag}",
                        level=Level.MEDIUM,
                        modalities=modalities,
                        scenario=scenario,
                        gold_per_modality=1 if tag != "T" else 3,
                        use_real_assets=True,
                        packet_id=f"vis_{scenario}_{tag}",
                    )
                )
            )
    return packets


SWEEPS = {
    "latency": sweep_latency,
    "dilution": sweep_dilution,
    "dilution-easy": sweep_dilution_easy,
    "integration": sweep_integration,
    "vision": sweep_vision,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="합성 RAG 패킷 생성")
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default="latency")
    parser.add_argument("--scenarios", nargs="*", help=f"기본: 전체 {list(SC.all_ids())}")
    parser.add_argument("--out", help="저장 경로 (없으면 stdout)")
    args = parser.parse_args(argv)

    packets = SWEEPS[args.sweep](args.scenarios)
    payload = json.dumps(packets, ensure_ascii=False, indent=2)

    if not args.out:
        print(payload)
        return 0

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(payload)
    print(f"{args.sweep}: 패킷 {len(packets)}개 -> {args.out}")
    for packet in packets:
        meta = packet["_meta"]
        print(
            f"  {meta['packet_id']:26s} {meta['group']:20s} "
            f"근거 {meta['evidence_total']:2d} "
            f"(gold {meta['gold_total']}, 무관 {meta['irrelevant_total']}, "
            f"중복 {meta['duplicate_total']}, 충돌 {meta['contradiction_total']})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
