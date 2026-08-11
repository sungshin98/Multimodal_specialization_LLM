"""품질 채점기.

datagen.py 가 근거마다 심어둔 역할 라벨(_role)을 기준으로 각 계층을 채점한다.

설계 원칙: LLM 심판을 최소로 쓴다.
- 1층·2층 지표는 전부 규칙으로 계산한다. 라벨이 정답이므로 판정에 모호함이 없고,
  재현 가능하며, "LLM 이 만든 것을 LLM 이 채점하는" 순환을 피한다.
- LLM 심판은 답변 정확도에만 쓴다. 자연어 답변이 정답 요지를 담았는지는
  문자열 매칭으로 판정할 수 없기 때문이다.

지표
  [1층] gold_recall            gold 근거 중 카드가 만들어진 비율 (높을수록 좋음)
        irrelevant_rejection   무관 근거 중 카드를 만들지 않은 비율 (높을수록 좋음)
        source_hallucination   없는 evidence_id 를 참조한 카드 비율 (낮을수록 좋음)

  [2층] duplicate_removal      중복 근거가 최종 근거에 남지 않은 비율 (결과)
        duplicate_recognized   중복으로 인지해 묶은 비율 (2층이 표방하는 기능)
        conflict_detection     모순 근거 카드가 충돌로 보고된 비율 (높을수록 좋음)
        reinforcing_preserved  타 모달 보강 카드가 살아남은 비율 (높을수록 좋음)
        final_gold_precision   최종 카드 중 gold 출처 비율 = 근거 희석 억제
        modality_loss          입력에 있었으나 최종 카드에서 사라진 모달리티 수

  [3층] citation_validity      인용이 실제 카드를 가리키는 비율
        citation_coverage      최종 gold 카드 중 인용된 비율
        key_point_coverage     정답 요지 중 답변이 담은 비율 (LLM 심판)
        contamination          무관 내용을 사실로 주장한 건수 (LLM 심판)
        conflict_disclosed     근거 충돌을 답변에서 밝혔는가 (LLM 심판)
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .clients import LLMError, StructuredLLMClient
from .schemas import DecoderOutput, Modality

_CARD_ID_RE = re.compile(r"\b(?:text|image|video|audio|table)_card_\d+\b")


def _norm(text: str) -> str:
    """심판이 돌려준 요지 문자열을 원본과 대조하기 위한 정규화."""
    return re.sub(r"\s+", " ", text).strip().lower()


GOLD = "gold"
IRRELEVANT = "irrelevant"
DUPLICATE = "duplicate"
CONTRADICTORY = "contradictory"
REINFORCING = "reinforcing"


@dataclass
class QualityScore:
    packet_id: str = ""
    arm: str = ""
    # 1층
    gold_recall: Optional[float] = None
    irrelevant_rejection: Optional[float] = None
    source_hallucination: Optional[float] = None
    # 2층
    duplicate_removal: Optional[float] = None
    duplicate_recognized: Optional[float] = None
    conflict_detection: Optional[float] = None
    reinforcing_preserved: Optional[float] = None
    final_gold_precision: Optional[float] = None
    modality_loss: int = 0
    # 3층
    citation_validity: Optional[float] = None
    citation_coverage: Optional[float] = None
    citation_precision: Optional[float] = None
    key_point_coverage: Optional[float] = None
    contamination: Optional[int] = None
    conflict_disclosed: Optional[bool] = None
    unsupported_claims: int = 0
    card_id_leak: int = 0
    # 메타
    degraded: bool = False
    judge_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 규칙 기반 채점
# ============================================================


def _roles(packet: Mapping[str, Any]) -> Dict[str, str]:
    """evidence_id -> 역할 라벨."""
    roles: Dict[str, str] = {}
    for result in (packet.get("retrieval_results") or {}).values():
        for item in (result.get("evidence") or []):
            role = (item.get("metadata") or {}).get("_role")
            if role:
                roles[str(item.get("evidence_id"))] = str(role)
    return roles


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    """분모가 0이면 None. 0.0 으로 두면 평균이 왜곡된다."""
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def score_rules(output: DecoderOutput, packet: Mapping[str, Any]) -> QualityScore:
    meta = packet.get("_meta") or {}
    roles = _roles(packet)
    score = QualityScore(
        packet_id=str(meta.get("packet_id", "")),
        degraded=bool(output.trace.degraded_backends),
        unsupported_claims=len(output.final_answer.unsupported_claims),
    )

    all_cards = [card for result in output.modality_results for card in result.cards]
    carded_sources = {card.source_evidence_id for card in all_cards}

    # 근거 카드를 만들지 않는 구성(외부 베이스라인 등)에서는 1층 지표가
    # 정의되지 않는다. 빈 카드 집합을 그대로 계산하면 gold채택 0.00,
    # 무관거부 1.00 이라는 무의미한 값이 나와 비교를 왜곡한다.
    card_based = bool(output.modality_results)

    # ---- 1층 -------------------------------------------------
    gold_ids = [eid for eid, role in roles.items() if role == GOLD]
    irrelevant_ids = [eid for eid, role in roles.items() if role == IRRELEVANT]

    if card_based:
        score.gold_recall = _ratio(
            sum(1 for eid in gold_ids if eid in carded_sources), len(gold_ids)
        )
        score.irrelevant_rejection = _ratio(
            sum(1 for eid in irrelevant_ids if eid not in carded_sources), len(irrelevant_ids)
        )
        score.source_hallucination = _ratio(
            sum(1 for card in all_cards if card.metadata.get("unmapped_source")), len(all_cards)
        )

    # ---- 2층 -------------------------------------------------
    integrated = output.integrated
    kept_ids = {card.card_id for card in integrated.cards}
    grouped_ids = {cid for group in integrated.duplicate_groups for cid in group}
    conflict_ids = {cid for note in integrated.conflicts for cid in note.card_ids}

    def cards_from(role: str):
        return [card for card in all_cards if roles.get(card.source_evidence_id) == role]

    duplicate_cards = cards_from(DUPLICATE)
    # 두 가지를 나눠 센다. 이전에는 하나로 묶어 재다가 "다른 이유로 버려진 것"을
    # 중복 처리 성공으로 오인했다.
    #   removal    최종 근거에 남지 않았는가 (결과)
    #   recognized 중복으로 인지해 묶었는가 (2층이 표방하는 기능)
    score.duplicate_removal = _ratio(
        sum(1 for card in duplicate_cards if card.card_id not in kept_ids),
        len(duplicate_cards),
    )
    score.duplicate_recognized = _ratio(
        sum(1 for card in duplicate_cards if card.card_id in grouped_ids),
        len(duplicate_cards),
    )

    contradictory_cards = cards_from(CONTRADICTORY)
    score.conflict_detection = _ratio(
        sum(1 for card in contradictory_cards if card.card_id in conflict_ids),
        len(contradictory_cards),
    )

    reinforcing_cards = cards_from(REINFORCING)
    score.reinforcing_preserved = _ratio(
        sum(1 for card in reinforcing_cards if card.card_id in kept_ids), len(reinforcing_cards)
    )

    # 근거 희석 억제: 최종 근거 집합에서 gold 가 차지하는 비율
    if integrated.cards:
        gold_kept = sum(
            1 for card in integrated.cards if roles.get(card.source_evidence_id) == GOLD
        )
        score.final_gold_precision = round(gold_kept / len(integrated.cards), 4)

    # 모달리티 소실: 검색 결과에 있었는데 최종 근거에는 없는 모달리티
    input_modalities = {
        m for m in (packet.get("retrieval_results") or {}) if (packet["retrieval_results"][m].get("evidence"))
    }
    kept_modalities = {card.modality.value for card in integrated.cards}
    # 1층이 카드를 아예 못 만든 모달리티는 통합 계층 책임이 아니므로 제외한다.
    carded_modalities = {card.modality.value for card in all_cards}
    score.modality_loss = len((input_modalities & carded_modalities) - kept_modalities)

    # ---- 3층 -------------------------------------------------
    answer = output.final_answer
    if answer.citations:
        score.citation_validity = round(
            sum(1 for cid in answer.citations if cid in kept_ids) / len(answer.citations), 4
        )
    elif integrated.cards:
        score.citation_validity = 0.0

    gold_kept_ids = {
        card.card_id
        for card in integrated.cards
        if roles.get(card.source_evidence_id) == GOLD
    }
    score.citation_coverage = _ratio(
        len(gold_kept_ids & set(answer.citations)), len(gold_kept_ids)
    )

    # card_id 누출: 사용자에게 보여줄 답변 본문에 내부 식별자가 남았는가.
    # 프롬프트에 금지 규칙이 있어도 지켜지지 않는 사례가 실측되었다.
    score.card_id_leak = len(_CARD_ID_RE.findall(answer.answer))

    # 인용 정밀도: 답변이 실제로 기댄 근거 중 gold 비율.
    # citation_validity 는 "존재하는 카드인가"만 보므로 잡음 카드를 인용해도
    # 1.00 이 나온다. 답변 수준의 근거 오염은 이 지표로만 드러난다.
    if answer.citations:
        role_by_card = {
            card.card_id: roles.get(card.source_evidence_id) for card in integrated.cards
        }
        cited_known = [cid for cid in answer.citations if cid in role_by_card]
        if cited_known:
            score.citation_precision = round(
                sum(1 for cid in cited_known if role_by_card[cid] == GOLD) / len(cited_known), 4
            )
    return score


# ============================================================
# LLM 심판 - 답변 정확도
# ============================================================

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "covered_key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key_point": {"type": "string"},
                    "covered": {"type": "boolean"},
                },
                "required": ["key_point", "covered"],
                "additionalProperties": False,
            },
        },
        "contaminated_statements": {"type": "array", "items": {"type": "string"}},
        "conflict_disclosed": {"type": "boolean"},
    },
    "required": ["covered_key_points", "contaminated_statements", "conflict_disclosed"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """너는 RAG 답변 채점자다. 답변을 생성하지 않고 판정만 한다.

판정 항목
1. covered_key_points: 주어진 정답 요지 각각에 대해, 답변이 그 내용을 담고 있으면
   covered=true. 표현이 달라도 의미가 같으면 true 다. 답변이 그 요지를 아예
   다루지 않았으면 false. 요지 목록을 빠짐없이 그대로 돌려줘라.
2. contaminated_statements: 답변이 질문과 무관한 주제를 사실로 끌어들여 서술한
   문장을 찾아 그대로 옮겨라. 무관 주제 목록이 주어진다. 없으면 빈 배열.
3. conflict_disclosed: 근거끼리 엇갈린다는 점을 답변이 밝혔으면 true.
   한쪽만 단정했으면 false.

엄격하게 판정하라. 애매하면 false 로 둔다."""


def score_answer_with_judge(
    output: DecoderOutput,
    packet: Mapping[str, Any],
    judge: StructuredLLMClient,
) -> Dict[str, Any]:
    meta = packet.get("_meta") or {}
    key_points: List[str] = list(meta.get("key_points") or [])
    irrelevant_topics: List[str] = list(meta.get("irrelevant_topics") or [])

    if not key_points:
        return {}

    lines = [
        f"[질문]\n{packet.get('original_query', '')}",
        f"[채점할 답변]\n{output.final_answer.answer}",
        "[정답 요지]\n" + "\n".join(f"- {kp}" for kp in key_points),
    ]
    if irrelevant_topics:
        lines.append("[무관 주제]\n" + "\n".join(f"- {t}" for t in irrelevant_topics))
    else:
        lines.append("[무관 주제]\n(없음)")
    lines.append(
        "[근거 충돌 여부]\n"
        + ("이 질문의 근거에는 서로 모순되는 진술이 포함되어 있다."
           if meta.get("has_conflict") else "모순되는 근거는 없다.")
    )

    raw = judge.generate_json(JUDGE_SYSTEM, "\n\n".join(lines), JUDGE_SCHEMA)

    entries = [e for e in (raw.get("covered_key_points") or []) if isinstance(e, Mapping)]
    # 심판이 요지를 쪼개거나 덧붙여 돌려주면 항목 수가 입력보다 많아진다. 그대로
    # 세면 비율이 1 을 넘으므로(HotpotQA 에서 1.08·1.17 관측), 돌려받은 항목을
    # 주어진 요지에 대응시켜 요지 단위로 한 번씩만 센다.
    covered_points = set()
    unmatched = []
    normalized = {_norm(kp): index for index, kp in enumerate(key_points)}
    for entry in entries:
        if not entry.get("covered"):
            continue
        index = normalized.get(_norm(str(entry.get("key_point", ""))))
        if index is None:
            unmatched.append(entry)
        else:
            covered_points.add(index)
    # 심판이 요지를 재서술한 경우. 순서는 보존되므로 남은 자리에 차례로 채운다.
    for index in range(len(key_points)):
        if not unmatched:
            break
        if index not in covered_points:
            covered_points.add(index)
            unmatched.pop(0)

    covered = len(covered_points)
    return {
        "key_point_coverage": round(covered / len(key_points), 4) if key_points else None,
        "contamination": len(raw.get("contaminated_statements") or []),
        "conflict_disclosed": bool(raw.get("conflict_disclosed"))
        if meta.get("has_conflict")
        else None,
    }


def score_output(
    output: DecoderOutput,
    packet: Mapping[str, Any],
    judge: Optional[StructuredLLMClient] = None,
    arm: str = "",
) -> QualityScore:
    score = score_rules(output, packet)
    score.arm = arm
    if judge is not None:
        try:
            for key, value in score_answer_with_judge(output, packet, judge).items():
                setattr(score, key, value)
        except LLMError as error:
            score.judge_error = str(error)[:200]
    return score


# ============================================================
# 집계
# ============================================================

METRIC_ORDER = [
    ("gold_recall", "gold채택", True),
    ("irrelevant_rejection", "무관거부", True),
    ("source_hallucination", "출처환각", False),
    ("duplicate_removal", "중복제거", True),
    ("duplicate_recognized", "중복인지", True),
    ("conflict_detection", "충돌탐지", True),
    ("reinforcing_preserved", "보강보존", True),
    ("final_gold_precision", "근거정밀", True),
    ("modality_loss", "모달소실", False),
    ("citation_validity", "인용유효", True),
    ("citation_coverage", "인용범위", True),
    ("citation_precision", "인용정밀", True),
    ("key_point_coverage", "요지충족", True),
    ("contamination", "오염서술", False),
    ("card_id_leak", "ID누출", False),
]


def aggregate(scores: Sequence[QualityScore]) -> Dict[str, Optional[float]]:
    """None 은 '해당 없음'이므로 평균에서 제외한다."""
    summary: Dict[str, Optional[float]] = {}
    for key, _, _ in METRIC_ORDER:
        values = [getattr(s, key) for s in scores]
        values = [v for v in values if v is not None]
        summary[key] = round(statistics.mean(values), 3) if values else None
    summary["n"] = len(scores)
    summary["degraded"] = sum(1 for s in scores if s.degraded)
    return summary


def print_scores(rows: Mapping[str, Sequence[QualityScore]]) -> None:
    """구성별 집계표. 화살표는 높을수록 좋은 지표를 뜻한다."""
    print("\n" + "=" * 108)
    header = "구성".ljust(22) + "n".ljust(4)
    for _, label, higher in METRIC_ORDER:
        header += (label + ("↑" if higher else "↓")).ljust(11)
    print(header)
    print("-" * 108)
    for name, scores in rows.items():
        summary = aggregate(scores)
        line = name.ljust(22) + str(summary["n"]).ljust(4)
        for key, _, _ in METRIC_ORDER:
            value = summary[key]
            line += ("-" if value is None else f"{value:.2f}").ljust(11)
        print(line)
        if summary["degraded"]:
            print(f"  [경고] {summary['degraded']}/{summary['n']}회가 폴백으로 오염됨. 결과 무효.")
    print("=" * 108)
