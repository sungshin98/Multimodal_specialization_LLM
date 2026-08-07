"""2층 - 근거 통합 계층.

모달리티별 카드를 하나의 근거 집합으로 합친다.
중복제거 / 충돌확인 / 우선순위 / 분량최적화 네 가지가 목적이다.

속도가 이 연구의 핵심 지표이므로 2단 구조로 만든다.
1. 규칙 기반 사전정리 - LLM 없이 처리 가능한 것은 여기서 끝낸다.
   (완전중복 제거, 우선순위 정렬, 분량 예산 절단)
2. LLM 통합 - 모달리티가 2개 이상이거나 충돌 가능성이 있을 때만 호출한다.
   단일 모달리티 + 카드 소수면 LLM 호출을 아예 건너뛴다.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .clients import LLMError, StructuredLLMClient
from .schemas import (
    ConflictNote,
    DecoderTask,
    EvidenceCard,
    IntegratedEvidence,
    Level,
    ModalityEvidenceResult,
)

INTEGRATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "kept_card_ids": {"type": "array", "items": {"type": "string"}},
        "dropped_card_ids": {"type": "array", "items": {"type": "string"}},
        # 열린 형태로 "중복을 찾아라" 하면 모델이 그냥 넘어간다. 실측에서
        # 공랭식 중복쌍(공랭식 vs 냉매 순환 없이 바람만으로)을 전혀 잡지 못했다.
        # 후보 쌍을 미리 주고 쌍마다 예/아니오를 강제하면 판정률이 올라간다.
        "pair_verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card_a": {"type": "string"},
                    "card_b": {"type": "string"},
                    "same_subject": {"type": "boolean"},
                    "same_fact": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["card_a", "card_b", "same_subject", "same_fact", "reason"],
                "additionalProperties": False,
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "card_ids": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "resolution": {"type": "string"},
                },
                "required": ["card_ids", "description", "resolution"],
                "additionalProperties": False,
            },
        },
        "coverage": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "covered": {"type": "boolean"},
                },
                "required": ["operation", "covered"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["kept_card_ids", "dropped_card_ids", "pair_verdicts", "conflicts", "coverage"],
    "additionalProperties": False,
}

INTEGRATION_SYSTEM = """너는 멀티모달 RAG 시스템의 근거 통합 계층이다.

여러 모달리티(텍스트/이미지/영상) 디코더가 만든 근거 카드를 받아
최종 답변 디코더가 쓸 근거 집합으로 정리한다. 답변은 작성하지 않는다.

할 일
1. 중복: 아래에 주어지는 [중복 후보 쌍] 목록의 **모든 쌍**에 대해 두 가지를
   판정해 pair_verdicts 에 넣어라. 한 쌍도 빠뜨리지 마라.
   - same_subject: 두 카드가 **같은 대상**을 말하는가. 인물, 작품, 기관, 제품이
     다르면 false 다. 문장 구조가 비슷하다는 이유로 true 로 하지 마라.
     반례: "Scott Derrickson은 미국 감독이다" 와 "Ed Wood는 미국 제작자이다" 는
     서술 형식이 같지만 대상이 다르므로 same_subject=false 다. 두 인물을 비교하는
     질문에서는 두 카드가 모두 필요하므로 절대 병합해서는 안 된다.
   - same_fact: 같은 대상에 대해 같은 사실을 말하는가.
   두 값이 **모두 true 일 때만** 중복으로 처리된다.
   판단 기준은 어휘가 아니라 의미다. 표현이 전혀 겹치지 않아도, 한쪽이 전문
   용어를 쓰고 다른 쪽이 그 원리를 풀어 썼어도, 가리키는 사실이 같으면 true 다.
   예시
   - "공랭식 열관리를 채택했다" 와 "냉매 순환 없이 바람만으로 열을 빼내는
     구조였다" -> true. 용어와 설명의 관계일 뿐 같은 사실이다.
   - "느린 호흡과 긴 컷을 유지했다" 와 "컷을 자주 나누지 않고 인물을 오래
     담아냈다" -> true.
   - 한쪽이 세부 설명을 덧붙였다는 이유로 false 로 하지 마라. 핵심 사실이
     같으면 덧붙은 설명은 중복 판정을 바꾸지 않는다.
   - 대상이 다르면(초기 모델 vs 개선 모델, 감독 A vs 감독 B) false 다.
   same_fact=true 인 쌍은 둘 중 하나만 kept_card_ids 에 남기고 나머지는
   dropped_card_ids 로 보내라. 더 구체적인 쪽을 남긴다.
2. 충돌: 서로 모순되는 카드를 찾아 무엇이 충돌인지, 어느 쪽을 우선할지 적는다.
   검색점수와 confidence 가 높은 쪽을 우선하되 판단 근거를 resolution 에 남겨라.
3. 선별: 질문에 답하는 데 불필요한 카드는 dropped_card_ids 로 보낸다.
   근거가 희석되지 않게 남길 카드는 최대 {max_cards}개로 제한한다.
4. 커버리지: 각 required_operation 이 남은 카드로 수행 가능한지 판정한다.

주의
- confidence 가 낮다는 것은 "원본 자료를 직접 확인하지 못했다"는 뜻일 수 있다.
  틀렸다는 뜻이 아니다. 질문에 직접 답하는 카드라면 confidence 가 낮아도 남겨라.
- 판단 기준은 confidence 가 아니라 "이 카드가 질문에 답하는가" 다.
  질문과 무관하지만 확신도가 높은 카드보다, 질문에 직접 답하는 불확실한 카드가 낫다.
- 각 모달리티에서 최소 1개는 남겨라. 한 모달리티를 통째로 버리면
  그 모달리티의 검색이 통째로 낭비된다.

kept_card_ids 와 dropped_card_ids 에는 실제로 주어진 card_id 만 쓴다."""


class EvidenceIntegrationLayer:
    def __init__(
        self,
        client: Optional[StructuredLLMClient] = None,
        max_cards: int = 12,
        char_budget: int = 4000,
        duplicate_threshold: float = 0.85,
        cross_modal_threshold: float = 0.35,
        min_relevance: Optional[float] = None,
        max_pairs: int = 20,
    ) -> None:
        self.client = client
        self.max_cards = max_cards
        self.char_budget = char_budget
        # 같은 모달리티 내 근사 중복 판정 기준 (규칙으로 바로 제거)
        self.duplicate_threshold = duplicate_threshold
        # 모달리티를 가로지르는 주장이 이만큼 겹치면 LLM 통합을 부른다.
        # 낮출수록 LLM 호출이 늘고 중복/충돌을 더 잡는다.
        self.cross_modal_threshold = cross_modal_threshold
        # 관련도 하한. 1층이 무관 근거를 relevance 로는 낮게 매기면서도
        # 카드 생성 자체는 막지 못하는 경우가 있어(특히 verify 질문) 여기서 자른다.
        # None 이면 끈다. 값은 데이터에 맞춰 보정해야 한다 - 합성 세트에서는
        # gold 0.81~0.95, 중복 0.74, 충돌 0.67, 무관 0.53~0.60 이라 0.65 가 분리점이지만
        # 이 값은 그 세트에 맞춘 것이지 일반값이 아니다.
        self.min_relevance = min_relevance
        # 중복 후보 쌍 상한. 카드가 많으면 쌍이 제곱으로 늘어 프롬프트가 커진다.
        self.max_pairs = max_pairs

    # ------------------------------------------------------------------

    def integrate(
        self,
        results: Sequence[ModalityEvidenceResult],
        context: DecoderTask,
    ) -> IntegratedEvidence:
        started = time.perf_counter()
        cards = [card for result in results for card in result.cards]

        if not cards:
            return IntegratedEvidence(
                coverage={op: False for op in context.required_operations},
                missing_operations=list(context.required_operations),
                char_budget=self.char_budget,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 0단계 - 관련도 하한 (설정된 경우에만)
        low_relevance: List[str] = []
        if self.min_relevance is not None:
            passing = [c for c in cards if c.relevance >= self.min_relevance]
            # 전부 잘려나가면 근거 없는 답변이 되므로 최상위 1장은 남긴다.
            if not passing and cards:
                passing = [max(cards, key=lambda c: c.relevance)]
            low_relevance = [c.card_id for c in cards if c not in passing]
            cards = passing

        # 1단계 - 규칙 기반 사전정리
        cards, duplicate_groups, dropped = self._prefilter(cards)
        dropped.extend(low_relevance)

        needs_llm = self.client is not None and self._needs_llm(cards, context)

        if not needs_llm:
            kept, over_budget = self._apply_budget(cards)
            dropped.extend(over_budget)
            return IntegratedEvidence(
                cards=kept,
                dropped_card_ids=dropped,
                duplicate_groups=duplicate_groups,
                coverage=self._rule_coverage(kept, context),
                missing_operations=self._missing(kept, context),
                char_budget=self.char_budget,
                char_used=sum(card.char_cost() for card in kept),
                bypassed=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # 2단계 - LLM 통합
        try:
            raw = self.client.generate_json(  # type: ignore[union-attr]
                INTEGRATION_SYSTEM.format(max_cards=self.max_cards),
                self._user_prompt(cards, results, context),
                INTEGRATION_SCHEMA,
            )
        except LLMError:
            # 통합에 실패해도 답변은 나와야 한다. 규칙 결과로 진행한다.
            kept, over_budget = self._apply_budget(cards)
            dropped.extend(over_budget)
            return IntegratedEvidence(
                cards=kept,
                dropped_card_ids=dropped,
                duplicate_groups=duplicate_groups,
                coverage=self._rule_coverage(kept, context),
                missing_operations=self._missing(kept, context),
                char_budget=self.char_budget,
                char_used=sum(card.char_cost() for card in kept),
                bypassed=True,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        return self._apply_llm_result(
            raw, cards, duplicate_groups, dropped, context, started
        )

    # ------------------------------------------------------------------
    # LLM 호출 여부 판단
    # ------------------------------------------------------------------

    def _needs_llm(self, cards: List[EvidenceCard], context: DecoderTask) -> bool:
        """LLM 통합이 실제로 할 일이 있을 때만 호출한다.

        이전에는 "모달리티 2개 이상"이면 무조건 호출했다. 그런데 멀티모달
        RAG 에서 그 조건은 거의 항상 참이라 사실상 최적화가 걸리지 않았다.
        LLM 이 판단할 거리가 있는지를 직접 본다.

        호출이 필요한 경우
        1. 선별 압력이 있다 - 카드가 상한을 넘거나 분량 예산을 초과한다.
        2. 중복/충돌 가능성이 있다 - 모달리티를 가로지르는 유사한 주장이 있다.
           (같은 모달리티 내 중복은 _prefilter 가 이미 규칙으로 제거했다)
        3. 검증/비교 작업이 요구된다 - 근거 간 대조가 답변의 핵심이다.
        4. 복잡도가 높다 - 안전망.
        """
        if len(cards) > self.max_cards:
            return True
        if sum(card.char_cost() for card in cards) > self.char_budget:
            return True
        if context.complexity_level == Level.HIGH:
            return True

        operations = {op.lower() for op in context.required_operations}
        if operations & {"verify", "compare", "analyze", "검증", "비교"}:
            return True

        # 모달리티를 가로지르는 유사 주장 = 중복 통합 또는 충돌 판정 대상
        for i, left in enumerate(cards):
            for right in cards[i + 1 :]:
                if left.modality == right.modality:
                    continue
                if _similarity(left.claim, right.claim) >= self.cross_modal_threshold:
                    return True
        return False

    # ------------------------------------------------------------------
    # 1단계
    # ------------------------------------------------------------------

    def _prefilter(self, cards: List[EvidenceCard]):
        """완전/근사 중복을 LLM 없이 제거하고 우선순위로 정렬한다."""
        cards = sorted(cards, key=lambda card: card.priority(), reverse=True)

        kept: List[EvidenceCard] = []
        groups: List[List[str]] = []
        dropped: List[str] = []

        for card in cards:
            duplicate_of = None
            for existing in kept:
                # 모달리티가 다르면 서로 보강하는 근거로 본다. 묶지 않는다.
                if existing.modality != card.modality:
                    continue
                if _similarity(existing.claim, card.claim) >= self.duplicate_threshold:
                    duplicate_of = existing
                    break

            if duplicate_of is None:
                kept.append(card)
                continue

            dropped.append(card.card_id)
            for group in groups:
                if duplicate_of.card_id in group:
                    group.append(card.card_id)
                    break
            else:
                groups.append([duplicate_of.card_id, card.card_id])

        return kept, groups, dropped

    def _apply_budget(self, cards: List[EvidenceCard]):
        """분량 예산. 근거 과부하로 답변 품질이 떨어지는 것을 막는다."""
        kept: List[EvidenceCard] = []
        dropped: List[str] = []
        used = 0
        for card in cards:
            cost = card.char_cost()
            if len(kept) >= self.max_cards or (used + cost > self.char_budget and kept):
                dropped.append(card.card_id)
                continue
            kept.append(card)
            used += cost
        return kept, dropped

    # ------------------------------------------------------------------
    # 2단계
    # ------------------------------------------------------------------

    def _user_prompt(
        self,
        cards: List[EvidenceCard],
        results: Sequence[ModalityEvidenceResult],
        context: DecoderTask,
    ) -> str:
        lines = [f"[원본 질문]\n{context.original_query}"]
        if context.required_operations:
            lines.append(f"[필요한 작업]\n{', '.join(context.required_operations)}")
        if context.constraints:
            lines.append(f"[제약]\n{', '.join(context.constraints)}")

        summaries = [
            f"- {result.modality.value}: {result.modality_summary}"
            for result in results
            if result.modality_summary
        ]
        if summaries:
            lines.append("[모달리티별 요약]\n" + "\n".join(summaries))

        insufficient = [
            f"- {result.modality.value}: {result.insufficient_reason}"
            for result in results
            if not result.is_sufficient and result.insufficient_reason
        ]
        if insufficient:
            lines.append("[근거 부족 신고]\n" + "\n".join(insufficient))

        card_lines = []
        for card in cards:
            card_lines.append(
                f"- card_id: {card.card_id} | 모달리티: {card.modality.value} | "
                f"관련도 {card.relevance:.2f} | 확신도 {card.confidence:.2f} | "
                f"검색점수 {card.retrieval_score:.4f}\n"
                f"  주장: {card.claim}\n"
                f"  상세: {card.detail}"
            )
        lines.append(f"[근거 카드 {len(cards)}개]\n" + "\n".join(card_lines))

        pairs = self._candidate_pairs(cards)
        if pairs:
            pair_lines = [
                f"- {a.card_id} vs {b.card_id}\n"
                f"    A: {a.claim}\n"
                f"    B: {b.claim}"
                for a, b in pairs
            ]
            lines.append(
                f"[중복 후보 쌍 {len(pairs)}개]\n" + "\n".join(pair_lines)
                + "\n\n위 쌍 전부에 대해 same_fact 를 판정해 pair_verdicts 에 넣어라."
            )
        return "\n\n".join(lines)

    def _candidate_pairs(self, cards: List[EvidenceCard]):
        """같은 모달리티 카드 쌍을 중복 후보로 만든다.

        모달리티가 다르면 서로 보강하는 근거이므로 후보에서 뺀다.
        쌍이 너무 많으면 어휘 유사도 상위만 남긴다. 유사도는 후보를 고르는
        데만 쓰고 판정에는 쓰지 않는다 - 판정은 의미로 해야 하기 때문이다.
        """
        pairs = [
            (a, b)
            for i, a in enumerate(cards)
            for b in cards[i + 1:]
            if a.modality == b.modality
        ]
        if len(pairs) > self.max_pairs:
            pairs.sort(key=lambda ab: _similarity(ab[0].claim, ab[1].claim), reverse=True)
            pairs = pairs[: self.max_pairs]
        return pairs

    def _apply_llm_result(
        self,
        raw: Mapping[str, Any],
        cards: List[EvidenceCard],
        duplicate_groups: List[List[str]],
        dropped: List[str],
        context: DecoderTask,
        started: float,
    ) -> IntegratedEvidence:
        by_id = {card.card_id: card for card in cards}

        # 쌍 판정을 묶음으로 합친다(연결 성분). 모델이 묶음을 직접 만들게 하면
        # 누락이 잦아, 쌍 단위 예/아니오를 받아 여기서 결정적으로 구성한다.
        same_pairs = [
            (str(v.get("card_a")), str(v.get("card_b")))
            for v in (raw.get("pair_verdicts") or [])
            if isinstance(v, Mapping)
            and v.get("same_fact")
            and v.get("same_subject")   # 대상이 같아야 병합한다
            and str(v.get("card_a")) in by_id
            and str(v.get("card_b")) in by_id
        ]
        llm_groups = _merge_pairs(same_pairs)
        duplicate_groups.extend(llm_groups)

        kept_ids = [cid for cid in (raw.get("kept_card_ids") or []) if cid in by_id]
        # LLM 이 전부 버리면 근거 없는 답변이 되므로 규칙 결과로 되돌린다.
        kept = [by_id[cid] for cid in kept_ids] if kept_ids else list(cards)

        # 선별 압력이 없으면 임의 제외를 허용하지 않는다.
        # 카드 수와 분량이 모두 예산 안에 들어오는데도 LLM 이 카드를 버리면
        # 다단계 추론에 필요한 근거가 사라진다. 실측에서 두 문서를 함께
        # 보아야 답할 수 있는 질문에서 한쪽이 제거되어 답변이 실패했다.
        # 이 경우 제외는 중복 병합과 분량 절단으로만 이루어진다.
        no_pressure = (
            len(cards) <= self.max_cards
            and sum(c.char_cost() for c in cards) <= self.char_budget
        )
        if no_pressure:
            kept = list(cards)

        # 중복 묶음에서 한쪽만 남긴다. 모델이 kept_card_ids 를 잘못 채워도
        # 중복이 함께 살아남지 않도록 규칙으로 강제한다.
        #
        # 생존자 기준을 처음에는 "더 구체적인(긴) 쪽"으로 두었으나 실패했다.
        # 중복본이 원본보다 길면 원본을 밀어내, 중복인지 1.00 인데 중복제거는
        # 0.33 에 그쳤다. 같은 사실이라면 어느 쪽을 남겨도 정보량은 같으므로,
        # 검색기와 디코더가 더 관련 있다고 판단한 쪽을 남기는 것이 맞다.
        deduped_out: List[str] = []
        for group in llm_groups:
            survivors = [card for card in kept if card.card_id in group]
            if len(survivors) > 1:
                best = max(
                    survivors,
                    key=lambda card: (card.priority(), card.retrieval_score, card.char_cost()),
                )
                for card in survivors:
                    if card.card_id != best.card_id:
                        kept.remove(card)
                        deduped_out.append(card.card_id)

        kept = self._ensure_modality_coverage(kept, cards)
        kept.sort(key=lambda card: card.priority(), reverse=True)
        kept, over_budget = self._apply_budget(kept)

        dropped = list(dict.fromkeys(dropped + deduped_out + over_budget +
                       [cid for cid in (raw.get("dropped_card_ids") or []) if cid in by_id]))
        kept_id_set = {card.card_id for card in kept}
        dropped = [cid for cid in dropped if cid not in kept_id_set]

        conflicts = [
            ConflictNote(
                card_ids=[cid for cid in (entry.get("card_ids") or []) if cid in by_id],
                description=str(entry.get("description", "") or ""),
                resolution=str(entry.get("resolution", "") or ""),
            )
            for entry in (raw.get("conflicts") or [])
            if isinstance(entry, Mapping) and entry.get("description")
        ]

        coverage: Dict[str, bool] = {}
        for entry in raw.get("coverage") or []:
            if isinstance(entry, Mapping) and entry.get("operation"):
                coverage[str(entry["operation"])] = bool(entry.get("covered"))
        for operation in context.required_operations:
            coverage.setdefault(operation, bool(kept))

        return IntegratedEvidence(
            cards=kept,
            dropped_card_ids=dropped,
            duplicate_groups=duplicate_groups,
            conflicts=conflicts,
            coverage=coverage,
            missing_operations=[op for op, ok in coverage.items() if not ok],
            char_budget=self.char_budget,
            char_used=sum(card.char_cost() for card in kept),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _ensure_modality_coverage(
        kept: List[EvidenceCard], candidates: List[EvidenceCard]
    ) -> List[EvidenceCard]:
        """LLM 이 한 모달리티를 통째로 버리면 최상위 카드 1개를 복원한다.

        비전 백엔드가 폴백으로 동작하면 image/video 카드의 confidence 가 낮게
        매겨진다. 이는 "원본을 못 봤다"는 뜻이지 "틀렸다"는 뜻이 아닌데,
        통합 LLM 이 이를 중요도로 오해해 질문에 직접 답하는 카드까지 버리는
        사례가 실측으로 확인되었다. 모달리티 소실은 규칙으로 막는다.
        """
        kept_modalities = {card.modality for card in kept}
        kept_ids = {card.card_id for card in kept}
        restored = list(kept)

        for modality in {card.modality for card in candidates}:
            if modality in kept_modalities:
                continue
            pool = [card for card in candidates if card.modality == modality]
            if not pool:
                continue
            best = max(pool, key=lambda card: card.relevance)
            if best.card_id not in kept_ids:
                best.metadata["restored_by_coverage_guard"] = True
                restored.append(best)
        return restored

    # ------------------------------------------------------------------

    @staticmethod
    def _rule_coverage(cards: List[EvidenceCard], context: DecoderTask) -> Dict[str, bool]:
        supported = {s for card in cards for s in card.supports}
        return {
            operation: (operation in supported) or bool(cards)
            for operation in context.required_operations
        }

    def _missing(self, cards: List[EvidenceCard], context: DecoderTask) -> List[str]:
        coverage = self._rule_coverage(cards, context)
        return [operation for operation, ok in coverage.items() if not ok]


def _merge_pairs(pairs: Sequence[tuple]) -> List[List[str]]:
    """같은 사실로 판정된 쌍들을 연결 성분으로 합친다.

    A=B, B=C 로 판정되면 {A,B,C} 한 묶음이다. 쌍 단위 판정을 묶음으로
    바꾸는 일은 규칙으로 하는 것이 안전하다 - 모델에게 묶음을 직접 만들게
    하면 일관성 없는 결과가 나온다.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: Dict[str, List[str]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(members) for members in groups.values() if len(members) > 1]


_TOKEN_RE = re.compile(r"[0-9a-zA-Z가-힣]+")


def _similarity(left: str, right: str) -> float:
    """어절 기반 Jaccard. 형태소 분석기 의존성 없이 근사 중복만 잡는다."""
    a = set(_TOKEN_RE.findall(left.lower()))
    b = set(_TOKEN_RE.findall(right.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
