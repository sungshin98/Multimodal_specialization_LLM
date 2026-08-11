"""공개 벤치마크 -> 실험 패킷 변환기.

합성 패킷은 통제가 쉬운 대신 외부 타당성이 없다. 공개 벤치마크를 같은 라벨
체계로 변환해 두 가지를 동시에 얻는다.

  gold        벤치마크가 지정한 정답 근거
  irrelevant  벤치마크가 함께 제공하는 distractor. 사람이 만든 합성 잡음이
              아니라 실제 검색기가 가져온 잡음이라 hard negative 의 기준이 된다.
  duplicate / contradictory
              공개 벤치마크에는 이 라벨이 없다. 필요하면 LLM 으로 gold 를
              바꿔 써서 주입한다(주입한 것이므로 정답 라벨이 확실하다).

HotpotQA distractor 설정을 기본으로 쓴다. 문단 10개 중 supporting_facts 가
가리키는 2개가 gold 이고 나머지 8개가 distractor 라, 근거 희석 실험에 필요한
구조가 이미 갖춰져 있다.

    python -m evidence_decoder.benchmarks --source hotpotqa --limit 20 \
           --out packets_hotpot.json
    python -m evidence_decoder.benchmarks --source hotpotqa --limit 20 \
           --inject --out packets_hotpot_injected.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .clients import LLMError, StructuredLLMClient
from .datagen import EvidenceRole
from .schemas import Level

HF_ROWS = "https://datasets-server.huggingface.co/rows"


@dataclass
class BenchmarkItem:
    """벤치마크 1문항을 라벨과 함께 정규화한 형태."""

    item_id: str
    question: str
    answer: str
    gold_texts: List[str] = field(default_factory=list)
    distractor_texts: List[str] = field(default_factory=list)
    duplicate_texts: List[str] = field(default_factory=list)
    contradictory_texts: List[str] = field(default_factory=list)
    level: Level = Level.MEDIUM
    source: str = ""


# ============================================================
# 로더
# ============================================================


def _fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> List[Dict[str, Any]]:
    params = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    request = urllib.request.Request(f"{HF_ROWS}?{params}", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return [row["row"] for row in json.load(response).get("rows", [])]
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HF rows API 오류 {error.code}: {error.read()[:200]!r}") from error


def load_hotpotqa(
    limit: int = 20,
    split: str = "validation",
    offset: int = 0,
    max_paragraph_chars: int = 1200,
) -> List[BenchmarkItem]:
    """HotpotQA distractor. 문단 10개 중 2개가 gold, 나머지가 distractor."""
    items: List[BenchmarkItem] = []
    fetched = 0
    # rows API 는 한 번에 100행까지 준다.
    while len(items) < limit:
        batch = _fetch_rows(
            "hotpotqa/hotpot_qa", "distractor", split, offset + fetched, min(100, limit - len(items))
        )
        if not batch:
            break
        fetched += len(batch)
        for row in batch:
            context = row.get("context") or {}
            titles = context.get("title") or []
            sentences = context.get("sentences") or []
            gold_titles = set((row.get("supporting_facts") or {}).get("title") or [])

            gold, distractors = [], []
            for title, sents in zip(titles, sentences):
                paragraph = f"{title}: {' '.join(sents)}"[:max_paragraph_chars]
                (gold if title in gold_titles else distractors).append(paragraph)

            if not gold:
                continue
            items.append(
                BenchmarkItem(
                    item_id=str(row.get("id", f"hotpot_{len(items)}")),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    gold_texts=gold,
                    distractor_texts=distractors,
                    # hard 는 다중 추론이 필요한 문항이다.
                    level=Level.HIGH if row.get("level") == "hard" else Level.MEDIUM,
                    source="hotpotqa",
                )
            )
            if len(items) >= limit:
                break
    return items


def load_jsonl(path: str, limit: Optional[int] = None) -> List[BenchmarkItem]:
    """직접 만든 벤치마크를 읽는다. 필드명은 BenchmarkItem 과 같다."""
    items: List[BenchmarkItem] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                BenchmarkItem(
                    item_id=str(data.get("item_id", len(items))),
                    question=str(data.get("question", "")),
                    answer=str(data.get("answer", "")),
                    gold_texts=list(data.get("gold_texts") or []),
                    distractor_texts=list(data.get("distractor_texts") or []),
                    duplicate_texts=list(data.get("duplicate_texts") or []),
                    contradictory_texts=list(data.get("contradictory_texts") or []),
                    source=str(data.get("source", "jsonl")),
                )
            )
            if limit and len(items) >= limit:
                break
    return items


# ============================================================
# 중복·모순 주입 (공개 벤치마크에는 라벨이 없다)
# ============================================================

INJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "paraphrase": {"type": "string"},
        "contradiction": {"type": "string"},
    },
    "required": ["paraphrase", "contradiction"],
    "additionalProperties": False,
}

INJECT_SYSTEM = """너는 RAG 실험용 데이터를 만드는 도구다.

주어진 문단 하나를 두 가지로 변형한다.
1. paraphrase: 같은 사실을 말하되 어휘를 최대한 바꿔 쓴 문단.
   전문 용어를 풀어 쓰거나 반대로 묶어 쓰는 식으로 표현을 크게 바꿔라.
   사실 자체는 절대 바꾸지 마라. 이것은 중복 탐지 시험용이다.
2. contradiction: 원문의 핵심 주장을 정면으로 부정하는 문단.
   같은 주제와 대상을 유지하되 결론만 뒤집어라. 실제 논문에서 반론이
   제기되는 것처럼 자연스럽게 써라. 이것은 충돌 탐지 시험용이다.

둘 다 원문과 비슷한 길이로 쓴다. 원문의 언어를 그대로 따른다."""


def inject_variants(
    items: Sequence[BenchmarkItem], client: StructuredLLMClient, verbose: bool = False
) -> List[BenchmarkItem]:
    """gold[0] 을 바꿔 써서 duplicate 와 contradictory 를 만든다.

    주입한 것이므로 라벨이 확실하다. 공개 벤치마크로 통합 계층(중복·충돌)을
    측정하려면 이 단계가 필요하다.
    """
    for index, item in enumerate(items):
        if not item.gold_texts:
            continue
        try:
            raw = client.generate_json(
                INJECT_SYSTEM, f"[원문]\n{item.gold_texts[0]}", INJECT_SCHEMA
            )
        except LLMError as error:
            if verbose:
                print(f"  [{item.item_id}] 주입 실패: {error}")
            continue
        paraphrase = str(raw.get("paraphrase", "") or "").strip()
        contradiction = str(raw.get("contradiction", "") or "").strip()
        if paraphrase:
            item.duplicate_texts = [paraphrase]
        if contradiction:
            item.contradictory_texts = [contradiction]
        if verbose:
            print(f"  [{index + 1}/{len(items)}] {item.item_id} 주입 완료")
    return items


# ============================================================
# 패킷 변환
# ============================================================


def to_packet(
    item: BenchmarkItem,
    max_distractors: Optional[int] = None,
    operations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """BenchmarkItem -> AdaptiveRAGOutput 형태의 패킷 (역할 라벨 포함)."""
    entries: List[tuple] = [(EvidenceRole.GOLD, text) for text in item.gold_texts]
    entries += [(EvidenceRole.DUPLICATE, t) for t in item.duplicate_texts]
    entries += [(EvidenceRole.CONTRADICTORY, t) for t in item.contradictory_texts]
    distractors = item.distractor_texts
    if max_distractors is not None:
        distractors = distractors[:max_distractors]
    entries += [(EvidenceRole.IRRELEVANT, t) for t in distractors]

    evidence = [
        {
            "evidence_id": f"text_{index}",
            "modality": "text",
            "score": round(0.95 - 0.05 * index, 4),
            "content": content,
            "metadata": {"rank": index + 1, "_role": role.value},
        }
        for index, (role, content) in enumerate(entries)
        if content
    ]

    ops = operations or (["compare", "verify"] if item.level == Level.HIGH else ["describe"])
    return {
        "schema_version": "3.0",
        "original_query": item.question,
        "normalized_query": item.question,
        "query_context": {
            "input_context": "",
            "identified_entities": [],
            "required_operations": ops,
            "constraints": [],
            "modality_focus": {"text": ["핵심 사실"]},
            "answer_constraints": ["질문에 직접 답하는 문장으로 시작할 것"],
            "retrieval_action": "retrieve",
            "sub_queries": [{"query": item.question, "modality": "text", "priority": "high"}],
        },
        "complexity": {
            "level": item.level.value,
            "score": 0.85 if item.level == Level.HIGH else 0.55,
            "retrieval_demand": {"text": item.level.value},
            "features": {},
            "reasons": [],
        },
        "retrieval_results": {
            "text": {
                "modality": "text",
                "query": item.question,
                "candidate_k": len(evidence),
                "final_k": len(evidence),
                "use_reranker": False,
                "uncertainty": {
                    "top1_score": evidence[0]["score"] if evidence else 0.0,
                    "top1_top2_gap": 0.05,
                    "score_variance": 0.01,
                    "shannon_entropy": 1.0,
                    "normalized_entropy": 0.9,
                    "level": "medium",
                },
                "evidence": evidence,
            }
        },
        "_meta": {
            "packet_id": f"{item.source}_{item.item_id}",
            "group": f"{item.source}/{item.level.value}",
            "scenario": item.source,
            "level": item.level.value,
            "modalities": ["text"],
            "evidence_total": len(evidence),
            "gold_total": len(item.gold_texts),
            "irrelevant_total": len(distractors),
            "duplicate_total": len(item.duplicate_texts),
            "contradiction_total": len(item.contradictory_texts),
            "hard_negatives": True,  # 벤치마크가 제공한 실제 distractor
            "use_real_assets": False,
            # 정답 요지. HotpotQA 는 짧은 스팬 정답이라 문장으로 감싼다.
            "key_points": [f'질문에 대한 정답은 "{item.answer}" 이다'],
            "irrelevant_topics": [t.split(":")[0][:60] for t in distractors[:6]],
            "has_conflict": bool(item.contradictory_texts),
        },
    }


SOURCES = {"hotpotqa": load_hotpotqa}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="공개 벤치마크 -> 실험 패킷")
    parser.add_argument("--source", choices=sorted(SOURCES) + ["jsonl"], default="hotpotqa")
    parser.add_argument("--path", help="--source jsonl 일 때 입력 파일")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-distractors", type=int, default=None, help="희석 비율 조절")
    parser.add_argument("--inject", action="store_true", help="중복·모순을 LLM 으로 주입")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.source == "jsonl":
        if not args.path:
            parser.error("--source jsonl 이면 --path 가 필요하다")
        items = load_jsonl(args.path, args.limit)
    else:
        items = SOURCES[args.source](limit=args.limit, offset=args.offset)
    print(f"{args.source}: {len(items)}문항 로드")

    if args.inject:
        from .clients import SolarStructuredClient

        print("중복·모순 주입 중...")
        items = inject_variants(items, SolarStructuredClient(), verbose=True)

    packets = [to_packet(item, args.max_distractors) for item in items]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(packets, handle, ensure_ascii=False, indent=1)

    total = sum(p["_meta"]["evidence_total"] for p in packets)
    gold = sum(p["_meta"]["gold_total"] for p in packets)
    print(f"패킷 {len(packets)}개 -> {args.out}")
    print(f"  근거 {total}건 (gold {gold}, 무관 {total - gold - sum(p['_meta']['duplicate_total'] + p['_meta']['contradiction_total'] for p in packets)})")
    print(f"  평균 희석률 {(1 - gold / total) * 100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
