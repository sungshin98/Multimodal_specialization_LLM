"""실제 LLM 을 붙여 다층 디코더를 1회 실행한다.

    python evidence_decoder/examples/run_live.py
    python evidence_decoder/examples/run_live.py --packet path/to/rag_output.json --asset-root ./data

필요한 환경변수
    UPSTAGE_API_KEY   (필수) 텍스트/통합/최종 디코더 - solar-pro3
    GOOGLE_API_KEY    (선택) 이미지/영상 디코더 - Gemini Flash
                      없으면 캡션 폴백으로 degraded 동작
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evidence_decoder import build_pipeline  # noqa: E402
from evidence_decoder.clients import build_default_clients  # noqa: E402


def load_env() -> None:
    """.env 를 흔한 위치에서 찾아 읽는다 (python-dotenv 없으면 수동 파싱)."""
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.expanduser("~/icac/.env"),
        os.path.expanduser("~/.env"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip().strip("'\"")
                if value and not os.getenv(key.strip()):
                    os.environ[key.strip()] = value
        print(f"환경변수 로드: {path}")
        return
    print("경고: .env 를 찾지 못했다. 셸 환경변수를 사용한다.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", help="Adaptive RAG 패킷 JSON 경로")
    parser.add_argument("--asset-root", help="이미지/영상 원본 루트 디렉터리")
    parser.add_argument("--json", action="store_true", help="전체 출력을 JSON 으로")
    args = parser.parse_args()

    load_env()

    if args.packet:
        with open(args.packet, encoding="utf-8") as handle:
            packet = json.load(handle)
    else:
        from evidence_decoder.test_offline import FULL_PACKET

        packet = FULL_PACKET
        print("샘플 패킷 사용 (텍스트 2건 + 이미지 1건)")

    backend = build_default_clients()["vision_backend"]
    print(f"비전 백엔드: {backend}")
    if backend == "caption_fallback":
        print("  -> GOOGLE_API_KEY 가 없어 캡션 폴백으로 동작한다 (degraded).")

    pipeline = build_pipeline(asset_root=args.asset_root)
    output = pipeline.run(packet)

    if args.json:
        print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print("\n" + "=" * 70)
    print(f"질문: {output.original_query}")
    print("=" * 70)

    print("\n[1층] 모달리티별 근거 디코더")
    for result in output.modality_results:
        flag = " (바이패스)" if result.bypassed else ""
        flag += " (실패)" if result.failed else ""
        print(f"  {result.modality.value}{flag}: 카드 {len(result.cards)}개, {result.latency_ms:.0f}ms")
        if result.modality_summary:
            print(f"    요약: {result.modality_summary}")
        for card in result.cards:
            print(f"    - [{card.card_id}] {card.claim}")
            print(f"      관련도 {card.relevance:.2f} 확신도 {card.confidence:.2f}")
        if not result.is_sufficient and result.insufficient_reason:
            print(f"    근거부족: {result.insufficient_reason}")

    integrated = output.integrated
    print(f"\n[2층] 근거 통합 계층 ({integrated.latency_ms:.0f}ms"
          f"{', 규칙만' if integrated.bypassed else ''})")
    print(f"  유지 {len(integrated.cards)}개 / 제외 {len(integrated.dropped_card_ids)}개 "
          f"/ 분량 {integrated.char_used}자 (예산 {integrated.char_budget})")
    for group in integrated.duplicate_groups:
        print(f"  중복: {' = '.join(group)}")
    for note in integrated.conflicts:
        print(f"  충돌: {note.description}")
        if note.resolution:
            print(f"        처리: {note.resolution}")
    if integrated.missing_operations:
        print(f"  근거부족 작업: {', '.join(integrated.missing_operations)}")

    answer = output.final_answer
    print(f"\n[3층] 최종 답변 디코더 ({answer.latency_ms:.0f}ms)")
    print("-" * 70)
    print(answer.answer)
    print("-" * 70)
    print(f"인용: {', '.join(answer.citations) or '없음'}")
    if answer.unsupported_claims:
        print(f"근거없는 서술: {answer.unsupported_claims}")
    print(f"신뢰도: {answer.confidence:.2f}")

    trace = output.trace
    print(f"\n[계측] 총 {trace.total_ms:.0f}ms "
          f"(1층 {trace.modality_stage_ms:.0f} / 2층 {trace.integration_ms:.0f} / "
          f"최종 {trace.final_ms:.0f}), LLM 호출 {trace.llm_calls}회")
    print(f"       모달별: {trace.modality_latency_ms}")
    print(f"       카드 {trace.cards_before_integration} -> {trace.cards_after_integration}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
