"""실험 하네스 - 속도저하 억제 효과 측정.

    python -m evidence_decoder.bench --repeat 3

비교군
  plain-rag   : 외부 베이스라인. 검색 결과를 그대로 프롬프트에 이어 붙여
                단일 호출로 답변한다. 근거 카드·인용·충돌 처리가 없다.
  full        : 모달 디코더 + 통합 계층 전부 사용 (제안 구조)
  no-integ    : 통합 계층 제거 (카드 단순 이어붙이기)
  bypass      : 저복잡도 바이패스 허용
  raw         : 1층/2층 모두 없이 검색 결과를 최종 디코더에 직접 투입 (베이스라인)

측정: 총 지연, 단계별 지연, LLM 호출 수, 통합 전후 카드 수, 최종 답변 길이.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clients import LLMError, build_default_clients
from .final_decoder import FinalAnswerDecoder
from .integration import EvidenceIntegrationLayer
from .modality import build_modality_decoders
from .packet import PacketAdapter
from .pipeline import MultiLayerDecoderPipeline, PipelineConfig
from .schemas import DecoderOutput, Level, Modality
from .scoring import QualityScore, print_scores, score_output

# plain-rag 는 파이프라인 밖의 외부 베이스라인이다(baselines.py).
ARMS = ("plain-rag", "raw", "no-integ", "bypass", "full")


@dataclass
class ArmResult:
    arm: str
    total_ms: List[float] = field(default_factory=list)
    modality_ms: List[float] = field(default_factory=list)
    integration_ms: List[float] = field(default_factory=list)
    final_ms: List[float] = field(default_factory=list)
    llm_calls: List[int] = field(default_factory=list)
    cards_before: List[int] = field(default_factory=list)
    cards_after: List[int] = field(default_factory=list)
    answer_chars: List[int] = field(default_factory=list)
    citations: List[int] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)
    failed_runs: List[str] = field(default_factory=list)

    def record(self, output: DecoderOutput) -> None:
        trace = output.trace
        if trace.failed_modalities:
            self.failed_runs.append("; ".join(trace.failed_modalities))
        if trace.degraded_backends:
            # 폴백으로 내려간 실행은 정상 경로의 수치가 아니다.
            self.degraded.append("; ".join(trace.degraded_backends))
        self.total_ms.append(trace.total_ms)
        self.modality_ms.append(trace.modality_stage_ms)
        self.integration_ms.append(trace.integration_ms)
        self.final_ms.append(trace.final_ms)
        self.llm_calls.append(trace.llm_calls)
        self.cards_before.append(trace.cards_before_integration)
        self.cards_after.append(trace.cards_after_integration)
        self.answer_chars.append(len(output.final_answer.answer))
        self.citations.append(len(output.final_answer.citations))

    def summary(self) -> Dict[str, Any]:
        def med(values: Sequence[float]) -> float:
            return round(statistics.median(values), 1) if values else 0.0

        return {
            "arm": self.arm,
            "n": len(self.total_ms),
            "total_ms_median": med(self.total_ms),
            "modality_ms_median": med(self.modality_ms),
            "integration_ms_median": med(self.integration_ms),
            "final_ms_median": med(self.final_ms),
            "llm_calls_median": med(self.llm_calls),
            "cards_before_median": med(self.cards_before),
            "cards_after_median": med(self.cards_after),
            "answer_chars_median": med(self.answer_chars),
            "citations_median": med(self.citations),
            "errors": len(self.errors),
            "degraded": len(self.degraded),
            "실패": len(self.failed_runs),
        }


def build_arm(arm: str, clients: Dict[str, Any], asset_root: Optional[str]):
    from .assets import AssetLoader
    from .baselines import PlainRAGPipeline

    text_client = clients["text"]
    if arm == "plain-rag":
        return PlainRAGPipeline(client=text_client)

    loader = AssetLoader(asset_root=asset_root)
    decoders = build_modality_decoders(
        text_client, clients["vision"], loader,
        (Modality.TEXT, Modality.IMAGE, Modality.VIDEO),
    )

    if arm == "raw":
        # 1층/2층 없음. 검색 결과가 그대로 최종 디코더로 간다.
        # 이미지/영상도 강제로 바이패스해야 진짜 "디코더 없음" 베이스라인이 된다.
        config = PipelineConfig(force_modality_bypass=True, enable_integration=False)
    elif arm == "no-integ":
        config = PipelineConfig(enable_modality_bypass=False, enable_integration=False)
    elif arm == "bypass":
        config = PipelineConfig(enable_modality_bypass=True, enable_integration=True)
    else:  # full
        config = PipelineConfig(enable_modality_bypass=False, enable_integration=True)

    return MultiLayerDecoderPipeline(
        modality_decoders=decoders,
        integration_layer=EvidenceIntegrationLayer(client=text_client),
        final_decoder=FinalAnswerDecoder(text_client),
        config=config,
    )


def run_bench(
    packets: List[Dict[str, Any]],
    arms: Sequence[str] = ARMS,
    repeat: int = 1,
    asset_root: Optional[str] = None,
    group_by_meta: bool = False,
    score: bool = False,
) -> Tuple[Dict[str, ArmResult], Dict[str, List[QualityScore]]]:
    clients = build_default_clients()
    print(f"비전 백엔드: {clients['vision_backend']}")
    # 심판은 생성 디코더와 같은 모델을 쓰되 별도 호출이다. 채점 기준의
    # 대부분은 규칙이고, 심판은 답변 요지 충족 여부만 판정한다.
    judge = clients["text"] if score else None
    scores: Dict[str, List[QualityScore]] = {}

    # 워밍업. 첫 호출은 TCP/TLS 핸드셰이크와 모델 콜드스타트를 떠안으므로
    # 측정에 넣으면 첫 번째 구성만 불리해진다.
    if packets:
        try:
            build_arm("raw", clients, asset_root).run(packets[0])
            print("  (워밍업 1회 완료)")
        except Exception as error:  # noqa: BLE001
            print(f"  (워밍업 실패, 무시: {error})")

    # 구성을 바깥 루프에 두면 마지막 구성이 누적 부하와 속도 제한을 떠안는다.
    # 실측에서 마지막 구성의 40%가 호출 실패한 적이 있으므로, 패킷을 바깥에
    # 두고 구성을 번갈아 실행하여 시간대 효과를 상쇄한다.
    pipelines = {arm: build_arm(arm, clients, asset_root) for arm in arms}
    results: Dict[str, ArmResult] = {}
    for index, packet in enumerate(packets):
        for arm in arms:
            pipeline = pipelines[arm]
            meta = packet.get("_meta") or {}
            label = meta.get("group") if group_by_meta else None
            key = f"{arm}@{label}" if label else arm
            result = results.setdefault(key, ArmResult(key))
            name = meta.get("packet_id", f"패킷{index}")

            for _ in range(repeat):
                try:
                    output = pipeline.run(packet)
                    result.record(output)
                    if score:
                        scores.setdefault(key, []).append(
                            score_output(output, packet, judge, arm=arm)
                        )
                    print(
                        f"  [{arm}] {name:22s} "
                        f"{output.trace.total_ms:7.0f}ms  "
                        f"호출{output.trace.llm_calls}  "
                        f"카드{output.trace.cards_before_integration}->"
                        f"{output.trace.cards_after_integration}"
                    )
                except Exception as error:  # noqa: BLE001
                    result.errors.append(repr(error))
                    print(f"  [{arm}] {name} 실패: {error}")
    return results, scores


def print_table(results: Dict[str, ArmResult]) -> None:
    rows = [result.summary() for result in results.values()]
    headers = [
        ("arm", "구성", 26),
        ("total_ms_median", "총지연ms", 10),
        ("modality_ms_median", "1층ms", 9),
        ("integration_ms_median", "2층ms", 9),
        ("final_ms_median", "최종ms", 9),
        ("llm_calls_median", "호출", 6),
        ("cards_before_median", "카드전", 7),
        ("cards_after_median", "카드후", 7),
        ("answer_chars_median", "답변자", 7),
        ("citations_median", "인용", 6),
        ("errors", "오류", 5),
        ("degraded", "오염", 5),
        ("실패", "실패", 5),
    ]
    print("\n" + "=" * 100)
    print("".join(title.ljust(width) for _, title, width in headers))
    print("-" * 100)
    for row in rows:
        print("".join(str(row[key]).ljust(width) for key, _, width in headers))
    print("=" * 100)

    broken = {k: r for k, r in results.items() if r.failed_runs}
    if broken:
        print("\n[경고] 아래 구성은 디코더 호출이 실패한 실행이 섞여 있다.")
        print("       실패한 실행은 카드가 0개가 되어 규칙 지표를 끌어내리는 반면,")
        print("       심판 호출도 함께 실패하면 품질 지표에서는 제외되어 행이 모순된다.")
        for key, result in broken.items():
            print(f"  - {key}: {len(result.failed_runs)}/{len(result.total_ms)}회 실패 | {result.failed_runs[0][:80]}")

    polluted = {k: r for k, r in results.items() if r.degraded}
    if polluted:
        print("\n[경고] 아래 구성은 백엔드가 폴백으로 내려간 실행이 섞여 있다.")
        print("       정상 경로의 수치가 아니므로 실험 결과로 쓰면 안 된다.")
        for key, result in polluted.items():
            print(f"  - {key}: {len(result.degraded)}/{len(result.total_ms)}회 오염 | {result.degraded[0][:90]}")

    baseline = results.get("raw")
    full = results.get("full")
    if baseline and full and baseline.total_ms and full.total_ms:
        base = statistics.median(baseline.total_ms)
        proposed = statistics.median(full.total_ms)
        overhead = (proposed - base) / base * 100 if base else 0.0
        print(f"\n제안 구조의 지연 증가율: {overhead:+.1f}%  (raw {base:.0f}ms -> full {proposed:.0f}ms)")


def load_packets(path: Optional[str]) -> List[Dict[str, Any]]:
    if path:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else [data]
    from .test_offline import FULL_PACKET

    return [FULL_PACKET]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="다층 디코더 지연/품질 비교")
    parser.add_argument("--packets", help="RAG 패킷 JSON 경로 (객체 또는 배열)")
    parser.add_argument("--asset-root", help="이미지/영상 원본 루트")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--group", action="store_true", help="_meta.group 별로 나눠 집계")
    parser.add_argument("--score", action="store_true", help="품질 채점까지 수행")
    args = parser.parse_args(argv)

    packets = load_packets(args.packets)
    print(f"패킷 {len(packets)}개 x 반복 {args.repeat}회 x 구성 {len(args.arms)}종")
    started = time.perf_counter()
    results, scores = run_bench(
        packets, args.arms, args.repeat, args.asset_root, args.group, args.score
    )
    print_table(results)
    if scores:
        print_scores(scores)
    print(f"총 소요 {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
