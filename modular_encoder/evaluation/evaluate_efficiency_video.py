import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


# ============================================================
# 프로젝트 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# evaluation 폴더에서 실행해도 router, encoders 등을 찾을 수 있게 설정
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 실험 입력
# ============================================================

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "frames"
    / "charade"
    / "charade_00000.jpg"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clips"
    / "charade"
    / "charade_00000.mp4"
)

QUERY_TEXT = "A person is moving inside a room."


# ============================================================
# 메모리 측정
# ============================================================

def get_peak_memory_mb():
    """
    현재 프로세스의 Peak Memory를 MB 단위로 반환한다.

    Windows에서는 psutil의 peak_wset이
    Peak Working Set에 해당한다.
    """

    process = psutil.Process(os.getpid())
    info = process.memory_info()

    peak_memory = getattr(info, "peak_wset", info.rss)

    return peak_memory / (1024 * 1024)


# ============================================================
# 입력 파일 확인
# ============================================================

def check_input_files():

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image file not found:\n{IMAGE_PATH}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video file not found:\n{VIDEO_PATH}"
        )


# ============================================================
# 실제 실험을 수행하는 Worker
# ============================================================

def run_worker(mode, input_type):

    # 프로젝트 루트가 import path에 들어간 뒤 import
    from router.input_router import InputRouter

    check_input_files()

    # --------------------------------------------------------
    # 전체 실행 시간 측정 시작
    # --------------------------------------------------------

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # 1. 초기화
    # --------------------------------------------------------

    init_start = time.perf_counter()

    router = InputRouter()

    # Eager Loading
    #
    # 실제 입력 종류와 관계없이
    # 모든 Text / Image / Video 관련 모델을 미리 로드한다.
    #
    # Audio는 최종 실험에서 제외한다.
    if mode == "eager":

        router.get_query_encoder()
        router.get_document_encoder()
        router.get_image_encoder()
        router.get_video_encoder()
        router.get_alignment_model()

    # Lazy Loading에서는 여기서 아무 Encoder도 강제로 로드하지 않는다.
    # 실제 입력이 들어왔을 때 Router가 필요한 Encoder만 로드한다.

    initialization_time = time.perf_counter() - init_start

    # --------------------------------------------------------
    # 2. 입력 처리
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    if input_type == "text":

        output = router.route_query(
            QUERY_TEXT
        )

    elif input_type == "image":

        output = router.route_file(
            str(IMAGE_PATH)
        )

    elif input_type == "video":

        output = router.route_file(
            str(VIDEO_PATH)
        )

    elif input_type == "mixed":

        text_output = router.route_query(
            QUERY_TEXT
        )

        image_output = router.route_file(
            str(IMAGE_PATH)
        )

        video_output = router.route_file(
            str(VIDEO_PATH)
        )

        output = {
            "text": text_output,
            "image": image_output,
            "video": video_output,
        }

    else:
        raise ValueError(
            f"Unknown input type: {input_type}"
        )

    inference_time = time.perf_counter() - inference_start

    # --------------------------------------------------------
    # 3. 전체 시간 / 메모리
    # --------------------------------------------------------

    total_time = time.perf_counter() - total_start

    peak_memory_mb = get_peak_memory_mb()

    # output이 너무 일찍 해제되지 않도록 참조 유지
    _ = output

    result = {
        "mode": mode,
        "input": input_type,
        "initialization_time_sec": initialization_time,
        "inference_time_sec": inference_time,
        "total_time_sec": total_time,
        "peak_memory_mb": peak_memory_mb,
    }

    # 부모 프로세스가 이 줄을 읽는다.
    print(
        "RESULT_JSON="
        + json.dumps(result),
        flush=True
    )


# ============================================================
# 부모 프로세스
# ============================================================

def run_parent():

    check_input_files()

    modes = [
        "eager",
        "lazy",
    ]

    input_types = [
        "text",
        "image",
        "video",
        "mixed",
    ]

    results = []

    print()
    print("=" * 78)
    print("Text / Image / Video Encoder Efficiency Experiment")
    print("Eager Loading vs Router + Lazy Loading")
    print("=" * 78)

    print()
    print(f"Image : {IMAGE_PATH}")
    print(f"Video : {VIDEO_PATH}")
    print()

    # --------------------------------------------------------
    # 각각 별도 프로세스에서 실행
    # --------------------------------------------------------
    #
    # 같은 프로세스에서 계속 실행하면
    # 이전 실험에서 로드된 모델이 메모리에 남을 수 있다.
    #
    # 따라서 각 조건을 독립된 Python 프로세스로 실행한다.
    # --------------------------------------------------------

    for input_type in input_types:

        for mode in modes:

            print(
                f"[RUN] {mode.upper():5s} | {input_type.upper()}",
                flush=True
            )

            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--mode",
                mode,
                "--input",
                input_type,
            ]

            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            # ------------------------------------------------
            # 오류 처리
            # ------------------------------------------------

            if completed.returncode != 0:

                print()
                print("=" * 78)
                print("ERROR")
                print("=" * 78)

                if completed.stdout:
                    print()
                    print("[STDOUT]")
                    print(completed.stdout)

                if completed.stderr:
                    print()
                    print("[STDERR]")
                    print(completed.stderr)

                sys.exit(1)

            # ------------------------------------------------
            # RESULT_JSON 찾기
            # ------------------------------------------------

            result_line = None

            for line in completed.stdout.splitlines():

                if line.startswith("RESULT_JSON="):

                    result_line = line[
                        len("RESULT_JSON="):
                    ]

            if result_line is None:

                print()
                print("RESULT_JSON을 찾지 못했습니다.")
                print()
                print(completed.stdout)

                sys.exit(1)

            result = json.loads(result_line)

            results.append(result)

            # ------------------------------------------------
            # 개별 결과 출력
            # ------------------------------------------------

            print(
                f"      Peak Memory : "
                f"{result['peak_memory_mb']:.2f} MB"
            )

            print(
                f"      Init Time   : "
                f"{result['initialization_time_sec']:.3f} sec"
            )

            print(
                f"      Infer Time  : "
                f"{result['inference_time_sec']:.3f} sec"
            )

            print(
                f"      Total Time  : "
                f"{result['total_time_sec']:.3f} sec"
            )

            print()

    # ========================================================
    # 최종 비교
    # ========================================================

    print()
    print("=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)

    header = (
        f"{'Input':<10}"
        f"{'Eager Mem':>12}"
        f"{'Lazy Mem':>12}"
        f"{'Mem Down':>12}"
        f"{'Eager Time':>14}"
        f"{'Lazy Time':>14}"
        f"{'Time Down':>12}"
    )

    print(header)
    print("-" * len(header))

    comparison = []

    for input_type in input_types:

        eager = next(
            item
            for item in results
            if item["mode"] == "eager"
            and item["input"] == input_type
        )

        lazy = next(
            item
            for item in results
            if item["mode"] == "lazy"
            and item["input"] == input_type
        )

        eager_memory = eager["peak_memory_mb"]
        lazy_memory = lazy["peak_memory_mb"]

        eager_time = eager["total_time_sec"]
        lazy_time = lazy["total_time_sec"]

        # ----------------------------------------------------
        # 메모리 감소율
        # ----------------------------------------------------

        if eager_memory > 0:

            memory_reduction = (
                (eager_memory - lazy_memory)
                / eager_memory
                * 100
            )

        else:
            memory_reduction = 0.0

        # ----------------------------------------------------
        # 시간 감소율
        # ----------------------------------------------------

        if eager_time > 0:

            time_reduction = (
                (eager_time - lazy_time)
                / eager_time
                * 100
            )

        else:
            time_reduction = 0.0

        print(
            f"{input_type:<10}"
            f"{eager_memory:>11.1f}M"
            f"{lazy_memory:>11.1f}M"
            f"{memory_reduction:>11.2f}%"
            f"{eager_time:>13.3f}s"
            f"{lazy_time:>13.3f}s"
            f"{time_reduction:>11.2f}%"
        )

        comparison.append(
            {
                "input": input_type,

                "eager_peak_memory_mb":
                    eager_memory,

                "lazy_peak_memory_mb":
                    lazy_memory,

                "memory_reduction_percent":
                    memory_reduction,

                "eager_initialization_time_sec":
                    eager["initialization_time_sec"],

                "lazy_initialization_time_sec":
                    lazy["initialization_time_sec"],

                "eager_inference_time_sec":
                    eager["inference_time_sec"],

                "lazy_inference_time_sec":
                    lazy["inference_time_sec"],

                "eager_total_time_sec":
                    eager_time,

                "lazy_total_time_sec":
                    lazy_time,

                "time_reduction_percent":
                    time_reduction,
            }
        )

    # ========================================================
    # 결과 JSON 저장
    # ========================================================

    output_directory = (
        PROJECT_ROOT
        / "evaluation"
        / "results"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_directory
        / "efficiency_text_image_video.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "experiment":
                    "Eager vs Router + Lazy Loading",

                "modalities": [
                    "text",
                    "image",
                    "video",
                ],

                "audio_used": False,

                "raw_results":
                    results,

                "comparison":
                    comparison,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78)
    print("Experiment finished.")
    print(f"Saved: {output_path}")
    print("=" * 78)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--worker",
        action="store_true"
    )

    parser.add_argument(
        "--mode",
        choices=[
            "eager",
            "lazy",
        ]
    )

    parser.add_argument(
        "--input",
        choices=[
            "text",
            "image",
            "video",
            "mixed",
        ]
    )

    args = parser.parse_args()

    if args.worker:

        if args.mode is None:
            raise ValueError(
                "--worker 사용 시 --mode가 필요합니다."
            )

        if args.input is None:
            raise ValueError(
                "--worker 사용 시 --input이 필요합니다."
            )

        run_worker(
            args.mode,
            args.input
        )

    else:

        run_parent()