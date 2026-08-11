from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from router.input_router import InputRouter


TEST_MANIFEST = PROJECT_ROOT / "data" / "test_manifest.jsonl"
REPORT_DIR = PROJECT_ROOT / "reports" / "resplit_135"

TOP_K_VALUES = (1, 5, 10)

DIRECTIONS = (
    ("text", "image"),
    ("text", "video"),
    ("image", "text"),
    ("image", "video"),
    ("video", "text"),
    ("video", "image"),
)


def resolve_project_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_video_clip(row: dict[str, Any]) -> Path:
    """
    장면 단위 video clip을 찾는다.

    우선순위:
    1) manifest의 clip_path
    2) data/processed/clips/<movie>/<scene_id>.mp4

    원본 영화 전체를 가리킬 수 있는 video_path는 검색 평가에 사용하지 않는다.
    """
    clip_path = row.get("clip_path")

    if clip_path:
        path = resolve_project_path(str(clip_path))
        if path.is_file():
            return path

    movie = row.get("movie")
    scene_id = row.get("scene_id")

    if movie and scene_id:
        path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "clips"
            / str(movie)
            / f"{scene_id}.mp4"
        )

        if path.is_file():
            return path

    raise FileNotFoundError(
        "Scene video clip not found. "
        "Expected manifest field 'clip_path' or "
        "data/processed/clips/<movie>/<scene_id>.mp4"
    )


def load_test_rows(
    manifest_path: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Test manifest not found: {manifest_path}"
        )

    rows: list[dict[str, Any]] = []

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            required = ("text", "image_path")
            missing = [
                key
                for key in required
                if not row.get(key)
            ]

            if missing:
                raise ValueError(
                    f"Missing fields {missing} at "
                    f"{manifest_path}:{line_number}"
                )

            image_path = resolve_project_path(
                str(row["image_path"])
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found at line {line_number}: "
                    f"{image_path}"
                )

            video_path = resolve_video_clip(row)

            prepared = dict(row)
            prepared["_image_path"] = image_path
            prepared["_video_path"] = video_path
            prepared["_scene_id"] = (
                str(
                    row.get("scene_id")
                    or row.get("group_id")
                    or f"scene_{len(rows):05d}"
                )
            )

            rows.append(prepared)

            if limit is not None and len(rows) >= limit:
                break

    if len(rows) < 2:
        raise ValueError(
            "At least two test scenes are required."
        )

    return rows


@torch.no_grad()
def extract_embeddings(
    rows: list[dict[str, Any]],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    최종 InputRouter를 그대로 사용해
    Text / Image / Video 임베딩을 추출한다.

    따라서 실제 배포 구조와 동일하게:
    - Text projection
    - Image projection
    - Video projection
    - L2 normalization
    이 적용된 최종 출력으로 평가한다.
    """
    router = InputRouter(device=device)

    collected: dict[str, list[torch.Tensor]] = {
        "text": [],
        "image": [],
        "video": [],
    }

    total = len(rows)

    for index, row in enumerate(rows, start=1):
        text_output = router.route_query(
            str(row["text"]).strip()
        )

        image_output = router.route_file(
            str(row["_image_path"])
        )

        video_output = router.route_file(
            str(row["_video_path"])
        )

        text_embedding = F.normalize(
            text_output.pooled_embedding.detach().cpu(),
            p=2,
            dim=-1,
        )

        image_embedding = F.normalize(
            image_output.pooled_embedding.detach().cpu(),
            p=2,
            dim=-1,
        )

        video_embedding = F.normalize(
            video_output.pooled_embedding.detach().cpu(),
            p=2,
            dim=-1,
        )

        collected["text"].append(text_embedding)
        collected["image"].append(image_embedding)
        collected["video"].append(video_embedding)

        print(
            f"[{index:03d}/{total:03d}] "
            f"{row['_scene_id']} completed",
            flush=True,
        )

    return {
        modality: torch.cat(parts, dim=0)
        for modality, parts in collected.items()
    }


def compute_metrics(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
) -> dict[str, float]:
    query_embeddings = F.normalize(
        query_embeddings,
        p=2,
        dim=1,
    )
    candidate_embeddings = F.normalize(
        candidate_embeddings,
        p=2,
        dim=1,
    )

    similarities = (
        query_embeddings
        @ candidate_embeddings.T
    )

    rankings = torch.argsort(
        similarities,
        dim=1,
        descending=True,
    )

    targets = torch.arange(
        similarities.size(0)
    ).unsqueeze(1)

    matches = rankings.eq(targets)

    metrics: dict[str, float] = {}

    for k in TOP_K_VALUES:
        effective_k = min(
            k,
            similarities.size(1),
        )

        metrics[f"recall_at_{k}"] = (
            matches[:, :effective_k]
            .any(dim=1)
            .float()
            .mean()
            .item()
        )

    # 각 행에는 자기 정답 index가 반드시 하나 있으므로
    # argmax 결과가 정답 순위를 반환한다.
    ranks = (
        matches.float().argmax(dim=1).float()
        + 1.0
    )

    metrics["mrr"] = (
        (1.0 / ranks).mean().item()
    )

    positive_cosine = (
        similarities.diag().mean().item()
    )

    negative_mask = ~torch.eye(
        similarities.size(0),
        dtype=torch.bool,
    )

    negative_cosine = (
        similarities[negative_mask]
        .mean()
        .item()
    )

    metrics["mean_positive_cosine"] = (
        positive_cosine
    )
    metrics["mean_negative_cosine"] = (
        negative_cosine
    )
    metrics["cosine_margin"] = (
        positive_cosine - negative_cosine
    )

    return metrics


def save_results(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        REPORT_DIR
        / "joint_retrieval_135_text_image_video.csv"
    )

    json_path = (
        REPORT_DIR
        / "joint_retrieval_135_text_image_video.json"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    json_path.write_text(
        json.dumps(
            {
                "modalities": [
                    "text",
                    "image",
                    "video",
                ],
                "audio_used": False,
                "results": results,
                "mean_over_6_directions": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Saved result files:")
    print(csv_path)
    print(json_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "0이면 전체 test set. "
            "예: --limit 3 은 3개 장면 smoke test."
        ),
    )

    args = parser.parse_args()

    limit = (
        args.limit
        if args.limit > 0
        else None
    )

    rows = load_test_rows(
        TEST_MANIFEST,
        limit=limit,
    )

    print("=" * 72)
    print("Text / Image / Video Retrieval Evaluation")
    print("=" * 72)
    print(f"Test scenes : {len(rows)}")
    print(f"Manifest    : {TEST_MANIFEST}")
    print("Audio used  : False")
    print()

    embeddings = extract_embeddings(
        rows,
        device="cpu",
    )

    print()
    print("Embedding shapes:")
    print(
        f"  text : {tuple(embeddings['text'].shape)}"
    )
    print(
        f"  image: {tuple(embeddings['image'].shape)}"
    )
    print(
        f"  video: {tuple(embeddings['video'].shape)}"
    )

    results: list[dict[str, Any]] = []

    print()
    print("Retrieval results:")

    for query_modality, candidate_modality in DIRECTIONS:
        metrics = compute_metrics(
            embeddings[query_modality],
            embeddings[candidate_modality],
        )

        row = {
            "query_modality": query_modality,
            "candidate_modality": candidate_modality,
            "test_scenes": len(rows),
            **metrics,
        }

        results.append(row)

        print(
            f"{query_modality:>5} -> "
            f"{candidate_modality:<5} | "
            f"R@1={metrics['recall_at_1']:.4f} "
            f"R@5={metrics['recall_at_5']:.4f} "
            f"R@10={metrics['recall_at_10']:.4f} "
            f"MRR={metrics['mrr']:.4f} "
            f"margin={metrics['cosine_margin']:.4f}"
        )

    summary = {
        "recall_at_1": sum(
            row["recall_at_1"]
            for row in results
        ) / len(results),

        "recall_at_5": sum(
            row["recall_at_5"]
            for row in results
        ) / len(results),

        "recall_at_10": sum(
            row["recall_at_10"]
            for row in results
        ) / len(results),

        "mrr": sum(
            row["mrr"]
            for row in results
        ) / len(results),

        "cosine_margin": sum(
            row["cosine_margin"]
            for row in results
        ) / len(results),
    }

    print()
    print("Mean over 6 directions:")
    print(
        f"R@1 : {summary['recall_at_1']:.4f}"
    )
    print(
        f"R@5 : {summary['recall_at_5']:.4f}"
    )
    print(
        f"R@10: {summary['recall_at_10']:.4f}"
    )
    print(
        f"MRR : {summary['mrr']:.4f}"
    )
    print(
        f"Margin: {summary['cosine_margin']:.4f}"
    )

    if limit is None:
        save_results(
            results,
            summary,
        )

    else:
        print()
        print(
            "Smoke test mode: "
            "result files were not saved."
        )

    print()
    print(
        "Text / Image / Video retrieval "
        "evaluation completed."
    )


if __name__ == "__main__":
    main()
