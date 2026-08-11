from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from encoders.document.document_encoder import DocumentEncoder
from encoders.image.image_encoder import ImageEncoder
from encoders.video.video_encoder import VideoEncoder
from models.alignment_model import (
    MultimodalAlignmentModel,
    filtered_alignment_state_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "train_manifest.jsonl"
)

VAL_MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "val_manifest.jsonl"
)

BASE_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "resplit_135"
    / "joint_finetuned_best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "video_added"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "video_projection"
)

LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"

LOG_PATH = LOG_DIR / "video_projection_overnight.log"

BEST_CHECKPOINT_PATH = (
    OUTPUT_DIR / "video_projection_best.pt"
)

LAST_CHECKPOINT_PATH = (
    OUTPUT_DIR / "video_projection_last.pt"
)

SUMMARY_JSON_PATH = (
    REPORT_DIR / "video_projection_summary.json"
)

SUMMARY_TXT_PATH = (
    REPORT_DIR / "video_projection_summary.txt"
)


BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07
PATIENCE = 6
RANDOM_SEED = 42


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"

    print(line, flush=True)

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(line + "\n")


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            scene_id = (
                row.get("scene_id")
                or row.get("group_id")
            )

            if not scene_id:
                raise ValueError(
                    "Missing scene_id/group_id at "
                    f"{path}:{line_number}"
                )

            required_fields = (
                "text",
                "image_path",
                "clip_path",
            )

            for field in required_fields:
                if not row.get(field):
                    raise ValueError(
                        f"Missing {field} at "
                        f"{path}:{line_number}"
                    )

            row["scene_id"] = str(scene_id)
            rows.append(row)

    if not rows:
        raise ValueError(
            f"Manifest is empty: {path}"
        )

    return rows


def load_base_models(
    device: torch.device,
) -> tuple[
    DocumentEncoder,
    ImageEncoder,
    VideoEncoder,
    MultimodalAlignmentModel,
]:
    if not BASE_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Base checkpoint not found: "
            f"{BASE_CHECKPOINT_PATH}"
        )

    log("기존 체크포인트 불러오기")

    checkpoint = torch.load(
        BASE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "document_encoder_state_dict",
        "image_encoder_state_dict",
        "alignment_model_state_dict",
    }

    missing = required_keys - checkpoint.keys()

    if missing:
        raise KeyError(
            "Base checkpoint missing keys: "
            + ", ".join(sorted(missing))
        )

    log("DocumentEncoder 불러오기")
    document_encoder = DocumentEncoder()

    document_encoder.model.load_state_dict(
        checkpoint[
            "document_encoder_state_dict"
        ]
    )

    document_encoder.model.to(device)
    document_encoder.model.eval()

    log("ImageEncoder 불러오기")
    image_encoder = ImageEncoder()

    image_encoder.model.load_state_dict(
        checkpoint[
            "image_encoder_state_dict"
        ]
    )

    image_encoder.model.to(device)
    image_encoder.model.eval()

    log("AlignmentModel 불러오기")
    alignment_model = MultimodalAlignmentModel()

    base_alignment_state = filtered_alignment_state_dict(
        checkpoint["alignment_model_state_dict"]
    )

    # The historical base checkpoint may not contain video_projection.*.
    # Load the available Text/Image projection weights and keep the
    # newly initialized Video projection head for training.
    text_image_state = {
        key: value
        for key, value in base_alignment_state.items()
        if key.startswith(("text_projection.", "image_projection."))
    }

    load_result = alignment_model.load_state_dict(
        text_image_state,
        strict=False,
    )

    invalid_missing = [
        key
        for key in load_result.missing_keys
        if not key.startswith("video_projection.")
    ]

    if invalid_missing or load_result.unexpected_keys:
        raise RuntimeError(
            "Alignment checkpoint mismatch: "
            f"missing={invalid_missing}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    alignment_model.to(device)
    alignment_model.eval()

    for parameter in (
        alignment_model.text_projection.parameters()
    ):
        parameter.requires_grad = False

    for parameter in (
        alignment_model.image_projection.parameters()
    ):
        parameter.requires_grad = False

    log("VideoEncoder 불러오기")
    video_encoder = VideoEncoder(
        device=str(device)
    )

    return (
        document_encoder,
        image_encoder,
        video_encoder,
        alignment_model,
    )


def save_feature_cache(
    path: Path,
    scene_ids: list[str],
    text_anchors: list[torch.Tensor],
    image_anchors: list[torch.Tensor],
    video_features: list[torch.Tensor],
    expected_count: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {
        "scene_ids": list(scene_ids),
        "text_anchors": torch.stack(
            text_anchors
        ),
        "image_anchors": torch.stack(
            image_anchors
        ),
        "video_features": torch.stack(
            video_features
        ),
        "completed_count": len(scene_ids),
        "expected_count": expected_count,
    }

    torch.save(
        cache,
        path,
    )


def extract_split_features(
    rows: list[dict[str, Any]],
    label: str,
    cache_path: Path,
    document_encoder: DocumentEncoder,
    image_encoder: ImageEncoder,
    video_encoder: VideoEncoder,
    alignment_model: MultimodalAlignmentModel,
    device: torch.device,
    reset_cache: bool,
) -> dict[str, torch.Tensor]:
    expected_scene_ids = [
        row["scene_id"]
        for row in rows
    ]

    completed_scene_ids: list[str] = []
    text_anchors: list[torch.Tensor] = []
    image_anchors: list[torch.Tensor] = []
    video_features: list[torch.Tensor] = []

    if cache_path.is_file() and not reset_cache:
        existing = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )

        cached_ids = existing.get(
            "scene_ids",
            [],
        )

        valid_prefix = (
            cached_ids
            == expected_scene_ids[
                : len(cached_ids)
            ]
        )

        expected_matches = (
            existing.get("expected_count")
            == len(rows)
        )

        if valid_prefix and expected_matches:
            completed_scene_ids = list(
                cached_ids
            )

            text_anchors = list(
                existing["text_anchors"]
            )

            image_anchors = list(
                existing["image_anchors"]
            )

            video_features = list(
                existing["video_features"]
            )

            log(
                f"{label} 캐시 재개: "
                f"{len(completed_scene_ids)}"
                f"/{len(rows)}"
            )

        else:
            log(
                f"{label} 캐시가 현재 manifest와 "
                "일치하지 않아 새로 생성"
            )

    start_index = len(
        completed_scene_ids
    )

    started = time.time()

    for index in range(
        start_index,
        len(rows),
    ):
        row = rows[index]
        scene_id = row["scene_id"]

        image_path = (
            PROJECT_ROOT / row["image_path"]
        )

        video_path = (
            PROJECT_ROOT / row["clip_path"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        with torch.no_grad():
            text_output = (
                document_encoder.encode(
                    row["text"].strip()
                )
            )

            image_output = (
                image_encoder.encode(
                    image_path
                )
            )

            video_output = (
                video_encoder.encode(
                    video_path
                )
            )

            raw_text = (
                text_output
                .pooled_embedding
                .to(device)
            )

            raw_image = (
                image_output
                .pooled_embedding
                .to(device)
            )

            text_anchor = (
                alignment_model
                .text_projection(raw_text)
                .squeeze(0)
                .cpu()
            )

            image_anchor = (
                alignment_model
                .image_projection(raw_image)
                .squeeze(0)
                .cpu()
            )

            video_feature = (
                video_output
                .pooled_embedding
                .squeeze(0)
                .cpu()
            )

        completed_scene_ids.append(
            scene_id
        )

        text_anchors.append(
            text_anchor
        )

        image_anchors.append(
            image_anchor
        )

        video_features.append(
            video_feature
        )

        completed = index + 1

        if (
            completed == 1
            or completed % 10 == 0
            or completed == len(rows)
        ):
            elapsed = time.time() - started

            newly_completed = (
                completed - start_index
            )

            seconds_per_scene = (
                elapsed
                / max(1, newly_completed)
            )

            remaining = (
                len(rows) - completed
            )

            estimated_remaining = (
                remaining
                * seconds_per_scene
            )

            log(
                f"{label} {completed}/{len(rows)} | "
                f"scene={scene_id} | "
                f"평균={seconds_per_scene:.1f}초/장면 | "
                f"남은 예상="
                f"{estimated_remaining / 3600:.2f}시간"
            )

        if (
            completed % 25 == 0
            or completed == len(rows)
        ):
            save_feature_cache(
                path=cache_path,
                scene_ids=completed_scene_ids,
                text_anchors=text_anchors,
                image_anchors=image_anchors,
                video_features=video_features,
                expected_count=len(rows),
            )

            log(
                f"{label} 중간 캐시 저장: "
                f"{cache_path}"
            )

    return {
        "text_anchors": torch.stack(
            text_anchors
        ),
        "image_anchors": torch.stack(
            image_anchors
        ),
        "video_features": torch.stack(
            video_features
        ),
    }


def symmetric_contrastive_loss(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    labels = torch.arange(
        first.size(0),
        device=first.device,
    )

    logits = (
        first @ second.T
    ) / TEMPERATURE

    loss_first = F.cross_entropy(
        logits,
        labels,
    )

    loss_second = F.cross_entropy(
        logits.T,
        labels,
    )

    return (
        loss_first + loss_second
    ) / 2


def video_alignment_loss(
    text_anchor: torch.Tensor,
    image_anchor: torch.Tensor,
    video_embedding: torch.Tensor,
) -> torch.Tensor:
    text_video = symmetric_contrastive_loss(
        text_anchor,
        video_embedding,
    )

    image_video = symmetric_contrastive_loss(
        image_anchor,
        video_embedding,
    )

    return (
        text_video + image_video
    ) / 2


@torch.no_grad()
def evaluate(
    model: MultimodalAlignmentModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    losses: list[float] = []

    for (
        text_anchor,
        image_anchor,
        video_feature,
    ) in loader:
        text_anchor = text_anchor.to(device)
        image_anchor = image_anchor.to(device)
        video_feature = video_feature.to(device)

        video_embedding = (
            model.video_projection(
                video_feature
            )
        )

        loss = video_alignment_loss(
            text_anchor,
            image_anchor,
            video_embedding,
        )

        losses.append(
            loss.item()
        )

    return (
        sum(losses)
        / max(1, len(losses))
    )


def train_video_projection(
    alignment_model: MultimodalAlignmentModel,
    train_features: dict[str, torch.Tensor],
    val_features: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int,
) -> dict[str, Any]:
    train_dataset = TensorDataset(
        train_features["text_anchors"],
        train_features["image_anchors"],
        train_features["video_features"],
    )

    val_dataset = TensorDataset(
        val_features["text_anchors"],
        val_features["image_anchors"],
        val_features["video_features"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(
            BATCH_SIZE,
            len(train_dataset),
        ),
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(
            BATCH_SIZE,
            len(val_dataset),
        ),
        shuffle=False,
    )

    alignment_model.to(device)

    for parameter in alignment_model.parameters():
        parameter.requires_grad = False

    for parameter in (
        alignment_model
        .video_projection
        .parameters()
    ):
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        alignment_model
        .video_projection
        .parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("Video Projection 학습 시작")

    for epoch in range(
        1,
        epochs + 1,
    ):
        alignment_model.train()

        train_losses: list[float] = []

        for (
            text_anchor,
            image_anchor,
            video_feature,
        ) in train_loader:
            text_anchor = text_anchor.to(device)
            image_anchor = image_anchor.to(device)
            video_feature = video_feature.to(device)

            optimizer.zero_grad()

            video_embedding = (
                alignment_model
                .video_projection(
                    video_feature
                )
            )

            loss = video_alignment_loss(
                text_anchor,
                image_anchor,
                video_embedding,
            )

            loss.backward()
            optimizer.step()

            train_losses.append(
                loss.item()
            )

        train_loss = (
            sum(train_losses)
            / max(1, len(train_losses))
        )

        val_loss = evaluate(
            alignment_model,
            val_loader,
            device,
        )

        log(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f}"
        )

        torch.save(
            {
                "alignment_model_state_dict": (
                    alignment_model.state_dict()
                ),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "video_projection_trained": True,
                "train_count": len(train_dataset),
                "val_count": len(val_dataset),
            },
            LAST_CHECKPOINT_PATH,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0

            torch.save(
                {
                    "alignment_model_state_dict": (
                        alignment_model.state_dict()
                    ),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "video_projection_trained": True,
                    "train_count": len(train_dataset),
                    "val_count": len(val_dataset),
                    "temperature": TEMPERATURE,
                },
                BEST_CHECKPOINT_PATH,
            )

            log(
                "최적 체크포인트 저장: "
                f"{BEST_CHECKPOINT_PATH}"
            )

        else:
            stale_epochs += 1

            if stale_epochs >= PATIENCE:
                log(
                    "Early stopping: "
                    f"epoch {epoch}"
                )
                break

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
    }


def save_summary(
    summary: dict[str, Any],
) -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "Video Projection Overnight Summary",
        "=" * 50,
        f"Train scenes: {summary['train_count']}",
        f"Validation scenes: {summary['val_count']}",
        f"Best epoch: {summary['best_epoch']}",
        (
            "Best validation loss: "
            f"{summary['best_val_loss']:.6f}"
        ),
        (
            "Elapsed hours: "
            f"{summary['elapsed_hours']:.3f}"
        ),
        (
            "Best checkpoint: "
            f"{BEST_CHECKPOINT_PATH}"
        ),
        (
            "Train cache: "
            f"{summary['train_cache']}"
        ),
        (
            "Validation cache: "
            f"{summary['val_cache']}"
        ),
    ]

    SUMMARY_TXT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "각 split에서 사용할 최대 장면 수. "
            "0이면 전체 사용."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
    )

    parser.add_argument(
        "--reset-cache",
        action="store_true",
    )

    args = parser.parse_args()

    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    started = time.time()
    device = torch.device("cpu")

    log("=" * 70)
    log("Video Projection 야간 파이프라인 시작")
    log(f"Device: {device}")

    train_rows = load_jsonl(
        TRAIN_MANIFEST_PATH
    )

    val_rows = load_jsonl(
        VAL_MANIFEST_PATH
    )

    if args.limit > 0:
        train_rows = train_rows[
            : args.limit
        ]

        val_rows = val_rows[
            : min(
                args.limit,
                len(val_rows),
            )
        ]

    log(
        f"장면 수: train={len(train_rows)}, "
        f"val={len(val_rows)}"
    )

    (
        document_encoder,
        image_encoder,
        video_encoder,
        alignment_model,
    ) = load_base_models(device)

    cache_suffix = (
        "full"
        if args.limit <= 0
        else f"limit_{args.limit}"
    )

    train_cache_path = (
        CACHE_DIR
        / f"train_{cache_suffix}.pt"
    )

    val_cache_path = (
        CACHE_DIR
        / f"val_{cache_suffix}.pt"
    )

    train_features = extract_split_features(
        rows=train_rows,
        label="train",
        cache_path=train_cache_path,
        document_encoder=document_encoder,
        image_encoder=image_encoder,
        video_encoder=video_encoder,
        alignment_model=alignment_model,
        device=device,
        reset_cache=args.reset_cache,
    )

    val_features = extract_split_features(
        rows=val_rows,
        label="val",
        cache_path=val_cache_path,
        document_encoder=document_encoder,
        image_encoder=image_encoder,
        video_encoder=video_encoder,
        alignment_model=alignment_model,
        device=device,
        reset_cache=args.reset_cache,
    )

    result = train_video_projection(
        alignment_model=alignment_model,
        train_features=train_features,
        val_features=val_features,
        device=device,
        epochs=args.epochs,
    )

    elapsed = time.time() - started

    summary = {
        **result,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600,
        "train_cache": str(
            train_cache_path
        ),
        "val_cache": str(
            val_cache_path
        ),
        "base_checkpoint": str(
            BASE_CHECKPOINT_PATH
        ),
        "best_checkpoint": str(
            BEST_CHECKPOINT_PATH
        ),
    }

    save_summary(summary)

    log(
        "전체 완료 | "
        f"소요시간={elapsed / 3600:.2f}시간"
    )

    log(
        f"요약 파일: {SUMMARY_TXT_PATH}"
    )


if __name__ == "__main__":
    main()