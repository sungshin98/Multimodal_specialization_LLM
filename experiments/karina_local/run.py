"""One entry point from data setup to KARINA model execution checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from adaptive_rag.adaptive_multimodal_rag_V4 import (  # noqa: E402
    CallableVectorEncoder,
    InMemoryRetriever,
    Modality,
    RuleBasedComplexityAnalyzer,
    RuleBasedQuestionUnderstandingDecoder,
)
from adaptive_rag.role2_pipeline import KARINARole2Pipeline  # noqa: E402
from datasets.karina.adapter import KARINADataset, KARINAInputAdapter  # noqa: E402
from datasets.karina.config import load_config as load_data_config  # noqa: E402
from datasets.karina.io import iter_jsonl  # noqa: E402
from datasets.karina.movieqa import MovieQANormalizer  # noqa: E402
from datasets.karina.schema import validate_sample  # noqa: E402
from datasets.karina.setup import dataset_status, setup_movieqa  # noqa: E402


OFFLINE_TESTS = (
    "datasets.karina.smoke_test",
    "experiments.karina_local.test_runner",
    "adaptive_rag.test_pipeline_V4",
    "adaptive_rag.test_role2_pipeline",
    "evidence_decoder.test_offline",
    "adaptive_rag.integration_offline_test",
)

OFFICIAL_MOVIEQA_COUNTS = {"train": 9848, "val": 1958, "test": 3138}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_experiment_config(path: str | Path) -> Dict[str, Any]:
    config_path = _resolve(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("experiment configuration must be a mapping")
    config["_path"] = str(config_path)
    return config


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def package_versions() -> Dict[str, str | None]:
    names = (
        "torch",
        "torchvision",
        "transformers",
        "sentence-transformers",
        "numpy",
        "Pillow",
        "opencv-python",
        "opencv-python-headless",
        "PyYAML",
    )
    versions: Dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_preflight(config: Mapping[str, Any], data_config: Dict[str, Any]) -> Dict[str, Any]:
    checkpoints = {
        key: _resolve(value) for key, value in dict(config.get("checkpoints", {})).items()
    }
    cuda: Dict[str, Any] = {"available": False}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(index).total_memory / 1024**3, 2
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        cuda["reason"] = "torch is not installed"

    disk = shutil.disk_usage(REPOSITORY_ROOT)
    result = {
        "status": "passed",
        "repository_root": str(REPOSITORY_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda": cuda,
        "executables": {
            name: shutil.which(name) for name in ("git", "ffmpeg", "ffprobe", "nvidia-smi")
        },
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "data": dataset_status(data_config),
        "checkpoints": {
            key: {"path": str(path), "exists": path.is_file()}
            for key, path in checkpoints.items()
        },
    }
    result["full_model_ready"] = bool(checkpoints) and all(
        path.is_file() for path in checkpoints.values()
    )
    packages = result["packages"]
    video_reader_ready = bool(
        result["executables"].get("ffmpeg")
        or packages.get("opencv-python")
        or packages.get("opencv-python-headless")
    )
    result["pretrained_smoke_ready"] = bool(
        all(
            packages.get(name)
            for name in ("torch", "transformers", "sentence-transformers", "Pillow")
        )
        and video_reader_ready
    )
    issues = []
    if sys.version_info[:2] != (3, 11):
        issues.append("the reproducible Windows environment requires Python 3.11")
    if not result["pretrained_smoke_ready"]:
        issues.append("pretrained backbone smoke dependencies are incomplete")
    if not cuda.get("available"):
        issues.append("CUDA is unavailable; use CPU or repair the CUDA/PyTorch installation")
    if not result["full_model_ready"]:
        issues.append("learned KARINA checkpoints are absent")
    if issues:
        result["status"] = "warning"
    result["issues"] = issues
    return result


def run_data_setup(
    config: Mapping[str, Any], data_config: Dict[str, Any], limit: int | None
) -> Dict[str, Any]:
    setup_result = setup_movieqa(data_config)
    prepare_config = copy.deepcopy(data_config)
    if limit is not None:
        # A quick smoke run must never shrink a previously prepared full dataset.
        prepare_config["movieqa"]["output_dir"] = "processed/movieqa_smoke"
    start = time.perf_counter()
    counts = MovieQANormalizer(prepare_config, extract_frames=True).prepare(
        prepare_config["movieqa"].get("splits", ["train", "val", "test"]),
        limit=limit,
    )
    validation: Dict[str, Any] = {}
    valid = True
    # Reuse the normalizer's path resolver for custom absolute/relative configs.
    output_dir = MovieQANormalizer(prepare_config, extract_frames=False).output_dir
    for split, expected_count in counts.items():
        path = output_dir / f"{split}.jsonl"
        sample_count = 0
        schema_error_count = 0
        first_errors = []
        empty_initial_input = 0
        for line_number, sample in enumerate(iter_jsonl(path), 1):
            sample_count += 1
            errors = validate_sample(sample)
            schema_error_count += len(errors)
            if errors and len(first_errors) < 20:
                first_errors.extend(f"line {line_number}: {item}" for item in errors)
            if not any(
                sample.get("initial_input", {}).get(modality)
                for modality in ("text", "image", "video")
            ):
                empty_initial_input += 1
        official_count = OFFICIAL_MOVIEQA_COUNTS.get(split) if limit is None else None
        split_valid = (
            sample_count == expected_count
            and schema_error_count == 0
            and (official_count is None or sample_count == official_count)
        )
        valid = valid and split_valid
        validation[split] = {
            "path": str(path),
            "samples": sample_count,
            "expected": expected_count,
            "official_expected": official_count,
            "schema_errors": schema_error_count,
            "first_errors": first_errors[:20],
            "empty_initial_input": empty_initial_input,
            "valid": split_valid,
        }
    return {
        "status": "passed" if valid else "failed",
        "setup": setup_result,
        "normalized": counts,
        "normalized_output_dir": str(output_dir),
        "limit_per_split": limit,
        "scope": "full_official_metadata" if limit is None else "smoke_subset",
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "validation": validation,
        "availability_after": dataset_status(data_config),
    }


def load_movieqa_adapter_payload(
    data_config: Dict[str, Any], limit: int | None
) -> Dict[str, Any] | None:
    """Load one normalized MovieQA row through the real lazy Dataset/adapter."""

    candidates = []
    if limit is not None:
        smoke_config = copy.deepcopy(data_config)
        smoke_config["movieqa"]["output_dir"] = "processed/movieqa_smoke"
        candidates.append(MovieQANormalizer(smoke_config, extract_frames=False).output_dir)
    candidates.append(MovieQANormalizer(data_config, extract_frames=False).output_dir)
    for output_dir in candidates:
        train_path = output_dir / "train.jsonl"
        if not train_path.is_file():
            continue
        dataset = KARINADataset(train_path)
        if len(dataset) == 0:
            continue
        sample = dataset[0]
        payload = KARINAInputAdapter(data_config["data_root"]).to_router_payload(sample)
        return {
            "sample_id": sample["sample_id"],
            "source_jsonl": str(train_path),
            "answer_choice_count": len(sample.get("metadata", {}).get("answer_choices", [])),
            "available_file_count": len(payload["file_paths"]),
            "payload": payload,
        }
    return None


def run_normalized_movieqa_sample(
    router: Any, data_config: Dict[str, Any], limit: int | None
) -> Dict[str, Any]:
    """Pass one normalized row through KARINADataset, adapter, and router."""

    item = load_movieqa_adapter_payload(data_config, limit)
    if item is None:
        return {
            "status": "skipped",
            "reason": "no normalized MovieQA train.jsonl; run the data stage first",
        }
    payload = item.pop("payload")
    started = time.perf_counter()
    outputs = router.route(
        query=payload["question"], file_paths=payload["file_paths"]
    )
    file_count = int(item["available_file_count"])
    return {
        "status": "passed",
        **item,
        "mode": "normalized_movieqa_multimodal" if file_count else "metadata_query_only",
        "question_characters": len(payload["question"]),
        "encoder_outputs": summarize_encoder_outputs(outputs),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "performance_metrics_valid": False,
        "warning": (
            "No story/video assets were available; this verifies only the normalized "
            "MovieQA question -> text backbone path."
            if file_count == 0
            else "This verifies sample plumbing and forward execution, not QA accuracy."
        ),
    }


def run_offline_tests() -> Dict[str, Any]:
    results = []
    for module in OFFLINE_TESTS:
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
        )
        output = (process.stdout + "\n" + process.stderr).strip()
        print(output)
        results.append(
            {
                "module": module,
                "returncode": process.returncode,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "output_tail": output.splitlines()[-20:],
            }
        )
        if process.returncode != 0:
            return {"status": "failed", "tests": results}
    return {"status": "passed", "tests": results}


def deterministic_embedding(text: str, dim: int = 64) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed).normal(size=dim).astype(np.float32)


def build_role2() -> KARINARole2Pipeline:
    encoder = CallableVectorEncoder(deterministic_embedding)
    documents = {
        Modality.TEXT: [
            "The scene contains a person entering a room and opening a book.",
            "The room is blue and contains a table.",
        ],
        Modality.IMAGE: ["scene_reference.jpg"],
        Modality.VIDEO: ["scene_reference.mp4"],
    }
    retrievers = {
        modality: InMemoryRetriever(
            modality,
            encoder,
            values,
            np.vstack([deterministic_embedding(value) for value in values]),
        )
        for modality, values in documents.items()
    }
    return KARINARole2Pipeline(
        query_decoder=RuleBasedQuestionUnderstandingDecoder(),
        complexity_analyzer=RuleBasedComplexityAnalyzer(),
        retrievers=retrievers,
    )


def summarize_encoder_outputs(outputs: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for modality, items in outputs.items():
        summary[modality] = []
        for item in items:
            tensor = item.pooled_embedding
            summary[modality].append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "l2_norm": round(float(tensor.norm().item()), 6),
                }
            )
    return summary


def _run_router_and_role2(router: Any, assets: Mapping[str, Path]) -> Dict[str, Any]:
    question = "이 장면에서 사람이 무엇을 하는지 설명하고 근거로 검증해줘."
    file_paths = [assets["text"], assets["image"], assets["video"]]
    started = time.perf_counter()
    outputs = router.route(query=question, file_paths=file_paths)
    encoder_seconds = time.perf_counter() - started
    role2 = build_role2()
    packet = role2.run(
        question,
        encoder_outputs=outputs,
        file_paths=file_paths,
        content_hints={
            "document": ["A person enters a blue room and opens a book."],
            "image": ["A blue room containing a person and a table."],
            "video": ["A short sequence of a person moving across the room."],
        },
    )
    return {
        "encoder_outputs": summarize_encoder_outputs(outputs),
        "encoder_elapsed_seconds": round(encoder_seconds, 3),
        "retrieval_action": packet.retrieval_decision.action.value,
        "question_condition": packet.to_dict()["question_condition"],
        "retrieval_modalities": sorted(packet.retrieval_results),
        "packet_schema_version": packet.schema_version,
    }


def run_pretrained_smoke(
    config: Mapping[str, Any],
    device: str,
    data_config: Dict[str, Any],
    limit: int | None,
) -> Dict[str, Any]:
    try:
        import torch
        from experiments.karina_local.assets import ensure_smoke_assets
        from experiments.karina_local.pretrained_router import PretrainedSmokeRouter
    except ImportError as exc:
        return {"status": "failed", "reason": f"missing model dependency: {exc}"}

    if device == "cuda" and not torch.cuda.is_available():
        return {
            "status": "failed",
            "device": device,
            "reason": "CUDA was requested but torch.cuda.is_available() is false; rerun with --device cpu or repair the CUDA PyTorch installation",
        }
    try:
        assets = ensure_smoke_assets(_resolve(config["smoke_asset_dir"]))
        models = dict(config.get("pretrained_models", {}))
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        router = PretrainedSmokeRouter(
            device=device,
            text_model=models.get("text", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            image_model=models.get("image", "openai/clip-vit-base-patch32"),
            video_model=models.get("video", "MCG-NJU/videomae-base-finetuned-kinetics"),
            video_frames=int(models.get("video_frames", 16)),
        )
        result = _run_router_and_role2(router, assets)
        result["normalized_movieqa_sample"] = run_normalized_movieqa_sample(
            router, data_config, limit
        )
        result.update(
            {
                "status": "passed",
                "device": device,
                "checkpoint_free": True,
                "performance_metrics_valid": False,
                "synthetic_multimodal_assets": True,
                "warning": "The multimodal assets are synthetic and the backbones are real, but learned KARINA projection checkpoints are not used.",
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                if device == "cuda"
                else 0.0,
            }
        )
        return result
    except Exception as exc:
        return {"status": "failed", "device": device, "reason": repr(exc)}


def run_full_checkpoint_smoke(
    config: Mapping[str, Any],
    device: str,
    require_full: bool,
    data_config: Dict[str, Any],
    limit: int | None,
) -> Dict[str, Any]:
    checkpoints = {key: _resolve(value) for key, value in config["checkpoints"].items()}
    missing = [str(path) for path in checkpoints.values() if not path.is_file()]
    if missing:
        status = "failed" if require_full else "skipped"
        return {
            "status": status,
            "reason": "repository checkpoints are not stored in Git",
            "missing": missing,
        }
    try:
        import torch
        from experiments.karina_local.assets import ensure_smoke_assets

        if device == "cuda" and not torch.cuda.is_available():
            return {
                "status": "failed",
                "device": device,
                "reason": "CUDA was requested but torch.cuda.is_available() is false; rerun with --device cpu or repair the CUDA PyTorch installation",
            }

        modular_root = REPOSITORY_ROOT / "modular_encoder"
        if str(modular_root) not in sys.path:
            sys.path.insert(0, str(modular_root))
        from router.input_router import InputRouter

        assets = ensure_smoke_assets(_resolve(config["smoke_asset_dir"]))
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        router = InputRouter(
            checkpoint_path=checkpoints["base"],
            video_checkpoint_path=checkpoints["video"],
            device=device,
        )
        result = _run_router_and_role2(router, assets)
        result["normalized_movieqa_sample"] = run_normalized_movieqa_sample(
            router, data_config, limit
        )
        result.update(
            {
                "status": "passed",
                "device": device,
                "checkpoint_free": False,
                "performance_metrics_valid": False,
                "synthetic_multimodal_assets": True,
                "warning": "This is a forward/plumbing smoke test, not a MovieQA accuracy benchmark.",
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                if device == "cuda"
                else 0.0,
            }
        )
        return result
    except Exception as exc:
        return {"status": "failed", "device": device, "reason": repr(exc)}


def save_report(config: Mapping[str, Any], report: Dict[str, Any]) -> Path:
    output_dir = _resolve(config["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = output_dir / f"karina_local_run_{stamp}.json"
    latest = output_dir / "latest.json"
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    stamped.write_text(text, encoding="utf-8")
    latest.write_text(text, encoding="utf-8")
    return stamped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KARINA local experiment workflow")
    parser.add_argument("--config", default="configs/karina_local_experiment.yaml")
    parser.add_argument(
        "--stage",
        choices=["all", "preflight", "data", "offline", "pretrained", "full"],
        default="all",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="MovieQA samples per split; 0 means all; omitted uses config",
    )
    parser.add_argument("--require-full", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_experiment_config(args.config)
    data_config = load_data_config(_resolve(config["data_config"]))
    requested_device = args.device or str(config.get("device", "auto"))
    device = choose_device(requested_device)
    configured_limit = int(config.get("prepare_limit", 20))
    limit_value = configured_limit if args.limit is None else args.limit
    limit = None if limit_value == 0 else limit_value

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", str(REPOSITORY_ROOT / ".cache" / "huggingface"))

    report: Dict[str, Any] = {
        "schema_version": "karina.local-experiment.v1",
        "started_at": datetime.now().astimezone().isoformat(),
        "config": str(config["_path"]),
        "stage": args.stage,
        "device": device,
        "results": {},
    }
    failed = False
    stages = (
        ["preflight", "data", "offline", "pretrained", "full"]
        if args.stage == "all"
        else [args.stage]
    )
    try:
        for stage in stages:
            print(f"\n[KARINA] stage={stage}")
            if stage == "preflight":
                result = run_preflight(config, data_config)
            elif stage == "data":
                result = run_data_setup(config, data_config, limit)
            elif stage == "offline":
                result = run_offline_tests()
            elif stage == "pretrained":
                result = run_pretrained_smoke(config, device, data_config, limit)
            elif stage == "full":
                result = run_full_checkpoint_smoke(
                    config, device, args.require_full, data_config, limit
                )
            else:  # pragma: no cover
                raise AssertionError(stage)
            report["results"][stage] = result
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            if result.get("status") == "failed":
                failed = True
                break
    except Exception as exc:
        report["fatal_error"] = repr(exc)
        failed = True
    report["finished_at"] = datetime.now().astimezone().isoformat()
    result_statuses = [
        str(result.get("status")) for result in report["results"].values()
    ]
    full_result = report["results"].get("full")
    report["full_model_executed"] = (
        None if full_result is None else full_result.get("status") == "passed"
    )
    if failed:
        report["status"] = "failed"
    elif any(status in {"warning", "skipped"} for status in result_statuses):
        report["status"] = "passed_with_limitations"
    else:
        report["status"] = "passed"
    output = save_report(config, report)
    print(f"\nReport: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
