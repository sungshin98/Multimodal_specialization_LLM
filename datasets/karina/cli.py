"""Command-line entry point for setup, preparation and validation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from .config import data_path, load_config
from .moviechat import MovieChatNormalizer
from .movienet import build_movienet_pool
from .movieqa import MovieQANormalizer
from .setup import dataset_status, setup_movieqa
from .validate import validate_many


def _movieqa_ids(config: Dict[str, Any]) -> Set[str]:
    path = data_path(config, config["movieqa"]["benchmark_dir"]) / "data" / "movies.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["imdb_key"]).lower() for item in payload}


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def command_setup(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    if args.dataset not in {"movieqa", "all"}:
        raise SystemExit(
            "Only MovieQA metadata is automatically downloadable. MovieNet and MovieChat-1K require account/terms-aware manual setup; run status for paths."
        )
    result = setup_movieqa(config, force=args.force)
    _print(result)
    return 0


def command_status(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    _print(dataset_status(config))
    return 0


def command_prepare(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    requested = {args.dataset} if args.dataset != "all" else {"movieqa", "movienet", "moviechat_1k"}
    result: Dict[str, Any] = {}
    limit = None if args.limit in (None, 0) else int(args.limit)
    if limit is not None and limit < 0:
        raise SystemExit("--limit must be 0 (all) or a positive integer")
    prepare_config = copy.deepcopy(config)
    if limit is not None:
        # Never let a quick CLI check shrink canonical full-data artifacts.
        prepare_config["movieqa"]["output_dir"] = "processed/movieqa_smoke"
        prepare_config["moviechat_1k"]["output_file"] = (
            "processed/moviechat_1k_smoke/generalization_test.jsonl"
        )
    movienet_matches = None
    if "movienet" in requested and (args.dataset == "movienet" or config["movienet"].get("enabled")):
        count, movienet_matches = build_movienet_pool(
            prepare_config, _movieqa_ids(prepare_config)
        )
        result["movienet_evidence"] = count
    if "movieqa" in requested and (args.dataset == "movieqa" or config["movieqa"].get("enabled", True)):
        normalizer = MovieQANormalizer(prepare_config, extract_frames=not args.no_frames)
        result["movieqa"] = normalizer.prepare(
            prepare_config["movieqa"].get("splits", ["train", "val", "test"]),
            movienet_by_movie=movienet_matches,
            limit=limit,
        )
        result["movieqa_output_dir"] = str(normalizer.output_dir)
    if "moviechat_1k" in requested and (
        args.dataset == "moviechat_1k" or config["moviechat_1k"].get("enabled")
    ):
        moviechat = MovieChatNormalizer(
            prepare_config, extract_frames=not args.no_frames
        )
        result["moviechat_1k"] = moviechat.prepare(limit=limit)
        result["moviechat_1k_output_file"] = str(moviechat.output)
    result["scope"] = "smoke_subset" if limit is not None else "full_configured_data"
    result["limit_per_split"] = limit
    _print(result)
    return 0


def command_validate(args: argparse.Namespace, config: Dict[str, Any]) -> int:
    paths: List[Path]
    if args.path:
        paths = [Path(args.path)]
    else:
        processed = Path(config["data_root"]) / "processed"
        paths = sorted(processed.rglob("*.jsonl")) if processed.is_dir() else []
    if not paths:
        raise SystemExit("No normalized JSONL files found. Run prepare first or pass --path.")
    reports = validate_many(paths, config["data_root"])
    _print(reports)
    return 0 if all(report["valid"] for report in reports) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KARINA movie data pipeline")
    parser.add_argument("--config", default="configs/karina_data.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="download policy-safe metadata")
    setup_parser.add_argument("--dataset", choices=["movieqa", "all"], default="movieqa")
    setup_parser.add_argument("--force", action="store_true")
    setup_parser.set_defaults(handler=command_setup)

    status_parser = subparsers.add_parser("status", help="show local/manual setup status")
    status_parser.set_defaults(handler=command_status)

    prepare_parser = subparsers.add_parser("prepare", help="normalize configured datasets")
    prepare_parser.add_argument(
        "--dataset", choices=["movieqa", "movienet", "moviechat_1k", "all"], default="all"
    )
    prepare_parser.add_argument(
        "--limit",
        type=int,
        help="samples per split; 0 means all, a positive value writes smoke outputs",
    )
    prepare_parser.add_argument("--no-frames", action="store_true")
    prepare_parser.set_defaults(handler=command_prepare)

    validate_parser = subparsers.add_parser("validate", help="validate normalized JSONL")
    validate_parser.add_argument("--path")
    validate_parser.set_defaults(handler=command_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    return int(args.handler(args, config))


if __name__ == "__main__":
    raise SystemExit(main())
