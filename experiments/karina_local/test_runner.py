"""Dependency-light regression tests for the local experiment orchestrator."""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path

from datasets.karina.cli import command_prepare
from experiments.karina_local.run import (
    run_data_setup,
    run_normalized_movieqa_sample,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class _Scalar:
    def item(self) -> float:
        return 1.0


class _Tensor:
    shape = (1, 768)
    dtype = "float32"

    @staticmethod
    def norm() -> _Scalar:
        return _Scalar()


class _Output:
    pooled_embedding = _Tensor()


class _Router:
    @staticmethod
    def route(query: str, file_paths: list[Path]) -> dict[str, list[_Output]]:
        assert query
        assert file_paths == []
        return {"query": [_Output()]}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="karina_runner_test_") as temporary:
        root = Path(temporary)
        benchmark = root / "raw/movieqa/benchmark"
        movie_id = "tt0000001"
        _write_json(
            benchmark / "data/movies.json",
            [{"imdb_key": movie_id, "name": "Fixture", "text": {}}],
        )
        _write_json(
            benchmark / "data/qa.json",
            [
                {
                    "qid": "fixture:0",
                    "question": "What happens?",
                    "answers": ["A", "B", "C", "D", "E"],
                    "correct_index": 0,
                    "imdb_key": movie_id,
                    "plot_alignment": [],
                    "video_clips": [],
                }
            ],
        )
        _write_json(
            benchmark / "data/splits.json",
            {"train": [movie_id], "val": [movie_id], "test": [movie_id]},
        )
        config = {
            "data_root": str(root),
            "movieqa": {
                "benchmark_dir": "raw/movieqa/benchmark",
                "output_dir": "processed/movieqa",
                "splits": ["train", "val", "test"],
                "max_video_clips_per_sample": 1,
            },
            "movienet": {
                "root_dir": "raw/movienet",
                "evidence_output": "processed/evidence/movienet.jsonl",
            },
            "moviechat_1k": {
                "root_dir": "raw/moviechat_1k",
                "output_file": "processed/moviechat/generalization.jsonl",
            },
            "frame_cache": {
                "root_dir": "cache/frames",
                "frames_per_video": 1,
                "jpeg_quality": 90,
                "overwrite": False,
            },
        }

        canonical = root / "processed/movieqa/train.jsonl"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text("canonical-sentinel\n", encoding="utf-8")
        data_result = run_data_setup({}, config, limit=1)
        assert data_result["status"] == "passed", data_result
        assert data_result["scope"] == "smoke_subset", data_result
        assert "movieqa_smoke" in data_result["normalized_output_dir"], data_result
        assert canonical.read_text(encoding="utf-8") == "canonical-sentinel\n"

        cli_result = command_prepare(
            Namespace(dataset="movieqa", limit=1, no_frames=True), config
        )
        assert cli_result == 0
        assert canonical.read_text(encoding="utf-8") == "canonical-sentinel\n"

        model_result = run_normalized_movieqa_sample(_Router(), config, limit=1)
        assert model_result["status"] == "passed", model_result
        assert model_result["mode"] == "metadata_query_only", model_result
        assert model_result["available_file_count"] == 0, model_result
        assert model_result["encoder_outputs"]["query"][0]["shape"] == [1, 768]

        full_result = run_data_setup({}, config, limit=None)
        assert full_result["status"] == "failed", full_result
        assert full_result["validation"]["train"]["official_expected"] == 9848

    print("PASS: limited preparation preserves canonical full output")
    print("PASS: standalone CLI finite --limit also uses smoke output")
    print("PASS: full preparation enforces official MovieQA split counts")
    print("PASS: normalized MovieQA -> lazy Dataset -> adapter -> router")


if __name__ == "__main__":
    main()
