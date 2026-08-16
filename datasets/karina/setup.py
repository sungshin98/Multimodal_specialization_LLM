"""Policy-safe dataset setup and local availability checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from .config import data_path


MOVIEQA_REPOSITORY = "https://github.com/makarandtapaswi/MovieQA_benchmark.git"
MOVIEQA_COMMIT = "c9367df9f0039dd037856fe37a676fdc5ab7b037"


def setup_movieqa(config: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    target = data_path(config, config["movieqa"]["benchmark_dir"])
    required = [target / "data" / name for name in ("qa.json", "movies.json", "splits.json")]
    if all(path.is_file() for path in required) and not force:
        return {"dataset": "movieqa", "status": "ready", "path": str(target)}
    if target.exists() and any(target.iterdir()):
        if not force:
            raise RuntimeError(
                f"refusing to overwrite non-empty MovieQA directory: {target}; use --force"
            )
        shutil.rmtree(target)
    if target.exists():
        target.rmdir()
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-checkout", MOVIEQA_REPOSITORY, str(target)],
        check=True,
    )
    subprocess.run(["git", "checkout", MOVIEQA_COMMIT], cwd=target, check=True)
    return {"dataset": "movieqa", "status": "metadata_ready", "path": str(target)}


def dataset_status(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    root = Path(config["data_root"])
    movieqa = data_path(config, config["movieqa"]["benchmark_dir"])
    movienet = data_path(config, config["movienet"]["root_dir"])
    moviechat = data_path(config, config["moviechat_1k"]["root_dir"])
    return {
        "movieqa": {
            "metadata": all((movieqa / "data" / name).is_file() for name in ("qa.json", "movies.json", "splits.json")),
            "stories": (movieqa / "story").is_dir() and any((movieqa / "story").glob("*/*")),
            "videos": (movieqa / "story" / "video_clips").is_dir(),
            "path": str(movieqa),
        },
        "movienet": {
            "available": movienet.is_dir() and any(movienet.iterdir()),
            "path": str(movienet),
            "manual_action": "Register at OpenDataLab and accept its terms; full movies are not publicly distributed.",
        },
        "moviechat_1k": {
            "available": moviechat.is_dir() and any(moviechat.rglob("*.json")),
            "path": str(moviechat),
            "manual_action": "Download from the official Enxin Hugging Face repositories after accepting applicable access conditions.",
        },
        "data_root": {"path": str(root)},
    }


def reset_movieqa_metadata(config: Dict[str, Any]) -> None:
    """Test helper; not exposed by CLI because downloaded stories may coexist."""
    target = data_path(config, config["movieqa"]["benchmark_dir"])
    if target.exists():
        shutil.rmtree(target)
