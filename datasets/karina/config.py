"""Small YAML configuration loader with repository-relative path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")

    data_root = Path(config.get("data_root", "data/karina"))
    if not data_root.is_absolute():
        # The checked-in config lives in configs/, so relative paths are repo-relative.
        base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        data_root = base / data_root
    config["data_root"] = str(data_root.resolve())
    return config


def data_path(config: Dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["data_root"]) / path

