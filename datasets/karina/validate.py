"""Validation checks for normalized KARINA JSONL files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import iter_jsonl
from .schema import validate_sample


def validate_jsonl(path: str | Path, data_root: str | Path) -> Dict[str, Any]:
    root = Path(data_root)
    seen = set()
    errors: List[str] = []
    warnings: List[str] = []
    count = 0
    for line_number, sample in enumerate(iter_jsonl(path), 1):
        count += 1
        for error in validate_sample(sample):
            errors.append(f"line {line_number}: {error}")
        sample_id = sample.get("sample_id")
        if sample_id in seen:
            errors.append(f"line {line_number}: duplicate sample_id {sample_id}")
        seen.add(sample_id)
        for modality, assets in sample.get("initial_input", {}).items():
            for asset in assets:
                candidate = Path(asset.get("path", ""))
                candidate = candidate if candidate.is_absolute() else root / candidate
                if not candidate.is_file():
                    errors.append(f"line {line_number}: missing {modality} asset {candidate}")
        if not any(sample.get("initial_input", {}).get(modality) for modality in ("text", "image", "video")):
            warnings.append(
                f"line {line_number}: no initial modality assets (story/media may not be installed)"
            )
        if sample.get("split") in {"train", "val"} and not sample.get("answer"):
            warnings.append(f"line {line_number}: empty supervised answer")
    return {
        "path": str(path),
        "samples": count,
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }


def validate_many(paths: Iterable[str | Path], data_root: str | Path) -> List[Dict[str, Any]]:
    return [validate_jsonl(path, data_root) for path in paths]
