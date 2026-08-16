"""Serializable contract shared by MovieQA, MovieNet and MovieChat-1K."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


MODALITIES = ("text", "image", "video")


@dataclass
class Asset:
    source_id: str
    path: str
    role: str
    content_hint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KARINASample:
    """One normalized KARINA sample.

    Paths are POSIX-style and relative to the configured data root. Large media
    and text are resolved lazily by :class:`KARINAInputAdapter`.
    """

    sample_id: str
    split: str
    question: str
    initial_input: Dict[str, List[Dict[str, Any]]]
    answer: str
    question_condition: Optional[Dict[str, Any]] = None
    retrieval_target: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "karina.movie.v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def empty_initial_input() -> Dict[str, List[Dict[str, Any]]]:
    return {modality: [] for modality in MODALITIES}


def validate_sample(sample: Mapping[str, Any]) -> List[str]:
    """Return validation errors instead of raising on the first problem."""

    errors: List[str] = []
    for key in ("sample_id", "split", "question", "initial_input", "answer", "metadata"):
        if key not in sample:
            errors.append(f"missing required field: {key}")

    if not isinstance(sample.get("question", ""), str) or not sample.get("question", "").strip():
        errors.append("question must be a non-empty string")
    if not isinstance(sample.get("answer", ""), str):
        errors.append("answer must be a string")

    initial = sample.get("initial_input")
    if not isinstance(initial, Mapping):
        errors.append("initial_input must be an object")
    else:
        for modality in MODALITIES:
            assets = initial.get(modality)
            if not isinstance(assets, list):
                errors.append(f"initial_input.{modality} must be a list")
                continue
            for index, asset in enumerate(assets):
                if not isinstance(asset, Mapping):
                    errors.append(f"initial_input.{modality}[{index}] must be an object")
                    continue
                for key in ("source_id", "path", "role"):
                    if not isinstance(asset.get(key), str):
                        errors.append(
                            f"initial_input.{modality}[{index}].{key} must be a string"
                        )

    for key in ("retrieval_target", "evidence"):
        if key in sample and not isinstance(sample[key], list):
            errors.append(f"{key} must be a list")
    return errors

