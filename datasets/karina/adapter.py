"""Lazy Dataset/DataLoader adapter for existing KARINA module contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .io import line_offsets

try:  # The schema/validation CLI remains usable before installing training deps.
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised in the lightweight smoke test
    torch = None  # type: ignore[assignment]
    TorchDataset = object  # type: ignore[assignment,misc]


class KARINADataset(TorchDataset):
    """Random-access JSONL dataset that stores only byte offsets in RAM."""

    def __init__(self, jsonl_path: str | Path) -> None:
        self.path = Path(jsonl_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.offsets = line_offsets(self.path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += len(self.offsets)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline().decode("utf-8"))


class KARINAInputAdapter:
    """Map normalized fields to InputRouter and Role-2 pipeline arguments."""

    SOURCE_MODALITY = {"text": "document", "image": "image", "video": "video"}

    def __init__(
        self,
        data_root: str | Path,
        max_text_assets: int = 3,
        max_image_assets: int = 4,
        max_video_assets: int = 1,
    ) -> None:
        self.data_root = Path(data_root)
        self.limits = {
            "text": max_text_assets,
            "image": max_image_assets,
            "video": max_video_assets,
        }

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    def to_router_payload(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        paths: List[Path] = []
        hints: Dict[str, List[str]] = {}
        metadata: Dict[str, Dict[str, Any]] = {}
        initial = sample["initial_input"]
        for modality in ("text", "image", "video"):
            source_modality = self.SOURCE_MODALITY[modality]
            assets = list(initial.get(modality, []))[: self.limits[modality]]
            hints[source_modality] = []
            for index, asset in enumerate(assets):
                path = self._resolve(str(asset["path"]))
                if not path.is_file():
                    continue
                paths.append(path)
                hints[source_modality].append(str(asset.get("content_hint") or ""))
                metadata[f"{source_modality}_{index}"] = {
                    "dataset_sample_id": sample["sample_id"],
                    "source_id": asset.get("source_id"),
                    "role": asset.get("role"),
                    **dict(asset.get("metadata") or {}),
                }
        return {
            "question": str(sample["question"]),
            "file_paths": paths,
            "content_hints": hints,
            "source_metadata": metadata,
        }

    def encode(self, router: Any, sample: Mapping[str, Any]) -> Dict[str, Any]:
        payload = self.to_router_payload(sample)
        return router.route(query=payload["question"], file_paths=payload["file_paths"])

    def run_role2(self, role2_pipeline: Any, router: Any, sample: Mapping[str, Any]) -> Any:
        payload = self.to_router_payload(sample)
        return role2_pipeline.run_from_router(
            router,
            payload["question"],
            payload["file_paths"],
            content_hints=payload["content_hints"],
            source_metadata=payload["source_metadata"],
        )

    @staticmethod
    def decoder_target(sample: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "answer": sample.get("answer", ""),
            "retrieval_target": sample.get("retrieval_target", []),
            "evidence": sample.get("evidence", []),
            "metadata": sample.get("metadata", {}),
        }


def collate_karina_samples(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate lazy paths and scalar supervision without decoding media in workers."""

    modality_mask = [
        [bool(sample["initial_input"].get(modality)) for modality in ("text", "image", "video")]
        for sample in samples
    ]
    correct_index = [
        int(sample.get("metadata", {}).get("correct_index"))
        if isinstance(sample.get("metadata", {}).get("correct_index"), int)
        else -100
        for sample in samples
    ]
    if torch is not None:
        modality_mask_value: Any = torch.tensor(modality_mask, dtype=torch.bool)
        correct_index_value: Any = torch.tensor(correct_index, dtype=torch.int64)
    else:
        modality_mask_value = modality_mask
        correct_index_value = correct_index
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "question": [sample["question"] for sample in samples],
        "initial_input": [sample["initial_input"] for sample in samples],
        "modality_mask": modality_mask_value,
        "answer": [sample.get("answer", "") for sample in samples],
        "correct_index": correct_index_value,
        "retrieval_target": [sample.get("retrieval_target", []) for sample in samples],
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }


def collate_encoder_outputs(
    outputs: Sequence[Mapping[str, Sequence[Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Pad InputRouter outputs to ``[B, M_max, D]`` plus a boolean mask."""

    if torch is None:
        raise RuntimeError("collate_encoder_outputs requires PyTorch")
    result: Dict[str, Dict[str, Any]] = {}
    modalities = sorted({key for sample in outputs for key in sample})
    for modality in modalities:
        max_items = max(len(sample.get(modality, [])) for sample in outputs)
        first = next(
            item
            for sample in outputs
            for item in sample.get(modality, [])
        )
        embedding = first.pooled_embedding
        dim = int(embedding.shape[-1])
        values = torch.zeros((len(outputs), max_items, dim), dtype=torch.float32)
        mask = torch.zeros((len(outputs), max_items), dtype=torch.bool)
        for batch_index, sample in enumerate(outputs):
            for item_index, item in enumerate(sample.get(modality, [])):
                vector = item.pooled_embedding.detach().cpu().to(dtype=torch.float32).reshape(-1, dim)[0]
                values[batch_index, item_index] = vector
                mask[batch_index, item_index] = True
        result[modality] = {"embeddings": values, "attention_mask": mask}
    return result

