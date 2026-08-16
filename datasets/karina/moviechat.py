"""MovieChat-1K conversion for a held-out generalization/evaluation split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

from .config import data_path
from .io import write_jsonl
from .media import FrameCache, VIDEO_EXTENSIONS
from .schema import KARINASample, empty_initial_input


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


class MovieChatNormalizer:
    def __init__(self, config: Dict[str, Any], extract_frames: bool = True) -> None:
        self.config = config
        self.data_root = Path(config["data_root"])
        section = config["moviechat_1k"]
        self.root = data_path(config, section["root_dir"])
        self.output = data_path(config, section["output_file"])
        self.split = str(section.get("split", "generalization_test"))
        frame_config = config["frame_cache"]
        self.frame_cache = FrameCache(
            data_path(config, frame_config["root_dir"]),
            int(frame_config.get("frames_per_video", 4)),
            int(frame_config.get("jpeg_quality", 90)),
            bool(frame_config.get("overwrite", False)),
        )
        self.extract_frames = extract_frames

    def _annotation_files(self) -> List[Path]:
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"MovieChat-1K directory not found: {self.root}. Use the official Hugging Face repositories."
            )
        candidates = sorted(self.root.rglob("*.json"))
        return [path for path in candidates if path.name not in {"dataset_info.json"}]

    def _find_video(self, name: str) -> Optional[Path]:
        relative = Path(name)
        direct_candidates = [
            self.root / relative,
            self.root / "raw_videos" / relative,
            self.root / "movies" / relative,
            self.root / "videos" / relative,
            self.root / "test" / relative,
        ]
        for candidate in direct_candidates:
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
                return candidate
        matches = list(self.root.rglob(relative.name))
        return next((path for path in matches if path.suffix.lower() in VIDEO_EXTENSIONS), None)

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        emitted = 0
        for annotation_path in self._annotation_files():
            try:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(payload, Mapping) or not isinstance(payload.get("info"), Mapping):
                continue
            info = dict(payload["info"])
            video_name = str(info.get("video_path") or "")
            video = self._find_video(video_name) if video_name else None
            video_id = Path(video_name).stem or annotation_path.stem
            caption = str(payload.get("caption") or "").strip()
            frames: List[Path] = []
            if video:
                frames = self.frame_cache.list_frames("moviechat_1k", video_id, video)
                if self.extract_frames and not frames:
                    frames = self.frame_cache.get_or_create("moviechat_1k", video_id, video)

            for scope in ("global", "breakpoint"):
                qa_items = payload.get(scope) or []
                if not isinstance(qa_items, list):
                    continue
                for qa_index, qa in enumerate(qa_items):
                    if not isinstance(qa, Mapping) or not str(qa.get("question") or "").strip():
                        continue
                    initial = empty_initial_input()
                    if caption:
                        caption_file = (
                            Path(self.config["data_root"])
                            / "processed" / "moviechat_1k" / "text" / f"{video_id}.txt"
                        )
                        caption_file.parent.mkdir(parents=True, exist_ok=True)
                        if not caption_file.is_file():
                            caption_file.write_text(caption, encoding="utf-8")
                        initial["text"].append(
                            {
                                "source_id": f"moviechat:{video_id}:caption",
                                "path": _relative(caption_file, self.data_root),
                                "role": "dense_caption",
                                "content_hint": caption[:500],
                                "metadata": {},
                            }
                        )
                    if video:
                        initial["video"].append(
                            {
                                "source_id": f"moviechat:{video_id}:video",
                                "path": _relative(video, self.data_root),
                                "role": "long_video",
                                "content_hint": f"MovieChat-1K {scope} video",
                                "metadata": {"timestamp_frame": qa.get("time")},
                            }
                        )
                    for frame_index, frame in enumerate(frames):
                        initial["image"].append(
                            {
                                "source_id": f"moviechat:{video_id}:frame:{frame_index}",
                                "path": _relative(frame, self.data_root),
                                "role": "sampled_frame",
                                "content_hint": f"Uniformly sampled long-video frame {frame_index}",
                                "metadata": {"parent_video": video_name},
                            }
                        )
                    answer = str(qa.get("answer") or "")
                    evidence = []
                    if caption:
                        evidence.append(
                            {
                                "evidence_id": f"moviechat:{video_id}:caption",
                                "modality": "text",
                                "content": caption,
                                "source": "moviechat_dense_caption",
                            }
                        )
                    sample = KARINASample(
                        sample_id=f"moviechat:{video_id}:{scope}:{qa_index}",
                        split=self.split,
                        question=str(qa["question"]).strip(),
                        initial_input=initial,
                        answer=answer,
                        question_condition={
                            "scope": scope,
                            "required_modalities": ["video"],
                            "timestamp_frame": qa.get("time"),
                        },
                        retrieval_target=evidence,
                        evidence=evidence,
                        metadata={
                            "dataset": "moviechat_1k",
                            "annotation_path": _relative(annotation_path, self.data_root),
                            "video_id": video_id,
                            "video_info": info,
                            "evaluation_only": True,
                            "missing_video": video is None,
                        },
                    )
                    yield sample.to_dict()
                    emitted += 1
                    if limit is not None and emitted >= limit:
                        return

    def prepare(self, limit: Optional[int] = None) -> int:
        return write_jsonl(self.output, self.iter_samples(limit=limit))

