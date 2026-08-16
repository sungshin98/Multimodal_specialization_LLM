"""MovieQA metadata/story/video normalization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from .config import data_path
from .io import write_jsonl
from .media import FrameCache, read_text
from .schema import KARINASample, empty_initial_input


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


class MovieQANormalizer:
    def __init__(self, config: Dict[str, Any], extract_frames: bool = True) -> None:
        self.config = config
        self.data_root = Path(config["data_root"])
        section = config["movieqa"]
        self.benchmark = data_path(config, section["benchmark_dir"])
        self.output_dir = data_path(config, section["output_dir"])
        self.max_clips = int(section.get("max_video_clips_per_sample", 2))
        frame_config = config["frame_cache"]
        self.frame_cache = FrameCache(
            data_path(config, frame_config["root_dir"]),
            int(frame_config.get("frames_per_video", 4)),
            int(frame_config.get("jpeg_quality", 90)),
            bool(frame_config.get("overwrite", False)),
        )
        self.extract_frames = extract_frames
        self._movies = self._load_json("data/movies.json")
        self._qa = self._load_json("data/qa.json")
        self._splits = self._load_json("data/splits.json")
        self._movie_map = {item["imdb_key"]: item for item in self._movies}

    def _load_json(self, relative: str) -> Any:
        path = self.benchmark / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"MovieQA metadata missing: {path}. Run the setup command first."
            )
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _text_assets(self, movie: Mapping[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
        assets: List[Dict[str, Any]] = []
        full_text: Dict[str, str] = {}
        movie_id = str(movie["imdb_key"])
        cache_dir = self.output_dir / "text" / movie_id
        for role in ("plot", "subtitle", "script", "dvs"):
            relative = (movie.get("text") or {}).get(role)
            if not relative:
                continue
            source = self.benchmark / relative
            if not source.is_file():
                continue
            content = read_text(source)
            if not content:
                continue
            cache = cache_dir / f"{role}.txt"
            cache.parent.mkdir(parents=True, exist_ok=True)
            if not cache.is_file() or cache.read_text(encoding="utf-8") != content:
                cache.write_text(content, encoding="utf-8")
            source_id = f"movieqa:{movie_id}:{role}"
            assets.append(
                {
                    "source_id": source_id,
                    "path": _relative(cache, self.data_root),
                    "role": role,
                    "content_hint": content[:500],
                    "metadata": {"original_path": str(relative)},
                }
            )
            full_text[role] = content

        # plot_alignment in qa.json points to sentence indices in the official
        # split_plot file, not to lines from the unsplit Wikipedia plot.
        split_plot_source = self.benchmark / "story" / "split_plot" / f"{movie_id}.split.wiki"
        if split_plot_source.is_file():
            split_plot = read_text(split_plot_source)
            if split_plot:
                cache = cache_dir / "split_plot.txt"
                if not cache.is_file() or cache.read_text(encoding="utf-8") != split_plot:
                    cache.write_text(split_plot, encoding="utf-8")
                assets.append(
                    {
                        "source_id": f"movieqa:{movie_id}:split_plot",
                        "path": _relative(cache, self.data_root),
                        "role": "split_plot",
                        "content_hint": split_plot[:500],
                        "metadata": {
                            "original_path": f"story/split_plot/{movie_id}.split.wiki"
                        },
                    }
                )
                full_text["split_plot"] = split_plot
        return assets, full_text

    def _video_path(self, movie_id: str, clip_name: str) -> Path:
        return self.benchmark / "story" / "video_clips" / movie_id / clip_name

    def iter_split(
        self,
        split: str,
        movienet_by_movie: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        if split not in self._splits:
            raise ValueError(f"unknown MovieQA split: {split}")
        movie_ids = set(self._splits[split])
        emitted = 0
        text_cache: Dict[str, tuple[List[Dict[str, Any]], Dict[str, str]]] = {}
        for qa in self._qa:
            movie_id = str(qa["imdb_key"])
            if movie_id not in movie_ids:
                continue
            movie = self._movie_map[movie_id]
            if movie_id not in text_cache:
                text_cache[movie_id] = self._text_assets(movie)
            text_assets, full_text = text_cache[movie_id]
            initial = empty_initial_input()
            initial["text"] = [dict(asset) for asset in text_assets]

            evidence: List[Dict[str, Any]] = []
            plot_lines = [
                line for line in full_text.get("split_plot", "").splitlines() if line
            ]
            for line_index in qa.get("plot_alignment") or []:
                # The official release uses direct Python-list indices.
                match = int(line_index) if 0 <= int(line_index) < len(plot_lines) else None
                if match is not None:
                    evidence.append(
                        {
                            "evidence_id": f"movieqa:{qa['qid']}:plot:{line_index}",
                            "modality": "text",
                            "content": plot_lines[match],
                            "source": "movieqa_plot_alignment",
                        }
                    )

            for clip_name in (qa.get("video_clips") or [])[: self.max_clips]:
                video = self._video_path(movie_id, str(clip_name))
                if not video.is_file():
                    continue
                initial["video"].append(
                    {
                        "source_id": f"movieqa:{movie_id}:video:{video.stem}",
                        "path": _relative(video, self.data_root),
                        "role": "aligned_clip",
                        "content_hint": f"Question-aligned MovieQA clip for {movie['name']}",
                        "metadata": {"clip_name": str(clip_name)},
                    }
                )
                frames = self.frame_cache.list_frames("movieqa", movie_id, video)
                if self.extract_frames and not frames:
                    frames = self.frame_cache.get_or_create("movieqa", movie_id, video)
                for frame_index, frame in enumerate(frames):
                    initial["image"].append(
                        {
                            "source_id": f"movieqa:{movie_id}:frame:{video.stem}:{frame_index}",
                            "path": _relative(frame, self.data_root),
                            "role": "sampled_frame",
                            "content_hint": f"Uniformly sampled frame {frame_index} from {video.name}",
                            "metadata": {"parent_video": _relative(video, self.data_root)},
                        }
                    )

            external = list((movienet_by_movie or {}).get(movie_id, []))
            evidence.extend(external)
            correct_index = qa.get("correct_index")
            answers = list(qa.get("answers") or [])
            answer = (
                str(answers[int(correct_index)])
                if isinstance(correct_index, int) and 0 <= correct_index < len(answers)
                else ""
            )
            sample = KARINASample(
                sample_id=f"movieqa:{qa['qid']}",
                split=split,
                question=str(qa["question"]),
                initial_input=initial,
                answer=answer,
                retrieval_target=list(evidence),
                evidence=list(evidence),
                metadata={
                    "dataset": "movieqa",
                    "qid": qa["qid"],
                    "imdb_key": movie_id,
                    "movie_name": movie.get("name", ""),
                    "year": movie.get("year", ""),
                    "genre": movie.get("genre", ""),
                    "answer_choices": answers,
                    "correct_index": correct_index,
                    "plot_alignment": qa.get("plot_alignment") or [],
                    "missing_video_count": sum(
                        not self._video_path(movie_id, str(name)).is_file()
                        for name in (qa.get("video_clips") or [])[: self.max_clips]
                    ),
                    "external_evidence_source": "movienet" if external else None,
                },
            )
            yield sample.to_dict()
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    def prepare(
        self,
        splits: Iterable[str],
        movienet_by_movie: Optional[Mapping[str, List[Dict[str, Any]]]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for split in splits:
            counts[split] = write_jsonl(
                self.output_dir / f"{split}.jsonl",
                self.iter_split(split, movienet_by_movie=movienet_by_movie, limit=limit),
            )
        return counts
