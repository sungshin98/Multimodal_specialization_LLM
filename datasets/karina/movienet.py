"""Build a streaming MovieNet external evidence pool from user-provided files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Set, Tuple

from .config import data_path
from .io import write_jsonl
from .media import IMAGE_EXTENSIONS, TEXT_EXTENSIONS, VIDEO_EXTENSIONS, read_text


IMDB_PATTERN = re.compile(r"tt\d{5,10}", re.IGNORECASE)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _aliases(path: Path, payload: Optional[Mapping[str, Any]] = None) -> Set[str]:
    values = {path.stem, *path.parts}
    if payload:
        for key in ("imdb_id", "imdb_key", "imdb", "id", "movie_id"):
            if payload.get(key) is not None:
                values.add(str(payload[key]))
    aliases: Set[str] = set()
    for value in values:
        match = IMDB_PATTERN.search(str(value))
        aliases.add(match.group(0).lower() if match else str(value).lower())
    return aliases


def _files_in_named_dirs(root: Path, names: List[str]) -> Iterator[Path]:
    seen: Set[Path] = set()
    for name in names:
        directory = root / name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def build_movienet_pool(
    config: Dict[str, Any],
    match_movie_ids: Optional[Set[str]] = None,
) -> Tuple[int, Dict[str, List[Dict[str, Any]]]]:
    section = config["movienet"]
    root = data_path(config, section["root_dir"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"MovieNet directory not found: {root}. Download it through the official OpenDataLab route first."
        )
    data_root = Path(config["data_root"])
    requested = {item.lower() for item in (match_movie_ids or set())}
    matched: Dict[str, List[Dict[str, Any]]] = {item: [] for item in requested}

    def records() -> Iterator[Dict[str, Any]]:
        specifications = (
            ("text", list(section.get("text_dirs", []))),
            ("image", list(section.get("image_dirs", []))),
            ("video", list(section.get("video_dirs", []))),
        )
        counter = 0
        for modality, directories in specifications:
            for path in _files_in_named_dirs(root, directories):
                suffix = path.suffix.lower()
                allowed = (
                    TEXT_EXTENSIONS if modality == "text" else
                    IMAGE_EXTENSIONS if modality == "image" else VIDEO_EXTENSIONS
                )
                if suffix not in allowed:
                    continue
                payload: Optional[Mapping[str, Any]] = None
                content = ""
                if modality == "text":
                    if suffix == ".json":
                        try:
                            loaded = json.loads(path.read_text(encoding="utf-8"))
                            payload = loaded if isinstance(loaded, Mapping) else None
                            if payload:
                                content = str(
                                    payload.get("synopsis")
                                    or payload.get("plot")
                                    or payload.get("description")
                                    or payload.get("title")
                                    or ""
                                )
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            payload = None
                    else:
                        content = read_text(path, max_chars=12000)
                aliases = _aliases(path.relative_to(root), payload)
                record = {
                    "evidence_id": f"movienet:{counter}",
                    "modality": modality,
                    "path": _relative(path, data_root),
                    "content": content,
                    "source": "movienet",
                    "metadata": {"movie_aliases": sorted(aliases)},
                }
                counter += 1
                for movie_id in aliases & requested:
                    if len(matched[movie_id]) < 64:
                        matched[movie_id].append(dict(record))
                yield record

    output = data_path(config, section["evidence_output"])
    count = write_jsonl(output, records())
    return count, {key: value for key, value in matched.items() if value}

