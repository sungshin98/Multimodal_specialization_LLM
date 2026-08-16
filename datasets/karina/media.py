"""Media helpers with optional OpenCV dependency and deterministic caches."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List


VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".ogv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEXT_EXTENSIONS = {".txt", ".wiki", ".srt", ".json"}


def read_text(path: Path, max_chars: int | None = None) -> str:
    raw = ""
    for encoding in ("utf-8-sig", "utf-8", "iso-8859-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if path.suffix.lower() == ".srt":
        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.isdigit() or "-->" in stripped:
                continue
            lines.append(re.sub(r"<[^>]+>", "", stripped))
        raw = " ".join(lines)
    else:
        raw = "\n".join(line.strip() for line in raw.splitlines() if line.strip())
    return raw[:max_chars] if max_chars is not None else raw


class FrameCache:
    def __init__(
        self,
        root: Path,
        frames_per_video: int = 4,
        jpeg_quality: int = 90,
        overwrite: bool = False,
    ) -> None:
        if frames_per_video <= 0:
            raise ValueError("frames_per_video must be positive")
        self.root = root
        self.frames_per_video = frames_per_video
        self.jpeg_quality = jpeg_quality
        self.overwrite = overwrite

    def directory(self, dataset: str, movie_id: str, video_path: Path) -> Path:
        safe_movie = re.sub(r"[^A-Za-z0-9_.-]", "_", movie_id)
        safe_video = re.sub(r"[^A-Za-z0-9_.-]", "_", video_path.stem)
        return self.root / dataset / safe_movie / safe_video

    def list_frames(self, dataset: str, movie_id: str, video_path: Path) -> List[Path]:
        return sorted(self.directory(dataset, movie_id, video_path).glob("frame_*.jpg"))

    def get_or_create(self, dataset: str, movie_id: str, video_path: Path) -> List[Path]:
        existing = self.list_frames(dataset, movie_id, video_path)
        if len(existing) >= self.frames_per_video and not self.overwrite:
            return existing[: self.frames_per_video]
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            return self._extract_with_ffmpeg(dataset, movie_id, video_path)

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                raise RuntimeError(f"video has no readable frames: {video_path}")
            indices = np.linspace(0, total - 1, self.frames_per_video, dtype=np.int64)
            output_dir = self.directory(dataset, movie_id, video_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            written: List[Path] = []
            for order, frame_index in enumerate(indices):
                output = output_dir / f"frame_{order:03d}.jpg"
                if output.is_file() and not self.overwrite:
                    written.append(output)
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError(f"failed to read frame {frame_index}: {video_path}")
                if not cv2.imwrite(
                    str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                ):
                    raise RuntimeError(f"failed to write frame: {output}")
                written.append(output)
            return written
        finally:
            capture.release()

    def _extract_with_ffmpeg(
        self, dataset: str, movie_id: str, video_path: Path
    ) -> List[Path]:
        """CPU-only fallback for setup machines without OpenCV."""

        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError(
                "frame extraction requires opencv-python-headless or ffmpeg/ffprobe"
            )
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=nokey=1:noprint_wrappers=1", str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(probe.stdout.strip())
        if duration <= 0:
            raise RuntimeError(f"video has invalid duration: {video_path}")
        output_dir = self.directory(dataset, movie_id, video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        for order in range(self.frames_per_video):
            output = output_dir / f"frame_{order:03d}.jpg"
            if output.is_file() and not self.overwrite:
                written.append(output)
                continue
            # Midpoints avoid seeking exactly to the container end where ffmpeg can
            # return success without emitting a frame.
            timestamp = duration * (order + 0.5) / self.frames_per_video
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{timestamp:.6f}", "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "2", str(output),
                ],
                check=True,
            )
            if not output.is_file():
                raise RuntimeError(f"ffmpeg did not emit frame: {output}")
            written.append(output)
        return written
