"""Create tiny local assets used only for an encoder execution smoke test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict

from PIL import Image, ImageDraw


def ensure_smoke_assets(root: Path) -> Dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    text_path = root / "scene.txt"
    image_path = root / "scene.jpg"
    video_path = root / "scene.mp4"

    if not text_path.is_file():
        text_path.write_text(
            "A person enters a blue room, walks toward a table, and opens a book.",
            encoding="utf-8",
        )
    if not image_path.is_file():
        image = Image.new("RGB", (224, 224), (36, 74, 120))
        draw = ImageDraw.Draw(image)
        draw.rectangle((72, 40, 152, 205), fill=(220, 190, 150))
        draw.rectangle((45, 160, 180, 190), fill=(92, 58, 37))
        draw.text((18, 12), "KARINA smoke scene", fill=(255, 255, 255))
        image.save(image_path, quality=92)
    if not _video_is_readable(video_path):
        _create_video(video_path)
    return {"text": text_path, "image": image_path, "video": video_path}


def _create_video(path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.building{path.suffix}")
    temporary.unlink(missing_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    try:
        if ffmpeg:
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=224x224:rate=8:duration=2",
                "-pix_fmt",
                "yuv420p",
                str(temporary),
            ]
            subprocess.run(command, check=True)
        else:
            _create_video_with_opencv(temporary)
        if not _video_is_readable(temporary):
            raise RuntimeError(f"generated smoke video is unreadable: {temporary}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _create_video_with_opencv(path: Path) -> None:

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("smoke video creation requires ffmpeg or OpenCV") from exc
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (224, 224)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create smoke video: {path}")
    try:
        for index in range(16):
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame[:, :] = (120, 74, 36)
            x = 20 + index * 10
            cv2.rectangle(frame, (x, 80), (x + 36, 180), (150, 190, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"OpenCV did not emit smoke video: {path}")


def _video_is_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            text=True,
            capture_output=True,
        )
        return result.returncode == 0 and "video" in result.stdout
    try:
        import cv2  # type: ignore
    except ImportError:
        # Non-empty is the strongest check available until setup installs FFmpeg/OpenCV.
        return True
    capture = cv2.VideoCapture(str(path))
    try:
        return capture.isOpened() and capture.read()[0]
    finally:
        capture.release()
