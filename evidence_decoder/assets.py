"""근거 원본 로더.

Adaptive RAG 의 evidence.content 는 'poster_003.jpg' 같은 파일명뿐이다.
실제 픽셀/영상을 비전 모델에 넣으려면 이 계층에서 파일을 찾아 열어야 한다.

탐색 순서
1. metadata 의 path / file_path / uri 절대경로
2. asset_root 하위 재귀 탐색 (결과 캐시)
3. 실패 시 None -> 디코더는 caption 힌트만으로 degraded 동작
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from .clients import MediaAsset
from .schemas import EvidenceItem, Modality

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Gemini inline_data 는 요청 전체 20MB 제한. 넘으면 프레임 샘플링으로 전환한다.
INLINE_VIDEO_LIMIT_BYTES = 15 * 1024 * 1024


@dataclass
class AssetLoadResult:
    """근거 1건이 만들어내는 자료 묶음.

    영상을 프레임으로 쪼개면 assets 가 여러 장이 된다. 호출측은
    항상 리스트로 다루면 되고 이미지/영상을 구분할 필요가 없다.
    """

    assets: List[MediaAsset] = field(default_factory=list)
    resolved_path: Optional[str] = None
    degraded_reason: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.assets)


@dataclass
class AssetLoader:
    """asset_root 를 기준으로 파일명을 실제 경로로 해석한다."""

    asset_root: Optional[str] = None
    max_video_frames: int = 6
    prefer_native_video: bool = True
    _index: Optional[Dict[str, str]] = field(default=None, init=False, repr=False)
    _temp_dirs: List[str] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------

    def load(self, item: EvidenceItem) -> AssetLoadResult:
        hint = item.caption_hint()
        path = self.resolve(item)

        if path is None:
            return AssetLoadResult([], None, "원본 파일을 찾지 못함")

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in VIDEO_EXTS:
                return self._load_video(path, hint)
            asset = MediaAsset.from_path(path, label=os.path.basename(path), text_hint=hint)
            return AssetLoadResult([asset], path)
        except OSError as error:
            return AssetLoadResult([], path, f"파일 열기 실패: {error}")

    def load_many(self, items: List[EvidenceItem]) -> Dict[str, AssetLoadResult]:
        return {item.evidence_id: self.load(item) for item in items}

    # ------------------------------------------------------------------

    def resolve(self, item: EvidenceItem) -> Optional[str]:
        for key in ("path", "file_path", "filepath", "uri", "url", "source_path"):
            value = item.metadata.get(key)
            if isinstance(value, str) and os.path.isfile(value):
                return value

        name = self._content_name(item.content)
        if not name:
            return None
        if os.path.isfile(name):
            return name
        if self.asset_root:
            direct = os.path.join(self.asset_root, name)
            if os.path.isfile(direct):
                return direct
            return self._index_lookup(os.path.basename(name))
        return None

    @staticmethod
    def _content_name(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, Mapping):
            for key in ("path", "file", "filename", "name"):
                value = content.get(key)
                if isinstance(value, str):
                    return value.strip()
        return ""

    def _index_lookup(self, basename: str) -> Optional[str]:
        if self._index is None:
            self._index = {}
            root = self.asset_root or ""
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    self._index.setdefault(filename, os.path.join(dirpath, filename))
        return self._index.get(basename)

    # ------------------------------------------------------------------
    # 영상
    # ------------------------------------------------------------------

    def _load_video(self, path: str, hint: str) -> AssetLoadResult:
        size = os.path.getsize(path)
        if self.prefer_native_video and size <= INLINE_VIDEO_LIMIT_BYTES:
            # Gemini 는 영상을 그대로 이해한다. 프레임 분해가 필요 없다.
            asset = MediaAsset.from_path(path, label=os.path.basename(path), text_hint=hint)
            return AssetLoadResult([asset], path)

        frames = self._sample_frames(path)
        if not frames:
            reason = (
                f"영상 {size // (1024 * 1024)}MB 가 인라인 한도를 넘고 ffmpeg 도 없어 "
                "프레임 샘플링 불가"
            )
            return AssetLoadResult([], path, reason)

        frames[0].text_hint = hint
        return AssetLoadResult(frames, path, f"영상을 {len(frames)}개 프레임으로 대체")

    def _sample_frames(self, path: str) -> List[MediaAsset]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return []

        outdir = tempfile.mkdtemp(prefix="evdec_frames_")
        self._temp_dirs.append(outdir)
        pattern = os.path.join(outdir, "frame_%03d.jpg")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path,
            "-vf", f"thumbnail,fps=1/2", "-frames:v", str(self.max_video_frames),
            "-q:v", "4", pattern,
        ]
        try:
            subprocess.run(command, check=True, timeout=120, capture_output=True)
        except (subprocess.SubprocessError, OSError):
            return []

        assets: List[MediaAsset] = []
        for filename in sorted(os.listdir(outdir)):
            full = os.path.join(outdir, filename)
            try:
                assets.append(
                    MediaAsset.from_path(full, label=f"{os.path.basename(path)} :: {filename}")
                )
            except OSError:
                continue
        return assets

    def cleanup(self) -> None:
        for directory in self._temp_dirs:
            shutil.rmtree(directory, ignore_errors=True)
        self._temp_dirs.clear()


def modality_of_path(path: str) -> Optional[Modality]:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return Modality.IMAGE
    if ext in VIDEO_EXTS:
        return Modality.VIDEO
    return None
