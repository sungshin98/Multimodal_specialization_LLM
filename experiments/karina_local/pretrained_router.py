"""Checkpoint-free router for execution testing of the real pretrained backbones.

This router is deliberately restricted to smoke testing. Text embeddings are
zero-padded from 384 to 768 and no learned cross-modal projection is applied,
so its scores must never be reported as KARINA retrieval performance.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODULAR_ENCODER_ROOT = REPOSITORY_ROOT / "modular_encoder"
if str(MODULAR_ENCODER_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULAR_ENCODER_ROOT))

from encoders.document.document_loader import DocumentLoader  # noqa: E402
from encoders.image.image_encoder import ImageEncoder  # noqa: E402
from encoders.video.video_encoder import VideoEncoder  # noqa: E402
from interfaces.encoder_output import EncoderOutput  # noqa: E402


class PretrainedSmokeRouter:
    """Lazy real-backbone router without repository-specific checkpoints."""

    DOCUMENT_EXTENSIONS = {".txt", ".pdf", ".docx"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".ogv"}

    def __init__(
        self,
        device: str,
        text_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        image_model: str = "openai/clip-vit-base-patch32",
        video_model: str = "MCG-NJU/videomae-base-finetuned-kinetics",
        video_frames: int = 16,
    ) -> None:
        self.device = torch.device(device)
        self.text_model_name = text_model
        self.image_model_name = image_model
        self.video_model_name = video_model
        self.video_frames = video_frames
        self.document_loader = DocumentLoader()
        self.text_encoder: SentenceTransformer | None = None
        self.image_encoder: ImageEncoder | None = None
        self.video_encoder: VideoEncoder | None = None

    def _text(self) -> SentenceTransformer:
        if self.text_encoder is None:
            self.text_encoder = SentenceTransformer(
                self.text_model_name, device=str(self.device)
            )
        return self.text_encoder

    def _image(self) -> ImageEncoder:
        if self.image_encoder is None:
            encoder = ImageEncoder(self.image_model_name)
            encoder.model.to(self.device).eval()
            self.image_encoder = encoder
        return self.image_encoder

    def _video(self) -> VideoEncoder:
        if self.video_encoder is None:
            self.video_encoder = VideoEncoder(
                self.video_model_name,
                num_frames=self.video_frames,
                device=str(self.device),
            )
        return self.video_encoder

    @staticmethod
    def _common(embedding: torch.Tensor, modality: str) -> EncoderOutput:
        value = embedding.detach().to(dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape[-1] == 384:
            value = F.pad(value, (0, 384))
        if value.shape[-1] != 768:
            raise RuntimeError(
                f"unexpected {modality} embedding dimension: {value.shape[-1]}"
            )
        return EncoderOutput(
            pooled_embedding=F.normalize(value, p=2, dim=-1).cpu(),
            modality=modality,
        )

    @torch.no_grad()
    def route_query(self, query: str) -> EncoderOutput:
        if not query.strip():
            raise ValueError("query must be non-empty")
        embedding = self._text().encode(
            [query], convert_to_tensor=True, normalize_embeddings=False
        )
        return self._common(embedding, "query")

    @torch.no_grad()
    def route_file(self, file_path: str | Path) -> EncoderOutput:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in self.DOCUMENT_EXTENSIONS:
            text = self.document_loader.load(path)
            embedding = self._text().encode(
                [text], convert_to_tensor=True, normalize_embeddings=False
            )
            return self._common(embedding, "document")
        if suffix in self.IMAGE_EXTENSIONS:
            return self._common(self._image().encode(path).pooled_embedding, "image")
        if suffix in self.VIDEO_EXTENSIONS:
            return self._common(self._video().encode(path).pooled_embedding, "video")
        raise ValueError(f"unsupported file extension: {suffix}")

    def route(
        self,
        query: str | None = None,
        file_paths: Sequence[str | Path] | None = None,
    ) -> Dict[str, list[EncoderOutput]]:
        if query is None and not file_paths:
            raise ValueError("at least one query or file is required")
        outputs: Dict[str, list[EncoderOutput]] = {}
        if query is not None:
            item = self.route_query(query)
            outputs.setdefault(item.modality, []).append(item)
        for file_path in file_paths or []:
            item = self.route_file(file_path)
            outputs.setdefault(item.modality, []).append(item)
        return outputs

