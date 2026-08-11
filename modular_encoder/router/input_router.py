from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from encoders.document.document_encoder import DocumentEncoder
from encoders.document.document_loader import DocumentLoader
from encoders.image.image_encoder import ImageEncoder
from encoders.query.query_encoder import QueryEncoder
from encoders.video.video_encoder import VideoEncoder
from interfaces.encoder_output import EncoderOutput
from models.alignment_model import (
    MultimodalAlignmentModel,
    filtered_alignment_state_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "resplit_135" / "joint_finetuned_best.pt"
)
DEFAULT_VIDEO_CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "video_added" / "video_projection_best.pt"
)


class InputRouter:
    """Lazy-loading router for the final Text/Image/Video encoder system."""

    DOCUMENT_EXTENSIONS = {".txt", ".pdf", ".docx"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov", ".ogv"}

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_BASE_CHECKPOINT_PATH,
        video_checkpoint_path: str | Path = DEFAULT_VIDEO_CHECKPOINT_PATH,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.video_checkpoint_path = Path(video_checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Base checkpoint not found: {self.checkpoint_path}"
            )
        if not self.video_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Video projection checkpoint not found: {self.video_checkpoint_path}"
            )

        self.query_encoder: QueryEncoder | None = None
        self.document_encoder: DocumentEncoder | None = None
        self.image_encoder: ImageEncoder | None = None
        self.video_encoder: VideoEncoder | None = None
        self.alignment_model: MultimodalAlignmentModel | None = None

        self.document_loader = DocumentLoader()
        self._base_checkpoint: dict[str, Any] | None = None
        self._video_checkpoint: dict[str, Any] | None = None

    def _get_base_checkpoint(self) -> dict[str, Any]:
        if self._base_checkpoint is None:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            required = {
                "document_encoder_state_dict",
                "image_encoder_state_dict",
            }
            missing = required - checkpoint.keys()
            if missing:
                raise KeyError(
                    "Base checkpoint missing keys: " + ", ".join(sorted(missing))
                )
            self._base_checkpoint = checkpoint
        return self._base_checkpoint

    def _get_video_checkpoint(self) -> dict[str, Any]:
        if self._video_checkpoint is None:
            checkpoint = torch.load(
                self.video_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            required = {
                "alignment_model_state_dict",
                "video_projection_trained",
            }
            missing = required - checkpoint.keys()
            if missing:
                raise KeyError(
                    "Video checkpoint missing keys: " + ", ".join(sorted(missing))
                )
            if not bool(checkpoint["video_projection_trained"]):
                raise RuntimeError("Video projection checkpoint is not marked as trained.")
            self._video_checkpoint = checkpoint
        return self._video_checkpoint

    def get_alignment_model(self) -> MultimodalAlignmentModel:
        if self.alignment_model is None:
            model = MultimodalAlignmentModel()
            raw_state = self._get_video_checkpoint()["alignment_model_state_dict"]
            state = filtered_alignment_state_dict(raw_state)
            result = model.load_state_dict(state, strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    "Alignment checkpoint mismatch: "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
                )
            model.to(self.device).eval()
            self.alignment_model = model
        return self.alignment_model

    def get_query_encoder(self) -> QueryEncoder:
        if self.query_encoder is None:
            encoder = QueryEncoder()
            encoder.model.load_state_dict(
                self._get_base_checkpoint()["document_encoder_state_dict"]
            )
            encoder.model.to(self.device)
            encoder.model.eval()
            self.query_encoder = encoder
        return self.query_encoder

    def get_document_encoder(self) -> DocumentEncoder:
        if self.document_encoder is None:
            encoder = DocumentEncoder()
            encoder.model.load_state_dict(
                self._get_base_checkpoint()["document_encoder_state_dict"]
            )
            encoder.model.to(self.device)
            encoder.model.eval()
            self.document_encoder = encoder
        return self.document_encoder

    def get_image_encoder(self) -> ImageEncoder:
        if self.image_encoder is None:
            encoder = ImageEncoder()
            encoder.model.load_state_dict(
                self._get_base_checkpoint()["image_encoder_state_dict"]
            )
            encoder.model.to(self.device)
            encoder.model.eval()
            self.image_encoder = encoder
        return self.image_encoder

    def get_video_encoder(self) -> VideoEncoder:
        if self.video_encoder is None:
            self.video_encoder = VideoEncoder(device=str(self.device))
        return self.video_encoder

    @torch.no_grad()
    def _project_text(self, output: EncoderOutput, modality: str) -> EncoderOutput:
        embedding = output.pooled_embedding.to(self.device)
        projected = self.get_alignment_model().text_projection(embedding)
        return EncoderOutput(
            pooled_embedding=F.normalize(projected, p=2, dim=-1).cpu(),
            modality=modality,
        )

    @torch.no_grad()
    def _project_image(self, output: EncoderOutput) -> EncoderOutput:
        embedding = output.pooled_embedding.to(self.device)
        projected = self.get_alignment_model().image_projection(embedding)
        return EncoderOutput(
            pooled_embedding=F.normalize(projected, p=2, dim=-1).cpu(),
            modality="image",
        )

    @torch.no_grad()
    def _project_video(self, output: EncoderOutput) -> EncoderOutput:
        embedding = output.pooled_embedding.to(self.device)
        projected = self.get_alignment_model().video_projection(embedding)
        return EncoderOutput(
            pooled_embedding=F.normalize(projected, p=2, dim=-1).cpu(),
            modality="video",
        )

    def route_query(self, query: str) -> EncoderOutput:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        return self._project_text(
            self.get_query_encoder().encode(query),
            modality="query",
        )

    def route_file(self, file_path: str | Path) -> EncoderOutput:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        extension = path.suffix.lower()

        if extension in self.DOCUMENT_EXTENSIONS:
            text = self.document_loader.load(path)
            return self._project_text(
                self.get_document_encoder().encode(text),
                modality="document",
            )
        if extension in self.IMAGE_EXTENSIONS:
            return self._project_image(
                self.get_image_encoder().encode(path)
            )
        if extension in self.VIDEO_EXTENSIONS:
            return self._project_video(
                self.get_video_encoder().encode(path)
            )

        raise ValueError(f"Unsupported file extension: {extension}")

    def route(
        self,
        query: str | None = None,
        file_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    ) -> dict[str, list[EncoderOutput]]:
        if query is None and not file_paths:
            raise ValueError("At least one query or file is required.")
        if file_paths is not None and not isinstance(file_paths, (list, tuple)):
            raise TypeError("file_paths must be a list or tuple.")

        outputs: dict[str, list[EncoderOutput]] = {}

        if query is not None:
            query_output = self.route_query(query)
            outputs.setdefault(query_output.modality, []).append(query_output)

        for file_path in file_paths or []:
            output = self.route_file(file_path)
            outputs.setdefault(output.modality, []).append(output)

        return outputs
