from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEModel,
)

from interfaces.encoder_output import EncoderOutput


class VideoEncoder:
    """VideoMAE 기반 장면 단위 영상 인코더."""

    SUPPORTED_EXTENSIONS = {
        ".mp4",
        ".webm",
        ".mkv",
        ".avi",
        ".mov",
        ".ogv",
    }

    def __init__(
        self,
        model_name: str = (
            "MCG-NJU/"
            "videomae-base-finetuned-kinetics"
        ),
        num_frames: int = 16,
        device: str = "cpu",
    ) -> None:
        if num_frames <= 0:
            raise ValueError(
                "num_frames must be greater than 0."
            )

        self.model_name = model_name
        self.num_frames = num_frames
        self.device = torch.device(device)

        self.processor = (
            VideoMAEImageProcessor.from_pretrained(
                model_name
            )
        )

        self.model = VideoMAEModel.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def _sample_frames(
        self,
        video_path: Path,
    ) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        try:
            total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if total_frames <= 0:
                raise RuntimeError(
                    "Video has no readable frames: "
                    f"{video_path}"
                )

            frame_indices = np.linspace(
                0,
                total_frames - 1,
                num=self.num_frames,
                dtype=np.int64,
            )

            frames: list[np.ndarray] = []

            for frame_index in frame_indices:
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_index),
                )

                success, frame = capture.read()

                if not success or frame is None:
                    raise RuntimeError(
                        "Failed to read frame "
                        f"{frame_index} from {video_path}"
                    )

                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frames.append(frame_rgb)

            return frames

        finally:
            capture.release()

    @torch.no_grad()
    def encode(
        self,
        video_path: str | Path,
    ) -> EncoderOutput:
        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        extension = video_path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(
                sorted(self.SUPPORTED_EXTENSIONS)
            )

            raise ValueError(
                f"Unsupported video type: {extension}. "
                f"Supported types: {supported}"
            )

        frames = self._sample_frames(video_path)

        inputs = self.processor(
            frames,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(
            self.device
        )

        output = self.model(
            pixel_values=pixel_values
        )

        hidden_state = output.last_hidden_state

        pooled_embedding = hidden_state.mean(dim=1)

        pooled_embedding = F.normalize(
            pooled_embedding,
            p=2,
            dim=-1,
        )

        return EncoderOutput(
            pooled_embedding=pooled_embedding.cpu(),
            modality="video",
        )