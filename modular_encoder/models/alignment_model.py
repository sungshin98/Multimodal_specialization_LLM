from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


COMMON_DIM = 768


class ProjectionHead(nn.Module):
    """Map native encoder features into the shared embedding space."""

    def __init__(self, input_dim: int, output_dim: int = COMMON_DIM) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.network(features)
        return F.normalize(projected, p=2, dim=-1)


class MultimodalAlignmentModel(nn.Module):
    """Text/Image/Video projection heads for the final evaluated system."""

    def __init__(self) -> None:
        super().__init__()
        self.text_projection = ProjectionHead(384, COMMON_DIM)
        self.image_projection = ProjectionHead(768, COMMON_DIM)
        self.video_projection = ProjectionHead(768, COMMON_DIM)

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        video_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.text_projection(text_features),
            self.image_projection(image_features),
            self.video_projection(video_features),
        )


def filtered_alignment_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Keep only Text/Image/Video projection weights.

    Older development checkpoints may also contain audio_projection.* keys.
    Those keys are intentionally ignored by the final Text/Image/Video model.
    """
    allowed_prefixes = (
        "text_projection.",
        "image_projection.",
        "video_projection.",
    )
    return {
        key: value
        for key, value in state_dict.items()
        if key.startswith(allowed_prefixes)
    }
