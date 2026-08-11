from dataclasses import dataclass

import torch


@dataclass
class EncoderOutput:
    pooled_embedding: torch.Tensor
    modality: str
