from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel

from interfaces.encoder_output import EncoderOutput


class ImageEncoder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
    ) -> None:
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPVisionModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, image_path: str | Path) -> EncoderOutput:
        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        with Image.open(image_path) as image:
            inputs = self.processor(
                images=image.convert("RGB"),
                return_tensors="pt",
            )

        with torch.no_grad():
            pooled_embedding = self.model(
                **inputs
            ).pooler_output

        return EncoderOutput(
            pooled_embedding=pooled_embedding,
            modality="image",
        )
