from pathlib import Path
from encoders.image.image_encoder import ImageEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "processed" / "frames" / "charade" / "charade_00000.jpg"

def main() -> None:
    output = ImageEncoder().encode(IMAGE_PATH)
    assert output.pooled_embedding.shape == (1, 768)
    assert output.modality == "image"
    print("ImageEncoder test passed:", output.pooled_embedding.shape)

if __name__ == "__main__":
    main()
