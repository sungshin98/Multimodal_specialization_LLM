from pathlib import Path
from router.input_router import InputRouter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = PROJECT_ROOT / "data" / "processed" / "frames" / "charade" / "charade_00000.jpg"
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "clips" / "charade" / "charade_00000.mp4"

def main() -> None:
    router = InputRouter()
    outputs = router.route(
        query="A person is moving inside a room.",
        file_paths=[IMAGE_PATH, VIDEO_PATH],
    )
    for modality in ("query", "image", "video"):
        output = outputs[modality][0]
        assert output.pooled_embedding.shape == (1, 768)
        assert abs(output.pooled_embedding.norm().item() - 1.0) < 1e-4
    print("Final router test passed.")

if __name__ == "__main__":
    main()
