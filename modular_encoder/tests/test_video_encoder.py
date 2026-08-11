from pathlib import Path
from encoders.video.video_encoder import VideoEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT / "data" / "processed" / "clips" / "charade" / "charade_00000.mp4"

def main() -> None:
    output = VideoEncoder().encode(VIDEO_PATH)
    assert output.pooled_embedding.shape == (1, 768)
    assert output.modality == "video"
    assert abs(output.pooled_embedding.norm().item() - 1.0) < 1e-4
    print("VideoEncoder test passed:", output.pooled_embedding.shape)

if __name__ == "__main__":
    main()
