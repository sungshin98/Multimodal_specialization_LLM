"""Dependency-light end-to-end smoke test using a tiny MovieQA fixture."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from PIL import Image

from .adapter import KARINADataset, KARINAInputAdapter, collate_karina_samples
from .io import iter_jsonl, write_jsonl
from .media import FrameCache
from .moviechat import MovieChatNormalizer
from .movienet import build_movienet_pool
from .movieqa import MovieQANormalizer
from .validate import validate_jsonl


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="karina_smoke_") as temporary:
        root = Path(temporary)
        benchmark = root / "raw/movieqa/benchmark"
        movie_id = "tt0000001"
        clip_name = "tt0000001.sf-0.ef-9.video.mp4"
        _write_json(
            benchmark / "data/movies.json",
            [{
                "imdb_key": movie_id,
                "name": "Fixture Movie",
                "year": "2026",
                "genre": "Drama",
                "text": {
                    "plot": f"story/plot/{movie_id}.wiki",
                    "subtitle": f"story/subtt/{movie_id}.srt",
                    "script": None,
                    "dvs": None,
                },
            }],
        )
        _write_json(
            benchmark / "data/qa.json",
            [{
                "qid": "train:0",
                "question": "Why does the character leave?",
                "answers": ["To get help", "To sleep", "To hide", "To eat", "To dance"],
                "correct_index": 0,
                "imdb_key": movie_id,
                "plot_alignment": [0],
                "video_clips": [clip_name],
            }],
        )
        _write_json(benchmark / "data/splits.json", {"train": [movie_id], "val": [], "test": []})
        plot = benchmark / f"story/plot/{movie_id}.wiki"
        plot.parent.mkdir(parents=True, exist_ok=True)
        plot.write_text("The character leaves to get help.\n", encoding="utf-8")
        split_plot = benchmark / f"story/split_plot/{movie_id}.split.wiki"
        split_plot.parent.mkdir(parents=True, exist_ok=True)
        split_plot.write_text("The character leaves to get help.\n", encoding="utf-8")
        subtitle = benchmark / f"story/subtt/{movie_id}.srt"
        subtitle.parent.mkdir(parents=True, exist_ok=True)
        subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nI will get help.\n", encoding="utf-8")
        video = benchmark / "story/video_clips" / movie_id / clip_name
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"fixture-not-decoded")

        config = {
            "data_root": str(root),
            "movieqa": {
                "benchmark_dir": "raw/movieqa/benchmark",
                "output_dir": "processed/movieqa",
                "max_video_clips_per_sample": 1,
            },
            "frame_cache": {
                "root_dir": "cache/frames",
                "frames_per_video": 1,
                "jpeg_quality": 90,
                "overwrite": False,
            },
            "movienet": {
                "root_dir": "raw/movienet",
                "evidence_output": "processed/evidence/movienet.jsonl",
                "text_dirs": ["subtitle"],
                "image_dirs": ["keyframes"],
                "video_dirs": ["videos"],
            },
            "moviechat_1k": {
                "root_dir": "raw/moviechat_1k",
                "output_file": "processed/moviechat_1k/generalization_test.jsonl",
                "split": "generalization_test",
            },
        }
        frame_cache = FrameCache(root / "cache/frames", frames_per_video=1)
        frame = frame_cache.directory("movieqa", movie_id, video) / "frame_000.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color=(20, 40, 60)).save(frame)

        normalizer = MovieQANormalizer(config, extract_frames=False)
        counts = normalizer.prepare(["train"])
        output = root / "processed/movieqa/train.jsonl"
        report = validate_jsonl(output, root)
        dataset = KARINADataset(output)
        sample = dataset[0]
        payload = KARINAInputAdapter(root).to_router_payload(sample)
        batch = collate_karina_samples([sample])

        assert counts == {"train": 1}
        assert report["valid"], report
        assert len(dataset) == 1
        assert sample["answer"] == "To get help"
        assert len(sample["initial_input"]["text"]) == 3
        assert len(sample["initial_input"]["image"]) == 1
        assert len(sample["initial_input"]["video"]) == 1
        # Adapter conservatively caps text assets at 3, plus image and video.
        assert len(payload["file_paths"]) == 5
        assert sample["retrieval_target"][0]["source"] == "movieqa_plot_alignment"
        assert batch["sample_id"] == ["movieqa:train:0"]

        movienet = root / "raw/movienet"
        mn_subtitle = movienet / f"subtitle/{movie_id}.srt"
        mn_subtitle.parent.mkdir(parents=True, exist_ok=True)
        mn_subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nExternal evidence.\n", encoding="utf-8")
        mn_frame = movienet / f"keyframes/{movie_id}/000.jpg"
        mn_frame.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(mn_frame)
        mn_video = movienet / f"videos/{movie_id}.mp4"
        mn_video.parent.mkdir(parents=True, exist_ok=True)
        mn_video.write_bytes(b"fixture")
        evidence_count, matched = build_movienet_pool(config, {movie_id})
        assert evidence_count == 3
        assert len(matched[movie_id]) == 3

        moviechat = root / "raw/moviechat_1k/test"
        chat_video = moviechat / "raw_videos/1.mp4"
        chat_video.parent.mkdir(parents=True, exist_ok=True)
        chat_video.write_bytes(b"fixture-not-decoded")
        chat_annotation = moviechat / "gt/gt/1.json"
        _write_json(
            chat_annotation,
            {
                "info": {"video_path": "1.mp4", "fps": 25, "num_frame": 100},
                "caption": "A person crosses a room and opens a door.",
                "global": [{"question": "What happens?", "answer": "A person opens a door."}],
                "breakpoint": [{"question": "What is opened?", "answer": "A door.", "time": 50}],
            },
        )
        chat_cache = FrameCache(root / "cache/frames", frames_per_video=1)
        chat_frame = chat_cache.directory("moviechat_1k", "1", chat_video) / "frame_000.jpg"
        chat_frame.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(chat_frame)
        chat_count = MovieChatNormalizer(config, extract_frames=False).prepare()
        chat_output = root / "processed/moviechat_1k/generalization_test.jsonl"
        chat_report = validate_jsonl(chat_output, root)
        assert chat_count == 2
        assert chat_report["valid"], chat_report

        atomic_path = root / "processed/atomic.jsonl"
        write_jsonl(atomic_path, [{"version": "complete"}])

        def interrupted_records():
            yield {"version": "partial"}
            raise RuntimeError("simulated interruption")

        try:
            write_jsonl(atomic_path, interrupted_records())
        except RuntimeError:
            pass
        assert list(iter_jsonl(atomic_path)) == [{"version": "complete"}]
        print("PASS: KARINA MovieQA normalize -> validate -> lazy Dataset -> adapter")
        print("PASS: MovieNet evidence merge fixture and MovieChat-1K generalization fixture")
        print("PASS: interrupted JSONL write preserves the previous complete file")


if __name__ == "__main__":
    main()
