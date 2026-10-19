from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asl_recognizer.features.video import VideoSample
from asl_recognizer.translation.manifest import build_manifest
from asl_recognizer.translation.person_validation import (
    annotate_caption_rows,
    inspect_clip,
)


def test_person_validation_resume_is_bound_to_video_and_model(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import asl_recognizer.translation.person_validation as people

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        people, "sampled_video_segment", lambda *_a, **_k: VideoSample(((frame, 0),), 1)
    )
    video = tmp_path / "one.mp4"
    video.write_bytes(b"first")
    rows = [{"clip_id": "one", "video_id": "one", "start_ms": "0", "end_ms": "1000"}]
    calls = []

    def detect(_frame):  # type: ignore[no-untyped-def]
        calls.append(1)
        return 1

    kwargs = dict(
        rows=rows,
        video_root=tmp_path,
        model_sha256="model",
        detect_people=detect,
        resume_root=tmp_path / "resume",
    )
    first, _ = annotate_caption_rows(**kwargs)
    second, _ = annotate_caption_rows(**kwargs)
    assert first == second and len(calls) == 1
    video.write_bytes(b"second")
    annotate_caption_rows(**kwargs)
    assert len(calls) == 2
    annotate_caption_rows(**{**kwargs, "model_sha256": "new-model"})
    assert len(calls) == 3


def test_person_evidence_requires_every_sample_to_decode(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import asl_recognizer.translation.person_validation as people

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        people,
        "sampled_video_segment",
        lambda *_args, **_kwargs: VideoSample(((frame, 0), (frame, 1000)), 2),
    )
    counts = iter((1, 2))
    evidence = inspect_clip(
        tmp_path / "a.mp4", start_ms=0, end_ms=2000, detect_people=lambda _frame: next(counts)
    )
    assert (evidence.person_count, evidence.samples, evidence.minimum, evidence.maximum) == (
        2,
        2,
        1,
        2,
    )
    monkeypatch.setattr(
        people,
        "sampled_video_segment",
        lambda *_args, **_kwargs: VideoSample(((frame, 0),), 2),
    )
    with pytest.raises(ValueError, match="every sampled frame"):
        inspect_clip(tmp_path / "a.mp4", start_ms=0, end_ms=2000, detect_people=lambda _frame: 1)


def test_missing_or_uncertain_youtube_clips_never_enter_manifest(
    monkeypatch, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    import asl_recognizer.translation.person_validation as people

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    monkeypatch.setattr(
        people,
        "sampled_video_segment",
        lambda *_args, **_kwargs: VideoSample(((frame, 0),), 1),
    )
    (tmp_path / "present.mp4").write_bytes(b"video")
    rows = [
        {
            "clip_id": "one",
            "video_id": "present",
            "start_ms": "0",
            "end_ms": "1000",
            "caption": "hello",
        },
        {
            "clip_id": "gone",
            "video_id": "gone",
            "start_ms": "0",
            "end_ms": "1000",
            "caption": "bye",
        },
    ]
    annotated, counts = annotate_caption_rows(
        rows, video_root=tmp_path, model_sha256="abc", detect_people=lambda _frame: 1
    )
    assert counts == {"one": 1, "unavailable": 1}
    assert annotated[0]["person_samples"] == "1"
    assert annotated[1]["availability"] == "unavailable"
    manifest = build_manifest(annotated, source="youtube-asl", require_person_count=True)
    assert [item.status for item in manifest.records] == ["accepted", "unavailable"]
