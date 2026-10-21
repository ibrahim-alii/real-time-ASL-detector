import json

import numpy as np
import pytest

from asl_recognizer.features.mmpose_backend import MMPoseWholeBodyBackend
from asl_recognizer.translation import worker
from asl_recognizer.translation.manifest import build_manifest, write_manifest


def test_batch_extraction_streams_and_preserves_raw_quality(tmp_path, monkeypatch):
    import cv2

    from asl_recognizer.features.video import iter_every_other_video_frame
    from asl_recognizer.translation.pose_cache import PoseCache

    root = tmp_path / "videos"
    root.mkdir()
    video = root / "recording.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 4.0, (32, 24))
    for _ in range(4):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    backend = MMPoseWholeBodyBackend.__new__(MMPoseWholeBodyBackend)
    backend.confidence_threshold = 0.5

    def infer(*a, **k):
        yield {
            "predictions": [
                [
                    {
                        "keypoints": np.ones((133, 2), np.float32),
                        "keypoint_scores": np.full(133, 0.8, np.float32),
                    }
                ]
            ]
        }

    backend._inferencer = infer
    monkeypatch.setattr(
        worker,
        "_extraction_backend",
        lambda args: (
            backend,
            {"feature_schema": "mmpose-coco-wholebody-133-xy-v1", "extractor": {"fixture": True}},
        ),
    )
    monkeypatch.setattr(
        worker,
        "every_other_video_frame",
        lambda *a, **k: pytest.fail("buffered video path used"),
        raising=False,
    )
    closed = []

    def frames(*args, **kwargs):
        try:
            yield from iter_every_other_video_frame(*args, **kwargs)
        finally:
            closed.append(True)

    monkeypatch.setattr(worker, "iter_every_other_video_frame", frames, raising=False)
    manifest = build_manifest(
        [
            {
                "id": "clip",
                "video_id": "recording",
                "caption": "sentence",
                "split": "train",
                "start": 0,
                "end": 1,
            }
        ],
        source="how2sign",
    )
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    output = tmp_path / "features"
    overlays = tmp_path / "overlays"
    assert (
        worker.main(
            [
                "extract",
                "--manifest",
                str(path),
                "--video-root",
                str(root),
                "--output",
                str(output),
                "--overlay-root",
                str(overlays),
                "--shard-count",
                "1",
                "--shard-index",
                "0",
            ]
        )
        == 0
    )
    values, metadata = PoseCache(output / "raw-pose").load("clip")
    assert [frame.timestamp_ms for frame in values] == [0, 500]
    assert metadata["preprocessing"]["frame_cap"] == 256
    assert metadata["quality"]["regions"]["left_hand"]["visible_fraction"] == 1
    report = json.loads(next(overlays.glob("*.json")).read_text())
    assert report["playback_fps"] == 2 and closed == [True]
