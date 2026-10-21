from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from asl_recognizer.features.mmpose_backend import WholeBodyFrame


def test_stream_overlay_releases_writer_and_preserves_timing(tmp_path, monkeypatch):
    from asl_recognizer.translation.local_runtime import OverlayWriter

    writer = MagicMock()
    monkeypatch.setattr("cv2.VideoWriter", MagicMock(return_value=writer))
    frame = WholeBodyFrame(np.ones((133, 2), np.float32), np.ones(133, np.bool_), 120)
    output = tmp_path / "overlay.mp4"
    with OverlayWriter(Path("source.mp4"), output, fps=5) as overlay:
        overlay.append(np.zeros((24, 32, 3), np.uint8), frame)
    import json

    value = json.loads(output.with_suffix(".json").read_text())
    assert value["frames"][0]["timestamp_ms"] == 120
    assert value["playback_fps"] == 5
    assert writer.release.called


def test_translation_engine_loads_once_and_checks_identity(tmp_path, monkeypatch):
    import torch
    from transformers import AutoTokenizer

    from asl_recognizer.translation import runtime

    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": {
                "feature_schema": "mmpose-coco-wholebody-133-xy-v1",
                "extractor": {"fixture": True},
            },
            "model": {},
        },
        path,
    )
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr(runtime, "LandmarkT5", constructor)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    engine = runtime.TranslationEngine(path, device="cpu")
    request = object()
    engine.translate(request, {"extractor": {"fixture": True}})
    engine.translate(request, {"extractor": {"fixture": True}})
    assert constructor.call_count == 1
    assert model.translate.call_count == 2
    with pytest.raises(ValueError, match="extractor"):
        engine.translate(request, {"extractor": {"changed": True}})


def test_extraction_engine_streams_and_reuses_backend(tmp_path, monkeypatch):
    import cv2

    from asl_recognizer.translation import runtime
    from asl_recognizer.translation.cache import TranslationFeatureCache

    writer = cv2.VideoWriter(
        str(tmp_path / "video.mp4"), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (32, 24)
    )
    for _ in range(10):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    backend = MagicMock()

    def detect(image, timestamp):
        return WholeBodyFrame(
            np.ones((133, 2), np.float32),
            np.ones(133, np.bool_),
            timestamp,
            np.ones(133, np.float32),
            np.ones((133, 2), np.float32),
        )

    backend.detect_translation.side_effect = detect
    constructor = MagicMock(return_value=backend)
    monkeypatch.setattr(runtime, "MMPoseWholeBodyBackend", constructor)
    monkeypatch.setattr(runtime, "extractor_identity", lambda *a: {"fixture": True})
    engine = runtime.ExtractionEngine({}, device="cpu")
    for name in ("one", "two"):
        report = engine.extract(
            video=tmp_path / "video.mp4", output=tmp_path / name, start_ms=200, end_ms=600
        )
        request, metadata = TranslationFeatureCache(tmp_path / name).load("recorded")
        assert len(request.values) == 2
        assert metadata["preprocessing"]["frame_cap"] == 256
        assert report["quality"]["regions"]["left_hand"]["visible_fraction"] == 1
    assert constructor.call_count == 1


def test_protocol_recovers_after_bad_request_and_keeps_ids():
    import io
    import json

    from asl_recognizer.translation.runtime import serve

    incoming = io.StringIO("bad json\n" + json.dumps({"id": "one", "value": 2}) + "\n")
    outgoing = io.StringIO()
    serve(lambda item: {"answer": item["value"] + 1}, stdin=incoming, stdout=outgoing)
    rows = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert rows[0]["ready"] is True
    assert rows[1]["ok"] is False
    assert rows[2] == {"id": "one", "ok": True, "result": {"answer": 3}}


def test_worker_paths_cannot_escape_session(tmp_path):
    from asl_recognizer.translation.runtime import request_path

    with pytest.raises(ValueError, match="session"):
        request_path(tmp_path, str(tmp_path / ".." / "escape"))
    assert request_path(tmp_path, str(tmp_path / "result")) == tmp_path / "result"


@pytest.mark.parametrize("role", ["extract", "translate"])
def test_runtime_cli_dispatches_synthetic_requests(tmp_path, monkeypatch, role):
    import io
    import json
    import sys

    from asl_recognizer.translation import runtime
    from asl_recognizer.translation.contracts import DirectTranslationResult

    monkeypatch.setattr(runtime, "require_sol", lambda root: None)
    engine = MagicMock()
    engine.extract.return_value = {"sampled_frames": 2}
    engine.translate.return_value = DirectTranslationResult("fixture", None, "tiny", "test", 1.0)
    engine.checkpoint_sha256 = "fixture"
    engine.initialization_seconds = 1.0
    monkeypatch.setattr(runtime, "ExtractionEngine", lambda *a, **k: engine)
    monkeypatch.setattr(runtime, "TranslationEngine", lambda *a, **k: engine)
    cache = MagicMock()
    cache.load.return_value = (object(), {})
    monkeypatch.setattr(runtime, "TranslationFeatureCache", lambda *a: cache)
    assets = tmp_path / "assets.json"
    assets.write_text(json.dumps({"assets": {}}))
    item = {
        "id": "one",
        "video": str(tmp_path / "video"),
        "output": str(tmp_path / "output"),
        "features": str(tmp_path / "features"),
        "overlay": str(tmp_path / "overlay"),
    }
    incoming = io.StringIO(json.dumps(item) + "\n")
    outgoing = io.StringIO()
    monkeypatch.setattr(sys, "stdin", incoming)
    monkeypatch.setattr(sys, "stdout", outgoing)
    runtime.main(
        [
            "--role",
            role,
            "--work-root",
            str(tmp_path),
            "--assets-json",
            str(assets),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
        ]
    )
    rows = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert rows[-1]["ok"] is True
    assert rows[-1]["id"] == "one"
    if role == "extract":
        assert engine.close.called


def test_runtime_refuses_real_work_without_allocation(tmp_path, monkeypatch):
    from asl_recognizer.translation.runtime import require_sol

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Sol Slurm"):
        require_sol(tmp_path)


def test_native_pose_cache_preserves_all_frames_and_baseline_stride(tmp_path, monkeypatch):
    import cv2

    from asl_recognizer.translation import runtime
    from asl_recognizer.translation.cache import TranslationFeatureCache
    from asl_recognizer.translation.pose_cache import PoseCache

    video = tmp_path / "native.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (32, 24))
    for _ in range(10):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    backend = MagicMock()
    backend.detect_translation.side_effect = lambda image, time: WholeBodyFrame(
        np.full((133, 2), time, np.float32),
        np.ones(133, np.bool_),
        time,
        np.ones(133, np.float32),
        np.full((133, 2), time, np.float32),
    )
    monkeypatch.setattr(runtime, "MMPoseWholeBodyBackend", lambda **kwargs: backend)
    monkeypatch.setattr(runtime, "extractor_identity", lambda *args: {"fixture": True})
    engine = runtime.ExtractionEngine({}, device="cpu")
    output = tmp_path / "features"
    report = engine.extract(video=video, output=output, start_ms=200, end_ms=600, native_pose=True)
    frames, metadata = PoseCache(output / "raw-pose").load("recorded")
    request, _ = TranslationFeatureCache(output).load("recorded")
    assert [f.timestamp_ms for f in frames] == [200, 300, 400, 500]
    assert len(request.values) == 2
    assert report["raw_pose_frames"] == 4
    assert metadata["raw_pose_sampling"] == "every-native-frame-v1"


def test_native_iterator_keeps_all_frames(monkeypatch):
    import cv2

    from asl_recognizer.features.video import iter_video_frames

    class Capture:
        position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_COUNT: 4, cv2.CAP_PROP_FPS: 10}[prop]

        def set(self, prop, value):
            self.position = value

        def read(self):
            self.position += 1
            return True, np.zeros((2, 2, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: Capture())
    frames = list(iter_video_frames(Path("synthetic"), start_ms=0, end_ms=400))
    assert [time for _, time in frames] == [0, 100, 200, 300]
