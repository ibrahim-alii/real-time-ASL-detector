from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from asl_recognizer.features.mmpose_backend import MMPoseWholeBodyBackend
from asl_recognizer.features.video import VideoSample
from asl_recognizer.translation import worker
from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import TranslationRequest
from asl_recognizer.translation.features import mmpose_translation_request
from asl_recognizer.translation.manifest import build_manifest, write_manifest
from asl_recognizer.translation.schema import MMPOSE_SCHEMA


def _prediction() -> dict[str, Any]:
    return {"predictions": [[{
        "keypoints": np.ones((133, 2), dtype=np.float32),
        "keypoint_scores": np.ones(133, dtype=np.float32),
    }]]}


def test_mmpose_masks_low_confidence_and_nonfinite_points() -> None:
    result = _prediction()
    person = result["predictions"][0][0]
    person["keypoint_scores"][112] = 0.1
    person["keypoints"][113, 0] = np.nan
    frame = MMPoseWholeBodyBackend.frame_from_result(result, 10)
    assert frame.coordinates.shape == (133, 2)
    assert frame.presence.sum() == 131
    assert np.all(frame.coordinates[[112, 113]] == 0)


def test_mmpose_missing_person_is_masked_and_multiple_people_are_rejected() -> None:
    frame = MMPoseWholeBodyBackend.frame_from_result({"predictions": [[]]}, 0)
    assert not frame.presence.any()
    result = _prediction()
    result["predictions"][0].append(result["predictions"][0][0])
    with pytest.raises(ValueError, match="multiple-person"):
        MMPoseWholeBodyBackend.frame_from_result(result, 0)


def test_mmpose_rejects_body_only_and_depth_or_confidence_as_coordinates() -> None:
    result = _prediction()
    result["predictions"][0][0]["keypoints"] = np.ones((133, 3))
    with pytest.raises(ValueError, match="133-point 2D"):
        MMPoseWholeBodyBackend.frame_from_result(result, 0)
    with pytest.raises(ValueError, match="266"):
        TranslationRequest(np.ones((2, 255), np.float32), np.ones((2, 85), np.bool_))
    request = TranslationRequest(
        np.ones((2, 266), np.float32), np.ones((2, 133), np.bool_), MMPOSE_SCHEMA
    )
    assert request.feature_schema == MMPOSE_SCHEMA


def test_mmpose_normalization_uses_common_scale_and_preserves_missing(tmp_path: Path) -> None:
    result = _prediction()
    person = result["predictions"][0][0]
    person["keypoints"][0] = [10, 20]
    person["keypoints"][1] = [110, 70]
    person["keypoint_scores"][112] = 0
    frames = [MMPoseWholeBodyBackend.frame_from_result(result, time) for time in (0, 10, 20)]
    request = mmpose_translation_request(frames, max_frames=2)
    xy = request.values.reshape(2, 133, 2)
    assert np.max(xy[request.landmark_mask]) == pytest.approx(1)
    assert xy[0, 1, 0] - xy[0, 0, 0] == pytest.approx(2 * (xy[0, 1, 1] - xy[0, 0, 1]))
    assert np.all(xy[:, 112] == -1000)
    cache = TranslationFeatureCache(tmp_path)
    cache.store("clip", request, metadata={"feature_schema": MMPOSE_SCHEMA})
    restored, _ = cache.load("clip")
    np.testing.assert_array_equal(restored.values, request.values)
    assert restored.feature_schema == MMPOSE_SCHEMA
    with pytest.raises(ValueError, match="schemas differ"):
        cache.store("bad", request, metadata={"feature_schema": "youtube-asl-holistic-85-v2"})
    with pytest.raises(ValueError, match="strictly increasing"):
        mmpose_translation_request([frames[1], frames[0]])


def test_worker_mmpose_extraction_resumes_and_recomputes_changed_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    backend = MMPoseWholeBodyBackend.__new__(MMPoseWholeBodyBackend)
    backend.confidence_threshold = 0.5

    def inferencer(*_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(1)
        yield _prediction()

    backend._inferencer = inferencer
    identity: dict[str, object] = {
        "feature_schema": MMPOSE_SCHEMA, "extractor": {"pose_weights": "first"},
    }
    monkeypatch.setattr(worker, "_extraction_backend", lambda _args: (backend, identity))
    image = np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(worker, "every_other_video_frame", lambda *_args, **_kwargs:
                        VideoSample(((image, 0), (image, 10)), 2))
    manifest = build_manifest([
        {"id": "clip", "video_id": "recording", "caption": "hello", "split": "train",
         "start": 0, "end": 1}
    ], source="how2sign")
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "recording.mp4").write_bytes(b"fixture")
    output = tmp_path / "features"
    args = ["extract", "--manifest", str(path), "--video-root", str(videos),
            "--output", str(output), "--shard-count", "1", "--shard-index", "0"]
    real_clock = worker.time.monotonic
    times = iter((0.0, 2.0))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(times))
    assert worker.main([*args, "--wall-time-seconds", "1"]) == 0
    report = json.loads((output / "extract-shard-00000.json").read_text())
    assert report["paused"] and not report["completed"] and not report["failures"]
    assert not calls
    monkeypatch.setattr(worker.time, "monotonic", real_clock)
    assert worker.main(args) == 0
    assert not json.loads((output / "extract-shard-00000.json").read_text())["paused"]
    assert worker.main(args) == 0
    assert len(calls) == 2
    identity["extractor"] = {"pose_weights": "second"}
    assert worker.main(args) == 0
    assert len(calls) == 4
    request, metadata = TranslationFeatureCache(output).load("clip")
    assert request.feature_schema == MMPOSE_SCHEMA
    assert metadata["extractor"] == {"pose_weights": "second"}


def test_explicit_detector_never_invents_pose_when_no_person(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    pose_calls: list[int] = []

    class Instances:
        def __init__(self, count: int) -> None:
            self.bboxes = np.ones((count, 4), np.float32)
            self.scores = np.ones(count, np.float32)
            self.labels = np.zeros(count, np.int64)

        def cpu(self) -> Instances:
            return self

        def numpy(self) -> Instances:
            return self

    modules = {name: ModuleType(name) for name in (
        "mmdet", "mmdet.apis", "mmpose", "mmpose.apis", "mmpose.utils",
        "mmpose.evaluation", "mmpose.evaluation.functional")}
    modules["mmdet.apis"].init_detector = (  # type: ignore[attr-defined]
        lambda *_a, **_k: SimpleNamespace(cfg={}))
    modules["mmdet.apis"].inference_detector = (  # type: ignore[attr-defined]
        lambda _model, image: SimpleNamespace(pred_instances=Instances(int(image[0, 0, 0]))))
    modules["mmpose.apis"].init_model = lambda *_a, **_k: object()  # type: ignore[attr-defined]
    modules["mmpose.utils"].adapt_mmdet_pipeline = lambda cfg: cfg  # type: ignore[attr-defined]
    modules["mmpose.evaluation.functional"].nms = (  # type: ignore[attr-defined]
        lambda boxes, _threshold: np.arange(len(boxes)))

    def pose(*_args: object) -> list[object]:
        pose_calls.append(1)
        instance = SimpleNamespace(keypoints=np.ones((1, 133, 2), np.float32),
                                   keypoint_scores=np.ones((1, 133), np.float32))
        instance.cpu = lambda: instance
        instance.numpy = lambda: instance
        return [SimpleNamespace(pred_instances=instance)]

    modules["mmpose.apis"].inference_topdown = pose  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    assets = {}
    for name in ("pose_config", "pose_weights", "detector_config", "detector_weights"):
        assets[name] = tmp_path / name
        assets[name].write_text("fixture")
    detector = MMPoseWholeBodyBackend(**assets)
    empty = np.zeros((10, 10, 3), np.uint8)
    assert detector.count_people(empty) == 0
    assert not detector.detect_translation(empty, 0).presence.any()
    assert not pose_calls
    one = np.ones_like(empty)
    assert detector.count_people(one) == 1
    assert detector.detect_translation(one, 0).presence.all()
    assert len(pose_calls) == 1
    multiple = np.full_like(empty, 2)
    assert detector.count_people(multiple) == 2
    with pytest.raises(ValueError, match="multiple-person"):
        detector.detect_translation(multiple, 0)
    assert len(pose_calls) == 1
