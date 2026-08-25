from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from asl_recognizer.features.mediapipe_backend import MediaPipeHolisticBackend
from asl_recognizer.features.schema import LANDMARK_COUNT
from asl_recognizer.features.video import sampled_video_frames


class _Capture:
    def __init__(self) -> None:
        self.position_ms = 0.0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        if property_id == 1:
            return 10.0
        if property_id == 2:
            return 5.0
        if property_id == 3:
            return self.position_ms + 7.0
        raise AssertionError(property_id)

    def set(self, property_id: int, value: float) -> bool:
        assert property_id == 3
        self.position_ms = value
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class _StaleTimestampCapture(_Capture):
    def get(self, property_id: int) -> float:
        if property_id == 3:
            return 0.0
        return super().get(property_id)


def test_video_sampling_uses_media_timestamps_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _Capture()
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_COUNT=1,
        CAP_PROP_FPS=2,
        CAP_PROP_POS_MSEC=3,
        VideoCapture=lambda _path: capture,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    fake_path = SimpleNamespace(__str__=lambda _self: "clip.mp4")
    sampled = sampled_video_frames(fake_path, sample_count=3)  # type: ignore[arg-type]

    assert sampled.requested_frames == 3
    assert [timestamp for _, timestamp in sampled.frames] == [7, 907, 1807]
    assert capture.released


def test_video_sampling_enforces_increasing_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _StaleTimestampCapture()
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            CAP_PROP_FRAME_COUNT=1,
            CAP_PROP_FPS=2,
            CAP_PROP_POS_MSEC=3,
            VideoCapture=lambda _path: capture,
        ),
    )

    sampled = sampled_video_frames(SimpleNamespace(), sample_count=3)  # type: ignore[arg-type]

    timestamps = [timestamp for _, timestamp in sampled.frames]
    assert timestamps == [0, 900, 1800]


def test_mediapipe_landmark_copy_validates_count_and_placement() -> None:
    landmarks = [SimpleNamespace(x=0.1, y=0.2, z=0.3) for _ in range(21)]
    coordinates = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    presence = np.zeros(LANDMARK_COUNT, dtype=np.bool_)

    MediaPipeHolisticBackend._copy_landmarks(landmarks, coordinates, presence, 21)

    assert presence[21:42].all()
    np.testing.assert_allclose(coordinates[21], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="21 landmarks"):
        MediaPipeHolisticBackend._copy_landmarks(landmarks[:-1], coordinates, presence, 0)
