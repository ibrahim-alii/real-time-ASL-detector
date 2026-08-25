from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from asl_recognizer.app.webcam import (
    CameraError,
    ModelError,
    OpenCVCapture,
    OpenCVDisplay,
    PerformanceTracker,
    Prediction,
    RuntimeStatus,
    WebcamRuntime,
    build_benchmark,
    current_rss_bytes,
    load_model_asset,
    model_size_bytes,
)


class FakeCapture:
    def __init__(self, frames: list[object], *, opened: bool = True) -> None:
        self.frames = iter(frames)
        self.opened = opened
        self.released = False

    def is_opened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


class FakeDisplay:
    def __init__(self, keys: list[int], *, closes_after: int | None = None) -> None:
        self.keys = iter(keys)
        self.closes_after = closes_after
        self.shown: list[tuple[object, object, object | None]] = []
        self.closed = False

    def show(self, frame: object, status: object, landmarks: object | None) -> None:
        self.shown.append((frame, status, landmarks))

    def poll_key(self, _delay_ms: int) -> int:
        return next(self.keys, -1)

    def is_open(self) -> bool:
        return self.closes_after is None or len(self.shown) < self.closes_after

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def detect(self, frame: object, timestamp_ms: int) -> tuple[object, int]:
        self.calls += 1
        return frame, timestamp_ms

    def close(self) -> None:
        self.closed = True


class FakeRecognizer:
    def __init__(self) -> None:
        self.calls = 0
        self.resets = 0
        self.interruptions = 0
        self.closed = False

    def update(self, features: object) -> Prediction:
        self.calls += 1
        return Prediction("HELLO", 0.91, "HELLO" if self.calls == 1 else None)

    def reset(self) -> None:
        self.resets += 1

    def interrupt(self) -> None:
        self.interruptions += 1

    def close(self) -> None:
        self.closed = True


@dataclass
class TickingClock:
    value: float = 0.0
    step: float = 0.01

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


def test_runtime_toggles_recognition_and_builds_status_history() -> None:
    capture = FakeCapture(["one", "two", "three", "four"])
    display = FakeDisplay([32, -1, 32, 27])
    detector = FakeDetector()
    recognizer = FakeRecognizer()

    runtime = WebcamRuntime(capture, display, detector, recognizer, clock=TickingClock())
    final_status = runtime.run()

    assert detector.calls == 2
    assert recognizer.calls == 2
    assert recognizer.resets == 1
    assert recognizer.interruptions == 1
    assert final_status.current_sign == "UNCERTAIN"
    assert final_status.confidence == pytest.approx(0.0)
    assert final_status.history == ("HELLO",)
    assert final_status.fps > 0
    assert final_status.extraction_ms.median > 0
    assert final_status.inference_ms.p95 > 0
    assert capture.released and display.closed
    assert detector.closed and recognizer.closed


def test_runtime_releases_resources_when_camera_read_fails() -> None:
    capture = FakeCapture([])
    display = FakeDisplay([])

    with pytest.raises(CameraError, match="Could not read a frame"):
        WebcamRuntime(
            capture, display, FakeDetector(), FakeRecognizer(), clock=TickingClock()
        ).run()

    assert capture.released and display.closed


def test_runtime_reports_unavailable_camera_and_still_closes() -> None:
    capture = FakeCapture([], opened=False)
    display = FakeDisplay([])

    with pytest.raises(CameraError, match="permission"):
        WebcamRuntime(
            capture, display, FakeDetector(), FakeRecognizer(), clock=TickingClock()
        ).run()

    assert capture.released and display.closed


def test_runtime_stops_when_window_is_closed() -> None:
    capture = FakeCapture(["one", "two"])
    display = FakeDisplay([-1, -1], closes_after=1)

    status = WebcamRuntime(
        capture, display, FakeDetector(), FakeRecognizer(), clock=TickingClock()
    ).run()

    assert status.frame_count == 1
    assert capture.released and display.closed


def test_runtime_interrupts_active_recognition_on_exit() -> None:
    capture = FakeCapture(["one", "two"])
    display = FakeDisplay([32, 27])
    recognizer = FakeRecognizer()

    status = WebcamRuntime(
        capture, display, FakeDetector(), recognizer, clock=TickingClock()
    ).run()

    assert recognizer.resets == 1
    assert recognizer.interruptions == 1
    assert not status.recognizing
    assert status.current_sign == "UNCERTAIN"


def test_runtime_wraps_model_failures_and_releases_resources() -> None:
    class BrokenRecognizer(FakeRecognizer):
        def update(self, features: object) -> Prediction:
            raise ValueError("wrong tensor shape")

    capture = FakeCapture(["one", "two"])
    display = FakeDisplay([32])

    with pytest.raises(ModelError, match="incompatible model"):
        WebcamRuntime(
            capture, display, FakeDetector(), BrokenRecognizer(), clock=TickingClock()
        ).run()

    assert capture.released and display.closed


def test_performance_tracker_reports_percentiles_and_cpu() -> None:
    tracker = PerformanceTracker(max_samples=3)
    tracker.observe(1.0, 2.0, 4.0, timestamp=1.0, process_time=0.10)
    tracker.observe(3.0, 4.0, 8.0, timestamp=1.5, process_time=0.20)
    tracker.observe(5.0, 6.0, 12.0, timestamp=2.0, process_time=0.30)
    tracker.observe(7.0, 8.0, 16.0, timestamp=2.5, process_time=0.40)

    snapshot = tracker.snapshot()

    assert snapshot.frame_count == 4
    assert snapshot.fps == pytest.approx(2.0)
    assert snapshot.extraction_ms.median == pytest.approx(5.0)
    assert snapshot.extraction_ms.p95 == pytest.approx(6.8)
    assert snapshot.end_to_end_ms.median == pytest.approx(12.0)
    assert snapshot.cpu_percent == pytest.approx(20.0)


def test_model_asset_helpers_validate_and_measure_files(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"12345")

    assert model_size_bytes(model) == 5
    assert load_model_asset(model, lambda path: path.name) == "model.bin"

    with pytest.raises(ModelError, match="does not exist"):
        load_model_asset(tmp_path / "missing.bin", lambda path: path)
    with pytest.raises(ModelError, match="could not be loaded"):
        load_model_asset(model, lambda _path: (_ for _ in ()).throw(ValueError("bad")))


def test_model_size_counts_directory_contents(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"123")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"4567")

    assert model_size_bytes(tmp_path) == 7


def test_benchmark_combines_runtime_memory_and_model_size(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"12345")
    performance = PerformanceTracker().snapshot()

    benchmark = build_benchmark(performance, model, ram_sampler=lambda: 4096)

    assert benchmark.performance == performance
    assert benchmark.ram_bytes == 4096
    assert benchmark.model_size_bytes == 5
    assert current_rss_bytes() > 0


class FakeCV2:
    WND_PROP_VISIBLE = 1
    FONT_HERSHEY_SIMPLEX = 2
    LINE_AA = 3

    def __init__(self) -> None:
        self.text: list[str] = []
        self.destroyed: list[str] = []

    def VideoCapture(self, index: int) -> object:  # noqa: N802
        assert index == 2
        return "capture"

    def flip(self, frame: object, axis: int) -> object:
        assert axis == 1
        return frame

    def putText(self, frame: object, text: str, *_args: object) -> object:  # noqa: N802
        self.text.append(text)
        return frame

    def imshow(self, name: str, frame: object) -> None:
        assert name == "ASL Recognizer"

    def waitKey(self, delay: int) -> int:  # noqa: N802
        return 288

    def getWindowProperty(self, name: str, prop: int) -> float:  # noqa: N802
        return 1.0

    def destroyWindow(self, name: str) -> None:  # noqa: N802
        self.destroyed.append(name)


def test_opencv_adapters_are_lazy_and_render_status(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = FakeCV2()
    monkeypatch.setattr("asl_recognizer.app.webcam.draw_landmarks", lambda frame, _lm: frame)
    capture = OpenCVCapture(2, cv2_backend=backend)
    display = OpenCVDisplay(cv2_backend=backend)
    status = RuntimeStatus()

    display.show(np.zeros((8, 8, 3), dtype=np.uint8), status, object())

    assert capture.backend == "capture"
    assert display.poll_key(1) == 32
    assert display.is_open()
    assert any("UNCERTAIN" in line for line in backend.text)
    assert any("FPS" in line for line in backend.text)
    display.close()
    assert backend.destroyed == ["ASL Recognizer"]
