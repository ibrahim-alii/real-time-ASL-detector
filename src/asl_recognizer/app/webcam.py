from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from asl_recognizer.features.schema import LandmarkFrame
from asl_recognizer.features.visualize import draw_landmarks

Frame = npt.NDArray[np.generic]
T = TypeVar("T")


class CameraError(RuntimeError):
    """The camera could not provide a usable video stream."""


class ModelError(RuntimeError):
    """A model asset or live inference operation failed."""


@dataclass(frozen=True, slots=True)
class Prediction:
    """Minimal result contract shared with the streaming recognizer."""

    label: str
    confidence: float
    emitted_label: str | None = None


class RecognitionResult(Protocol):
    label: str
    confidence: float
    emitted_label: str | None


class Capture(Protocol):
    def is_opened(self) -> bool: ...

    def read(self) -> tuple[bool, Frame | None]: ...

    def release(self) -> None: ...


class Display(Protocol):
    def show(
        self, frame: Frame, status: RuntimeStatus, landmarks: LandmarkFrame | None
    ) -> None: ...

    def poll_key(self, delay_ms: int) -> int: ...

    def is_open(self) -> bool: ...

    def close(self) -> None: ...


class FrameDetector(Protocol):
    def detect(self, frame: Frame, timestamp_ms: int) -> LandmarkFrame: ...


class StreamingRecognizer(Protocol):
    def update(self, frame: LandmarkFrame) -> RecognitionResult: ...

    def reset(self) -> None: ...

    def interrupt(self) -> None: ...


class OpenCVCapture:
    """Small OpenCV capture adapter; OpenCV is imported only when instantiated."""

    def __init__(self, device_index: int = 0, *, cv2_backend: Any | None = None) -> None:
        cv2 = cv2_backend if cv2_backend is not None else _import_cv2()
        self.backend = cv2.VideoCapture(device_index)

    def is_opened(self) -> bool:
        return bool(self.backend.isOpened())

    def read(self) -> tuple[bool, Frame | None]:
        ok, frame = self.backend.read()
        return bool(ok), frame if ok else None

    def release(self) -> None:
        self.backend.release()


class OpenCVDisplay:
    """Mirrored live display with prediction and performance overlays."""

    def __init__(
        self,
        *,
        window_name: str = "ASL Recognizer",
        cv2_backend: Any | None = None,
    ) -> None:
        self._cv2 = cv2_backend if cv2_backend is not None else _import_cv2()
        self._window_name = window_name
        self._shown = False

    def show(
        self, frame: Frame, status: RuntimeStatus, landmarks: LandmarkFrame | None
    ) -> None:
        rendered = draw_landmarks(frame, landmarks) if landmarks is not None else frame.copy()
        rendered = self._cv2.flip(rendered, 1)
        lines = (
            f"{'RUNNING' if status.recognizing else 'STOPPED'}  {status.current_sign}",
            f"Confidence: {status.confidence:.1%}",
            f"History: {' -> '.join(status.history) if status.history else '-'}",
            f"FPS: {status.fps:.1f}  CPU: {status.cpu_percent:.1f}%",
            (
                f"Latency p95 ms: extract {status.extraction_ms.p95:.1f}  "
                f"model {status.inference_ms.p95:.1f}  e2e {status.end_to_end_ms.p95:.1f}"
            ),
        )
        for index, line in enumerate(lines):
            self._cv2.putText(
                rendered,
                line,
                (12, 28 + index * 24),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                self._cv2.LINE_AA,
            )
        self._cv2.imshow(self._window_name, rendered)
        self._shown = True

    def poll_key(self, delay_ms: int) -> int:
        return int(self._cv2.waitKey(delay_ms)) & 0xFF

    def is_open(self) -> bool:
        if not self._shown:
            return True
        return bool(
            self._cv2.getWindowProperty(self._window_name, self._cv2.WND_PROP_VISIBLE) >= 1
        )

    def close(self) -> None:
        if self._shown:
            self._cv2.destroyWindow(self._window_name)
            self._shown = False


@dataclass(frozen=True, slots=True)
class LatencySummary:
    median: float = 0.0
    p95: float = 0.0


@dataclass(frozen=True, slots=True)
class PerformanceSnapshot:
    frame_count: int = 0
    fps: float = 0.0
    cpu_percent: float = 0.0
    extraction_ms: LatencySummary = LatencySummary()
    inference_ms: LatencySummary = LatencySummary()
    end_to_end_ms: LatencySummary = LatencySummary()


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    recognizing: bool = False
    current_sign: str = "UNCERTAIN"
    confidence: float = 0.0
    history: tuple[str, ...] = ()
    frame_count: int = 0
    fps: float = 0.0
    cpu_percent: float = 0.0
    extraction_ms: LatencySummary = LatencySummary()
    inference_ms: LatencySummary = LatencySummary()
    end_to_end_ms: LatencySummary = LatencySummary()


@dataclass(frozen=True, slots=True)
class RuntimeBenchmark:
    performance: PerformanceSnapshot
    ram_bytes: int
    model_size_bytes: int


class PerformanceTracker:
    def __init__(self, *, max_samples: int = 300) -> None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        self._extraction: deque[float] = deque(maxlen=max_samples)
        self._inference: deque[float] = deque(maxlen=max_samples)
        self._end_to_end: deque[float] = deque(maxlen=max_samples)
        self._frames = 0
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._first_process_time: float | None = None
        self._last_process_time: float | None = None

    def observe(
        self,
        extraction_ms: float,
        inference_ms: float,
        end_to_end_ms: float,
        *,
        timestamp: float,
        process_time: float,
    ) -> None:
        if min(extraction_ms, inference_ms, end_to_end_ms) < 0:
            raise ValueError("latencies cannot be negative")
        self._frames += 1
        self._extraction.append(extraction_ms)
        self._inference.append(inference_ms)
        self._end_to_end.append(end_to_end_ms)
        if self._first_timestamp is None:
            self._first_timestamp = timestamp
            self._first_process_time = process_time
        self._last_timestamp = timestamp
        self._last_process_time = process_time

    def snapshot(self) -> PerformanceSnapshot:
        elapsed = _difference(self._first_timestamp, self._last_timestamp)
        cpu_elapsed = _difference(self._first_process_time, self._last_process_time)
        intervals = max(0, self._frames - 1)
        return PerformanceSnapshot(
            frame_count=self._frames,
            fps=intervals / elapsed if elapsed > 0 else 0.0,
            cpu_percent=100.0 * cpu_elapsed / elapsed if elapsed > 0 else 0.0,
            extraction_ms=_summarize(self._extraction),
            inference_ms=_summarize(self._inference),
            end_to_end_ms=_summarize(self._end_to_end),
        )


class WebcamRuntime:
    """Dependency-injected webcam loop with no import-time hardware dependencies."""

    def __init__(
        self,
        capture: Capture,
        display: Display,
        detector: FrameDetector,
        recognizer: StreamingRecognizer,
        *,
        clock: Callable[[], float] = time.perf_counter,
        process_clock: Callable[[], float] = time.process_time,
        history_limit: int = 8,
    ) -> None:
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self._capture = capture
        self._display = display
        self._detector = detector
        self._recognizer = recognizer
        self._clock = clock
        self._process_clock = process_clock
        self._history_limit = history_limit

    def run(self) -> RuntimeStatus:
        status = RuntimeStatus()
        tracker = PerformanceTracker()
        frame_count = 0
        try:
            if not self._capture.is_opened():
                raise CameraError(
                    "Camera is unavailable; check the device index, connection, "
                    "and camera permission."
                )
            while True:
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise CameraError(
                        "Could not read a frame from the camera; check the connection "
                        "or permissions."
                    )
                frame_count += 1
                landmarks: LandmarkFrame | None = None
                if status.recognizing:
                    frame_started = self._clock()
                    timestamp_ms = int(round(frame_started * 1000.0))
                    extraction_started = self._clock()
                    try:
                        landmarks = self._detector.detect(frame, timestamp_ms)
                        extraction_ms = (self._clock() - extraction_started) * 1000.0
                        inference_started = self._clock()
                        prediction = self._recognizer.update(landmarks)
                        inference_ms = (self._clock() - inference_started) * 1000.0
                    except Exception as error:
                        raise ModelError(
                            "Live inference failed; verify that the detector and model assets "
                            "match the configured feature schema (incompatible model or input)."
                        ) from error
                    ended = self._clock()
                    tracker.observe(
                        extraction_ms,
                        inference_ms,
                        (ended - frame_started) * 1000.0,
                        timestamp=ended,
                        process_time=self._process_clock(),
                    )
                    history = status.history
                    if prediction.emitted_label is not None:
                        history = (*history, prediction.emitted_label)[-self._history_limit :]
                    status = replace(
                        status,
                        current_sign=prediction.label or "UNCERTAIN",
                        confidence=float(prediction.confidence),
                        history=history,
                    )
                performance = tracker.snapshot()
                status = replace(
                    status,
                    frame_count=frame_count,
                    fps=performance.fps,
                    cpu_percent=performance.cpu_percent,
                    extraction_ms=performance.extraction_ms,
                    inference_ms=performance.inference_ms,
                    end_to_end_ms=performance.end_to_end_ms,
                )
                self._display.show(frame, status, landmarks)
                if not self._display.is_open():
                    break
                key = self._display.poll_key(1)
                if key == 27:
                    break
                if key == 32:
                    if status.recognizing:
                        self._recognizer.interrupt()
                        status = replace(
                            status,
                            recognizing=False,
                            current_sign="UNCERTAIN",
                            confidence=0.0,
                        )
                    else:
                        self._recognizer.reset()
                        status = replace(status, recognizing=True)
        finally:
            try:
                if status.recognizing:
                    self._recognizer.interrupt()
            finally:
                try:
                    _close_if_supported(self._detector)
                finally:
                    try:
                        _close_if_supported(self._recognizer)
                    finally:
                        try:
                            self._capture.release()
                        finally:
                            self._display.close()
        return replace(status, recognizing=False, current_sign="UNCERTAIN", confidence=0.0)


def load_model_asset(path: Path, loader: Callable[[Path], T]) -> T:
    if not path.is_file():
        raise ModelError(f"Model asset does not exist or is not a file: {path}")
    try:
        return loader(path)
    except Exception as error:
        raise ModelError(
            f"Model asset could not be loaded; verify its format and compatibility: {path}"
        ) from error


def model_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise ModelError(f"Model asset does not exist: {path}")


def current_rss_bytes() -> int:
    """Return resident memory using only the standard library."""
    if sys.platform == "win32":
        return _windows_rss_bytes()
    try:
        import resource
    except ImportError as error:
        raise RuntimeError("RAM measurement is unsupported on this platform") from error
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


def build_benchmark(
    performance: PerformanceSnapshot,
    model_path: Path,
    *,
    ram_sampler: Callable[[], int] = current_rss_bytes,
) -> RuntimeBenchmark:
    return RuntimeBenchmark(performance, ram_sampler(), model_size_bytes(model_path))


def _windows_rss_bytes() -> int:
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = wintypes.HANDLE
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    get_memory_info.restype = wintypes.BOOL
    handle = get_current_process()
    if not get_memory_info(
        handle, ctypes.byref(counters), counters.cb
    ):
        raise OSError("Windows could not report process memory")
    return int(counters.WorkingSetSize)


def _summarize(values: deque[float]) -> LatencySummary:
    if not values:
        return LatencySummary()
    samples = np.asarray(values, dtype=np.float64)
    return LatencySummary(float(np.median(samples)), float(np.percentile(samples, 95)))


def _difference(start: float | None, end: float | None) -> float:
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


def _close_if_supported(component: object) -> None:
    close = getattr(component, "close", None)
    if callable(close):
        close()


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for webcam capture and display") from error
    return cv2
