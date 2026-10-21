from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class VideoSample:
    frames: tuple[tuple[npt.NDArray[np.uint8], int], ...]
    requested_frames: int


def sampled_video_segment(
    path: Path,
    *,
    start_ms: int,
    end_ms: int,
    sample_rate_hz: float = 15.0,
) -> VideoSample:
    """Sample a time-bounded video segment without forcing a fixed sequence length."""
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("video segment timing is invalid")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for video decoding") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"unable to open video: {path.name}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid video metadata: {path.name}")
        video_end_ms = frame_count * 1000.0 / fps
        bounded_end = min(float(end_ms), video_end_ms)
        if bounded_end <= start_ms:
            raise ValueError(f"video segment lies outside {path.name}")
        interval_ms = 1000.0 / sample_rate_hz
        target_times = np.arange(float(start_ms), bounded_end, interval_ms)
        frames: list[tuple[npt.NDArray[np.uint8], int]] = []
        previous_timestamp = start_ms - 1
        for target_ms in target_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(target_ms))
            ok, frame = capture.read()
            if not ok:
                continue
            actual_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(actual_ms) or not start_ms <= actual_ms < bounded_end:
                continue
            timestamp = round(actual_ms)
            if timestamp <= previous_timestamp or timestamp >= end_ms:
                continue
            frames.append((cast(npt.NDArray[np.uint8], frame), timestamp))
            previous_timestamp = timestamp
        return VideoSample(tuple(frames), len(target_times))
    finally:
        capture.release()


def every_other_video_frame(path: Path, *, start_ms: int, end_ms: int) -> VideoSample:
    """Decode every second native frame within a caption's half-open time span."""
    frames = tuple(iter_every_other_video_frame(path, start_ms=start_ms, end_ms=end_ms))
    return VideoSample(frames, len(frames))


def iter_every_other_video_frame(
    path: Path, *, start_ms: int, end_ms: int
) -> Iterator[tuple[npt.NDArray[np.uint8], int]]:
    """Decode incrementally; closing the iterator also releases the video handle."""
    yield from iter_video_frames(path, start_ms=start_ms, end_ms=end_ms, frame_stride=2)


def iter_video_frames(
    path: Path, *, start_ms: int, end_ms: int, frame_stride: int = 1
) -> Iterator[tuple[npt.NDArray[np.uint8], int]]:
    """Yield native frames with bounded image memory and an explicit sampling stride."""
    if start_ms < 0 or end_ms <= start_ms:
        raise ValueError("video segment timing is invalid")
    if frame_stride < 1:
        raise ValueError("frame stride must be positive")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for video decoding") from error
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"unable to open video: {path.name}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if count <= 0 or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"invalid video metadata: {path.name}")
        first = max(0, int(np.ceil(start_ms * fps / 1000)))
        stop = min(count, int(np.ceil(end_ms * fps / 1000)))
        if stop <= first:
            raise ValueError(f"video segment lies outside {path.name}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for index in range(first, stop):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"unable to decode frame {index}: {path.name}")
            if (index - first) % frame_stride == 0:
                yield cast(npt.NDArray[np.uint8], frame), round(index * 1000 / fps)
    finally:
        capture.release()
