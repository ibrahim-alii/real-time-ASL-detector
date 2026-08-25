from __future__ import annotations

from pathlib import Path

import numpy as np

from .extract import VideoSample


def sampled_video_frames(
    path: Path, *, sample_count: int = 32
) -> VideoSample:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
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
        if frame_count <= 0 or fps <= 0:
            raise ValueError(f"invalid video metadata: {path.name}")
        requested = min(sample_count, frame_count)
        duration_ms = max(0.0, (frame_count - 1) * 1000.0 / fps)
        target_times = np.linspace(0.0, duration_ms, requested)
        frames: list[tuple[np.ndarray, int]] = []
        previous_timestamp = -1
        for target_ms in target_times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(target_ms))
            ok, frame = capture.read()
            if not ok:
                continue
            actual_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp = int(round(actual_ms if actual_ms >= 0 else target_ms))
            if timestamp <= previous_timestamp:
                timestamp = max(int(round(float(target_ms))), previous_timestamp + 1)
            frames.append((frame, timestamp))
            previous_timestamp = timestamp
        return VideoSample(tuple(frames), requested)
    finally:
        capture.release()
