from __future__ import annotations

from collections import deque

import numpy as np

from asl_recognizer.features.normalize import FramePreprocessor
from asl_recognizer.features.schema import FeatureSequence, LandmarkFrame, pack_sequence


class TimestampedFeatureBuffer:
    """Causally preprocess and retain a fixed-duration window of landmark frames."""

    def __init__(
        self,
        *,
        window_ms: int = 2_000,
        sequence_length: int = 32,
        inference_stride: int = 4,
    ) -> None:
        if window_ms < 1 or sequence_length < 1 or inference_stride < 1:
            raise ValueError("buffer dimensions and stride must be positive")
        self.window_ms = window_ms
        self.sequence_length = sequence_length
        self.inference_stride = inference_stride
        self._frames: deque[LandmarkFrame] = deque()
        self._preprocessor = FramePreprocessor()
        self._frames_since_inference = 0

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> tuple[LandmarkFrame, ...]:
        return tuple(self._frames)

    def append(self, frame: LandmarkFrame) -> bool:
        if self._frames and frame.timestamp_ms <= self._frames[-1].timestamp_ms:
            raise ValueError("frame timestamps must be strictly increasing")
        normalized = self._preprocessor.process(frame)
        self._frames.append(normalized)
        cutoff = frame.timestamp_ms - self.window_ms
        while self._frames and self._frames[0].timestamp_ms < cutoff:
            self._frames.popleft()
        self._frames_since_inference += 1
        if self._frames_since_inference < self.inference_stride:
            return False
        self._frames_since_inference = 0
        return True

    def sequence(self) -> FeatureSequence:
        if not self._frames:
            raise ValueError("cannot create a sequence from an empty buffer")
        source = list(self._frames)
        if len(source) == 1:
            selected = [source[0]] * self.sequence_length
        else:
            timestamps = np.asarray([frame.timestamp_ms for frame in source], dtype=np.int64)
            targets = np.linspace(
                timestamps[0], timestamps[-1], self.sequence_length, dtype=np.float64
            )
            right = np.searchsorted(timestamps, targets, side="left")
            right = np.clip(right, 0, len(source) - 1)
            left = np.maximum(right - 1, 0)
            use_left = np.abs(targets - timestamps[left]) <= np.abs(timestamps[right] - targets)
            indices = np.where(use_left, left, right)
            selected = [source[int(index)] for index in indices]
        return pack_sequence(selected, length=self.sequence_length)

    def reset(self) -> None:
        self._frames.clear()
        self._preprocessor = FramePreprocessor()
        self._frames_since_inference = 0

