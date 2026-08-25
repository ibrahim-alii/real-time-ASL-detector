from __future__ import annotations

import numpy as np

from .schema import LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX, LandmarkFrame

PREPROCESSING_VERSION = "shoulder-causal-v1"


class CausalShoulderNormalizer:
    def __init__(self, *, minimum_scale: float = 1e-6) -> None:
        self._minimum_scale = minimum_scale
        self._center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self._scale = 1.0

    def normalize(self, frame: LandmarkFrame) -> LandmarkFrame:
        anchor_present = frame.pose_anchor_present
        if anchor_present:
            left = frame.coordinates[LEFT_SHOULDER_INDEX]
            right = frame.coordinates[RIGHT_SHOULDER_INDEX]
            candidate_scale = float(np.linalg.norm((left - right)[:2]))
            if candidate_scale >= self._minimum_scale:
                self._center = ((left + right) / 2.0).astype(np.float32)
                self._scale = candidate_scale
            else:
                anchor_present = False

        normalized = np.zeros_like(frame.coordinates, dtype=np.float32)
        normalized[frame.presence] = (
            frame.coordinates[frame.presence] - self._center
        ) / self._scale
        presence = frame.presence.copy()
        if not anchor_present:
            presence[[LEFT_SHOULDER_INDEX, RIGHT_SHOULDER_INDEX]] = False
        return LandmarkFrame(normalized, presence, frame.timestamp_ms)


class FramePreprocessor:
    """Shared stateful preprocessing path for offline and streaming frames."""

    def __init__(self) -> None:
        self._normalizer = CausalShoulderNormalizer()

    def process(self, frame: LandmarkFrame) -> LandmarkFrame:
        return self._normalizer.normalize(frame)
