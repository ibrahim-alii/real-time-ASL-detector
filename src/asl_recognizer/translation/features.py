from __future__ import annotations

from itertools import pairwise

import numpy as np

from ..features.mmpose_backend import WholeBodyFrame
from .contracts import TranslationRequest
from .schema import MMPOSE_SCHEMA

TRANSLATION_FEATURE_SCHEMA = MMPOSE_SCHEMA
MAX_ENCODER_FRAMES = 256
MISSING_LANDMARK_VALUE = -1000.0


def mmpose_translation_request(
    frames: list[WholeBodyFrame], *, max_frames: int = MAX_ENCODER_FRAMES
) -> TranslationRequest:
    """Normalize pixel xy with one common scale, preserving hand/body geometry."""
    if not frames or max_frames < 1:
        raise ValueError("non-empty frames and a positive frame cap are required")
    timestamps = [frame.timestamp_ms for frame in frames]
    if any(left >= right for left, right in pairwise(timestamps)):
        raise ValueError("frame timestamps must be strictly increasing")
    values = np.stack([frame.coordinates for frame in frames])
    mask = np.stack([frame.presence for frame in frames])
    if not mask.any():
        raise ValueError("utterance has no observed MMPose landmarks")
    # Normalize the complete clip before downsampling; confidence is never depth.
    observed = values[mask]
    minimum = observed.min(axis=0)
    scale = max(float(np.max(observed.max(axis=0) - minimum)), 1e-6)
    values[mask] = (observed - minimum) / scale
    values[~mask] = MISSING_LANDMARK_VALUE
    indices = np.linspace(0, len(frames) - 1, min(len(frames), max_frames), dtype=np.int64)
    return TranslationRequest(values[indices].reshape(-1, 266), mask[indices], MMPOSE_SCHEMA)
