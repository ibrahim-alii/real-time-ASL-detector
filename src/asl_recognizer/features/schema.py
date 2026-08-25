from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

HAND_LANDMARKS = 21
POSE_LANDMARKS = 7
LANDMARK_COUNT = HAND_LANDMARKS * 2 + POSE_LANDMARKS
FEATURE_DIMENSION = LANDMARK_COUNT * 3
FEATURE_SCHEMA_VERSION = "landmarks-v1"
LEFT_SHOULDER_INDEX = 43
RIGHT_SHOULDER_INDEX = 44

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    coordinates: FloatArray
    presence: BoolArray
    timestamp_ms: int

    def __post_init__(self) -> None:
        if self.coordinates.shape != (LANDMARK_COUNT, 3):
            raise ValueError(f"coordinates must have shape ({LANDMARK_COUNT}, 3)")
        if self.presence.shape != (LANDMARK_COUNT,):
            raise ValueError(f"presence must have shape ({LANDMARK_COUNT},)")
        if not np.isfinite(self.coordinates).all():
            raise ValueError("landmark coordinates must be finite")

    @property
    def pose_anchor_present(self) -> bool:
        return bool(self.presence[LEFT_SHOULDER_INDEX] and self.presence[RIGHT_SHOULDER_INDEX])

    @property
    def hand_present(self) -> bool:
        return bool(self.presence[: HAND_LANDMARKS * 2].any())


@dataclass(frozen=True, slots=True)
class FeatureSequence:
    values: FloatArray
    landmark_mask: BoolArray
    temporal_mask: BoolArray
    timestamps_ms: IntArray
    schema_version: str = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        time_steps = self.values.shape[0] if self.values.ndim == 2 else -1
        if self.values.shape != (time_steps, FEATURE_DIMENSION):
            raise ValueError(f"values must have shape [T, {FEATURE_DIMENSION}]")
        if self.landmark_mask.shape != (time_steps, LANDMARK_COUNT):
            raise ValueError(f"landmark_mask must have shape [T, {LANDMARK_COUNT}]")
        if self.temporal_mask.shape != (time_steps,):
            raise ValueError("temporal_mask must have shape [T]")
        if self.timestamps_ms.shape != (time_steps,):
            raise ValueError("timestamps_ms must have shape [T]")
        if not np.isfinite(self.values).all():
            raise ValueError("feature values must be finite")
        if self.values.dtype != np.float32:
            raise ValueError("feature values must use float32")
        if self.landmark_mask.dtype != np.bool_ or self.temporal_mask.dtype != np.bool_:
            raise ValueError("feature masks must use bool")
        if self.timestamps_ms.dtype != np.int64:
            raise ValueError("timestamps must use int64")
        coordinates = self.values.reshape(time_steps, LANDMARK_COUNT, 3)
        if np.any(coordinates[~self.landmark_mask] != 0):
            raise ValueError("masked landmarks must have zero coordinates")
        if np.any(self.landmark_mask[~self.temporal_mask]):
            raise ValueError("padded timesteps cannot contain landmark presence")
        if np.any(self.values[~self.temporal_mask] != 0):
            raise ValueError("padded timesteps must have zero values")
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema: {self.schema_version}")


def pack_sequence(frames: list[LandmarkFrame], *, length: int) -> FeatureSequence:
    if length < 1:
        raise ValueError("sequence length must be positive")
    if not frames:
        raise ValueError("cannot pack an empty frame sequence")
    if len(frames) > length:
        indices = np.linspace(0, len(frames) - 1, length, dtype=np.int64)
        selected = [frames[int(index)] for index in indices]
    else:
        selected = frames

    values = np.zeros((length, FEATURE_DIMENSION), dtype=np.float32)
    landmark_mask = np.zeros((length, LANDMARK_COUNT), dtype=np.bool_)
    temporal_mask = np.zeros(length, dtype=np.bool_)
    timestamps = np.zeros(length, dtype=np.int64)
    for index, frame in enumerate(selected):
        coordinates = frame.coordinates.copy()
        coordinates[~frame.presence] = 0
        values[index] = coordinates.reshape(-1)
        landmark_mask[index] = frame.presence
        temporal_mask[index] = True
        timestamps[index] = frame.timestamp_ms
    return FeatureSequence(values, landmark_mask, temporal_mask, timestamps)
