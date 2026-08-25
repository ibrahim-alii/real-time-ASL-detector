from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ..features.schema import FEATURE_DIMENSION, FeatureSequence


def aggregate_sequence(sequence: FeatureSequence) -> npt.NDArray[np.float32]:
    """Return per-coordinate masked mean and standard deviation."""
    landmark_presence = np.repeat(sequence.landmark_mask, 3, axis=1)
    valid = landmark_presence & sequence.temporal_mask[:, None]
    counts = valid.sum(axis=0)
    means = np.divide(
        (sequence.values * valid).sum(axis=0),
        counts,
        out=np.zeros(FEATURE_DIMENSION, dtype=np.float32),
        where=counts > 0,
    )
    centered = np.where(valid, sequence.values - means, 0.0)
    variances = np.divide(
        (centered * centered).sum(axis=0),
        counts,
        out=np.zeros(FEATURE_DIMENSION, dtype=np.float32),
        where=counts > 0,
    )
    return np.concatenate((means, np.sqrt(variances))).astype(np.float32, copy=False)


def aggregate_sequences(
    sequences: Sequence[FeatureSequence],
) -> npt.NDArray[np.float32]:
    if not sequences:
        return np.empty((0, FEATURE_DIMENSION * 2), dtype=np.float32)
    return np.stack([aggregate_sequence(sequence) for sequence in sequences])
