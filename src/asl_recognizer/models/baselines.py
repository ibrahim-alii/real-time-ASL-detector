from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..features.schema import FeatureSequence
from ..training.features import aggregate_sequences

IntArray = npt.NDArray[np.int64]


class MajorityBaseline:
    """Deterministic classifier that always predicts the most frequent class."""

    def __init__(self) -> None:
        self.majority_class_: int | None = None

    def fit(self, labels: Sequence[int] | IntArray) -> MajorityBaseline:
        values = np.asarray(labels, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("labels must be a non-empty one-dimensional sequence")
        classes, counts = np.unique(values, return_counts=True)
        self.majority_class_ = int(classes[np.argmax(counts)])
        return self

    def predict(self, sample_count: int) -> IntArray:
        if self.majority_class_ is None:
            raise RuntimeError("baseline must be fitted before prediction")
        if sample_count < 0:
            raise ValueError("sample_count cannot be negative")
        return np.full(sample_count, self.majority_class_, dtype=np.int64)


class LogisticBaseline:
    """Linear reference model over masked temporal mean and standard deviation."""

    def __init__(self, *, random_state: int = 0, max_iter: int = 1_000) -> None:
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(random_state=random_state, max_iter=max_iter),
                ),
            ]
        )

    def fit(
        self, sequences: Sequence[FeatureSequence], labels: Sequence[int] | IntArray
    ) -> LogisticBaseline:
        if len(sequences) == 0:
            raise ValueError("sequences cannot be empty")
        values = np.asarray(labels, dtype=np.int64)
        if values.shape != (len(sequences),):
            raise ValueError("one label is required per sequence")
        self._pipeline.fit(aggregate_sequences(sequences), values)
        return self

    def predict(self, sequences: Sequence[FeatureSequence]) -> IntArray:
        if len(sequences) == 0:
            return np.empty(0, dtype=np.int64)
        return np.asarray(self._pipeline.predict(aggregate_sequences(sequences)), dtype=np.int64)

    def predict_proba(self, sequences: Sequence[FeatureSequence]) -> npt.NDArray[np.float64]:
        if len(sequences) == 0:
            raise ValueError("sequences cannot be empty")
        return np.asarray(
            self._pipeline.predict_proba(aggregate_sequences(sequences)), dtype=np.float64
        )
