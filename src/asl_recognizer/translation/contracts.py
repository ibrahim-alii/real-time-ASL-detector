from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .schema import MMPOSE_SCHEMA, translation_schema


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """A locally buffered utterance represented only by selected landmarks."""

    values: npt.NDArray[np.float32]
    landmark_mask: npt.NDArray[np.bool_]
    feature_schema: str = MMPOSE_SCHEMA

    def __post_init__(self) -> None:
        schema = translation_schema(self.feature_schema)
        if self.values.ndim != 2 or self.values.shape[1] != schema.dimension:
            raise ValueError(f"translation values must have shape [frames, {schema.dimension}]")
        if self.landmark_mask.shape != (self.values.shape[0], schema.landmarks):
            raise ValueError("translation landmark mask does not match the feature schema")
        if self.values.dtype != np.float32 or self.landmark_mask.dtype != np.bool_:
            raise ValueError("translation inputs must use float32 values and bool masks")
        if not len(self.values) or not np.isfinite(self.values).all():
            raise ValueError("translation input must be non-empty and finite")


@dataclass(frozen=True, slots=True)
class DirectTranslationResult:
    """Generated English plus auditable decoding and runtime identity."""

    english: str
    decoder_score: float | None
    model_id: str
    model_version: str
    latency_ms: float
    beam_width: int = 5

    def __post_init__(self) -> None:
        if self.decoder_score is not None and not np.isfinite(self.decoder_score):
            raise ValueError("decoder score must be finite when provided")
        if not self.model_id or not self.model_version:
            raise ValueError("model identity and version are required")
        if self.latency_ms < 0 or self.beam_width < 1:
            raise ValueError("latency and beam width are invalid")
