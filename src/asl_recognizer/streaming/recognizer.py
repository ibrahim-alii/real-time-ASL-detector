from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from asl_recognizer.features.schema import HAND_LANDMARKS, FeatureSequence, LandmarkFrame

from .buffer import TimestampedFeatureBuffer


@dataclass(frozen=True, slots=True)
class Prediction:
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        values = dict(self.probabilities)
        if not values:
            raise ValueError("prediction probabilities must be non-empty")
        invalid = (
            not label or not isfinite(value) or value < 0.0 or value > 1.0
            for label, value in values.items()
        )
        if any(invalid):
            raise ValueError("prediction probabilities must use labels and finite values in [0, 1]")
        if not np.isclose(sum(values.values()), 1.0, atol=1e-6):
            raise ValueError("prediction probabilities must sum to 1")
        object.__setattr__(self, "probabilities", MappingProxyType(values))

    @property
    def label(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]


class Predictor(Protocol):
    def predict(self, sequence: FeatureSequence) -> Prediction: ...


@dataclass(frozen=True, slots=True)
class RecognitionEvent:
    label: str
    timestamp_ms: int
    confidence: float

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("recognition label cannot be blank")
        if self.timestamp_ms < 0:
            raise ValueError("recognition timestamp cannot be negative")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("recognition confidence must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class RecognitionUpdate:
    label: str | None
    confidence: float
    emitted_label: str | None = None
    event: RecognitionEvent | None = None


@dataclass(frozen=True, slots=True)
class StreamingConfig:
    window_ms: int = 2_000
    sequence_length: int = 32
    inference_stride: int = 4
    ema_alpha: float = 0.4
    activation_threshold: float = 0.8
    release_threshold: float = 0.5
    stable_windows: int = 3
    cooldown_ms: int = 1_000
    minimum_frames: int = 4
    minimum_hand_rate: float = 0.1
    minimum_motion: float = 0.01
    background_label: str = "BACKGROUND"

    def __post_init__(self) -> None:
        positive_values = (
            self.window_ms,
            self.sequence_length,
            self.inference_stride,
            self.stable_windows,
        )
        if min(positive_values) < 1:
            raise ValueError("window, sequence, stride, and stability values must be positive")
        if self.cooldown_ms < 0 or self.minimum_frames < 1 or self.minimum_motion < 0.0:
            raise ValueError(
                "cooldown and motion must be non-negative; minimum frames must be positive"
            )
        for name in (
            "ema_alpha",
            "activation_threshold",
            "release_threshold",
            "minimum_hand_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.release_threshold > self.activation_threshold:
            raise ValueError("release threshold cannot exceed activation threshold")
        if not self.background_label.strip():
            raise ValueError("background label cannot be blank")


class StreamingRecognizer:
    """State machine for gated, stabilized recognition over a live feature stream."""

    def __init__(self, predictor: Predictor, config: StreamingConfig | None = None) -> None:
        self.predictor = predictor
        self.config = config or StreamingConfig()
        self.buffer = TimestampedFeatureBuffer(
            window_ms=self.config.window_ms,
            sequence_length=self.config.sequence_length,
            inference_stride=self.config.inference_stride,
        )
        self._smoothed: dict[str, float] = {}
        self._candidate: str | None = None
        self._candidate_count = 0
        self._active_label: str | None = None
        self._last_emitted_at: int | None = None
        self._last_update = RecognitionUpdate(None, 0.0)

    def update(self, frame: LandmarkFrame) -> RecognitionUpdate:
        should_infer = self.buffer.append(frame)
        if not should_infer or len(self.buffer) < self.config.minimum_frames:
            if self._last_update.event is not None:
                self._last_update = RecognitionUpdate(
                    self._last_update.label,
                    self._last_update.confidence,
                )
            return self._last_update
        sequence = self.buffer.sequence()
        if not self._passes_activity_gate(sequence):
            self._clear_observation()
            self._last_update = RecognitionUpdate(None, 0.0)
            return self._last_update

        prediction = self.predictor.predict(sequence)
        self._update_ema(prediction)
        label = max(self._smoothed, key=self._smoothed.__getitem__)
        confidence = self._smoothed[label]

        if label == self.config.background_label:
            self._candidate = None
            self._candidate_count = 0
            self._active_label = None
            self._last_update = RecognitionUpdate(None, confidence)
            return self._last_update

        if self._active_label is not None:
            active_confidence = self._smoothed.get(self._active_label, 0.0)
            if active_confidence < self.config.release_threshold:
                self._active_label = None

        if confidence >= self.config.activation_threshold:
            if label == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = label
                self._candidate_count = 1
        else:
            self._candidate = None
            self._candidate_count = 0

        event: RecognitionEvent | None = None
        cooldown_complete = (
            self._last_emitted_at is None
            or frame.timestamp_ms - self._last_emitted_at >= self.config.cooldown_ms
        )
        if (
            self._candidate_count >= self.config.stable_windows
            and label != self._active_label
            and cooldown_complete
        ):
            event = RecognitionEvent(label, frame.timestamp_ms, confidence)
            self._active_label = label
            self._last_emitted_at = frame.timestamp_ms
            self._candidate = None
            self._candidate_count = 0

        self._last_update = RecognitionUpdate(
            label=label,
            confidence=confidence,
            emitted_label=event.label if event else None,
            event=event,
        )
        return self._last_update

    def interrupt(self) -> None:
        """Clear partial observations while retaining cross-gap cooldown state."""
        self.buffer.reset()
        self._clear_observation()
        self._last_update = RecognitionUpdate(None, 0.0)

    def reset(self) -> None:
        """Start a fully independent recognition session."""
        self.interrupt()
        self._last_emitted_at = None

    def _clear_observation(self) -> None:
        self._smoothed.clear()
        self._candidate = None
        self._candidate_count = 0
        self._active_label = None

    def _update_ema(self, prediction: Prediction) -> None:
        labels = self._smoothed.keys() | prediction.probabilities.keys()
        alpha = self.config.ema_alpha
        self._smoothed = {
            label: alpha * prediction.probabilities.get(label, 0.0)
            + (1.0 - alpha) * self._smoothed.get(label, 0.0)
            for label in labels
        }
        total = sum(self._smoothed.values())
        if total > 0.0:
            self._smoothed = {label: value / total for label, value in self._smoothed.items()}

    def _passes_activity_gate(self, sequence: FeatureSequence) -> bool:
        valid = sequence.temporal_mask
        if not valid.any():
            return False
        hands = sequence.landmark_mask[:, : HAND_LANDMARKS * 2]
        hand_rate = float(hands[valid].any(axis=1).mean())
        if hand_rate < self.config.minimum_hand_rate:
            return False
        if self.config.minimum_motion == 0.0:
            return True
        coordinates = sequence.values.reshape(sequence.values.shape[0], -1, 3)
        movements: list[float] = []
        for index in range(1, len(sequence.values)):
            common = hands[index - 1] & hands[index]
            if common.any():
                movements.extend(
                    np.linalg.norm(
                        coordinates[index, : HAND_LANDMARKS * 2][common]
                        - coordinates[index - 1, : HAND_LANDMARKS * 2][common],
                        axis=1,
                    ).tolist()
                )
        return bool(movements) and float(np.mean(movements)) >= self.config.minimum_motion


ContinuousRecognizer = StreamingRecognizer
