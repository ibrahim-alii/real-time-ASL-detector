from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from asl_recognizer.calibration import (
    AnnotatedInterval,
    CalibrationRecorder,
    CalibrationSession,
    evaluate_events,
)
from asl_recognizer.features.schema import LANDMARK_COUNT, FeatureSequence, LandmarkFrame
from asl_recognizer.streaming import (
    Prediction,
    RecognitionEvent,
    StreamingConfig,
    StreamingRecognizer,
    TimestampedFeatureBuffer,
)


def _frame(timestamp_ms: int, *, hand_x: float = 0.5, hand_present: bool = True) -> LandmarkFrame:
    coordinates = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    presence = np.zeros(LANDMARK_COUNT, dtype=np.bool_)
    coordinates[43] = [0.4, 0.5, 0.0]
    coordinates[44] = [0.6, 0.5, 0.0]
    presence[[43, 44]] = True
    if hand_present:
        coordinates[0] = [hand_x, 0.4, 0.0]
        presence[0] = True
    return LandmarkFrame(coordinates, presence, timestamp_ms)


class _SequencePredictor:
    def __init__(self, predictions: list[Prediction]) -> None:
        self.predictions = iter(predictions)
        self.calls: list[FeatureSequence] = []

    def predict(self, sequence: FeatureSequence) -> Prediction:
        self.calls.append(sequence)
        return next(self.predictions)


def test_buffer_keeps_two_seconds_and_resamples_to_32_steps() -> None:
    buffer = TimestampedFeatureBuffer(window_ms=2_000, sequence_length=32, inference_stride=4)
    for timestamp in range(0, 2_501, 100):
        buffer.append(_frame(timestamp, hand_x=0.5 + timestamp / 10_000))

    sequence = buffer.sequence()

    assert sequence.values.shape == (32, LANDMARK_COUNT * 3)
    assert sequence.temporal_mask.all()
    assert sequence.timestamps_ms[0] >= 500
    assert sequence.timestamps_ms[-1] == 2_500
    assert np.all(np.diff(sequence.timestamps_ms) >= 0)


def test_buffer_enforces_timestamps_stride_and_reset() -> None:
    buffer = TimestampedFeatureBuffer(window_ms=2_000, sequence_length=32, inference_stride=4)

    assert not buffer.append(_frame(0))
    assert not buffer.append(_frame(10))
    assert not buffer.append(_frame(20))
    assert buffer.append(_frame(30))
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append(_frame(30))

    buffer.reset()
    assert len(buffer) == 0
    assert not buffer.append(_frame(0))


def test_prediction_rejects_invalid_probability_distributions() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        Prediction({"HELLO": 0.8, "THANKS": 0.8})
    with pytest.raises(ValueError, match="non-empty"):
        Prediction({})


def test_recognition_event_rejects_invalid_public_values() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        RecognitionEvent("HELLO", -1, 0.9)
    with pytest.raises(ValueError, match="confidence"):
        RecognitionEvent("HELLO", 1, 1.1)


def test_recognizer_ema_stability_emits_once_and_suppresses_duplicates() -> None:
    predictor = _SequencePredictor(
        [Prediction({"HELLO": 0.9, "THANKS": 0.1}) for _ in range(5)]
    )
    recognizer = StreamingRecognizer(
        predictor,
        StreamingConfig(
            inference_stride=1,
            stable_windows=3,
            ema_alpha=1.0,
            activation_threshold=0.8,
            release_threshold=0.5,
            cooldown_ms=500,
            minimum_frames=2,
            minimum_hand_rate=0.1,
            minimum_motion=0.0,
        ),
    )

    updates = [recognizer.update(_frame(index * 100)) for index in range(6)]

    assert [update.emitted_label for update in updates] == [None, None, None, "HELLO", None, None]
    assert updates[3].event == RecognitionEvent("HELLO", 300, 0.9)
    assert updates[-1].label == "HELLO"


def test_emitted_label_is_not_repeated_between_inference_steps() -> None:
    predictor = _SequencePredictor([Prediction({"HELLO": 1.0})])
    recognizer = StreamingRecognizer(
        predictor,
        StreamingConfig(
            inference_stride=2,
            stable_windows=1,
            ema_alpha=1.0,
            cooldown_ms=0,
            minimum_frames=1,
            minimum_hand_rate=0.0,
            minimum_motion=0.0,
        ),
    )

    recognizer.update(_frame(0))
    assert recognizer.update(_frame(100)).emitted_label == "HELLO"
    between_steps = recognizer.update(_frame(200))
    assert between_steps.label == "HELLO"
    assert between_steps.emitted_label is None


def test_recognizer_releases_on_low_confidence_and_respects_cooldown() -> None:
    predictor = _SequencePredictor(
        [
            Prediction({"HELLO": 0.9, "THANKS": 0.1}),
            Prediction({"HELLO": 0.9, "THANKS": 0.1}),
            Prediction({"HELLO": 0.2, "THANKS": 0.8}),
            Prediction({"HELLO": 0.9, "THANKS": 0.1}),
            Prediction({"HELLO": 0.9, "THANKS": 0.1}),
            Prediction({"HELLO": 0.9, "THANKS": 0.1}),
        ]
    )
    recognizer = StreamingRecognizer(
        predictor,
        StreamingConfig(
            inference_stride=1,
            stable_windows=2,
            ema_alpha=1.0,
            activation_threshold=0.8,
            release_threshold=0.4,
            cooldown_ms=500,
            minimum_frames=1,
            minimum_hand_rate=0.0,
            minimum_motion=0.0,
        ),
    )

    updates = [recognizer.update(_frame(timestamp)) for timestamp in (0, 100, 200, 300, 400, 700)]

    assert updates[1].emitted_label == "HELLO"
    assert updates[4].emitted_label is None
    assert updates[5].emitted_label == "HELLO"


def test_background_prediction_releases_without_emitting_background() -> None:
    predictor = _SequencePredictor(
        [
            Prediction({"HELLO": 0.9, "BACKGROUND": 0.1}),
            Prediction({"HELLO": 0.1, "BACKGROUND": 0.9}),
            Prediction({"HELLO": 0.9, "BACKGROUND": 0.1}),
        ]
    )
    recognizer = StreamingRecognizer(
        predictor,
        StreamingConfig(
            inference_stride=1,
            stable_windows=1,
            ema_alpha=1.0,
            cooldown_ms=0,
            minimum_frames=1,
            minimum_hand_rate=0.0,
            minimum_motion=0.0,
        ),
    )

    assert recognizer.update(_frame(0)).emitted_label == "HELLO"
    background = recognizer.update(_frame(100))
    assert background.label is None
    assert background.emitted_label is None
    assert recognizer.update(_frame(200)).emitted_label == "HELLO"


def test_motion_gate_and_interruption_clear_pending_state() -> None:
    predictor = _SequencePredictor(
        [Prediction({"HELLO": 0.95, "THANKS": 0.05}) for _ in range(3)]
    )
    recognizer = StreamingRecognizer(
        predictor,
        StreamingConfig(
            inference_stride=1,
            stable_windows=2,
            ema_alpha=1.0,
            activation_threshold=0.8,
            release_threshold=0.5,
            minimum_frames=2,
            minimum_hand_rate=0.1,
            minimum_motion=0.01,
        ),
    )

    assert recognizer.update(_frame(0, hand_x=0.5)).label is None
    assert recognizer.update(_frame(100, hand_x=0.5)).label is None
    assert not predictor.calls
    assert recognizer.update(_frame(200, hand_x=0.7)).emitted_label is None
    recognizer.interrupt()
    assert recognizer.update(_frame(300, hand_x=0.8)).label is None
    assert recognizer.update(_frame(400, hand_x=1.0)).emitted_label is None
    assert recognizer.update(_frame(500, hand_x=1.2)).emitted_label == "HELLO"


def test_reset_starts_a_new_session_without_old_cooldown() -> None:
    predictor = _SequencePredictor(
        [Prediction({"HELLO": 1.0}), Prediction({"HELLO": 1.0})]
    )
    recognizer = StreamingRecognizer(
        predictor,
        StreamingConfig(
            inference_stride=1,
            stable_windows=1,
            ema_alpha=1.0,
            cooldown_ms=10_000,
            minimum_frames=1,
            minimum_hand_rate=0.0,
            minimum_motion=0.0,
        ),
    )

    assert recognizer.update(_frame(0)).emitted_label == "HELLO"
    recognizer.reset()
    assert recognizer.update(_frame(0)).emitted_label == "HELLO"


def test_event_evaluation_is_one_to_one_and_counts_duplicates() -> None:
    annotations = (
        AnnotatedInterval("HELLO", 1_000, 1_500),
        AnnotatedInterval("THANKS", 4_000, 4_500),
    )
    events = (
        RecognitionEvent("HELLO", 1_200, 0.9),
        RecognitionEvent("HELLO", 1_300, 0.8),
        RecognitionEvent("THANKS", 5_200, 0.9),
        RecognitionEvent("OTHER", 7_000, 0.9),
    )

    metrics = evaluate_events(events, annotations, duration_ms=60_000, tolerance_ms=1_000)

    assert metrics.true_positives == 2
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 0
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0
    assert metrics.detection_precision == metrics.precision
    assert metrics.detection_recall == metrics.recall
    assert metrics.classification_accuracy == 1.0
    assert metrics.false_activations_per_minute == 2.0
    assert metrics.duplicate_rate == 0.25


def test_calibration_recorder_writes_only_relative_local_metadata(tmp_path: Path) -> None:
    recorder = CalibrationRecorder(tmp_path / "calibration")
    recorder.start("session-1", split="calibration", kind="transitions", started_at_ms=100)
    recorder.annotate("HELLO", start_ms=200, end_ms=500)

    path = recorder.finish(media_filename="session-1.mp4", ended_at_ms=1_000)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path / "calibration"
    assert payload["media_filename"] == "session-1.mp4"
    assert str(tmp_path) not in path.read_text(encoding="utf-8")
    assert "landmarks" not in payload
    assert "images" not in payload
    loaded = CalibrationSession.from_json(path)
    assert loaded.annotations == (AnnotatedInterval("HELLO", 200, 500),)


def test_calibration_recorder_rejects_overlap_and_path_escape(tmp_path: Path) -> None:
    recorder = CalibrationRecorder(tmp_path)
    recorder.start("valid", split="evaluation", kind="idle", started_at_ms=0)
    recorder.annotate("BACKGROUND", start_ms=10, end_ms=20)
    with pytest.raises(ValueError, match="overlap"):
        recorder.annotate("HELLO", start_ms=15, end_ms=30)
    with pytest.raises(ValueError, match="plain filename"):
        recorder.finish(media_filename="../escape.mp4", ended_at_ms=100)
