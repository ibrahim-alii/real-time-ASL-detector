from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from asl_recognizer.streaming import RecognitionEvent

from .models import AnnotatedInterval


@dataclass(frozen=True, slots=True)
class EventMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    missed_sign_rate: float
    false_activations_per_minute: float
    duplicate_rate: float
    classification_accuracy: float

    @property
    def detection_precision(self) -> float:
        return self.precision

    @property
    def detection_recall(self) -> float:
        return self.recall


def evaluate_events(
    events: Sequence[RecognitionEvent],
    annotations: Sequence[AnnotatedInterval],
    *,
    duration_ms: int,
    tolerance_ms: int = 1_000,
) -> EventMetrics:
    if duration_ms <= 0:
        raise ValueError("evaluation duration must be positive")
    if tolerance_ms < 0:
        raise ValueError("event tolerance cannot be negative")
    ordered_events = sorted(events, key=lambda event: event.timestamp_ms)
    ordered_annotations = sorted(annotations, key=lambda item: item.start_ms)
    used_events: set[int] = set()
    matched_annotations: set[int] = set()

    for annotation_index, annotation in enumerate(ordered_annotations):
        for event_index, event in enumerate(ordered_events):
            if event_index in used_events:
                continue
            if event.timestamp_ms < annotation.start_ms:
                continue
            if event.timestamp_ms > annotation.end_ms + tolerance_ms:
                break
            if event.label == annotation.label:
                used_events.add(event_index)
                matched_annotations.add(annotation_index)
                break

    duplicates = 0
    for event_index, event in enumerate(ordered_events):
        if event_index in used_events:
            continue
        if any(
            event.label == annotation.label
            and annotation.start_ms <= event.timestamp_ms <= annotation.end_ms + tolerance_ms
            for annotation in ordered_annotations
        ):
            duplicates += 1

    true_positives = len(used_events)
    false_positives = len(ordered_events) - true_positives
    false_negatives = len(ordered_annotations) - len(matched_annotations)
    temporal_matches = _count_temporal_matches(ordered_events, ordered_annotations, tolerance_ms)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    return EventMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=true_positives / precision_denominator if precision_denominator else 0.0,
        recall=true_positives / recall_denominator if recall_denominator else 0.0,
        missed_sign_rate=false_negatives / recall_denominator if recall_denominator else 0.0,
        false_activations_per_minute=false_positives / (duration_ms / 60_000),
        duplicate_rate=duplicates / len(ordered_events) if ordered_events else 0.0,
        classification_accuracy=true_positives / temporal_matches if temporal_matches else 0.0,
    )


def _count_temporal_matches(
    events: Sequence[RecognitionEvent],
    annotations: Sequence[AnnotatedInterval],
    tolerance_ms: int,
) -> int:
    used: set[int] = set()
    for annotation in annotations:
        for event_index, event in enumerate(events):
            if event_index in used or event.timestamp_ms < annotation.start_ms:
                continue
            if event.timestamp_ms > annotation.end_ms + tolerance_ms:
                break
            used.add(event_index)
            break
    return len(used)
