from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from .normalize import PREPROCESSING_VERSION, FramePreprocessor
from ..data.schema import SampleRecord
from .cache import FeatureCache, cache_key
from .schema import FeatureSequence, LandmarkFrame, pack_sequence


class LandmarkDetector(Protocol):
    def detect(self, frame: npt.NDArray[np.generic], timestamp_ms: int) -> LandmarkFrame: ...


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    requested_frames: int
    decoded_frames: int
    pose_anchor_rate: float
    hand_detection_rate: float
    left_hand_rate: float
    right_hand_rate: float
    elapsed_ms: float

    @property
    def decode_rate(self) -> float:
        return self.decoded_frames / self.requested_frames


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    sequence: FeatureSequence
    report: ExtractionReport


@dataclass(frozen=True, slots=True)
class DatasetExtractionSummary:
    attempted: int
    succeeded: int
    failed: int
    cache_hits: int
    failures: dict[str, str]
    success_rate: float
    frame_decode_rate: float
    pose_anchor_rate: float
    hand_detection_rate: float
    median_elapsed_ms: float
    p95_elapsed_ms: float
    per_gloss_hand_rate: dict[str, float]
    per_signer_hand_rate: dict[str, float]
    by_gloss: dict[str, GroupExtractionMetrics]
    by_signer: dict[str, GroupExtractionMetrics]
    by_split: dict[str, GroupExtractionMetrics]


@dataclass(frozen=True, slots=True)
class GroupExtractionMetrics:
    attempted: int
    succeeded: int
    failed: int
    frame_decode_rate: float
    pose_anchor_rate: float
    left_hand_rate: float
    right_hand_rate: float
    median_elapsed_ms: float
    p95_elapsed_ms: float


@dataclass(frozen=True, slots=True)
class VideoSample:
    frames: tuple[tuple[npt.NDArray[np.generic], int], ...]
    requested_frames: int


def extract_frames(
    frames: Iterable[tuple[npt.NDArray[np.generic], int]],
    detector: LandmarkDetector,
    *,
    sequence_length: int = 32,
    requested_frames: int | None = None,
    minimum_decode_rate: float = 0.95,
) -> ExtractionResult:
    started = time.perf_counter()
    preprocessor = FramePreprocessor()
    normalized: list[LandmarkFrame] = []
    anchors = 0
    hands = 0
    left_hands = 0
    right_hands = 0
    for image, timestamp_ms in frames:
        detected = detector.detect(image, timestamp_ms)
        anchors += int(detected.pose_anchor_present)
        hands += int(detected.hand_present)
        left_hands += int(detected.presence[:21].any())
        right_hands += int(detected.presence[21:42].any())
        normalized.append(preprocessor.process(detected))
    if not normalized:
        raise ValueError("video produced no decodable frames")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    count = len(normalized)
    requested = requested_frames if requested_frames is not None else count
    if requested < count or requested < 1:
        raise ValueError("requested frame count is invalid")
    if count / requested < minimum_decode_rate:
        raise ValueError(f"video decode rate {count / requested:.1%} is below minimum")
    return ExtractionResult(
        pack_sequence(normalized, length=sequence_length),
        ExtractionReport(
            requested,
            count,
            anchors / count,
            hands / count,
            left_hands / count,
            right_hands / count,
            elapsed_ms,
        ),
    )


FrameLoader = Callable[
    [Path, int], VideoSample
]


def extract_dataset(
    records: Sequence[SampleRecord],
    detector: LandmarkDetector,
    frame_loader: FrameLoader,
    cache: FeatureCache,
    *,
    sequence_length: int,
    extractor_identity: Mapping[str, object],
) -> DatasetExtractionSummary:
    failures: dict[str, str] = {}
    reports: list[tuple[SampleRecord, ExtractionReport]] = []
    cache_hits = 0
    config = dict(extractor_identity)
    config["sequence_length"] = sequence_length
    config["preprocessing_version"] = PREPROCESSING_VERSION
    config["minimum_decode_rate"] = 0.95
    for record in records:
        try:
            key = cache_key(record.path, config, source_id=record.video_id)
            if cache.contains(key):
                try:
                    _sequence, metadata = cache.load(key)
                    if metadata.get("video_id") != record.video_id:
                        raise ValueError("cached video ID does not match manifest")
                    cached_report = metadata.get("extraction_report")
                    if not isinstance(cached_report, dict):
                        raise ValueError("cached extraction report is missing")
                    report = ExtractionReport(**cached_report)
                except (KeyError, OSError, TypeError, ValueError):
                    cache.discard(key)
                else:
                    reports.append((record, report))
                    cache_hits += 1
                    continue
            sampled = frame_loader(record.path, sequence_length)
            reset = getattr(detector, "reset", None)
            if callable(reset):
                reset()
            result = extract_frames(
                sampled.frames,
                detector,
                sequence_length=sequence_length,
                requested_frames=sampled.requested_frames,
            )
            cache.store(
                key,
                result.sequence,
                metadata={
                    "video_id": record.video_id,
                    "gloss": record.gloss,
                    "class_id": record.class_id,
                    "signer_id": record.signer_id,
                    "split": record.split,
                    "cache_key": key,
                    "preprocessing_identity": config,
                    "extraction_report": {
                        "requested_frames": result.report.requested_frames,
                        "decoded_frames": result.report.decoded_frames,
                        "pose_anchor_rate": result.report.pose_anchor_rate,
                        "hand_detection_rate": result.report.hand_detection_rate,
                        "left_hand_rate": result.report.left_hand_rate,
                        "right_hand_rate": result.report.right_hand_rate,
                        "elapsed_ms": result.report.elapsed_ms,
                    },
                },
            )
            reports.append((record, result.report))
        except (OSError, RuntimeError, ValueError) as error:
            failures[record.video_id] = _failure_code(error)
    if reports:
        decoded_total = sum(report.decoded_frames for _, report in reports)
        requested_total = sum(report.requested_frames for _, report in reports)
        pose_rate = _weighted_rate(reports, "pose_anchor_rate", decoded_total)
        hand_rate = _weighted_rate(reports, "hand_detection_rate", decoded_total)
        elapsed = [report.elapsed_ms for _, report in reports if report.elapsed_ms > 0]
        median_ms = float(np.median(elapsed)) if elapsed else 0.0
        p95_ms = float(np.percentile(elapsed, 95)) if elapsed else 0.0
        frame_decode_rate = decoded_total / requested_total
        per_gloss = _grouped_hand_rates(reports, "gloss")
        per_signer = _grouped_hand_rates(reports, "signer_id")
    else:
        pose_rate = hand_rate = median_ms = p95_ms = frame_decode_rate = 0.0
        per_gloss = {}
        per_signer = {}
    attempted = len(records)
    return DatasetExtractionSummary(
        attempted=attempted,
        succeeded=len(reports),
        failed=len(failures),
        cache_hits=cache_hits,
        failures=failures,
        success_rate=len(reports) / attempted if attempted else 0.0,
        frame_decode_rate=frame_decode_rate,
        pose_anchor_rate=pose_rate,
        hand_detection_rate=hand_rate,
        median_elapsed_ms=median_ms,
        p95_elapsed_ms=p95_ms,
        per_gloss_hand_rate=per_gloss,
        per_signer_hand_rate=per_signer,
        by_gloss=_group_metrics(records, reports, "gloss"),
        by_signer=_group_metrics(records, reports, "signer_id"),
        by_split=_group_metrics(records, reports, "split"),
    )


def _weighted_rate(
    reports: Sequence[tuple[SampleRecord, ExtractionReport]],
    attribute: str,
    decoded_total: int,
) -> float:
    return float(
        sum(getattr(report, attribute) * report.decoded_frames for _, report in reports)
        / decoded_total
    )


def _grouped_hand_rates(
    reports: Sequence[tuple[SampleRecord, ExtractionReport]], attribute: str
) -> dict[str, float]:
    totals: dict[str, tuple[float, int]] = {}
    for record, report in reports:
        key = str(getattr(record, attribute))
        numerator, denominator = totals.get(key, (0.0, 0))
        totals[key] = (
            numerator + report.hand_detection_rate * report.decoded_frames,
            denominator + report.decoded_frames,
        )
    return {
        key: numerator / denominator
        for key, (numerator, denominator) in sorted(totals.items())
    }


def _failure_code(error: Exception) -> str:
    if isinstance(error, OSError):
        return "io_error"
    if isinstance(error, RuntimeError):
        return "detector_error"
    if "decode rate" in str(error):
        return "underdecoded_video"
    if "no decodable frames" in str(error):
        return "no_frames"
    return "invalid_video"


def _group_metrics(
    records: Sequence[SampleRecord],
    reports: Sequence[tuple[SampleRecord, ExtractionReport]],
    attribute: str,
) -> dict[str, GroupExtractionMetrics]:
    reports_by_id = {record.video_id: report for record, report in reports}
    grouped: dict[str, list[SampleRecord]] = {}
    for record in records:
        grouped.setdefault(str(getattr(record, attribute)), []).append(record)
    result: dict[str, GroupExtractionMetrics] = {}
    for key, group_records in sorted(grouped.items()):
        group_reports = [
            reports_by_id[record.video_id]
            for record in group_records
            if record.video_id in reports_by_id
        ]
        decoded = sum(report.decoded_frames for report in group_reports)
        requested = sum(report.requested_frames for report in group_reports)
        elapsed = [report.elapsed_ms for report in group_reports if report.elapsed_ms > 0]

        def weighted(name: str) -> float:
            if decoded == 0:
                return 0.0
            return float(
                sum(getattr(report, name) * report.decoded_frames for report in group_reports)
                / decoded
            )

        result[key] = GroupExtractionMetrics(
            attempted=len(group_records),
            succeeded=len(group_reports),
            failed=len(group_records) - len(group_reports),
            frame_decode_rate=decoded / requested if requested else 0.0,
            pose_anchor_rate=weighted("pose_anchor_rate"),
            left_hand_rate=weighted("left_hand_rate"),
            right_hand_rate=weighted("right_hand_rate"),
            median_elapsed_ms=float(np.median(elapsed)) if elapsed else 0.0,
            p95_elapsed_ms=float(np.percentile(elapsed, 95)) if elapsed else 0.0,
        )
    return result
