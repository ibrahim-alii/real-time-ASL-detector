from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from asl_recognizer.features.cache import FeatureCache, cache_key
from asl_recognizer.data.schema import SampleRecord
from asl_recognizer.features.extract import VideoSample, extract_dataset, extract_frames
from asl_recognizer.features.normalize import CausalShoulderNormalizer, FramePreprocessor
from asl_recognizer.features.schema import (
    LANDMARK_COUNT,
    FeatureSequence,
    LandmarkFrame,
    pack_sequence,
)


def _frame(offset: float = 0.0, scale: float = 1.0) -> LandmarkFrame:
    coordinates = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    presence = np.zeros(LANDMARK_COUNT, dtype=np.bool_)
    coordinates[43] = [offset + 0.4 * scale, offset + 0.5 * scale, 0.0]
    coordinates[44] = [offset + 0.6 * scale, offset + 0.5 * scale, 0.0]
    coordinates[0] = [offset + 0.5 * scale, offset + 0.4 * scale, 0.1 * scale]
    presence[[0, 43, 44]] = True
    return LandmarkFrame(coordinates, presence, timestamp_ms=0)


def test_normalization_is_translation_and_scale_invariant() -> None:
    first = CausalShoulderNormalizer().normalize(_frame())
    transformed = CausalShoulderNormalizer().normalize(_frame(offset=2.0, scale=3.0))

    np.testing.assert_allclose(first.coordinates, transformed.coordinates, atol=1e-5)
    assert np.all(first.coordinates[~first.presence] == 0)


def test_normalization_reuses_only_past_valid_transform() -> None:
    normalizer = CausalShoulderNormalizer()
    valid = normalizer.normalize(_frame())
    missing = _frame()
    missing.presence[[43, 44]] = False
    missing.coordinates[0] = [0.5, 0.4, 0.1]

    normalized_missing = normalizer.normalize(missing)

    np.testing.assert_allclose(normalized_missing.coordinates[0], valid.coordinates[0])
    assert not normalized_missing.pose_anchor_present


def test_pack_sequence_pads_short_inputs_and_masks_padding() -> None:
    frames = [CausalShoulderNormalizer().normalize(_frame()) for _ in range(3)]

    sequence = pack_sequence(frames, length=5)

    assert sequence.values.shape == (5, LANDMARK_COUNT * 3)
    assert sequence.landmark_mask.shape == (5, LANDMARK_COUNT)
    assert sequence.temporal_mask.tolist() == [True, True, True, False, False]
    assert np.all(sequence.values[3:] == 0)


def test_pack_sequence_uniformly_downsamples_long_inputs() -> None:
    frames = []
    for index in range(5):
        frame = _frame()
        frame.coordinates[0, 0] = float(index)
        frames.append(frame)

    sequence = pack_sequence(frames, length=3)

    assert sequence.timestamps_ms.tolist() == [0, 0, 0]
    assert sequence.temporal_mask.all()
    assert sequence.values[0, 0] != sequence.values[-1, 0]


def test_feature_sequence_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        FeatureSequence(
            values=np.full((1, LANDMARK_COUNT * 3), np.nan, dtype=np.float32),
            landmark_mask=np.zeros((1, LANDMARK_COUNT), dtype=np.bool_),
            temporal_mask=np.ones(1, dtype=np.bool_),
            timestamps_ms=np.zeros(1, dtype=np.int64),
        )


def test_feature_sequence_rejects_masked_nonzero_values() -> None:
    values = np.zeros((1, LANDMARK_COUNT * 3), dtype=np.float32)
    values[0, 0] = 1.0
    with pytest.raises(ValueError, match="masked landmarks"):
        FeatureSequence(
            values=values,
            landmark_mask=np.zeros((1, LANDMARK_COUNT), dtype=np.bool_),
            temporal_mask=np.ones(1, dtype=np.bool_),
            timestamps_ms=np.zeros(1, dtype=np.int64),
        )


def test_feature_sequence_rejects_landmarks_on_padding() -> None:
    with pytest.raises(ValueError, match="padded timesteps"):
        FeatureSequence(
            values=np.zeros((1, LANDMARK_COUNT * 3), dtype=np.float32),
            landmark_mask=np.ones((1, LANDMARK_COUNT), dtype=np.bool_),
            temporal_mask=np.zeros(1, dtype=np.bool_),
            timestamps_ms=np.zeros(1, dtype=np.int64),
        )


def test_feature_cache_round_trip_and_config_invalidation(tmp_path: Path) -> None:
    sequence = pack_sequence([_frame()], length=2)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    cache = FeatureCache(tmp_path / "cache")
    key = cache_key(source, {"sequence_length": 2})

    cache.store(key, sequence, metadata={"video_id": "clip"})

    loaded, metadata = cache.load(key)
    np.testing.assert_array_equal(loaded.values, sequence.values)
    assert metadata["video_id"] == "clip"
    assert cache_key(source, {"sequence_length": 3}) != key


class _FakeDetector:
    def detect(self, _image: np.ndarray, timestamp_ms: int) -> LandmarkFrame:
        frame = _frame()
        return LandmarkFrame(frame.coordinates, frame.presence, timestamp_ms)


class _ResettableDetector(_FakeDetector):
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


def test_extract_frames_reports_detection_coverage() -> None:
    frames = [(np.zeros((4, 4, 3), dtype=np.uint8), 0), (np.zeros((4, 4, 3)), 40)]

    result = extract_frames(frames, _FakeDetector(), sequence_length=3)

    assert result.sequence.values.shape == (3, LANDMARK_COUNT * 3)
    assert result.report.decoded_frames == 2
    assert result.report.pose_anchor_rate == 1.0
    assert result.report.hand_detection_rate > 0.0


def test_extract_frames_rejects_underdecoded_video() -> None:
    frames = [(np.zeros((4, 4, 3), dtype=np.uint8), 0)]

    with pytest.raises(ValueError, match="decode rate"):
        extract_frames(frames, _FakeDetector(), sequence_length=3, requested_frames=3)


def test_shared_frame_preprocessor_matches_batch_path() -> None:
    frames = [_frame(), _frame()]
    frames[1].presence[[43, 44]] = False
    direct = FramePreprocessor()
    expected = [direct.process(frame) for frame in frames]

    actual = extract_frames(
        [(np.zeros((2, 2, 3), dtype=np.uint8), index) for index in range(2)],
        type(
            "Detector",
            (),
            {"detect": lambda _self, _image, timestamp: frames[timestamp]},
        )(),
        sequence_length=2,
    ).sequence

    np.testing.assert_allclose(actual.values, pack_sequence(expected, length=2).values)


def test_extract_dataset_caches_successes_and_reports_failures(tmp_path: Path) -> None:
    good = tmp_path / "good.mp4"
    bad = tmp_path / "bad.mp4"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")
    records = [
        SampleRecord("good", good, "A", "p1", "train", 0),
        SampleRecord("bad", bad, "A", "p1", "train", 0),
    ]

    loader_calls = 0

    def loader(path: Path, _count: int) -> VideoSample:
        nonlocal loader_calls
        loader_calls += 1
        if path == bad:
            raise ValueError("corrupt video")
        return VideoSample(((np.zeros((4, 4, 3), dtype=np.uint8), 0),), 1)

    detector = _ResettableDetector()
    summary = extract_dataset(
        records,
        detector,
        loader,
        FeatureCache(tmp_path / "cache"),
        sequence_length=2,
        extractor_identity={"model_sha256": "abc"},
    )

    assert summary.succeeded == 1
    assert summary.failed == 1
    assert summary.failures == {"bad": "invalid_video"}
    assert summary.success_rate == 0.5
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1

    cached = extract_dataset(
        records[:1],
        _FakeDetector(),
        loader,
        FeatureCache(tmp_path / "cache"),
        sequence_length=2,
        extractor_identity={"model_sha256": "abc"},
    )
    assert cached.cache_hits == 1
    assert loader_calls == 2


def test_identical_video_bytes_do_not_collide_between_records(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    records = [
        SampleRecord("first", first, "A", "p1", "train", 0),
        SampleRecord("second", second, "A", "p1", "train", 0),
    ]
    calls = 0

    def loader(_path: Path, _count: int) -> VideoSample:
        nonlocal calls
        calls += 1
        return VideoSample(((np.zeros((2, 2, 3), dtype=np.uint8), 0),), 1)

    detector = _ResettableDetector()
    summary = extract_dataset(
        records,
        detector,
        loader,
        FeatureCache(tmp_path / "cache"),
        sequence_length=1,
        extractor_identity={"model": "x"},
    )

    assert summary.succeeded == 2
    assert calls == 2
    assert detector.reset_calls == 2
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 2


def test_corrupt_cache_is_recomputed(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    record = SampleRecord("clip", source, "A", "p1", "train", 0)
    cache = FeatureCache(tmp_path / "cache")
    identity = {"model": "x"}
    key = cache_key(
        source,
        {
            **identity,
            "sequence_length": 1,
            "preprocessing_version": "shoulder-causal-v1",
            "minimum_decode_rate": 0.95,
        },
        source_id="clip",
    )
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / f"{key}.npz").write_bytes(b"not an archive")
    calls = 0

    def loader(_path: Path, _count: int) -> VideoSample:
        nonlocal calls
        calls += 1
        return VideoSample(((np.zeros((2, 2, 3), dtype=np.uint8), 0),), 1)

    summary = extract_dataset(
        [record],
        _FakeDetector(),
        loader,
        cache,
        sequence_length=1,
        extractor_identity=identity,
    )

    assert summary.succeeded == 1
    assert summary.cache_hits == 0
    assert calls == 1
