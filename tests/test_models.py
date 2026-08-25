from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from asl_recognizer.data.schema import SampleRecord
from asl_recognizer.features.cache import FeatureCache, cache_key
from asl_recognizer.features.schema import FEATURE_DIMENSION, LANDMARK_COUNT, FeatureSequence
from asl_recognizer.models.baselines import LogisticBaseline, MajorityBaseline
from asl_recognizer.models.checkpoint import CheckpointCompatibilityError, load_bundle, save_bundle
from asl_recognizer.models.gru import GRUConfig, LandmarkGRU
from asl_recognizer.training.data import (
    SequenceExample,
    collate_examples,
    discover_cache_index,
    load_cached_examples,
)
from asl_recognizer.training.engine import TrainingConfig, evaluate, train_model
from asl_recognizer.training.features import aggregate_sequence


def _sequence(value: float, *, valid_steps: int = 3, length: int = 4) -> FeatureSequence:
    values = np.zeros((length, FEATURE_DIMENSION), dtype=np.float32)
    landmark_mask = np.zeros((length, LANDMARK_COUNT), dtype=np.bool_)
    temporal_mask = np.zeros(length, dtype=np.bool_)
    values[:valid_steps] = value
    landmark_mask[:valid_steps] = True
    temporal_mask[:valid_steps] = True
    return FeatureSequence(
        values=values,
        landmark_mask=landmark_mask,
        temporal_mask=temporal_mask,
        timestamps_ms=np.arange(length, dtype=np.int64),
    )


def test_aggregate_sequence_ignores_padding_and_missing_landmarks() -> None:
    sequence = _sequence(2.0, valid_steps=2)
    sequence.landmark_mask[1, 0] = False
    sequence.values[1, :3] = 0

    aggregated = aggregate_sequence(sequence)

    assert aggregated.shape == (FEATURE_DIMENSION * 2,)
    np.testing.assert_allclose(aggregated[:3], np.full(3, 2.0, dtype=np.float32))
    np.testing.assert_allclose(
        aggregated[FEATURE_DIMENSION : FEATURE_DIMENSION + 3],
        np.zeros(3, dtype=np.float32),
    )
    assert np.isfinite(aggregated).all()


def test_baselines_fit_and_predict() -> None:
    sequences = [_sequence(-2.0), _sequence(-1.0), _sequence(1.0), _sequence(2.0)]
    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    majority = MajorityBaseline().fit(labels)
    logistic = LogisticBaseline(random_state=7).fit(sequences, labels)

    assert majority.predict(3).tolist() == [0, 0, 0]
    assert logistic.predict(sequences).tolist() == labels.tolist()


def test_collation_and_gru_respect_temporal_masks() -> None:
    batch = collate_examples(
        [SequenceExample(_sequence(1.0, valid_steps=2), 0), SequenceExample(_sequence(2.0), 1)]
    )
    model = LandmarkGRU(GRUConfig(num_classes=2, hidden_size=8, num_layers=1, dropout=0.0))

    logits = model(batch.values, batch.landmark_mask, batch.temporal_mask)

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    with pytest.raises(ValueError, match="at least one valid timestep"):
        model(batch.values, batch.landmark_mask, torch.zeros_like(batch.temporal_mask))


def test_load_cached_examples_preserves_manifest_identity(tmp_path: Path) -> None:
    key = "a" * 64
    FeatureCache(tmp_path).store(key, _sequence(1.0), metadata={"video_id": "clip"})
    record = SampleRecord("clip", tmp_path / "clip.mp4", "A", "signer", "train", 3)

    examples = load_cached_examples([record], tmp_path, {"clip": key})

    assert len(examples) == 1
    assert examples[0].class_id == 3
    assert examples[0].signer_id == "signer"
    with pytest.raises(ValueError, match="no feature cache key"):
        load_cached_examples([record], tmp_path, {})


def test_discover_cache_index_requires_one_preprocessing_identity(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    cache.store(
        "a" * 64,
        _sequence(1.0),
        metadata={"video_id": "one", "preprocessing_identity": {"schema": "v1"}},
    )
    cache.store(
        "b" * 64,
        _sequence(2.0),
        metadata={"video_id": "two", "preprocessing_identity": {"schema": "v1"}},
    )

    index, identity = discover_cache_index(tmp_path)

    assert index == {"one": "a" * 64, "two": "b" * 64}
    assert identity == {"schema": "v1"}


def test_discover_cache_index_ignores_incomplete_stale_generation(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    cache.store(
        "a" * 64,
        _sequence(1.0),
        metadata={"video_id": "one", "preprocessing_identity": {"schema": "old"}},
    )
    cache.store(
        "b" * 64,
        _sequence(2.0),
        metadata={"video_id": "one", "preprocessing_identity": {"schema": "current"}},
    )
    cache.store(
        "c" * 64,
        _sequence(3.0),
        metadata={"video_id": "two", "preprocessing_identity": {"schema": "current"}},
    )

    index, identity = discover_cache_index(tmp_path, {"one", "two"})

    assert index == {"one": "b" * 64, "two": "c" * 64}
    assert identity == {"schema": "current"}


def test_discover_cache_index_rejects_ambiguous_complete_generations(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path)
    for key, schema in (("a", "v1"), ("b", "v2")):
        cache.store(
            key * 64,
            _sequence(1.0),
            metadata={"video_id": "one", "preprocessing_identity": {"schema": schema}},
        )

    with pytest.raises(ValueError, match="multiple preprocessing generations"):
        discover_cache_index(tmp_path, {"one"})


def test_discover_cache_index_selects_current_source_after_replacement(tmp_path: Path) -> None:
    cache = FeatureCache(tmp_path / "cache")
    source = tmp_path / "clip.mp4"
    identity = {"schema": "v1"}
    source.write_bytes(b"old")
    old_key = cache_key(source, identity, source_id="clip")
    cache.store(
        old_key,
        _sequence(1.0),
        metadata={"video_id": "clip", "preprocessing_identity": identity},
    )
    source.write_bytes(b"current")
    current_key = cache_key(source, identity, source_id="clip")
    cache.store(
        current_key,
        _sequence(2.0),
        metadata={"video_id": "clip", "preprocessing_identity": identity},
    )

    index, _ = discover_cache_index(
        tmp_path / "cache", {"clip"}, {"clip": source}
    )

    assert index == {"clip": current_key}


def test_training_is_finite_and_learns_tiny_dataset() -> None:
    examples = [
        SequenceExample(_sequence(-2.0), 0),
        SequenceExample(_sequence(-1.0), 0),
        SequenceExample(_sequence(1.0), 1),
        SequenceExample(_sequence(2.0), 1),
    ]
    model = LandmarkGRU(GRUConfig(num_classes=2, hidden_size=12, dropout=0.0))

    history = train_model(
        model,
        examples,
        examples,
        TrainingConfig(epochs=30, batch_size=4, learning_rate=0.02, patience=30, seed=11),
    )
    result = evaluate(model, examples, batch_size=4)

    assert all(np.isfinite(history.train_loss))
    assert history.best_epoch >= 0
    assert max(history.validation_macro_f1) == 1.0
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0


def test_safe_checkpoint_round_trip_and_rejects_incompatibility(tmp_path: Path) -> None:
    config = GRUConfig(num_classes=2, hidden_size=8, dropout=0.0)
    model = LandmarkGRU(config)
    destination = tmp_path / "model.pt"
    save_bundle(
        destination,
        model,
        labels=("A", "B"),
        preprocessing_hash="preprocess-123",
        dataset_hash="dataset-123",
    )

    loaded, metadata = load_bundle(
        destination,
        expected_labels=("A", "B"),
        expected_preprocessing_hash="preprocess-123",
    )

    assert metadata.labels == ("A", "B")
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)

    with pytest.raises(CheckpointCompatibilityError, match="labels"):
        load_bundle(destination, expected_labels=("B", "A"))
    with pytest.raises(CheckpointCompatibilityError, match="preprocessing"):
        load_bundle(destination, expected_preprocessing_hash="other")
    with pytest.raises(CheckpointCompatibilityError, match="dataset"):
        load_bundle(destination, expected_dataset_hash="other")

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch bundle")
    with pytest.raises(CheckpointCompatibilityError, match="safe model bundle"):
        load_bundle(corrupt)
