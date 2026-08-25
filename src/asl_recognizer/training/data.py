from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from ..data.schema import SampleRecord
from ..features.cache import FeatureCache, cache_key
from ..features.schema import FeatureSequence


@dataclass(frozen=True, slots=True)
class SequenceExample:
    sequence: FeatureSequence
    class_id: int
    signer_id: str = ""

    def __post_init__(self) -> None:
        if self.class_id < 0:
            raise ValueError("class_id cannot be negative")


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    values: Tensor
    landmark_mask: Tensor
    temporal_mask: Tensor
    labels: Tensor

    def to(self, device: torch.device | str) -> SequenceBatch:
        return SequenceBatch(
            self.values.to(device),
            self.landmark_mask.to(device),
            self.temporal_mask.to(device),
            self.labels.to(device),
        )


def collate_examples(examples: list[SequenceExample]) -> SequenceBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    lengths = {example.sequence.values.shape[0] for example in examples}
    if len(lengths) != 1:
        raise ValueError("all sequences in a batch must have the same length")
    return SequenceBatch(
        values=torch.from_numpy(np.stack([example.sequence.values for example in examples])),
        landmark_mask=torch.from_numpy(
            np.stack([example.sequence.landmark_mask for example in examples])
        ),
        temporal_mask=torch.from_numpy(
            np.stack([example.sequence.temporal_mask for example in examples])
        ),
        labels=torch.tensor([example.class_id for example in examples], dtype=torch.long),
    )


def load_cached_examples(
    records: list[SampleRecord], cache_root: Path, cache_keys: dict[str, str]
) -> list[SequenceExample]:
    """Load extracted sequences in manifest order using explicit video-to-cache keys."""
    cache = FeatureCache(cache_root)
    examples: list[SequenceExample] = []
    for record in records:
        if record.class_id is None:
            raise ValueError(f"record {record.video_id} does not have a class_id")
        try:
            key = cache_keys[record.video_id]
        except KeyError as error:
            raise ValueError(f"no feature cache key for {record.video_id}") from error
        sequence, metadata = cache.load(key)
        if metadata.get("video_id") != record.video_id:
            raise ValueError(f"cache metadata does not match record {record.video_id}")
        examples.append(SequenceExample(sequence, record.class_id, record.signer_id))
    return examples


def discover_cache_index(
    cache_root: Path,
    video_ids: set[str] | None = None,
    video_sources: Mapping[str, Path] | None = None,
) -> tuple[dict[str, str], dict[str, object]]:
    """Select one complete preprocessing generation from a content-addressed cache."""
    cache = FeatureCache(cache_root)
    generations: dict[str, tuple[dict[str, object], dict[str, str]]] = {}
    for path in sorted(cache_root.glob("*.npz")):
        key = path.stem
        _sequence, metadata = cache.load(key)
        video_id = metadata.get("video_id")
        identity = metadata.get("preprocessing_identity")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"feature cache entry {key} has no video ID")
        if not isinstance(identity, dict):
            raise ValueError(f"feature cache entry {key} has no preprocessing identity")
        normalized_identity = dict(identity)
        identity_key = json.dumps(normalized_identity, sort_keys=True, separators=(",", ":"))
        _generation_identity, index = generations.setdefault(
            identity_key, (normalized_identity, {})
        )
        if video_id in index:
            if video_sources is None or video_id not in video_sources:
                raise ValueError(
                    f"duplicate feature cache entry for {video_id} and preprocessing identity"
                )
            expected_key = cache_key(
                video_sources[video_id], normalized_identity, source_id=video_id
            )
            if key == expected_key:
                index[video_id] = key
            elif index[video_id] != expected_key:
                raise ValueError(f"no cache entry matches the current source for {video_id}")
        else:
            index[video_id] = key
    if not generations:
        raise ValueError("feature cache contains no usable entries")
    required = video_ids or set()
    candidates = list(generations.values())
    if required:
        scored = [
            (len(required.intersection(index)), identity, index)
            for identity, index in candidates
        ]
        best_coverage = max(score for score, _identity, _index in scored)
        if best_coverage / len(required) < 0.95:
            raise ValueError("no preprocessing generation covers at least 95% of requested videos")
        candidates = [
            (identity, index)
            for score, identity, index in scored
            if score == best_coverage
        ]
    if len(candidates) > 1:
        raise ValueError("multiple preprocessing generations have equal best coverage")
    identity, index = candidates[0]
    selected = (
        {video_id: index[video_id] for video_id in required if video_id in index}
        if required
        else index
    )
    return selected, identity
