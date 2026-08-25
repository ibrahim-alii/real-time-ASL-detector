from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .schema import FEATURE_SCHEMA_VERSION, FeatureSequence


def cache_key(
    source: Path, extractor_config: Mapping[str, object], *, source_id: str | None = None
) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(FEATURE_SCHEMA_VERSION.encode())
    if source_id is not None:
        digest.update(source_id.encode())
    digest.update(json.dumps(extractor_config, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


class FeatureCache:
    def __init__(self, root: Path) -> None:
        self._root = root

    def store(
        self, key: str, sequence: FeatureSequence, *, metadata: Mapping[str, object]
    ) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        destination = self._path(key)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            values=sequence.values,
            landmark_mask=sequence.landmark_mask,
            temporal_mask=sequence.temporal_mask,
            timestamps_ms=sequence.timestamps_ms,
            schema_version=np.array(sequence.schema_version),
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        temporary.replace(destination)
        return destination

    def load(self, key: str) -> tuple[FeatureSequence, dict[str, Any]]:
        with np.load(self._path(key), allow_pickle=False) as archive:
            sequence = FeatureSequence(
                values=archive["values"].astype(np.float32, copy=False),
                landmark_mask=archive["landmark_mask"].astype(np.bool_, copy=False),
                temporal_mask=archive["temporal_mask"].astype(np.bool_, copy=False),
                timestamps_ms=archive["timestamps_ms"].astype(np.int64, copy=False),
                schema_version=str(archive["schema_version"].item()),
            )
            metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("feature cache metadata must be an object")
        return sequence, metadata

    def contains(self, key: str) -> bool:
        return self._path(key).is_file()

    def discard(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self._root / f"{key}.npz"
