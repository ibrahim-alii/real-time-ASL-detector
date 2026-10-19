from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .contracts import TranslationRequest
from .youtube import file_hash


class TranslationFeatureCache:
    """Checksummed per-clip cache; a corrupt shard is safely recomputed on resume."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def store(
        self, clip_id: str, request: TranslationRequest, *, metadata: Mapping[str, object]
    ) -> Path:
        if metadata.get("feature_schema", request.feature_schema) != request.feature_schema:
            raise ValueError("cache metadata and request feature schemas differ")
        destination = self._path(clip_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            values=request.values,
            landmark_mask=request.landmark_mask,
            feature_schema=request.feature_schema,
            metadata=json.dumps(metadata, sort_keys=True),
        )
        temporary.replace(destination)
        self._digest_path(clip_id).write_text(file_hash(destination) + "\n", encoding="ascii")
        return destination

    def load(self, clip_id: str) -> tuple[TranslationRequest, dict[str, object]]:
        path = self._path(clip_id)
        expected = self._digest_path(clip_id).read_text(encoding="ascii").strip()
        if expected != file_hash(path):
            raise ValueError("feature shard checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            request = TranslationRequest(
                archive["values"].astype(np.float32), archive["landmark_mask"].astype(np.bool_),
                str(archive["feature_schema"].item()),
            )
            metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict):
            raise ValueError("feature metadata is invalid")
        if metadata.get("feature_schema", request.feature_schema) != request.feature_schema:
            raise ValueError("cached metadata and feature schemas differ")
        return request, metadata

    def find_valid(
        self, clip_id: str, *, expected_metadata: Mapping[str, object] | None = None
    ) -> tuple[TranslationRequest, dict[str, object]] | None:
        try:
            result = self.load(clip_id)
            if expected_metadata is not None and any(
                result[1].get(key) != value for key, value in expected_metadata.items()
            ):
                return None
            return result
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    def _path(self, clip_id: str) -> Path:
        digest = hashlib.sha256(clip_id.encode()).hexdigest()
        return self.root / f"{digest}.npz"

    def _digest_path(self, clip_id: str) -> Path:
        return self._path(clip_id).with_suffix(".npz.sha256")
