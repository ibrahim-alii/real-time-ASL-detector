from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np

from .contracts import TranslationRequest
from .youtube import file_hash


def _cache_version(path: Path) -> tuple[int, ...] | None:
    if sys.platform != "linux":
        return None
    value = path.stat()
    # BeeGFS timestamps can be coarse; recent writes must pass full integrity checks.
    if time.time_ns() - value.st_ctime_ns < 2_000_000_000:
        return None
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class TranslationFeatureCache:
    """Checksummed per-clip cache; a corrupt shard is safely recomputed on resume."""

    def __init__(self, root: Path, *, max_memory_bytes: int = 64 * 1024**2) -> None:
        if max_memory_bytes < 0:
            raise ValueError("feature memory budget cannot be negative")
        self.root = root
        self.max_memory_bytes = max_memory_bytes
        self.memory_bytes = 0
        self._memory: OrderedDict[str, tuple[
            tuple[tuple[int, ...], tuple[int, ...]], TranslationRequest, dict[str, object], int
        ]] = OrderedDict()

    def _forget(self, clip_id: str) -> None:
        previous = self._memory.pop(clip_id, None)
        if previous is not None:
            self.memory_bytes -= previous[3]

    def store(
        self, clip_id: str, request: TranslationRequest, *, metadata: Mapping[str, object]
    ) -> Path:
        if metadata.get("feature_schema", request.feature_schema) != request.feature_schema:
            raise ValueError("cache metadata and request feature schemas differ")
        self._forget(clip_id)
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
        digest_path = self._digest_path(clip_id)
        versions = (_cache_version(path), _cache_version(digest_path))
        cached = self._memory.get(clip_id)
        if cached is not None and cached[0] == versions:
            self._memory.move_to_end(clip_id)
            request = cached[1]
            return (TranslationRequest(request.values.copy(), request.landmark_mask.copy(),
                                       request.feature_schema), deepcopy(cached[2]))
        self._forget(clip_id)
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
        payload_bytes = (request.values.nbytes + request.landmark_mask.nbytes
                         + len(json.dumps(metadata).encode()))
        first, second = versions
        if (first is not None and second is not None and payload_bytes <= self.max_memory_bytes
                and versions == (_cache_version(path), _cache_version(digest_path))):
            while self._memory and self.memory_bytes + payload_bytes > self.max_memory_bytes:
                self._forget(next(iter(self._memory)))
            snapshot = TranslationRequest(request.values.copy(), request.landmark_mask.copy(),
                                          request.feature_schema)
            self._memory[clip_id] = ((first, second), snapshot, deepcopy(metadata), payload_bytes)
            self.memory_bytes += payload_bytes
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
