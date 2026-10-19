from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import TranslationRequest
from asl_recognizer.translation.youtube import file_hash


@pytest.mark.parametrize("hash_file", [file_hash])
@pytest.mark.parametrize(
    "payload", [b"", b"abc", b"abc" * (1024 * 1024)], ids=["empty", "small", "multi-chunk"]
)
def test_file_hash_compatibility(
    tmp_path: Path, hash_file: Callable[[Path], str], payload: bytes
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    assert hash_file(source) == hashlib.sha256(payload).hexdigest()
    source.unlink()
    with pytest.raises(FileNotFoundError):
        hash_file(source)


def test_translation_cache_preserves_checksums_and_recovers_corrupt_shards(tmp_path: Path) -> None:
    cache = TranslationFeatureCache(tmp_path)
    request = TranslationRequest(
        np.ones((2, 266), dtype=np.float32), np.ones((2, 133), dtype=np.bool_)
    )
    path = cache.store("clip", request, metadata={"source": "how2sign"})
    digest_path = path.with_suffix(".npz.sha256")
    assert digest_path.read_text(encoding="ascii").strip() == file_hash(path)
    loaded, metadata = cache.load("clip")
    np.testing.assert_array_equal(loaded.values, request.values)
    np.testing.assert_array_equal(loaded.landmark_mask, request.landmark_mask)
    assert loaded.feature_schema == request.feature_schema
    assert metadata == {"source": "how2sign"}
    assert cache.find_valid("clip", expected_metadata={"source": "changed"}) is None
    path.write_bytes(b"corrupt")
    assert cache.find_valid("clip") is None
    cache.store("clip", request, metadata=metadata)
    assert cache.find_valid("clip", expected_metadata=metadata) is not None
    digest_path.unlink()
    assert cache.find_valid("clip") is None


@pytest.mark.parametrize("schema", [None, "youtube-asl-holistic-85-v2"])
def test_translation_cache_rejects_retired_or_unidentified_features(
    tmp_path: Path, schema: str | None
) -> None:
    cache = TranslationFeatureCache(tmp_path)
    request = TranslationRequest(
        np.ones((2, 266), np.float32), np.ones((2, 133), np.bool_)
    )
    path = cache.store("clip", request, metadata={})
    arrays = dict(values=np.ones((2, 255), np.float32),
                  landmark_mask=np.ones((2, 85), np.bool_), metadata="{}")
    if schema is not None:
        arrays["feature_schema"] = schema
    np.savez_compressed(path, **arrays)
    path.with_suffix(".npz.sha256").write_text(file_hash(path), encoding="ascii")
    assert cache.find_valid("clip") is None
