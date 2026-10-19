from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from asl_recognizer.translation import manifest as manifest_module
from asl_recognizer.translation.manifest import (
    ClipManifest,
    ManifestRecord,
    build_manifest,
    read_manifest,
    write_manifest,
)


def test_manifest_fingerprint_is_computed_once_and_preserves_identity(tmp_path: Path) -> None:
    row = ManifestRecord("a", "how2sign", "train", "video", 0, 1000, "Hello")
    with patch.object(
        manifest_module.hashlib, "sha256", wraps=manifest_module.hashlib.sha256
    ) as sha:
        manifest = ClipManifest((row,))
        for _ in range(10):
            assert manifest.fingerprint == (
                "8cb69352e6127ba0ceb145e9eeb60f3d5d0138507b452d2f2e905bcd9455df77"
            )
        assert sha.call_count == 1
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    assert read_manifest(path) == manifest
    assert read_manifest(path).fingerprint == manifest.fingerprint
    path.write_text(path.read_text(encoding="utf-8").replace("Hello", "Changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(path)


def test_manifest_identity_tracks_records_and_is_order_independent() -> None:
    first = ManifestRecord("a", "how2sign", "train", "one", 0, 1000, "Hello")
    second = replace(first, clip_id="b", video_id="two")
    manifest = ClipManifest((first, second))
    assert ClipManifest((second, first)).fingerprint == manifest.fingerprint
    changed = replace(manifest, records=(replace(first, caption="Changed"), second))
    assert changed.fingerprint != manifest.fingerprint
    with pytest.raises(FrozenInstanceError):
        manifest.records = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="unique"):
        ClipManifest((first, first))


@pytest.mark.parametrize("start,end", [
    (-1, 1000), (0, -1), (1000, 500), (0, 0),
    (float("inf"), 1000), (0, float("inf")),
    (float("nan"), 1000), ("unknown", 1000), (None, 1000),
])
@pytest.mark.parametrize("unit", ["ms", "s"])
def test_bad_caption_times_are_audited_without_losing_valid_rows(
    start: object, end: object, unit: str,
) -> None:
    start_key, end_key = ("start_ms", "end_ms") if unit == "ms" else ("start", "end")
    rows = [
        {"id": "bad", "video_id": "one", "caption": "Bad", start_key: start, end_key: end},
        {"id": "good", "video_id": "two", "caption": "Good", "start": 0, "end": 1},
    ]
    manifest = build_manifest(rows, source="how2sign")
    bad, good = manifest.records
    assert bad.status == "malformed"
    assert bad.reason in {"invalid-caption-boundary", "non-positive-duration"}
    assert good.status == "accepted"
    assert manifest.accepted() == (good,)


def test_direct_manifest_records_still_reject_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="boundaries"):
        ManifestRecord("a", "how2sign", "train", "video", -1, 1000, "Hello")
