from __future__ import annotations

import csv
from pathlib import Path

import pytest

from asl_recognizer.translation.data_audit import audit_manifest, read_signer_map
from asl_recognizer.translation.how2sign import build_how2sign_manifest
from asl_recognizer.translation.manifest import build_manifest


def test_realigned_how2sign_uses_recording_names_and_absolute_boundaries(tmp_path: Path) -> None:
    paths = {}
    for split in ("train", "dev", "test"):
        path = tmp_path / f"{split}.tsv"
        row = {
            "VIDEO_ID": split, "VIDEO_NAME": f"{split}_rgb_front",
            "SENTENCE_NAME": f"{split}_sentence", "SENTENCE": "I want water.",
            "START_REALIGNED": "12.5", "END_REALIGNED": "16.0",
            "START": "10", "END": "14",
        }
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
            writer.writeheader()
            writer.writerow(row)
        paths[split] = path
    manifest = build_how2sign_manifest(
        **paths, alignment="realigned", video_layout="recordings"
    )
    assert manifest.accepted("train")[0].video_id == "train_rgb_front"
    assert manifest.accepted("train")[0].start_ms == 12_500
    assert manifest.accepted("train")[0].end_ms == 16_000
    with pytest.raises(ValueError, match="full recordings"):
        build_how2sign_manifest(**paths, alignment="realigned")


def test_audit_reports_source_caption_and_signer_overlap() -> None:
    manifest = build_manifest([
        {"id": "a", "video_id": "same", "caption": "Hello  there", "split": "train",
         "start": 0, "end": 1},
        {"id": "b", "video_id": "same", "caption": "hello there", "split": "dev",
         "start": 2, "end": 3},
        {"id": "c", "video_id": "other", "caption": "Goodbye", "split": "test",
         "start": 0, "end": 1},
    ], source="how2sign")
    report = audit_manifest(manifest, signer_map={"same": "signer-1", "other": "signer-1"})
    assert report["cross_split_sources"] == {"how2sign:same": ["dev", "train"]}
    assert report["cross_split_captions"] == {"hello there": ["dev", "train"]}
    assert report["cross_split_signers"] == {"signer-1": ["dev", "test", "train"]}
    assert report["signer_metadata_complete"] is True
    assert report["signer_independence_verified"] is False
    assert report["source_split_isolation_passed"] is False
    assert audit_manifest(manifest)["signer_independence_verified"] is False


def test_audit_verifies_known_distinct_signers_and_rejects_conflicting_metadata(
    tmp_path: Path,
) -> None:
    manifest = build_manifest([
        {"id": split, "video_id": split, "caption": split, "split": split,
         "start": 0, "end": 1} for split in ("train", "dev", "test")
    ], source="how2sign")
    report = audit_manifest(
        manifest, signer_map={split: split for split in ("train", "dev", "test")}
    )
    assert report["signer_independence_verified"] is True
    assert report["source_split_isolation_passed"] is True
    train_only = build_manifest([
        {"id": "train", "video_id": "train", "caption": "hello", "split": "train",
         "start": 0, "end": 1}
    ], source="how2sign")
    assert audit_manifest(train_only, signer_map={"train": "one"})[
        "signer_independence_verified"
    ] is False
    path = tmp_path / "signers.csv"
    path.write_text("video_id,signer_id\na,one\na,two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting"):
        read_signer_map(path)
