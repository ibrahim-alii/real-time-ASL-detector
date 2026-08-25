from __future__ import annotations

import csv
from pathlib import Path

import pytest

from asl_recognizer.data.index import (
    assign_class_ids,
    build_manifest,
    manifest_fingerprint,
    read_manifest,
    read_vocabulary,
    select_vocabulary,
    write_manifest,
    write_vocabulary,
)
from asl_recognizer.data.schema import SampleRecord, validate_manifest


def _write_split(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Participant ID", "Video file", "Gloss", "ASL-LEX Code"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_manifest_accepts_official_column_names(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    (video_root / "001-HELLO.mp4").touch()
    split_csv = tmp_path / "train.csv"
    _write_split(
        split_csv,
        [{"Participant ID": "p1", "Video file": "001-HELLO.mp4", "Gloss": "HELLO"}],
    )

    records = build_manifest(
        {"train": split_csv},
        video_root,
        require_files=True,
        require_all_splits=False,
        expected_signer_counts=None,
    )

    assert records == [
        SampleRecord("001-HELLO", video_root / "001-HELLO.mp4", "HELLO", "p1", "train")
    ]


def test_build_manifest_reports_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("Video file,Gloss\nclip.mp4,HELLO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Participant ID"):
        build_manifest(
            {"train": path},
            tmp_path,
            require_files=False,
            require_all_splits=False,
            expected_signer_counts=None,
        )


def test_validate_manifest_rejects_signer_leakage() -> None:
    records = [
        SampleRecord("a", Path("a.mp4"), "HELLO", "same", "train"),
        SampleRecord("b", Path("b.mp4"), "HELLO", "same", "test"),
    ]

    with pytest.raises(ValueError, match="signer leakage"):
        validate_manifest(records)


def test_select_vocabulary_is_deterministic_and_requires_all_splits() -> None:
    records: list[SampleRecord] = []
    for gloss, train_signers in (("ABLE", ["a", "b"]), ("BEFORE", ["c", "d", "e"])):
        for signer in train_signers:
            records.append(SampleRecord(f"{gloss}-{signer}", Path("x"), gloss, signer, "train"))
        records.append(SampleRecord(f"{gloss}-v", Path("x"), gloss, f"{gloss}-v", "val"))
        records.append(SampleRecord(f"{gloss}-t", Path("x"), gloss, f"{gloss}-t", "test"))
    records.append(SampleRecord("train-only", Path("x"), "EXCLUDED", "z", "train"))

    assert select_vocabulary(records, size=2) == ["BEFORE", "ABLE"]


def test_validate_manifest_rejects_duplicate_video_ids() -> None:
    records = [
        SampleRecord("same", Path("a.mp4"), "A", "p1", "train"),
        SampleRecord("same", Path("b.mp4"), "A", "p2", "val"),
    ]

    with pytest.raises(ValueError, match="duplicate video"):
        validate_manifest(records)


def test_build_manifest_requires_all_official_splits(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    _write_split(path, [])

    with pytest.raises(ValueError, match="exactly train, val, and test"):
        build_manifest({"train": path}, tmp_path, require_files=False)


@pytest.mark.parametrize("filename", ["../outside.mp4", "C:/outside.mp4"])
def test_build_manifest_rejects_paths_outside_video_root(tmp_path: Path, filename: str) -> None:
    path = tmp_path / "train.csv"
    _write_split(
        path,
        [{"Participant ID": "p1", "Video file": filename, "Gloss": "HELLO"}],
    )

    with pytest.raises(ValueError, match="outside video root"):
        build_manifest(
            {"train": path},
            tmp_path,
            require_files=False,
            require_all_splits=False,
            expected_signer_counts=None,
        )


def test_manifest_fingerprint_does_not_depend_on_dataset_location() -> None:
    first = SampleRecord("id", Path("C:/one/clip.mp4"), "A", "p1", "train", 0)
    second = SampleRecord("id", Path("D:/two/clip.mp4"), "A", "p1", "train", 0)

    assert manifest_fingerprint([first]) == manifest_fingerprint([second])


def test_manifest_and_vocabulary_round_trip_detects_tampering(tmp_path: Path) -> None:
    raw = [
        SampleRecord("a", tmp_path / "a.mp4", "A", "p1", "train"),
        SampleRecord("b", tmp_path / "b.mp4", "A", "p2", "val"),
        SampleRecord("c", tmp_path / "c.mp4", "A", "p3", "test"),
    ]
    indexed = assign_class_ids(raw, ["A"])
    for record in indexed:
        record.path.touch()
    manifest_path = tmp_path / "manifest.csv"
    vocabulary_path = tmp_path / "vocabulary.json"
    source_hashes = {"train.csv": "a" * 64, "val.csv": "b" * 64, "test.csv": "c" * 64}
    fingerprint = manifest_fingerprint(indexed)
    write_manifest(indexed, manifest_path, video_root=tmp_path, source_hashes=source_hashes)
    write_vocabulary(
        ["A"],
        vocabulary_path,
        dataset_version="1.0",
        manifest_fingerprint_value=fingerprint,
        source_hashes=source_hashes,
    )

    vocabulary = read_vocabulary(vocabulary_path)
    loaded = read_manifest(manifest_path, tmp_path, vocabulary)

    assert loaded == indexed
    assert loaded[0].class_id == 0
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(",A,", ",B,"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mismatch"):
        read_manifest(manifest_path, tmp_path, vocabulary)


def test_build_manifest_reports_malformed_short_row(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    path.write_text("Participant ID,Video file,Gloss\np1,clip.mp4\n", encoding="utf-8")

    with pytest.raises(ValueError, match="blank required value"):
        build_manifest(
            {"train": path},
            tmp_path,
            require_files=False,
            require_all_splits=False,
            expected_signer_counts=None,
        )


def test_nested_video_path_round_trips(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "clip.mp4"
    nested.parent.mkdir()
    nested.touch()
    records = [
        SampleRecord("clip", nested, "A", "p1", "train", 0),
        SampleRecord("v", tmp_path / "v.mp4", "A", "p2", "val", 0),
        SampleRecord("t", tmp_path / "t.mp4", "A", "p3", "test", 0),
    ]
    records[1].path.touch()
    records[2].path.touch()
    source_hashes = {"train.csv": "a", "val.csv": "b", "test.csv": "c"}
    manifest_path = tmp_path / "manifest.csv"
    fingerprint = manifest_fingerprint(records)
    write_manifest(records, manifest_path, video_root=tmp_path, source_hashes=source_hashes)
    vocabulary_path = tmp_path / "vocabulary.json"
    write_vocabulary(
        ["A"],
        vocabulary_path,
        dataset_version="1.0",
        manifest_fingerprint_value=fingerprint,
        source_hashes=source_hashes,
    )

    loaded = read_manifest(manifest_path, tmp_path, read_vocabulary(vocabulary_path))

    assert loaded[0].path == nested


def test_read_manifest_requires_video_files_by_default(tmp_path: Path) -> None:
    records = [
        SampleRecord("a", tmp_path / "a.mp4", "A", "p1", "train", 0),
        SampleRecord("b", tmp_path / "b.mp4", "A", "p2", "val", 0),
        SampleRecord("c", tmp_path / "c.mp4", "A", "p3", "test", 0),
    ]
    source_hashes = {"train.csv": "a", "val.csv": "b", "test.csv": "c"}
    path = tmp_path / "manifest.csv"
    write_manifest(records, path, video_root=tmp_path, source_hashes=source_hashes)
    vocabulary_path = tmp_path / "vocabulary.json"
    write_vocabulary(
        ["A"],
        vocabulary_path,
        dataset_version="1.0",
        manifest_fingerprint_value=manifest_fingerprint(records),
        source_hashes=source_hashes,
    )

    with pytest.raises(FileNotFoundError, match="video not found"):
        read_manifest(path, tmp_path, read_vocabulary(vocabulary_path))


def test_build_manifest_enforces_expected_signer_counts(tmp_path: Path) -> None:
    split_files: dict[str, Path] = {}
    for split, signer in (("train", "p1"), ("val", "p2"), ("test", "p3")):
        path = tmp_path / f"{split}.csv"
        _write_split(
            path,
            [{"Participant ID": signer, "Video file": f"{split}.mp4", "Gloss": "A"}],
        )
        split_files[split] = path

    records = build_manifest(
        split_files,
        tmp_path,
        require_files=False,
        expected_signer_counts={"train": 1, "val": 1, "test": 1},
    )
    assert len(records) == 3
    with pytest.raises(ValueError, match="expected official"):
        build_manifest(
            split_files,
            tmp_path,
            require_files=False,
            expected_signer_counts={"train": 2, "val": 1, "test": 1},
        )


def test_write_manifest_rejects_inconsistent_class_mapping(tmp_path: Path) -> None:
    records = [
        SampleRecord("a", tmp_path / "a.mp4", "A", "p1", "train", 0),
        SampleRecord("b", tmp_path / "b.mp4", "A", "p2", "val", 1),
    ]

    with pytest.raises(ValueError, match="one-to-one"):
        write_manifest(records, tmp_path / "manifest.csv", video_root=tmp_path, source_hashes={})
