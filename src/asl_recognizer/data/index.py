from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, cast

from .schema import SampleRecord, Split, VALID_SPLITS, validate_manifest

REQUIRED_COLUMNS = ("Participant ID", "Video file", "Gloss")
OFFICIAL_SIGNER_COUNTS = {"train": 35, "val": 6, "test": 11}


@dataclass(frozen=True, slots=True)
class Vocabulary:
    labels: tuple[str, ...]
    manifest_fingerprint: str
    source_hashes: dict[str, str]

    @property
    def class_ids(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.labels)}


def build_manifest(
    split_files: Mapping[str, Path],
    video_root: Path,
    *,
    require_files: bool = True,
    require_all_splits: bool = True,
    expected_signer_counts: Mapping[str, int] | None = OFFICIAL_SIGNER_COUNTS,
) -> list[SampleRecord]:
    if require_all_splits and set(split_files) != set(VALID_SPLITS):
        raise ValueError("split files must contain exactly train, val, and test")
    resolved_root = video_root.resolve()
    records: list[SampleRecord] = []
    for split, metadata_path in split_files.items():
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split name: {split}")
        with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = [column for column in REQUIRED_COLUMNS if column not in columns]
            if missing:
                raise ValueError(f"missing required columns: {', '.join(missing)}")
            for row_number, row in enumerate(reader, start=2):
                filename = (row.get("Video file") or "").strip()
                gloss = (row.get("Gloss") or "").strip()
                signer_id = (row.get("Participant ID") or "").strip()
                if not all((filename, gloss, signer_id)):
                    raise ValueError(f"blank required value in {metadata_path} row {row_number}")
                candidate = Path(filename)
                path = (video_root / candidate).resolve()
                try:
                    path.relative_to(resolved_root)
                except ValueError as error:
                    raise ValueError(
                        f"video path is outside video root in {metadata_path} row {row_number}"
                    ) from error
                if require_files and not path.is_file():
                    raise FileNotFoundError(f"video not found: {path}")
                records.append(
                    SampleRecord(path.stem, path, gloss, signer_id, cast(Split, split))
                )
    validate_manifest(records)
    if expected_signer_counts is not None:
        for split, expected in expected_signer_counts.items():
            actual = len({record.signer_id for record in records if record.split == split})
            if actual != expected:
                raise ValueError(
                    f"{split} has {actual} signers; expected official ASL Citizen count {expected}"
                )
    return records


def select_vocabulary(records: Sequence[SampleRecord], *, size: int = 25) -> list[str]:
    if size < 1:
        raise ValueError("vocabulary size must be positive")
    splits_by_gloss: dict[str, set[str]] = defaultdict(set)
    train_signers: dict[str, set[str]] = defaultdict(set)
    train_counts: Counter[str] = Counter()
    for record in records:
        splits_by_gloss[record.gloss].add(record.split)
        if record.split == "train":
            train_signers[record.gloss].add(record.signer_id)
            train_counts[record.gloss] += 1
    eligible = [
        gloss for gloss, splits in splits_by_gloss.items() if splits == set(VALID_SPLITS)
    ]
    eligible.sort(key=lambda gloss: (-len(train_signers[gloss]), -train_counts[gloss], gloss))
    if len(eligible) < size:
        raise ValueError(f"only {len(eligible)} signs occur in every split; requested {size}")
    return eligible[:size]


def assign_class_ids(
    records: Sequence[SampleRecord], vocabulary: Sequence[str]
) -> list[SampleRecord]:
    class_ids = {gloss: index for index, gloss in enumerate(vocabulary)}
    return [
        SampleRecord(
            record.video_id,
            record.path,
            record.gloss,
            record.signer_id,
            record.split,
            class_ids[record.gloss],
        )
        for record in records
        if record.gloss in class_ids
    ]


def manifest_fingerprint(records: Sequence[SampleRecord]) -> str:
    canonical = [
        {
            "video_id": item.video_id,
            "gloss": item.gloss,
            "signer_id": item.signer_id,
            "split": item.split,
            "class_id": item.class_id,
        }
        for item in sorted(records, key=lambda record: record.video_id)
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    records: Sequence[SampleRecord],
    output: Path,
    *,
    video_root: Path,
    source_hashes: Mapping[str, str],
) -> None:
    validate_manifest(records)
    if any(record.class_id is None for record in records):
        raise ValueError("all manifest records must have a class_id")
    gloss_to_id = {record.gloss: record.class_id for record in records}
    id_to_gloss = {record.class_id: record.gloss for record in records}
    if len(gloss_to_id) != len(id_to_gloss) or any(
        gloss_to_id[record.gloss] != record.class_id for record in records
    ):
        raise ValueError("gloss and class_id mapping must be one-to-one")
    class_ids = sorted(cast(int, value) for value in id_to_gloss)
    if class_ids != list(range(len(class_ids))):
        raise ValueError("class IDs must be contiguous from zero")
    resolved_root = video_root.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "relative_path",
                "gloss",
                "class_id",
                "signer_id",
                "split",
            ],
        )
        writer.writeheader()
        for record in records:
            try:
                relative_path = record.path.resolve().relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(f"video path is outside video root: {record.video_id}") from error
            writer.writerow(
                {
                    "video_id": record.video_id,
                    "relative_path": relative_path.as_posix(),
                    "gloss": record.gloss,
                    "class_id": record.class_id,
                    "signer_id": record.signer_id,
                    "split": record.split,
                }
            )
    metadata = {
        "dataset": "ASL Citizen",
        "dataset_version": "1.0",
        "manifest_fingerprint": manifest_fingerprint(records),
        "source_hashes": dict(sorted(source_hashes.items())),
    }
    _manifest_metadata_path(output).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def write_vocabulary(
    glosses: Sequence[str],
    output: Path,
    *,
    dataset_version: str,
    manifest_fingerprint_value: str,
    source_hashes: Mapping[str, str],
) -> None:
    payload = {
        "dataset": "ASL Citizen",
        "dataset_version": dataset_version,
        "selection": "all splits; train signer coverage; train count; gloss",
        "manifest_fingerprint": manifest_fingerprint_value,
        "source_hashes": dict(sorted(source_hashes.items())),
        "labels": [{"class_id": index, "gloss": gloss} for index, gloss in enumerate(glosses)],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_hash"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_vocabulary(path: Path) -> Vocabulary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.pop("content_hash", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    if not isinstance(stored_hash, str) or hashlib.sha256(canonical).hexdigest() != stored_hash:
        raise ValueError("vocabulary content hash mismatch")
    entries = payload.get("labels")
    if not isinstance(entries, list):
        raise ValueError("vocabulary labels must be a list")
    labels: list[str] = []
    for expected_id, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("class_id") != expected_id:
            raise ValueError("vocabulary class IDs must be contiguous and ordered")
        gloss = entry.get("gloss")
        if not isinstance(gloss, str) or not gloss:
            raise ValueError("vocabulary glosses must be non-empty strings")
        labels.append(gloss)
    if len(labels) != len(set(labels)):
        raise ValueError("vocabulary glosses must be unique")
    fingerprint = payload.get("manifest_fingerprint")
    source_hashes = payload.get("source_hashes")
    if not isinstance(fingerprint, str) or not isinstance(source_hashes, dict):
        raise ValueError("vocabulary provenance is missing")
    return Vocabulary(tuple(labels), fingerprint, dict(source_hashes))


def read_manifest(
    path: Path,
    video_root: Path,
    vocabulary: Vocabulary,
    *,
    require_files: bool = True,
) -> list[SampleRecord]:
    resolved_root = video_root.resolve()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {
            "video_id",
            "relative_path",
            "gloss",
            "class_id",
            "signer_id",
            "split",
        }
        if set(reader.fieldnames or ()) != expected_columns:
            raise ValueError("manifest schema mismatch")
        records: list[SampleRecord] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                class_id = int(row["class_id"])
                split = cast(Split, row["split"])
                relative_path = Path(row["relative_path"])
                resolved_path = (video_root / relative_path).resolve()
                resolved_path.relative_to(resolved_root)
                record = SampleRecord(
                    row["video_id"],
                    resolved_path,
                    row["gloss"],
                    row["signer_id"],
                    split,
                    class_id,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid manifest row {row_number}") from error
            if vocabulary.class_ids.get(record.gloss) != record.class_id:
                raise ValueError(f"class mapping mismatch in manifest row {row_number}")
            if require_files and not record.path.is_file():
                raise FileNotFoundError(f"video not found: {record.path}")
            records.append(record)
    validate_manifest(records)
    metadata = json.loads(_manifest_metadata_path(path).read_text(encoding="utf-8"))
    fingerprint = manifest_fingerprint(records)
    if metadata.get("manifest_fingerprint") != fingerprint:
        raise ValueError("manifest fingerprint mismatch")
    if vocabulary.manifest_fingerprint != fingerprint:
        raise ValueError("manifest fingerprint does not match vocabulary")
    if metadata.get("source_hashes") != vocabulary.source_hashes:
        raise ValueError("manifest source hashes do not match vocabulary")
    return records


def _manifest_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")
