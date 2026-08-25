from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

Split = Literal["train", "val", "test"]
VALID_SPLITS = frozenset({"train", "val", "test"})


@dataclass(frozen=True, slots=True)
class SampleRecord:
    video_id: str
    path: Path
    gloss: str
    signer_id: str
    split: Split
    class_id: int | None = None


def validate_manifest(records: Iterable[SampleRecord]) -> None:
    materialized = list(records)
    if not materialized:
        raise ValueError("manifest is empty")

    video_ids = [record.video_id for record in materialized]
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("duplicate video IDs found in manifest")

    signers_by_split = {
        split: {record.signer_id for record in materialized if record.split == split}
        for split in VALID_SPLITS
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = signers_by_split[left] & signers_by_split[right]
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"signer leakage between {left} and {right}: {sample}")

    for record in materialized:
        if not all((record.video_id, record.gloss, record.signer_id)):
            raise ValueError("manifest fields cannot be blank")
        if record.split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {record.split}")
        if record.class_id is not None and record.class_id < 0:
            raise ValueError("class_id cannot be negative")
