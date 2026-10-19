"""Bounded, deterministic pilots that preserve source split assignments."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from .manifest import ClipManifest, read_manifest, write_manifest


def subset_manifest(
    manifest_path: Path, output: Path, *, train_limit: int, dev_limit: int,
    seed: int = 42,
) -> None:
    if min(train_limit, dev_limit) < 1:
        raise ValueError("train and dev limits must be positive")
    if output.exists():
        raise FileExistsError(output)
    manifest = read_manifest(manifest_path)
    selected: set[str] = set()
    for split, limit in (("train", train_limit), ("dev", dev_limit)):
        rows = sorted(manifest.accepted(split), key=lambda row: hashlib.sha256(
            f"{seed}:{row.clip_id}".encode()).hexdigest())
        if not rows:
            raise ValueError(f"subset requires accepted {split} rows")
        selected.update(row.clip_id for row in rows[:limit])
    # Preserve test rows and rejected/unavailable entries. Extraction can select
    # train+dev so test features need not be materialized until final evaluation.
    records = tuple(
        replace(row, status="rejected", reason="outside-bounded-subset")
        if row.status == "accepted" and row.split in {"train", "dev"}
        and row.clip_id not in selected else row for row in manifest.records
    )
    write_manifest(output, ClipManifest(records))
