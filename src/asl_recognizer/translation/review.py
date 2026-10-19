"""Deterministic reviewer queues and an explicit semantic-quality gate."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .manifest import read_manifest

REVIEW_FIELDS = ("clip_id", "video_id", "start_ms", "end_ms", "caption", "split",
                 "meaning_ok", "timing_ok", "visibility_ok", "extraction_ok", "reviewer", "notes")
CHECKS = ("meaning_ok", "timing_ok", "visibility_ok", "extraction_ok")


def export_review(manifest_path: Path, output: Path, *, count: int = 100, seed: int = 42) -> None:
    if count < 1:
        raise ValueError("review count must be positive")
    if output.exists():
        raise FileExistsError(output)
    manifest = read_manifest(manifest_path)
    # Training pairs only: review must not become a test-set tuning exercise.
    rows = sorted(manifest.accepted("train"), key=lambda row: hashlib.sha256(
        f"{seed}:{row.clip_id}".encode()).hexdigest())[:count]
    if not rows:
        raise ValueError("no accepted training pairs to review")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: getattr(row, name) for name in REVIEW_FIELDS[:6]})
    output.with_suffix(".sha256").write_text(manifest.fingerprint + "\n", encoding="ascii")


def review_gate(
    path: Path, *, manifest_path: Path, minimum: int = 100, threshold: float = 0.9
) -> dict[str, object]:
    if minimum < 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid review gate settings")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    manifest = read_manifest(manifest_path)
    if path.with_suffix(".sha256").read_text(encoding="ascii").strip() != manifest.fingerprint:
        raise ValueError("review queue belongs to a different manifest version")
    accepted = {row.clip_id: row for row in manifest.accepted("train")}
    for row in rows:
        original = accepted.get(row["clip_id"])
        if original is None or any(str(getattr(original, name)) != row[name]
                                   for name in REVIEW_FIELDS[:6]):
            raise ValueError("review pair differs from the frozen training manifest")
    if len({row["clip_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate review IDs")
    completed = [row for row in rows if row.get("reviewer", "").strip()
                 and all(row.get(check) in {"yes", "no"} for check in CHECKS)]
    acceptable = sum(all(row[check] == "yes" for check in CHECKS) for row in completed)
    rate = acceptable / len(completed) if completed else 0.0
    return {"manifest_hash": manifest.fingerprint,
            "sample_size": len(rows), "completed": len(completed), "acceptable": acceptable,
            "acceptable_rate": rate, "passed": len(completed) == len(rows)
            and len(completed) >= minimum and rate >= threshold}
