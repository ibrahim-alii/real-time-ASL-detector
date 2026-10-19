"""Import How2Sign with explicit alignment and source-video layout."""

from __future__ import annotations

import csv
from pathlib import Path

from .manifest import ClipManifest, build_manifest

REQUIRED_COLUMNS = {
    "VIDEO_ID", "SENTENCE_NAME", "START", "END", "SENTENCE",
}


def _read_table(path: Path, *, alignment: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = REQUIRED_COLUMNS
        if alignment == "realigned":
            required = (REQUIRED_COLUMNS - {"START", "END"}) | {
                "START_REALIGNED", "END_REALIGNED", "VIDEO_NAME"
            }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"How2Sign annotation columns are incomplete: {path.name}")
        return list(reader)


def build_how2sign_manifest(
    *, train: Path, dev: Path, test: Path, pilot_first_video: bool = False,
    pilot_clips: int = 32,
    alignment: str = "original", video_layout: str = "sentence-clips",
) -> ClipManifest:
    """Realigned timestamps address full recordings, never original sentence clips."""
    if pilot_clips < 1:
        raise ValueError("pilot clip limit must be positive")
    if alignment not in {"original", "realigned"}:
        raise ValueError("alignment must be original or realigned")
    if video_layout not in {"recordings", "sentence-clips"}:
        raise ValueError("video layout must be recordings or sentence-clips")
    if alignment == "realigned" and video_layout != "recordings":
        raise ValueError("realigned annotations require full recordings")
    tables = {
        "train": _read_table(train, alignment=alignment),
        "dev": _read_table(dev, alignment=alignment),
        "test": _read_table(test, alignment=alignment),
    }
    if any(not table for table in tables.values()):
        raise ValueError("official How2Sign train/dev/test annotations must be non-empty")
    if pilot_first_video:
        video_id = min(row["VIDEO_ID"] for row in tables["train"])
        pilot_rows = [row for row in tables["train"] if row["VIDEO_ID"] == video_id]
        tables = {"train": pilot_rows[:pilot_clips]}
    rows: list[dict[str, object]] = []
    for split, table in tables.items():
        for row in table:
            start = float(row["START_REALIGNED" if alignment == "realigned" else "START"])
            end = float(row["END_REALIGNED" if alignment == "realigned" else "END"])
            if video_layout == "recordings" and not row.get("VIDEO_NAME", "").strip():
                raise ValueError("full recordings require VIDEO_NAME")
            rows.append({
                "clip_id": f"{split}:{row['SENTENCE_NAME']}",
                "video_id": Path(row["VIDEO_NAME"]).stem
                if video_layout == "recordings" else row["SENTENCE_NAME"],
                "start_ms": round(start * 1000) if video_layout == "recordings" else 0,
                "end_ms": round(end * 1000)
                if video_layout == "recordings" else round((end - start) * 1000),
                "caption": row["SENTENCE"],
                "split": split,
            })
    return build_manifest(rows, source="how2sign")
