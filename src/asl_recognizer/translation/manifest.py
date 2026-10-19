from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    clip_id: str
    source: str
    split: str
    video_id: str
    start_ms: int
    end_ms: int
    caption: str
    status: str = "accepted"
    reason: str = ""

    def __post_init__(self) -> None:
        if (
            not self.clip_id
            or not self.source
            or self.split not in {"train", "dev", "test", "unassigned"}
        ):
            raise ValueError("manifest identity or split is invalid")
        if self.start_ms < 0 or self.end_ms < self.start_ms:
            raise ValueError("caption boundaries are invalid")
        if self.status not in {"accepted", "unavailable", "changed", "malformed", "rejected"}:
            raise ValueError("manifest status is invalid")


@dataclass(frozen=True, slots=True)
class ClipManifest:
    records: tuple[ManifestRecord, ...]
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        ids = [record.clip_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("manifest clip IDs must be unique")

        payload = [asdict(record) for record in sorted(self.records, key=lambda item: item.clip_id)]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        object.__setattr__(self, "_fingerprint", digest)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def accepted(self, split: str | None = None) -> tuple[ManifestRecord, ...]:
        return tuple(
            item
            for item in self.records
            if item.status == "accepted" and (split is None or item.split == split)
        )


def build_manifest(
    rows: list[dict[str, Any]],
    *,
    source: str,
    official_split: dict[str, str] | None = None,
    require_person_count: bool = False,
) -> ClipManifest:
    """Create a complete audit trail; inaccessible clips are never substituted."""
    records: list[ManifestRecord] = []
    for number, row in enumerate(rows):
        clip_id = str(row.get("clip_id") or row.get("id") or number)
        video_id = str(row.get("video_id") or row.get("youtube_id") or "")
        caption = str(row.get("caption") or row.get("text") or "")
        try:
            start_ms = (
                _milliseconds(row["start_ms"], unit="ms")
                if "start_ms" in row
                else _milliseconds(row.get("start", 0), unit="s")
            )
            end_ms = (
                _milliseconds(row["end_ms"], unit="ms")
                if "end_ms" in row
                else _milliseconds(row.get("end", 0), unit="s")
            )
            if start_ms < 0 or end_ms < start_ms:
                raise ValueError("caption boundaries are invalid")
        except (TypeError, ValueError, OverflowError):
            start_ms, end_ms = 0, 0
            row = {**row, "availability": "malformed-time"}
        split = (official_split or {}).get(clip_id, str(row.get("split", "unassigned")))
        status, reason = _status(
            row, source, video_id, caption, start_ms, end_ms, require_person_count
        )
        records.append(
            ManifestRecord(
                clip_id, source, split, video_id, start_ms, end_ms, caption, status, reason
            )
        )
    return ClipManifest(tuple(records))


def _milliseconds(value: object, *, unit: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError("caption time must be numeric")
    number = float(value)
    return round(number * 1000) if unit == "s" else round(number)


def _status(
    row: dict[str, Any],
    source: str,
    video_id: str,
    caption: str,
    start_ms: int,
    end_ms: int,
    require_person_count: bool,
) -> tuple[str, str]:
    availability = str(row.get("availability", "available"))
    if availability in {"unavailable", "changed"}:
        return availability, "reported by validator"
    if availability == "malformed-time":
        return "malformed", "invalid-caption-boundary"
    if not video_id:
        return "rejected", "missing-video-id"
    if end_ms <= start_ms:
        return "malformed", "non-positive-duration"
    if not caption.strip():
        return "malformed", "empty-caption"
    if source == "youtube-asl":
        if require_person_count and "person_count" not in row:
            return "rejected", "person-count-unverified"
        if len(caption) > 300:
            return "rejected", "caption-over-300-characters"
        if end_ms - start_ms < 200:
            return "rejected", "caption-under-200ms"
        if end_ms - start_ms > 60_000:
            return "rejected", "caption-over-60s"
        if "person_count" in row:
            try:
                count = int(row["person_count"])
            except (TypeError, ValueError):
                return "rejected", "person-count-invalid"
            if count != 1:
                return "rejected", "person-count-not-one"
    return "accepted", ""


def write_manifest(path: Path, manifest: ClipManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ManifestRecord.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(item) for item in manifest.records)
    path.with_suffix(path.suffix + ".sha256").write_text(
        manifest.fingerprint + "\n", encoding="ascii"
    )


def read_manifest(path: Path) -> ClipManifest:
    with path.open(newline="", encoding="utf-8") as handle:
        records = tuple(
            ManifestRecord(
                clip_id=row["clip_id"],
                source=row["source"],
                split=row["split"],
                video_id=row["video_id"],
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                caption=row["caption"],
                status=row["status"],
                reason=row["reason"],
            )
            for row in csv.DictReader(handle)
        )
    manifest = ClipManifest(records)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    if manifest.fingerprint != digest_path.read_text(encoding="ascii").strip():
        raise ValueError("manifest checksum mismatch")
    return manifest
