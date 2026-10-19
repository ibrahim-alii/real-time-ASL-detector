"""Import immutable, locally staged YouTube-ASL video/caption snapshots."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def staged_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not relative or not path.is_relative_to(root.resolve()):
        raise ValueError("snapshot path must stay inside the staging directory")
    return path


def channel_split(channel: str, seed: int = 42) -> str:
    if not channel:
        raise ValueError("channel identity is required for grouped splits")
    value = int(hashlib.sha256(f"{seed}:{channel}".encode()).hexdigest(), 16) % 100
    return "train" if value < 90 else ("dev" if value < 95 else "test")


def freeze_snapshot(
    *, index: Path, root: Path, official_ids: Path, output: Path, seed: int = 42,
    exclude_hashes: set[str] | None = None,
) -> dict[str, Any]:
    """Write captions.csv and snapshot.json; keep failures, never substitute videos.

    Index columns: video_id,channel_id,video_path,caption_path,downloaded_at,
    availability; optional split and signer_id. Captions are UTF-8 CSV with
    start_ms,end_ms,caption. Official splits override generated channel splits.
    """
    if output.exists():
        raise FileExistsError("snapshot output exists; choose a new version directory")
    ids = {line.strip() for line in official_ids.read_text(encoding="utf-8").splitlines()
           if line.strip()}
    if not ids:
        raise ValueError("official YouTube-ASL ID list is empty")
    with index.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "channel_id", "video_path", "caption_path",
                    "downloaded_at", "availability"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"snapshot index requires {sorted(required)}")
        videos = list(reader)
    if not videos or len({row["video_id"] for row in videos}) != len(videos):
        raise ValueError("snapshot index must contain unique video IDs")
    channels: dict[str, str] = {}
    signers: dict[str, str] = {}
    evidence: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in sorted(videos, key=lambda item: item["video_id"]):
        video_id = row["video_id"]
        if video_id not in ids or not video_id or any(c in video_id for c in "/\\"):
            raise ValueError(f"video ID outside official release: {video_id}")
        if not row["downloaded_at"].strip():
            raise ValueError("snapshot entries need a download/attempt date")
        split = row.get("split") or channel_split(row["channel_id"], seed)
        if split not in {"train", "dev", "test"}:
            raise ValueError("invalid snapshot split")
        for group, groups in ((row["channel_id"], channels),
                              (row.get("signer_id", ""), signers)):
            if group and groups.setdefault(group, split) != split:
                raise ValueError("channel or verified signer spans snapshot splits")
        current: dict[str, Any] = {**row, "split": split}
        availability = row["availability"]
        if availability not in {"available", "unavailable", "changed"}:
            raise ValueError("availability must be available, unavailable or changed")
        if availability == "available":
            video = staged_path(root, row["video_path"])
            caption_file = staged_path(root, row["caption_path"])
            if video.stem != video_id or video.suffix.lower() != ".mp4":
                raise ValueError("staged video must be named VIDEO_ID.mp4")
            if not video.is_file() or not caption_file.is_file():
                availability = "unavailable"
                current["reason"] = "missing-staged-file"
            else:
                current["video_sha256"] = file_hash(video)
                current["caption_sha256"] = file_hash(caption_file)
                if current["video_sha256"] in seen_hashes | (exclude_hashes or set()):
                    availability = "changed"
                    current["reason"] = "duplicate-or-evaluation-video"
                seen_hashes.add(current["video_sha256"])
        current["availability"] = availability
        evidence.append(current)
        if availability != "available":
            captions.append({"clip_id": f"{video_id}:unavailable", "video_id": video_id,
                             "start_ms": 0, "end_ms": 0, "caption": "",
                             "split": split, "availability": availability})
            continue
        with staged_path(root, row["caption_path"]).open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            reader = csv.DictReader(handle)
            if not {"start_ms", "end_ms", "caption"}.issubset(reader.fieldnames or []):
                raise ValueError("caption file requires start_ms,end_ms,caption")
            rows = list(reader)
        if not rows:
            rows = [{"start_ms": "0", "end_ms": "0", "caption": ""}]
        for number, caption in enumerate(rows):
            captions.append({"clip_id": f"{video_id}:{number:06d}", "video_id": video_id,
                             "start_ms": caption["start_ms"], "end_ms": caption["end_ms"],
                             "caption": caption["caption"], "split": split,
                             "availability": "available"})
    report = {"seed": seed, "official_ids_sha256": file_hash(official_ids),
              "index_sha256": file_hash(index), "videos": evidence,
              "signer_independence": "unverified-unless-all-identities-supplied"}
    output.mkdir(parents=True)
    with (output / "captions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(captions[0]))
        writer.writeheader()
        writer.writerows(captions)
    report["captions_sha256"] = file_hash(output / "captions.csv")
    (output / "snapshot.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def verify_snapshot(snapshot: Path, root: Path) -> None:
    report = json.loads(snapshot.read_text(encoding="utf-8"))
    if file_hash(snapshot.parent / "captions.csv") != report["captions_sha256"]:
        raise ValueError("frozen caption table changed")
    for row in report["videos"]:
        if row["availability"] != "available":
            continue
        for name in ("video", "caption"):
            if file_hash(staged_path(root, row[f"{name}_path"])) != row[f"{name}_sha256"]:
                raise ValueError(f"frozen {name} changed: {row['video_id']}")
