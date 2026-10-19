"""Report dataset overlap without changing annotations or official splits."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from .manifest import ClipManifest


def read_signer_map(path: Path) -> dict[str, str]:
    """CSV maps manifest video_id to an independently verified signer_id."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"video_id", "signer_id"}.issubset(reader.fieldnames):
            raise ValueError("signer CSV requires video_id and signer_id")
        result: dict[str, str] = {}
        for row in reader:
            video, signer = row["video_id"].strip(), row["signer_id"].strip()
            if not video or not signer or (video in result and result[video] != signer):
                raise ValueError("empty or conflicting signer identity")
            result[video] = signer
        return result


def _overlaps(groups: dict[str, set[str]]) -> dict[str, list[str]]:
    return {key: sorted(splits) for key, splits in sorted(groups.items()) if len(splits) > 1}


def audit_manifest(
    manifest: ClipManifest, *, signer_map: dict[str, str] | None = None
) -> dict[str, object]:
    """Exact caption repeats are a review signal, not proof of duplicated video."""
    sources: dict[str, set[str]] = defaultdict(set)
    captions: dict[str, set[str]] = defaultdict(set)
    signers: dict[str, set[str]] = defaultdict(set)
    splits: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    missing_signers: set[str] = set()
    unassigned: list[str] = []
    for record in manifest.accepted():
        splits[record.split] += 1
        sources[f"{record.source}:{record.video_id}"].add(record.split)
        caption = " ".join(record.caption.casefold().split())
        captions[caption].add(record.split)
        caption_counts[caption] += 1
        if record.split == "unassigned":
            unassigned.append(record.clip_id)
        signer = (signer_map or {}).get(record.video_id)
        if signer is None:
            missing_signers.add(record.video_id)
        else:
            signers[signer].add(record.split)
    source_overlap = _overlaps(sources)
    signer_overlap = _overlaps(signers)
    return {
        "manifest_hash": manifest.fingerprint,
        "accepted_by_split": dict(sorted(splits.items())),
        "status_counts": dict(sorted(Counter(row.status for row in manifest.records).items())),
        "cross_split_sources": source_overlap,
        "cross_split_captions": _overlaps(captions),
        "repeated_caption_groups": sum(count > 1 for count in caption_counts.values()),
        "cross_split_signers": signer_overlap,
        "signer_metadata_complete": not missing_signers,
        "videos_without_signer_metadata": sorted(missing_signers),
        "signer_independence_verified": (
            "train" in splits and bool({"dev", "test"} & splits.keys())
            and not missing_signers and not signer_overlap and not unassigned
        ),
        "source_split_isolation_passed": bool(splits) and not source_overlap and not unassigned,
        "unassigned_clips": sorted(unassigned),
        "limitations": (
            "Checks metadata only; no fuzzy text, visual duplicate, or caption alignment audit."
        ),
    }
