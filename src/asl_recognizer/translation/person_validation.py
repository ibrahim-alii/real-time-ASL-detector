"""Conservative, auditable sampled-frame person check for YouTube-ASL clips."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..features.video import sampled_video_segment
from .youtube import file_hash

VALIDATOR_NAME = "mmpose-person-1hz-v1"
EVIDENCE_FIELDS = (
    "person_count",
    "person_samples",
    "person_min",
    "person_max",
    "person_validator",
    "person_model_sha256",
    "person_error",
)


@dataclass(frozen=True, slots=True)
class PersonEvidence:
    person_count: int
    samples: int
    minimum: int
    maximum: int


def inspect_clip(
    path: Path,
    *,
    start_ms: int,
    end_ms: int,
    detect_people: Callable[[npt.NDArray[np.uint8]], int],
) -> PersonEvidence:
    """Accept one person only when every decoded 1 Hz frame detects exactly one."""
    sampled = sampled_video_segment(path, start_ms=start_ms, end_ms=end_ms, sample_rate_hz=1.0)
    if not sampled.frames or len(sampled.frames) != sampled.requested_frames:
        raise ValueError("person validation could not decode every sampled frame")
    counts = [detect_people(frame) for frame, _ in sampled.frames]
    if any(count not in (0, 1, 2) for count in counts):
        raise ValueError("person detector returned an invalid count")
    person_count = 2 if 2 in counts else (1 if all(count == 1 for count in counts) else 0)
    return PersonEvidence(person_count, len(counts), min(counts), max(counts))


def annotate_caption_rows(
    rows: list[dict[str, str]],
    *,
    video_root: Path,
    model_sha256: str,
    detect_people: Callable[[npt.NDArray[np.uint8]], int],
    validator_name: str = VALIDATOR_NAME,
    resume_root: Path | None = None,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Preserve source rows; leave uncertain clips unverified for manifest rejection."""
    annotated: list[dict[str, str]] = []
    outcomes: Counter[str] = Counter()
    videos: dict[str, Path] = {}
    video_hashes: dict[str, str] = {}
    for candidate in video_root.rglob("*.mp4"):
        if candidate.stem in videos:
            raise ValueError(f"ambiguous staged video ID: {candidate.stem}")
        videos[candidate.stem] = candidate
    for row in rows:
        current = {**row, **dict.fromkeys(EVIDENCE_FIELDS, "")}
        current["person_validator"] = validator_name
        current["person_model_sha256"] = model_sha256
        video_id = row.get("video_id") or row.get("youtube_id") or ""
        cached: Path | None = None
        try:
            if not video_id or Path(video_id).name != video_id or "\\" in video_id:
                raise ValueError("invalid video ID")
            start_ms = int(row["start_ms"])
            end_ms = int(row["end_ms"])
            path = videos.get(video_id)
            if path is not None and resume_root is not None:
                import hashlib

                from .sol_common import read_json, write_json

                if video_id not in video_hashes:
                    video_hashes[video_id] = file_hash(path)
                cache_key = hashlib.sha256(
                    json.dumps(
                        {
                            "row": row,
                            "model": model_sha256,
                            "validator": validator_name,
                            "video": video_hashes[video_id],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                cached = resume_root / (cache_key + ".json")
                if cached.exists():
                    saved = read_json(cached)
                    if saved.get("person_count") in {"0", "1", "2"}:
                        annotated.append(saved)
                        outcomes["one" if saved["person_count"] == "1" else "not-one"] += 1
                        continue
            if path is None:
                current["availability"] = "unavailable"
                current["person_error"] = "source-video-unavailable"
                outcomes["unavailable"] += 1
            else:
                evidence = inspect_clip(
                    path, start_ms=start_ms, end_ms=end_ms, detect_people=detect_people
                )
                current["person_count"] = str(evidence.person_count)
                current["person_samples"] = str(evidence.samples)
                current["person_min"] = str(evidence.minimum)
                current["person_max"] = str(evidence.maximum)
                outcomes["one" if evidence.person_count == 1 else "not-one"] += 1
                if cached is not None:
                    write_json(cached, current)
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            current["person_error"] = str(error)
            outcomes["unverified"] += 1
        annotated.append(current)
    return annotated, dict(outcomes)
