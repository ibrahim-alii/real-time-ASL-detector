"""Lossless pose evidence for quality audits and independent model preprocessing."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ..features.mmpose_backend import WholeBodyFrame
from .sol_common import write_json
from .youtube import file_hash

POSE_SCHEMA = "mmpose-coco-wholebody-133-raw-xy-score-ms-v1"
PREPROCESSING = {
    "sampling": "caption-half-open-every-second-native-frame-v1",
    "normalization": "complete-clip-common-xy-bounding-box-v1",
    "frame_cap": 256,
    "downsampling": "uniform-inclusive-indices-v1",
    "missing_coordinates": -1000.0,
}
REGIONS = {"body": slice(0, 17), "face": slice(23, 91),
           "left_hand": slice(91, 112), "right_hand": slice(112, 133)}


class PoseCache:
    """Raw coordinates and scores remain separate from thresholded model features."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, clip_id: str) -> Path:
        return self.root / (hashlib.sha256(clip_id.encode()).hexdigest() + ".npz")

    def store(self, clip_id: str, frames: Sequence[WholeBodyFrame], *,
              metadata: Mapping[str, object]) -> Path:
        if not frames or any(f.raw_coordinates is None or f.scores is None for f in frames):
            raise ValueError("raw pose coordinates and scores are required")
        times = [f.timestamp_ms for f in frames]
        if any(a >= b for a, b in pairwise(times)):
            raise ValueError("raw pose timestamps must increase")
        destination = self.path(clip_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, coordinates=np.stack([f.coordinates for f in frames]),
            raw_coordinates=np.stack([f.raw_coordinates for f in frames
                                      if f.raw_coordinates is not None]),
            scores=np.stack([f.scores for f in frames if f.scores is not None]),
            presence=np.stack([f.presence for f in frames]), timestamps_ms=np.array(times),
            metadata=json.dumps({**metadata, "pose_schema": POSE_SCHEMA}, sort_keys=True))
        temporary.replace(destination)
        write_json(destination.with_suffix(".json"), {"sha256": file_hash(destination)})
        return destination

    def load(self, clip_id: str, *, expected_metadata: Mapping[str, object] | None = None
             ) -> tuple[list[WholeBodyFrame], dict[str, Any]]:
        path = self.path(clip_id)
        digest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if file_hash(path) != digest["sha256"]:
            raise ValueError("raw pose checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if (not isinstance(metadata, dict) or metadata.get("pose_schema") != POSE_SCHEMA
                    or any(metadata.get(k) != v for k, v in (expected_metadata or {}).items())):
                raise ValueError("raw pose identity mismatch")
            times = data["timestamps_ms"]
            if (times.ndim != 1 or not len(times) or not np.issubdtype(times.dtype, np.integer)
                    or np.any(np.diff(times) <= 0)):
                raise ValueError("invalid raw pose timestamps")
            if any(len(data[key]) != len(times) for key in (
                    "coordinates", "raw_coordinates", "scores", "presence")):
                raise ValueError("raw pose arrays have different lengths")
            frames = [WholeBodyFrame(data["coordinates"][i], data["presence"][i], int(time),
                      data["scores"][i], data["raw_coordinates"][i])
                      for i, time in enumerate(times)]
        return frames, metadata


def pose_quality(frames: Sequence[WholeBodyFrame]) -> dict[str, Any]:
    """Regional visibility is evidence, never an invented semantic acceptance flag."""
    if not frames:
        raise ValueError("pose quality requires frames")
    masks = np.stack([frame.presence for frame in frames])
    coordinates = np.stack([frame.coordinates for frame in frames])
    regions: dict[str, Any] = {}
    for name, indices in REGIONS.items():
        present = masks[:, indices]
        longest = current = 0
        for index in range(len(frames)):
            missing = not bool(present[index].any())
            current = current + 1 if missing else 0
            longest = max(longest, current)
        paired = present[1:] & present[:-1]
        displacement = np.linalg.norm(np.diff(coordinates[:, indices], axis=0), axis=-1)
        regions[name] = {
            "visible_fraction": float(present.mean()),
            "longest_missing_frames": longest,
            "median_displacement_pixels": float(np.median(displacement[paired]))
            if paired.any() else None,
        }
    return {"frames": len(frames), "start_ms": frames[0].timestamp_ms,
            "end_ms": frames[-1].timestamp_ms, "regions": regions,
            "frames_with_either_hand": int(masks[:, 91:].any(axis=1).sum()),
            "semantic_review": "pending"}
