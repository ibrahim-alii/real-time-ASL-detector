"""Asset identities, overlays, and runtime checks shared by local commands."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ..features.mmpose_backend import MMPoseWholeBodyBackend, WholeBodyFrame
from .youtube import file_hash


def extractor_identity(assets: dict[str, Path], threshold: float) -> dict[str, Any]:
    return {"backend": "mmpose", "confidence_threshold": threshold,
            "implementation": "explicit-person-boxes-v2",
            "person_score_threshold": 0.3, "person_nms_iou": 0.3,
            "assets": {name: file_hash(path) for name, path in assets.items()},
            "packages": {name: version(name) for name in ("mmpose", "mmcv", "mmengine", "mmdet")}}


def write_overlay(video: Path, frames: list[tuple[Any, WholeBodyFrame]], output: Path) -> None:
    """Write sampled landmark frames; points stay in source pixel coordinates."""
    import cv2

    if not frames:
        raise ValueError("no overlay frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0][0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter.fourcc(*"mp4v"), 15.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("could not open overlay video writer")
    try:
        for image, frame in frames:
            canvas = image.copy()
            for index, point in enumerate(frame.coordinates):
                if frame.presence[index]:
                    color = (0, 255, 0) if index >= 91 else (0, 180, 255)
                    cv2.circle(canvas, (round(float(point[0])), round(float(point[1]))),
                               2, color, -1)
            cv2.putText(canvas, f"{video.name} {frame.timestamp_ms}ms", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(canvas)
    finally:
        writer.release()
    output.with_suffix(".json").write_text(json.dumps({
        "source": str(video), "timestamps_ms": [frame.timestamp_ms for _, frame in frames],
        "playback_fps": 15, "note": "sampled frames; timestamps define source timing",
        "width": width, "height": height,
        "frames": [{"timestamp_ms": frame.timestamp_ms,
                    "coordinates": frame.coordinates.tolist(),
                    "presence": frame.presence.tolist()} for _, frame in frames],
    }, indent=2) + "\n", encoding="utf-8")


def runtime_check(assets: dict[str, Path], *, device: str, threshold: float) -> dict[str, Any]:
    import torch

    with MMPoseWholeBodyBackend(**assets, device=device, confidence_threshold=threshold):
        pass
    return {"extractor": extractor_identity(assets, threshold), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": device,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
