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


class OverlayWriter:
    """Write review video incrementally; retain only small pose records in memory."""

    def __init__(self, video: Path, output: Path, *, fps: float = 15.0) -> None:
        if fps <= 0:
            raise ValueError("overlay fps must be positive")
        self.video, self.output, self.fps = video, output, fps
        self.writer: Any = None
        self.frames: list[dict[str, Any]] = []
        self.width = self.height = 0

    def append(self, image: Any, frame: WholeBodyFrame) -> None:
        import cv2

        if self.writer is None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.height, self.width = image.shape[:2]
            self.writer = cv2.VideoWriter(str(self.output), cv2.VideoWriter.fourcc(*"mp4v"),
                                          self.fps, (self.width, self.height))
            if not self.writer.isOpened():
                self.writer.release()
                raise RuntimeError("could not open overlay video writer")
        if self.frames and frame.timestamp_ms <= self.frames[-1]["timestamp_ms"]:
            raise ValueError("overlay timestamps must increase")
        canvas = image.copy()
        for index, point in enumerate(frame.coordinates):
            if frame.presence[index]:
                color = (0, 255, 0) if index >= 91 else (0, 180, 255)
                cv2.circle(canvas, (round(float(point[0])), round(float(point[1]))),
                           2, color, -1)
        cv2.putText(canvas, f"{self.video.name} {frame.timestamp_ms}ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.writer.write(canvas)
        self.frames.append({"timestamp_ms": frame.timestamp_ms,
                            "coordinates": frame.coordinates.tolist(),
                            "presence": frame.presence.tolist()})

    def __enter__(self) -> OverlayWriter:
        return self

    def __exit__(self, error_type: object, *_args: object) -> None:
        if self.writer is not None:
            self.writer.release()
        if error_type is not None:
            return
        if not self.frames:
            raise ValueError("no overlay frames")
        self.output.with_suffix(".json").write_text(json.dumps({
            "source": str(self.video), "timestamps_ms": [f["timestamp_ms"] for f in self.frames],
            "playback_fps": self.fps, "note": "sampled frames; timestamps define source timing",
            "width": self.width, "height": self.height, "frames": self.frames,
        }, indent=2) + "\n", encoding="utf-8")


def write_overlay(video: Path, frames: list[tuple[Any, WholeBodyFrame]], output: Path) -> None:
    """Write sampled landmark frames; points stay in source pixel coordinates."""
    with OverlayWriter(video, output) as writer:
        for image, frame in frames:
            writer.append(image, frame)


def runtime_check(assets: dict[str, Path], *, device: str, threshold: float) -> dict[str, Any]:
    import torch

    with MMPoseWholeBodyBackend(**assets, device=device, confidence_threshold=threshold):
        pass
    return {"extractor": extractor_identity(assets, threshold), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": device,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
