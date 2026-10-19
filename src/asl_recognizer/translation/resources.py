"""Resource estimates from measured local experiment and extraction reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def resource_estimate(
    *, progress: Path, extraction_report: Path, feature_root: Path,
    target_clips: int, target_frames: int, target_steps: int = 205000,
) -> dict[str, Any]:
    run = json.loads(progress.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_report.read_text(encoding="utf-8"))
    steps = int(run["global_step"])
    seconds = float(run["optimizer_seconds"])
    fps = float(extraction["processing_fps"])
    files = list(feature_root.rglob("*.npz"))
    if min(steps, target_clips, target_frames, target_steps, len(files)) < 1 \
            or seconds <= 0 or fps <= 0 or not math.isfinite(seconds + fps):
        raise ValueError("resource estimate requires positive measured work and targets")
    training_hours = seconds / steps * target_steps / 3600
    extraction_hours = target_frames / fps / 3600
    return {"measured_optimizer_steps": steps, "seconds_per_optimizer_step": seconds / steps,
            "target_optimizer_steps": target_steps, "training_gpu_hours": training_hours,
            "extraction_gpu_hours": extraction_hours,
            "estimated_total_gpu_hours": training_hours + extraction_hours,
            "estimated_feature_bytes": sum(p.stat().st_size for p in files) / len(files)
            * target_clips, "peak_training_vram_bytes": run["peak_vram_bytes"],
            "limitations": "excludes downloads, validation, overlays and checkpoint storage; "
            "short-clip measurements may underestimate full-corpus cost"}
