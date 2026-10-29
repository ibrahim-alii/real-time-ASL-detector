"""Recorded-video pose extraction for pretrained models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .runtime import ExtractionEngine
from .sol_common import write_json


def extract_video(
    *, video: Path, assets: dict[str, Path], output: Path, device: str = "cuda:0",
    threshold: float = 0.5, start_ms: int = 0, end_ms: int | None = None,
    overlay: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    engine = ExtractionEngine(assets, device=device, threshold=threshold)
    try:
        report = engine.extract(video=video, output=output, start_ms=start_ms,
                                end_ms=end_ms, overlay=overlay)
        report["peak_vram_bytes"] = (torch.cuda.max_memory_allocated(device)
                                     if device.startswith("cuda") else 0)
        write_json(output / "extraction.json", report)
        return report
    finally:
        engine.close()

