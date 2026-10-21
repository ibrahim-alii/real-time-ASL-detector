"""Recorded-sentence extraction and translation across isolated environments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .cache import TranslationFeatureCache
from .experiment import checkpoint_model_config
from .model import LandmarkT5
from .runtime import ExtractionEngine
from .schema import MMPOSE_SCHEMA
from .sol_common import write_json
from .youtube import file_hash


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


def infer_video(
    *, checkpoint: Path, video: Path, assets: dict[str, Path], output: Path,
    extraction_python: Path | None = None, device: str = "cuda:0",
    start_ms: int = 0, end_ms: int | None = None, overlay: Path | None = None,
) -> dict[str, Any]:
    """Extract in a child process, release its GPU memory, then load T5."""
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["config"]["feature_schema"] != MMPOSE_SCHEMA:
        raise ValueError("recorded inference requires an MMPose checkpoint")
    expected = saved["config"].get("extractor")
    if not isinstance(expected, dict):
        raise ValueError("checkpoint lacks extractor identity")
    started = time.perf_counter()
    feature_root = output.parent / (output.stem + "-features")
    command = [str(extraction_python or Path(sys.executable)), "-m",
               "asl_recognizer.translation.worker", "extract-video",
               "--video", str(video.resolve()),
               "--output", str(feature_root.resolve()), "--device", device,
               "--start-ms", str(start_ms), "--keypoint-threshold",
               str(expected["confidence_threshold"])]
    for name, path in assets.items():
        command.extend(["--" + name.replace("_", "-"), str(path.resolve())])
    if end_ms is not None:
        command.extend(["--end-ms", str(end_ms)])
    if overlay is not None:
        command.extend(["--overlay", str(overlay.resolve())])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    subprocess.run(command, env=env, check=True)
    request, metadata = TranslationFeatureCache(feature_root).load("recorded")
    if metadata["extractor"] != expected:
        raise ValueError("recorded extractor differs from checkpoint; translation refused")
    preprocessing = saved["config"].get("preprocessing")
    if preprocessing is not None and metadata.get("preprocessing") != preprocessing:
        raise ValueError("recorded preprocessing differs from checkpoint; translation refused")
    if device.startswith("cuda"):
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    model_config = checkpoint_model_config(saved)
    model = LandmarkT5(model_config).to(device)
    model.load_state_dict(saved["model"])
    tokenizer = AutoTokenizer.from_pretrained(model_config.pretrained_model,
                                              revision=model_config.pretrained_revision)
    result = model.translate(request, tokenizer, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    extraction = json.loads((feature_root / "extraction.json").read_text(encoding="utf-8"))
    report = {**asdict(result), "checkpoint_sha256": file_hash(checkpoint),
              "video": str(video), "extraction": extraction, "end_to_end_seconds": elapsed,
              "translation_peak_vram_bytes": torch.cuda.max_memory_allocated(device)
              if device.startswith("cuda") else 0,
              "quality_status": "unverified-decoder-score-is-not-calibrated-confidence"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
