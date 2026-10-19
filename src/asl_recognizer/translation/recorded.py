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

from ..features.mmpose_backend import MMPoseWholeBodyBackend
from ..features.video import every_other_video_frame
from .cache import TranslationFeatureCache
from .experiment import checkpoint_model_config
from .features import mmpose_translation_request
from .local_runtime import extractor_identity, write_overlay
from .model import LandmarkT5
from .schema import MMPOSE_SCHEMA
from .youtube import file_hash


def extract_video(
    *, video: Path, assets: dict[str, Path], output: Path, device: str = "cuda:0",
    threshold: float = 0.5, start_ms: int = 0, end_ms: int | None = None,
    overlay: Path | None = None,
) -> dict[str, Any]:
    import cv2

    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or count <= 0:
            raise ValueError("unreadable recorded video")
        duration = round(count * 1000 / fps)
    finally:
        capture.release()
    end_ms = duration if end_ms is None else end_ms
    if end_ms > duration:
        raise ValueError("sentence interval exceeds recording duration")
    sampled = every_other_video_frame(video, start_ms=start_ms, end_ms=end_ms)
    if not sampled.frames or len(sampled.frames) != sampled.requested_frames:
        raise ValueError("recording could not decode every requested frame")
    identity = extractor_identity(assets, threshold)
    if device.startswith("cuda"):
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    with MMPoseWholeBodyBackend(
        **assets, device=device, confidence_threshold=threshold
    ) as detector:
        frames = [detector.detect_translation(image, timestamp)
                  for image, timestamp in sampled.frames]
    request = mmpose_translation_request(frames)
    if overlay is not None:
        write_overlay(video, list(zip((image for image, _ in sampled.frames), frames, strict=True)),
                      overlay)
    elapsed = time.perf_counter() - started
    metadata = {"feature_schema": MMPOSE_SCHEMA, "extractor": identity,
                "source_video_hash": file_hash(video), "start_ms": start_ms, "end_ms": end_ms}
    TranslationFeatureCache(output).store("recorded", request, metadata=metadata)
    report = {**metadata, "sampled_frames": len(frames), "encoder_frames": len(request.values),
              "extraction_seconds": elapsed, "processing_fps": len(frames) / elapsed,
              "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
              if device.startswith("cuda") else 0}
    (output / "extraction.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


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
