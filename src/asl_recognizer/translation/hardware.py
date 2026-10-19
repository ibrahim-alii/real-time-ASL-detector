"""A synthetic memory/optimizer check, explicitly not an ASL experiment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .model import DirectTranslationConfig, LandmarkT5
from .schema import MMPOSE_SCHEMA


def check_training_gpu(
    *, steps: int = 10, frames: int = 256, precision: str = "bf16"
) -> dict[str, Any]:
    from transformers.optimization import Adafactor

    if steps < 1 or not 1 <= frames <= 256 or not torch.cuda.is_available():
        raise ValueError("GPU smoke check needs CUDA, positive steps and 1..256 frames")
    if precision not in {"fp16", "bf16"}:
        raise ValueError("GPU smoke precision must be fp16 or bf16")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    torch.empty(1, device=device)  # Initialize the allocator before resetting its counters.
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = LandmarkT5(DirectTranslationConfig(feature_schema=MMPOSE_SCHEMA)).to(device)
    model.t5.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
    model.t5.config.use_cache = False
    model.train()
    optimizer = Adafactor(  # type: ignore[no-untyped-call]
        model.parameters(), lr=0.001, scale_parameter=False, relative_step=False, warmup_init=False)
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    values = torch.rand((1, frames, 266), device=device)
    # Include the actual cache sentinel to expose instability from missing points.
    values[:, :, 182:190] = -1000
    attention = torch.ones((1, frames), dtype=torch.long, device=device)
    labels = torch.randint(2, 100, (1, 128), device=device)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16):
            loss = model(values, attention, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"synthetic T5-base {precision} loss is non-finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f"synthetic T5-base {precision} gradients are non-finite")
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    return {"purpose": "synthetic-hardware-smoke-not-ASL-quality", "optimizer_steps": steps,
            "frames": frames, "microbatch": 1, "precision": precision,
            "losses": losses, "seconds_including_model_load": time.perf_counter() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "weights_saved": False}


def write_failure(path: Path, error: Exception) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"purpose": "synthetic-hardware-smoke-not-ASL-quality",
                                "passed": False, "error": str(error)}, indent=2) + "\n",
                    encoding="utf-8")
