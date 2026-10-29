"""Shared development-set comparison metrics and worker requests."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ClipManifest, ManifestRecord
from .runtime_client import PersistentWorker


def development_rows(manifest: ClipManifest, limit: int) -> tuple[ManifestRecord, ...]:
    if not 1 <= limit <= 32:
        raise ValueError("comparison requires between 1 and 32 development examples")
    groups = [{r.video_id for r in manifest.accepted(split)} for split in ("train", "dev", "test")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("source recording overlaps held-out splits")
    rows = manifest.accepted("dev")[:limit]
    if not rows or any(row.source != "how2sign" for row in rows):
        raise ValueError("comparison requires accepted How2Sign development examples")
    return rows


def _scores(rows: list[dict[str, Any]], model: str) -> dict[str, float]:
    import sacrebleu

    outputs, refs = [r[model]["english"] for r in rows], [r["reference"] for r in rows]
    return {
        "bleu": float(sacrebleu.corpus_bleu(outputs, [refs]).score),
        "chrf": float(sacrebleu.corpus_chrf(outputs, [refs]).score),
    }


def comparison_report(
    rows: list[dict[str, Any]], *, bootstrap_samples: int = 200
) -> dict[str, Any]:
    if not rows or len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("comparison requires nonempty unique aligned clip IDs")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    models: dict[str, Any] = {}
    for model in ("baseline", "unisign"):
        latency = [float(r[model]["latency_ms"]) for r in rows]
        if any(not math.isfinite(value) or value < 0 for value in latency):
            raise ValueError("comparison latency must be finite and nonnegative")
        models[model] = {
            **_scores(rows, model),
            "latency_ms": {
                "p50": float(np.quantile(latency, 0.5)),
                "p95": float(np.quantile(latency, 0.95)),
            },
        }
    groups = sorted({r["video_id"] for r in rows})
    grouped = {key: [row for row in rows if row["video_id"] == key] for key in groups}
    rng = np.random.default_rng(42)
    differences: dict[str, list[float]] = {"bleu": [], "chrf": []}
    for _ in range(bootstrap_samples):
        sample = [
            row for key in rng.choice(groups, len(groups), replace=True) for row in grouped[key]
        ]
        first, second = _scores(sample, "baseline"), _scores(sample, "unisign")
        for metric in differences:
            differences[metric].append(second[metric] - first[metric])
    return {
        "examples": len(rows),
        "recordings": len(groups),
        "models": models,
        "paired_difference": {
            metric: {
                "estimate": models["unisign"][metric] - models["baseline"][metric],
                "lower": float(np.quantile(values, 0.025)),
                "upper": float(np.quantile(values, 0.975)),
            }
            for metric, values in differences.items()
        },
        "bootstrap": {"unit": "source recording", "samples": bootstrap_samples, "seed": 42},
        "semantic_review": "pending",
        "quality_validated": False,
        "test_examples_used": 0,
        "bleurt": {"status": "not_run", "reason": "optional scorer requires separate assets"},
        "limitations": [
            "Small engineering dev sample; not a benchmark or architecture ablation",
            "Released model and pilot have unequal training data and exposure",
            "Common RTMPose-M raw poses differ from the official online lightweight extractor",
            "Baseline uses half native frames; Uni-Sign uses native frames "
            "and official preprocessing",
            "Signer independence and external checkpoint training overlap are unverified",
        ],
        "samples": rows,
    }


def predict(worker: PersistentWorker, features: Path, *, timeout: float = 600) -> dict[str, Any]:
    started = time.perf_counter()
    result = worker.request({"features": str(features.resolve())}, timeout=timeout)
    return {**result, "request_ms": (time.perf_counter() - started) * 1000}

