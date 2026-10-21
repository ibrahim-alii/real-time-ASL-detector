"""Dev-only released Uni-Sign comparison with shared references and bounded Sol workers."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ClipManifest, ManifestRecord, read_manifest
from .runtime_client import PersistentWorker
from .sol_common import write_json


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


def run_comparison(args: argparse.Namespace) -> None:
    from .runtime import require_sol

    require_sol(args.work_root)
    if not args.output.resolve().is_relative_to(args.work_root.resolve()):
        raise ValueError("comparison output must be within the Sol work root")
    if not 60 <= args.wall_time_seconds <= 3600:
        raise ValueError("comparison wall-time budget must be between 60 and 3600 seconds")
    manifest = read_manifest(args.manifest)
    rows = development_rows(manifest, args.max_clips)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + float(args.wall_time_seconds) * 0.8

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 5:
            raise TimeoutError("comparison reserve reached; preserve partial results")
        return min(600, value)

    features: list[Path] = []
    details: list[dict[str, Any]] = []
    src = str(Path(__file__).parents[2])
    worker_env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    extract_env = {**worker_env, "PYTHONPATH": src}
    initialization: dict[str, Any] = {}
    with ExitStack() as stack:
        log = stack.enter_context((args.output / "workers.log").open("w", encoding="utf-8"))
        tick = time.perf_counter()
        extractor = stack.enter_context(
            PersistentWorker(
                [
                    str(args.extraction_python),
                    "-m",
                    "asl_recognizer.translation.runtime",
                    "--role",
                    "extract",
                    "--work-root",
                    str(args.work_root),
                    "--assets-json",
                    str(args.assets_json),
                ],
                timeout=remaining(),
                env=extract_env,
                stderr=log,
            )
        )
        initialization["extractor_process_seconds"] = time.perf_counter() - tick
        tick = time.perf_counter()
        baseline = stack.enter_context(
            PersistentWorker(
                [
                    str(args.translation_python),
                    "-m",
                    "asl_recognizer.translation.runtime",
                    "--role",
                    "translate",
                    "--work-root",
                    str(args.work_root),
                    "--checkpoint",
                    str(args.baseline_checkpoint),
                ],
                timeout=remaining(),
                env=worker_env,
                stderr=log,
            )
        )
        initialization["baseline_process_seconds"] = time.perf_counter() - tick
        for row in rows:
            if Path(row.video_id).name != row.video_id:
                raise ValueError("recording ID cannot contain a path")
            video = args.recordings / "dev" / (row.video_id + ".mp4")
            target = args.output / "features" / hashlib.sha256(row.clip_id.encode()).hexdigest()
            extraction = extractor.request(
                {
                    "video": str(video.resolve()),
                    "output": str(target),
                    "start_ms": row.start_ms,
                    "end_ms": row.end_ms,
                    "native_pose": True,
                },
                timeout=remaining(),
            )
            features.append(target)
            result = predict(baseline, target, timeout=remaining())
            details.append(
                {
                    "clip_id": row.clip_id,
                    "video_id": row.video_id,
                    "reference": row.caption,
                    "baseline": result,
                    "extraction": extraction,
                }
            )
            write_json(args.output / "progress.json", {"status": "partial", "samples": details})
        repeated = [predict(baseline, features[0], timeout=remaining()) for _ in range(3)]
        if any(item["english"] != details[0]["baseline"]["english"] for item in repeated):
            raise ValueError("warm baseline generation differs on identical input")
        initialization["baseline_warm_repeat_ms"] = [item["request_ms"] for item in repeated]
        initialization["warm_prediction_parity"] = True
        initialization["resident_gpu_observation"] = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    with (args.output / "unisign.log").open("w", encoding="utf-8") as log:
        tick = time.perf_counter()
        with PersistentWorker(
            [
                str(args.translation_python),
                "-m",
                "asl_recognizer.translation.unisign",
                "--source",
                str(args.unisign_source),
                "--checkpoint",
                str(args.unisign_checkpoint),
                "--tokenizer",
                str(args.tokenizer),
                "--work-root",
                str(args.work_root),
            ],
            timeout=remaining(),
            env=worker_env,
            stderr=log,
        ) as unisign:
            initialization["unisign_process_seconds"] = time.perf_counter() - tick
            for detail, feature in zip(details, features, strict=True):
                detail["unisign"] = predict(unisign, feature, timeout=remaining())
                write_json(args.output / "progress.json", {"status": "partial", "samples": details})
    report = comparison_report(details)
    write_json(
        args.output / "comparison.json",
        {
            **report,
            "status": "completed",
            "manifest_hash": manifest.fingerprint,
            "job": os.environ["SLURM_JOB_ID"],
            "initialization": initialization,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "work-root",
        "manifest",
        "recordings",
        "assets-json",
        "baseline-checkpoint",
        "extraction-python",
        "translation-python",
        "unisign-source",
        "unisign-checkpoint",
        "tokenizer",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--wall-time-seconds", type=int, default=1800)
    run_comparison(parser.parse_args(argv))


if __name__ == "__main__":
    main()
