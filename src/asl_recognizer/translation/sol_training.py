"""Real-feature smoke/resume, microbatch measurements and dev-only input controls."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .cache import TranslationFeatureCache
from .experiment import ExperimentConfig, checkpoint_model_config, run_experiment
from .manifest import read_manifest
from .model import LandmarkT5
from .sol import remaining
from .sol_common import GIB, capacity, read_json, tree_bytes, verify_seal, write_json
from .subsets import subset_manifest

if TYPE_CHECKING:
    from .sol_pipeline import Pipeline


def collect() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def smoke(pipeline: Pipeline) -> None:
    manifest_path = pipeline.manifests / "how2sign-smoke.csv"
    if not manifest_path.exists():
        subset_manifest(
            pipeline.manifests / "how2sign.csv", manifest_path, train_limit=32, dev_limit=8
        )
    original = read_manifest(pipeline.manifests / "how2sign.csv")
    manifest = read_manifest(manifest_path)
    source = TranslationFeatureCache(pipeline.features / "how2sign")
    target_root = pipeline.scratch / "smoke-features"
    target = TranslationFeatureCache(target_root / "how2sign")
    for row in manifest.accepted():
        if row.split not in {"train", "dev"}:
            continue
        artifact = source.find_valid(
            row.clip_id, expected_metadata={"manifest_hash": original.fingerprint}
        )
        if artifact is None:
            raise ValueError("smoke source feature missing")
        target.store(
            row.clip_id,
            artifact[0],
            metadata={**artifact[1], "manifest_hash": manifest.fingerprint},
        )
    config = ExperimentConfig(
        youtube_steps=0, how2sign_steps=10, validation_interval=5, checkpoint_interval=5
    )
    config_path = pipeline.root / "smoke-config.json"
    write_json(config_path, asdict(config))
    for name, steps in (("uninterrupted", 10), ("resumed", 5), ("resumed", 5)):
        pipeline.worker(
            pipeline.train,
            "train-local",
            "--features",
            target_root,
            "--how2sign-manifest",
            manifest_path,
            "--config",
            config_path,
            "--output",
            pipeline.root / "smoke" / name,
            "--job-steps",
            steps,
        )
    full = torch.load(
        pipeline.root / "smoke/uninterrupted/latest.pt", weights_only=True, map_location="cpu"
    )
    resumed = torch.load(
        pipeline.root / "smoke/resumed/latest.pt", weights_only=True, map_location="cpu"
    )
    if full["status"] != "completed" or resumed["global_step"] != full["global_step"]:
        raise ValueError("real-feature interrupted resume did not complete")
    for key, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    write_json(
        pipeline.reports / "smoke.json",
        {"passed": True, "steps": full["global_step"], "resume_model_exact": True},
    )


def benchmark(pipeline: Pipeline) -> None:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError("Sol training benchmark requires a native BF16 CUDA GPU")
    measurements: list[dict[str, Any]] = []
    total_memory = torch.cuda.get_device_properties(0).total_memory
    base = ExperimentConfig(
        youtube_steps=25, how2sign_steps=25, validation_interval=25, checkpoint_interval=25
    )
    for size in (1, 2, 4, 8):
        config = replace(base, microbatch_size=size)
        run = pipeline.scratch / "benchmarks" / f"microbatch-{size}"
        started = time.monotonic()
        try:
            run_experiment(
                features=pipeline.features,
                how2sign_manifest=pipeline.manifests / "how2sign.csv",
                youtube_manifest=pipeline.manifests / "youtube.csv",
                config=config,
                output=run,
            )
            progress = read_json(run / "progress.json")
            timings = progress["source_timings"]
            if set(timings) != {"youtube-asl", "how2sign"}:
                raise ValueError("benchmark must measure both corpora")
            update_seconds = max(
                t["optimizer_seconds"] / t["optimizer_updates"] for t in timings.values()
            )
            validation_seconds = max(t["validation_seconds_max"] for t in timings.values())
            checkpoint_seconds = progress["checkpoint_seconds_max"]
            if not all(math.isfinite(t) and t > 0 for t in (
                update_seconds, validation_seconds, checkpoint_seconds,
            )):
                raise ValueError("benchmark timings must be finite and positive")
            record = {
                "microbatch": size,
                "passed": True,
                "source_timings": timings,
                "seconds_per_update": update_seconds,
                "seconds_per_validation": validation_seconds,
                "seconds_per_checkpoint": checkpoint_seconds,
                "wall_seconds": time.monotonic() - started,
                "peak_vram_bytes": progress["peak_vram_bytes"],
                "total_vram_bytes": total_memory,
            }
            record["headroom_passed"] = record["peak_vram_bytes"] <= 0.75 * total_memory
        except torch.cuda.OutOfMemoryError as error:
            record = {"microbatch": size, "passed": False, "error": str(error)}
        measurements.append(record)
        write_json(pipeline.reports / "benchmark.json", {"candidates": measurements})
        collect()
    valid = [r for r in measurements if r["passed"] and r["headroom_passed"]]
    if not valid:
        raise ValueError("no stable microbatch with 25% VRAM headroom")
    # Effective batch size is fixed: faster updates mean more useful examples/sec.
    chosen = min(valid, key=lambda r: (r["seconds_per_update"], -r["microbatch"]))
    allocation = pipeline.config["allocation_seconds"] * 0.8
    # Cover phase-entry/end selection writes as well as the regular 50-update saves.
    usable = (
        allocation - 2 * chosen["seconds_per_validation"] - 5 * chosen["seconds_per_checkpoint"]
    )
    update_cost = chosen["seconds_per_update"] + chosen["seconds_per_checkpoint"] / 50
    if usable <= update_cost:
        raise ValueError("allocation too short for measured validation and optimizer work")
    job_steps = max(1, min(250, int(usable / update_cost)))
    config = ExperimentConfig(microbatch_size=chosen["microbatch"], checkpoint_interval=50)
    write_json(pipeline.root / "training-config.json", asdict(config))
    write_json(
        pipeline.reports / "training-budget.json",
        {
            **chosen,
            "job_steps": job_steps,
            "wall_time_seconds": usable,
            "estimated_training_hours": 2500 * update_cost / 3600,
            "estimated_validation_hours": 14 * chosen["seconds_per_validation"] / 3600,
            "validation_estimate_includes": "12 baseline evaluations and two input controls",
            "scratch_bytes": tree_bytes(pipeline.scratch),
            "durable_bytes": tree_bytes(pipeline.root),
            "timing_policy": "slower corpus, maximum validation/checkpoint, 20% wall reserve",
            "limitations": "Both corpora measured; batch composition and queue delays may differ",
        },
    )


def train(pipeline: Pipeline, *, ablation: bool = False) -> str | None:
    from .youtube import file_hash, verify_snapshot

    snapshot = read_json(pipeline.root / "selected-snapshot.json")
    path = Path(snapshot["path"])
    if file_hash(path / "snapshot.json") != snapshot["sha256"]:
        raise ValueError("selected YouTube snapshot metadata changed")
    verify_snapshot(path / "snapshot.json", pipeline.scratch / "youtube-asl")
    for row in read_manifest(pipeline.manifests / "how2sign.csv").accepted():
        if row.split in {"train", "dev"}:
            marker = pipeline.scratch / "how2sign/recordings" / row.split / (row.video_id + ".json")
            if not marker.exists() or not verify_seal(read_json(marker)):
                raise ValueError("selected How2Sign recording changed or is unavailable")
    if ablation and not read_json(pipeline.reports / "input-diagnostic.json")["ablation_eligible"]:
        write_json(
            pipeline.reports / "ablation.json",
            {
                "status": "skipped",
                "reason": "baseline did not improve dev BLEU and demonstrate visual-input benefit",
            },
        )
        return None
    capacity(pipeline.root, 10 * GIB, remaining_quota=remaining(pipeline.config, "home"))
    config = ExperimentConfig.read(pipeline.root / "training-config.json")
    if ablation:
        config = replace(config, input_encoding="mask-v1")
    budget = read_json(pipeline.reports / "training-budget.json")
    run = pipeline.root / "runs" / ("mask-ablation" if ablation else "baseline")
    run_experiment(
        features=pipeline.features,
        how2sign_manifest=pipeline.manifests / "how2sign.csv",
        youtube_manifest=pipeline.manifests / "youtube.csv",
        config=config,
        output=run,
        job_steps=budget["job_steps"],
        wall_time_seconds=budget["wall_time_seconds"],
    )
    return (
        None
        if read_json(run / "progress.json")["status"] == "completed"
        else ("ablation" if ablation else "train")
    )


def diagnostic(pipeline: Pipeline) -> None:
    from transformers import AutoTokenizer

    from .evaluation import evaluate_translations

    saved = torch.load(
        pipeline.root / "runs/baseline/selected.pt", weights_only=True, map_location="cpu"
    )
    config = checkpoint_model_config(saved)
    model = LandmarkT5(config).to("cuda:0")
    model.load_state_dict(saved["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model, revision=config.pretrained_revision
    )
    rows = read_manifest(pipeline.manifests / "how2sign.csv").accepted("dev")
    if len(rows) < 2:
        raise ValueError("shuffled-input control requires at least two dev pairs")
    cache = TranslationFeatureCache(pipeline.features / "how2sign")
    requests = [cache.load(row.clip_id)[0] for row in rows]
    # Fixed non-identity permutation: every reference gets another clip's landmarks.
    outputs = [
        model.translate(requests[(i + 1) % len(rows)], tokenizer, device="cuda:0")
        for i in range(len(rows))
    ]
    shuffled = evaluate_translations(tuple(outputs), tuple(row.caption for row in rows)).bleu
    actual = read_json(pipeline.root / "runs/baseline/dev/report.json")["bleu"]
    initial = saved["initial_dev"]["how2sign"]["bleu"]
    write_json(
        pipeline.reports / "input-diagnostic.json",
        {
            "actual_bleu": actual,
            "shuffled_bleu": shuffled,
            "initial_bleu": initial,
            "ablation_eligible": actual > initial and actual > shuffled,
            "semantic_quality_verified": False,
            "control": "cyclic-permutation-of-dev-clips-v1",
        },
    )
