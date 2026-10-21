"""Memory-bounded local experiments with dev selection and resumable phases."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import TranslationFeatureCache
from .evaluation import evaluate_translations
from .manifest import read_manifest
from .model import T5_MODEL_ID, T5_REVISION, DirectTranslationConfig, LandmarkT5
from .run_metadata import collect_run_metadata, write_run_metadata
from .schema import MMPOSE_SCHEMA
from .training import _batch, _examples
from .youtube import file_hash


@dataclass(frozen=True)
class ExperimentConfig:
    feature_schema: str = MMPOSE_SCHEMA
    batch_size: int = 16
    microbatch_size: int = 1
    precision: str = "bf16"
    gradient_checkpointing: bool = True
    youtube_steps: int = 2000
    how2sign_steps: int = 500
    validation_interval: int = 250
    early_stopping_patience: int = 4
    checkpoint_interval: int = 250
    learning_rate: float = 0.001
    seed: int = 42
    input_encoding: str = "sentinel-v1"

    def __post_init__(self) -> None:
        positive = (
            self.batch_size,
            self.microbatch_size,
            self.how2sign_steps,
            self.validation_interval,
            self.early_stopping_patience,
            self.checkpoint_interval,
        )
        if min(positive) < 1 or self.youtube_steps < 0:
            raise ValueError("experiment counts must be positive (YouTube steps may be zero)")
        if self.batch_size % self.microbatch_size:
            raise ValueError("effective batch must divide into complete microbatches")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16")
        if self.input_encoding not in {"sentinel-v1", "mask-v1"}:
            raise ValueError("unsupported translation input encoding")
        if (
            self.feature_schema != MMPOSE_SCHEMA
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("local experiments require MMPose and a positive learning rate")

    @classmethod
    def read(cls, path: Path) -> ExperimentConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def atomic_checkpoint(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def selected_checkpoint(run: Path, state: dict[str, Any], source: str) -> Path:
    """Resolve the immutable selection committed by this resume state."""
    selection = state.get("best_checkpoints", {}).get(source)
    if selection is None:
        return run / f"best-{source}.pt"
    name = selection["file"]
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError("invalid selected checkpoint reference")
    path = run / name
    if file_hash(path) != selection["sha256"]:
        raise ValueError("selected checkpoint checksum mismatch")
    return path


def checkpoint_model_config(saved: dict[str, Any]) -> DirectTranslationConfig:
    config = saved["config"]
    return DirectTranslationConfig(
        pretrained_model=config.get("model", T5_MODEL_ID),
        pretrained_revision=config.get("model_revision", T5_REVISION),
        feature_schema=config["feature_schema"],
        beam_width=config.get("beam_width", 5),
        decoder_max_tokens=config.get("decoder_tokens", 128),
        input_encoding=config.get("input_encoding", "sentinel-v1"),
    )


def preprocessing_contract(paths: dict[str, Path], features: Path) -> dict[str, Any] | None:
    """Require one input contract across training and development, without opening test data."""
    contracts: dict[str, dict[str, Any] | None] = {}
    for source, path in paths.items():
        cache = TranslationFeatureCache(features / source)
        for row in read_manifest(path).accepted():
            if row.split not in {"train", "dev"}:
                continue
            _, metadata = cache.load(row.clip_id)
            current = metadata.get("preprocessing")
            if current is not None and (not isinstance(current, dict) or not current):
                raise ValueError("invalid preprocessing contract")
            contracts[json.dumps(current, sort_keys=True, allow_nan=False)] = current
    if len(contracts) != 1:
        raise ValueError("train and dev features require one consistent preprocessing contract")
    return next(iter(contracts.values()))


def _dev(
    model: Any,
    tokenizer: Any,
    manifest_path: Path,
    features: Path,
    source: str,
    schema: str,
    identity: dict[str, Any],
    device: torch.device,
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = read_manifest(manifest_path)
    rows = manifest.accepted("dev")
    if not rows:
        raise ValueError(f"{source} requires a separate dev split")
    cache = TranslationFeatureCache(features / source)
    losses: list[float] = []
    predictions = []
    model.eval()
    with torch.inference_mode():
        for row in rows:
            artifact = cache.find_valid(
                row.clip_id,
                expected_metadata={
                    "manifest_hash": manifest.fingerprint,
                    "source": source,
                    "split": "dev",
                    "caption": row.caption,
                    "feature_schema": schema,
                    "extractor": identity,
                    "preprocessing": preprocessing,
                },
            )
            if artifact is None:
                raise ValueError(f"missing or mismatched dev feature: {row.clip_id}")
            values, attention, labels = _batch(
                [(row.clip_id, row.caption)],
                cache,
                tokenizer,
                device,
                getattr(getattr(model, "config", None), "input_encoding", "sentinel-v1"),
            )
            loss = float(model(values, attention, labels).detach().cpu())
            if not math.isfinite(loss):
                raise RuntimeError("non-finite validation loss")
            losses.append(loss)
            predictions.append(model.translate(artifact[0], tokenizer, device=str(device)))
    metric = evaluate_translations(tuple(predictions), tuple(row.caption for row in rows))
    model.train()
    return {
        "dev_loss": sum(losses) / len(losses),
        "bleu": metric.bleu,
        "examples": [
            {"clip_id": row.clip_id, "reference": row.caption, "prediction": prediction.english}
            for row, prediction in zip(rows, predictions, strict=True)
        ],
    }


def run_experiment(
    *,
    features: Path,
    how2sign_manifest: Path,
    youtube_manifest: Path | None,
    output: Path,
    config: ExperimentConfig,
    job_steps: int | None = None,
    device_name: str | None = None,
    wall_time_seconds: float | None = None,
    strict_deadline: bool = False,
) -> None:
    """Steps count optimizer updates, never individual accumulation microbatches."""
    from transformers import AutoTokenizer
    from transformers.optimization import Adafactor

    if job_steps is not None and job_steps < 1:
        raise ValueError("job steps must be positive")
    if wall_time_seconds is not None and (
        not math.isfinite(wall_time_seconds) or wall_time_seconds <= 0
    ):
        raise ValueError("wall time budget must be finite and positive")
    allocation_started = time.monotonic()
    paths = {"how2sign": how2sign_manifest}
    phases = [("how2sign", config.how2sign_steps)]
    if config.youtube_steps:
        if youtube_manifest is None:
            raise ValueError("YouTube phase requires its frozen manifest")
        paths["youtube-asl"] = youtube_manifest
        phases.insert(0, ("youtube-asl", config.youtube_steps))
    loaded = {
        source: _examples(path, source, features, config.feature_schema)
        for source, path in paths.items()
    }
    identity = loaded["how2sign"][2]
    if not identity or any(item[2] != identity for item in loaded.values()):
        raise ValueError("all experiment corpora require the same recorded extractor")
    preprocessing = preprocessing_contract(paths, features)
    for manifest, _, _ in loaded.values():
        if manifest.accepted("unassigned"):
            raise ValueError("local experiments require assigned splits")
        groups = [{row.video_id for row in manifest.accepted(split)}
                  for split in ("train", "dev", "test")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("source recording spans train and held-out splits")
    hashes = {source: item[0].fingerprint for source, item in loaded.items()}
    run_config = {
        **asdict(config),
        "model": T5_MODEL_ID,
        "model_revision": T5_REVISION,
        "extractor": identity,
        "beam_width": 5,
        "decoder_tokens": 128,
        "loss_normalization": "valid-target-tokens-v1",
        "preprocessing": preprocessing,
        "preprocessing_status": "versioned" if preprocessing is not None else "legacy-unversioned",
        "exposure_tracking": "per-source-clips-and-target-tokens-v1",
    }
    device = torch.device(device_name or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu" and config.precision != "fp32":
        raise ValueError("CPU experiments must explicitly select fp32")
    if device.type == "cuda" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("selected GPU does not support bf16")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = LandmarkT5(checkpoint_model_config({"config": run_config})).to(device)
    if config.gradient_checkpointing:
        model.t5.gradient_checkpointing_enable()  # type: ignore[no-untyped-call]
        model.t5.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_ID, revision=T5_REVISION)
    optimizer = Adafactor(  # type: ignore[no-untyped-call]
        model.parameters(),
        lr=config.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.precision == "fp16"
    )
    rng = np.random.default_rng(config.seed)
    state: dict[str, Any] = {
        "mode": "experiment",
        "config": run_config,
        "manifest_hashes": hashes,
        "phase": 0,
        "phase_step": 0,
        "global_step": 0,
        "status": "running",
        "best_bleu": None,
        "bad_evaluations": 0,
        "history": [],
        "initial_dev": {},
        "optimizer_seconds": 0.0,
        "validation_seconds": 0.0,
        "source_timings": {},
        "checkpoint_seconds_max": 0.0,
        "peak_vram_bytes": 0,
        "best_checkpoints": {},
        "exposure": {},
    }
    latest = output / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["config"] != run_config or state["manifest_hashes"] != hashes:
            raise ValueError("resume identity differs from the saved experiment")
        if state["status"] == "completed":
            return
        for source in state.get("best_checkpoints", {}):
            selected = torch.load(selected_checkpoint(output, state, source),
                                  map_location="cpu", weights_only=True)
            atomic_checkpoint(selected, output / f"best-{source}.pt")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state = state["numpy_rng"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_run_metadata(
            output / "run-metadata.json",
            collect_run_metadata(
                seed=config.seed, manifest_hashes=hashes, container="local-experiment"
            ),
        )
        (output / "config.json").write_text(
            json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
        )

    def evaluate_dev(source: str) -> dict[str, Any]:
        started = time.monotonic()
        result = _dev(
            model,
            tokenizer,
            paths[source],
            features,
            source,
            config.feature_schema,
            identity,
            device,
            preprocessing=preprocessing,
        )
        elapsed = time.monotonic() - started
        state["validation_seconds"] = state.get("validation_seconds", 0.0) + elapsed
        timing = state.setdefault("source_timings", {}).setdefault(source, {})
        timing["validation_seconds_max"] = max(timing.get("validation_seconds_max", 0.0), elapsed)
        if device.type == "cuda":
            state["peak_vram_bytes"] = max(
                state["peak_vram_bytes"], torch.cuda.max_memory_allocated(device)
            )
        return result

    def write_checkpoint(path: Path) -> None:
        started = time.monotonic()
        atomic_checkpoint(state, path)
        state["checkpoint_seconds_max"] = max(
            state.get("checkpoint_seconds_max", 0.0), time.monotonic() - started
        )

    def save(selection_source: str | None = None) -> None:
        state.update(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            numpy_rng=rng.bit_generator.state,
            python_rng=random.getstate(),
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        )
        if selection_source is not None:
            # The resume pointer advances only after both selection files exist.
            path = output / f"selected-{selection_source}-{state['global_step']:09d}.pt"
            write_checkpoint(path)
            state.setdefault("best_checkpoints", {})[selection_source] = {
                "file": path.name, "sha256": file_hash(path),
            }
            write_checkpoint(output / f"best-{selection_source}.pt")
        write_checkpoint(latest)
        summary = {
            key: value
            for key, value in state.items()
            if key
            not in {
                "model",
                "optimizer",
                "scaler",
                "numpy_rng",
                "python_rng",
                "torch_rng",
                "cuda_rng",
            }
        }
        from .sol_common import write_json

        write_json(output / "progress.json", summary)

    completed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while state["phase"] < len(phases):
        phase = int(state["phase"])
        source, maximum = phases[phase]
        examples = loaded[source][1]
        cache = TranslationFeatureCache(features / source)
        if strict_deadline and wall_time_seconds is not None:
            timing = state.get("source_timings", {}).get(source, {})
            reserve = state.get("checkpoint_seconds_max", 0.0)
            reserve += timing.get("optimizer_seconds_max", 0.0)
            if (state["phase_step"] + 1) % config.validation_interval == 0:
                reserve += timing.get("validation_seconds_max", 0.0)
            if time.monotonic() - allocation_started + reserve >= wall_time_seconds:
                save()
                return
        if source not in state["initial_dev"]:
            state["initial_dev"][source] = evaluate_dev(source)
            state["best_bleu"] = state["initial_dev"][source]["bleu"]
            save(source)
            if (strict_deadline and wall_time_seconds is not None
                    and time.monotonic() - allocation_started >= wall_time_seconds):
                return
        model.train()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        indices = rng.integers(0, len(examples), size=config.batch_size)
        selected = [examples[int(index)] for index in indices]
        all_values, all_attention, all_labels = _batch(
            selected, cache, tokenizer, torch.device("cpu"), config.input_encoding
        )
        total_tokens = int(all_labels.ne(-100).sum())
        if total_tokens == 0:
            raise ValueError("training batch has no target tokens")
        mean_loss = 0.0
        for begin in range(0, config.batch_size, config.microbatch_size):
            stop = begin + config.microbatch_size
            labels = all_labels[begin:stop].to(device)
            weight = int(all_labels[begin:stop].ne(-100).sum()) / total_tokens
            values = all_values[begin:stop].to(device)
            attention = all_attention[begin:stop].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if config.precision == "fp16" else torch.bfloat16,
                enabled=config.precision != "fp32",
            ):
                loss = model(values, attention, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss; checkpoint not advanced")
            mean_loss += float(loss.detach().cpu()) * weight
            scaler.scale(loss * weight).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("non-finite gradients; checkpoint not advanced")
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            state["peak_vram_bytes"] = max(
                state["peak_vram_bytes"], torch.cuda.max_memory_allocated(device)
            )
        elapsed = time.perf_counter() - started
        state["optimizer_seconds"] += elapsed
        timing = state.setdefault("source_timings", {}).setdefault(source, {})
        timing["optimizer_seconds"] = timing.get("optimizer_seconds", 0.0) + elapsed
        timing["optimizer_updates"] = timing.get("optimizer_updates", 0) + 1
        timing["optimizer_seconds_max"] = max(timing.get("optimizer_seconds_max", 0.0), elapsed)
        state["global_step"] += 1
        state["phase_step"] += 1
        exposure = state["exposure"].setdefault(source, {
            "examples_seen": 0, "target_tokens": 0, "clip_counts": {}})
        exposure["examples_seen"] += len(selected)
        exposure["target_tokens"] += total_tokens
        for clip_id, _caption in selected:
            counts = exposure["clip_counts"]
            counts[clip_id] = counts.get(clip_id, 0) + 1
        exposure["unique_examples"] = len(exposure["clip_counts"])
        exposure["effective_epochs"] = exposure["examples_seen"] / len(examples)
        state["train_loss"] = mean_loss
        completed += 1
        end_phase = state["phase_step"] >= maximum
        if state["phase_step"] % config.validation_interval == 0 or end_phase:
            result = evaluate_dev(source)
            state["history"].append(
                {"source": source, "step": state["global_step"], "train_loss": mean_loss, **result}
            )
            if state["best_bleu"] is None or result["bleu"] > state["best_bleu"]:
                state["best_bleu"] = result["bleu"]
                state["bad_evaluations"] = 0
                save(source)
            else:
                state["bad_evaluations"] += 1
            end_phase = end_phase or state["bad_evaluations"] >= config.early_stopping_patience
        if end_phase:
            # Start fine-tuning from the selected YouTube checkpoint, with fresh
            # optimizer moments. Its optimizer state is not a resumable phase state.
            best = torch.load(selected_checkpoint(output, state, source),
                              map_location="cpu", weights_only=True)
            model.load_state_dict(best["model"])
            state["selected_step"] = best["global_step"]
            state["selected_bleu"] = best["best_bleu"]
            state["phase"] += 1
            state["phase_step"] = 0
            state["best_bleu"] = None
            state["bad_evaluations"] = 0
            optimizer.state.clear()
            if state["phase"] == len(phases):
                state["status"] = "completed"
        time_exhausted = (
            wall_time_seconds is not None
            and time.monotonic() - allocation_started >= wall_time_seconds
        )
        if (
            end_phase
            or state["global_step"] % config.checkpoint_interval == 0
            or time_exhausted
            or (job_steps is not None and completed >= job_steps)
        ):
            save()
        if time_exhausted or (job_steps is not None and completed >= job_steps):
            return


def finalize_experiment(run: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    latest = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    if latest.get("mode") != "experiment" or latest.get("status") != "completed":
        raise ValueError("only a completed, dev-selected experiment can be finalized")
    selected_path = selected_checkpoint(run, latest, "how2sign")
    selected = torch.load(selected_path, map_location="cpu", weights_only=True)
    if (
        selected["config"] != latest["config"]
        or selected["manifest_hashes"] != latest["manifest_hashes"]
    ):
        raise ValueError("selected checkpoint identity differs from completed run")
    selected = {
        key: value
        for key, value in selected.items()
        if key not in {"optimizer", "scaler", "numpy_rng", "python_rng", "torch_rng", "cuda_rng"}
    }
    selected.update(
        status="finalized", selected_checkpoint_sha256=file_hash(selected_path)
    )
    atomic_checkpoint(selected, output)
