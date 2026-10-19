from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .cache import TranslationFeatureCache
from .contracts import DirectTranslationResult
from .manifest import read_manifest
from .model import T5_MODEL_ID, T5_REVISION, DirectTranslationConfig, LandmarkT5
from .schema import MMPOSE_SCHEMA

BLEURT_MODEL_ID = "lucadiliello/BLEURT-20"
BLEURT_REVISION = "e22b9eb071dcb939da9fff08b20a7e746c72a830"


@dataclass(frozen=True, slots=True)
class TranslationEvaluation:
    bleu: float
    bleurt: float | None
    exact_match: float
    mean_latency_ms: float
    failures: tuple[dict[str, str], ...]


def evaluate_translations(
    predictions: tuple[DirectTranslationResult, ...],
    references: tuple[str, ...],
    *,
    bleurt_scorer: Callable[[list[str], list[str]], list[float]] | None = None,
) -> TranslationEvaluation:
    if not predictions or len(predictions) != len(references):
        raise ValueError("predictions and references must be non-empty and aligned")
    outputs = [item.english for item in predictions]
    try:
        import sacrebleu
    except ImportError as error:
        raise RuntimeError("install asl-recognizer[hpc] for SacreBLEU v2 evaluation") from error
    bleu = float(sacrebleu.corpus_bleu(outputs, [list(references)]).score)
    matches = [
        output.strip().casefold() == reference.strip().casefold()
        for output, reference in zip(outputs, references, strict=True)
    ]
    scores = bleurt_scorer(outputs, list(references)) if bleurt_scorer else None
    if scores is not None and len(scores) != len(outputs):
        raise ValueError("BLEURT scorer returned an unexpected number of scores")
    return TranslationEvaluation(
        bleu=bleu,
        bleurt=(100 * sum(scores) / len(scores) if scores else None),
        exact_match=sum(matches) / len(matches),
        mean_latency_ms=sum(item.latency_ms for item in predictions) / len(predictions),
        failures=tuple(
            {"reference": reference, "prediction": output}
            for output, reference, matched in zip(outputs, references, matches, strict=True)
            if not matched
        ),
    )


def evaluation_report(evaluation: TranslationEvaluation) -> dict[str, object]:
    """Machine-readable report with reference values explicitly not treated as targets."""
    return {**asdict(evaluation), "published_bleu_reference_points": [1.22, 3.95, 11.89, 12.39]}


def _bleurt_scores(candidates: list[str], references: list[str], device: str) -> list[float]:
    try:
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    except ImportError as error:
        raise RuntimeError("install asl-recognizer[hpc] for BLEURT-20 evaluation") from error
    scorer = BleurtForSequenceClassification.from_pretrained(
        BLEURT_MODEL_ID, revision=BLEURT_REVISION
    ).to(device).eval()
    tokenizer = BleurtTokenizer.from_pretrained(BLEURT_MODEL_ID, revision=BLEURT_REVISION)
    values: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(candidates), 8):
            batch = tokenizer(
                references[start : start + 8],
                candidates[start : start + 8],
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            values.extend(
                scorer(**{key: value.to(device) for key, value in batch.items()})
                .logits.flatten()
                .tolist()
            )
    return values


def evaluate_checkpoint(
    *, checkpoint: Path, manifest_path: Path, features: Path, output: Path, split: str,
    with_bleurt: bool = True,
) -> None:
    """Monitor dev; score test only after experiment finalization."""
    from transformers import AutoTokenizer

    manifest = read_manifest(manifest_path)
    if split not in {"dev", "test"}:
        raise ValueError("evaluation split must be dev or test")
    rows = manifest.accepted(split)
    if not rows or any(record.source != "how2sign" for record in rows):
        raise ValueError(f"official How2Sign {split} manifest has no valid accepted clips")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["mode"] != "experiment" or (
        split == "test" and saved.get("status") != "finalized"
    ):
        raise ValueError("test evaluation requires a finalized experiment")
    if saved["manifest_hashes"]["how2sign"] != manifest.fingerprint:
        raise ValueError("checkpoint and official How2Sign manifest differ")
    report_path = output / "report.json"
    if report_path.exists():
        raise FileExistsError("sealed evaluation report already exists")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schema = saved["config"]["feature_schema"]
    config = DirectTranslationConfig(
        pretrained_model=saved["config"].get("model", T5_MODEL_ID),
        pretrained_revision=saved["config"].get("model_revision", T5_REVISION),
        feature_schema=schema, beam_width=saved["config"].get("beam_width", 5),
        decoder_max_tokens=saved["config"].get("decoder_tokens", 128),
        input_encoding=saved["config"].get("input_encoding", "sentinel-v1"),
    )
    model = LandmarkT5(config).to(device)
    model.load_state_dict(saved["model"])
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model,
                                              revision=config.pretrained_revision)
    cache = TranslationFeatureCache(features / "how2sign")
    predictions: list[DirectTranslationResult] = []
    references: list[str] = []
    details: list[dict[str, object]] = []
    for record in rows:
        expected: dict[str, object] = {
            "manifest_hash": manifest.fingerprint, "feature_schema": schema,
            "caption": record.caption, "source": "how2sign", "split": split,
        }
        if schema == MMPOSE_SCHEMA:
            expected["extractor"] = saved["config"]["extractor"]
        artifact = cache.find_valid(
            record.clip_id,
            expected_metadata=expected,
        )
        if artifact is None:
            raise ValueError(f"missing or mismatched test feature: {record.clip_id}")
        result = model.translate(artifact[0], tokenizer, device=device)
        predictions.append(result)
        references.append(record.caption)
        details.append(
            {
                "clip_id": record.clip_id,
                "reference": record.caption,
                "prediction": result.english,
                "latency_ms": result.latency_ms,
                "decoder_score": result.decoder_score,
            }
        )
    metric = evaluate_translations(
        tuple(predictions),
        tuple(references),
        bleurt_scorer=(lambda outputs, refs: _bleurt_scores(outputs, refs, device))
        if with_bleurt else None,
    )
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "report.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                **evaluation_report(metric),
                "checkpoint": str(checkpoint),
                "manifest_hash": manifest.fingerprint,
                "split": split,
                "feature_schema": schema,
                "bleurt_model": BLEURT_MODEL_ID if with_bleurt else None,
                "bleurt_revision": BLEURT_REVISION if with_bleurt else None,
                "container_sha256": os.environ.get("ASL_CONTAINER_SHA256"),
                "bleurt_scale": 100,
                "examples": details,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
