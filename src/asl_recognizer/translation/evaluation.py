from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from .contracts import DirectTranslationResult

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

