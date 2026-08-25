from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    labels: tuple[str, ...]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: npt.NDArray[np.int64]
    per_sign_accuracy: dict[str, float]
    per_signer_accuracy: dict[str, float]


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    signer_ids: Sequence[str],
) -> ClassificationReport:
    if not (len(y_true) == len(y_pred) == len(signer_ids)):
        raise ValueError("y_true, y_pred, and signer_ids must have the same length")
    if not y_true:
        raise ValueError("classification inputs cannot be empty")
    ordered_labels = tuple(labels)
    if len(ordered_labels) != len(set(ordered_labels)):
        raise ValueError("labels must be unique")
    label_index = {label: index for index, label in enumerate(ordered_labels)}
    unknown = (set(y_true) | set(y_pred)) - set(ordered_labels)
    if unknown:
        raise ValueError(f"unknown labels: {', '.join(sorted(unknown))}")

    matrix = np.zeros((len(ordered_labels), len(ordered_labels)), dtype=np.int64)
    signer_correct: CounterLike = {}
    signer_total: CounterLike = {}
    for truth, prediction, signer in zip(y_true, y_pred, signer_ids, strict=True):
        matrix[label_index[truth], label_index[prediction]] += 1
        signer_total[signer] = signer_total.get(signer, 0) + 1
        signer_correct[signer] = signer_correct.get(signer, 0) + int(truth == prediction)

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    per_sign: dict[str, float] = {}
    for index, label in enumerate(ordered_labels):
        true_positive = int(matrix[index, index])
        precision = _safe_divide(true_positive, int(matrix[:, index].sum()))
        recall = _safe_divide(true_positive, int(matrix[index, :].sum()))
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(_safe_divide(2 * precision * recall, precision + recall))
        per_sign[label] = recall

    return ClassificationReport(
        labels=ordered_labels,
        accuracy=_safe_divide(int(np.trace(matrix)), int(matrix.sum())),
        macro_precision=float(np.mean(precisions)),
        macro_recall=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1_scores)),
        confusion_matrix=matrix,
        per_sign_accuracy=per_sign,
        per_signer_accuracy={
            signer: _safe_divide(signer_correct[signer], total)
            for signer, total in sorted(signer_total.items())
        },
    )


CounterLike = dict[str, int]

