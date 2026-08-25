from __future__ import annotations

import numpy as np
import pytest

from asl_recognizer.evaluation.metrics import classification_report


def test_classification_report_uses_stable_labels_and_per_signer_accuracy() -> None:
    report = classification_report(
        y_true=["A", "B", "B", "A"],
        y_pred=["A", "A", "B", "A"],
        labels=["A", "B", "C"],
        signer_ids=["s1", "s1", "s2", "s2"],
    )

    assert report.accuracy == 0.75
    assert report.confusion_matrix.tolist() == [[2, 0, 0], [1, 1, 0], [0, 0, 0]]
    assert report.per_sign_accuracy == {"A": 1.0, "B": 0.5, "C": 0.0}
    assert report.per_signer_accuracy == {"s1": 0.5, "s2": 1.0}
    assert np.isclose(report.macro_recall, 0.5)


def test_classification_report_rejects_length_mismatch() -> None:
    try:
        classification_report(["A"], [], ["A"], ["s1"])
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("expected ValueError")


@pytest.mark.parametrize(
    ("truth", "prediction", "labels", "message"),
    [
        ([], [], ["A"], "cannot be empty"),
        (["A"], ["A"], ["A", "A"], "unique"),
        (["A"], ["B"], ["A"], "unknown labels"),
    ],
)
def test_classification_report_rejects_invalid_inputs(
    truth: list[str], prediction: list[str], labels: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        classification_report(truth, prediction, labels, ["s1"] * len(truth))
