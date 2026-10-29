"""The extraction CLI must not import legacy model evaluation at startup."""
import subprocess
import sys


def test_worker_import_avoids_legacy_evaluation():
    code = (
        "import sys; import asl_recognizer.translation.worker; "
        "assert 'asl_recognizer.translation.evaluation' not in sys.modules; "
        "assert 'asl_recognizer.translation.model' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90)


def test_worker_evaluate_dispatches_to_evaluation(monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock

    from asl_recognizer.translation import evaluation, worker

    evaluate = MagicMock()
    monkeypatch.setattr(evaluation, "evaluate_checkpoint", evaluate)
    status = worker.main([
        "evaluate", "--checkpoint", "checkpoint", "--manifest", "manifest",
        "--features", "features", "--output", "output", "--split", "dev",
        "--metrics", "basic",
    ])
    assert status == 0
    evaluate.assert_called_once_with(
        checkpoint=Path("checkpoint"), manifest_path=Path("manifest"),
        features=Path("features"), output=Path("output"), split="dev", with_bleurt=False,
    )
