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


def test_worker_retires_legacy_gpu_check():
    from asl_recognizer.translation import worker

    commands = worker._parser()._subparsers._group_actions[0].choices
    retired = {"check-training-gpu", "train-local", "finalize", "infer-video", "evaluate"}
    assert not retired & commands.keys()
    assert "runtime-check" in commands
