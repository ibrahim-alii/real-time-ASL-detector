"""Regression: pretrained inference must not load the legacy trainer on import."""
import subprocess
import sys


def test_unisign_import_does_not_import_legacy_experiment():
    code = (
        "import sys; import asl_recognizer.translation.unisign; "
        "assert 'asl_recognizer.translation.experiment' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=90)
