"""Keep the small documentation set navigable."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    ROOT / "README.md",
    ROOT / "PRD.md",
    ROOT / "docs/REPLICATION_PROTOCOL.md",
    ROOT / "docs/SOL_JOB_GUIDE.md",
    ROOT / "docs/PILOT_PLAN.md",
    ROOT / "docs/TEST_DRIVEN_DEPLOYMENT.md",
    ROOT / "docs/SOL_TRANSLATION_PLAN.md",
)


def test_local_markdown_links_resolve() -> None:
    for document in DOCS:
        if document != ROOT / "README.md" and not document.exists():
            continue  # Extra local docs are deliberately excluded from Git.
        body = document.read_text(encoding="utf-8")
        for destination in re.findall(r"\]\(([^)]+)\)", body):
            if "://" in destination or destination.startswith("#"):
                continue
            path = (document.parent / destination.split("#", 1)[0]).resolve()
            assert path.exists(), f"{document}: {destination}"


@pytest.mark.skipif(
    not (ROOT / "docs/SOL_JOB_GUIDE.md").exists(), reason="optional local-only guide"
)
def test_beginner_guide_names_real_pilot_scripts_and_stop_gate() -> None:
    guide = (ROOT / "docs/SOL_JOB_GUIDE.md").read_text(encoding="utf-8")
    for name in (
        "preflight.sh",
        "prepare-pilot-env.sbatch",
        "download-pilot-shard.sbatch",
        "submit-pilot.sh",
        "estimate_training.py",
    ):
        assert (ROOT / "hpc" / name).is_file()
        assert name in guide
    assert "--check-only" in guide
    assert "Stop after the functional pilot" in guide
