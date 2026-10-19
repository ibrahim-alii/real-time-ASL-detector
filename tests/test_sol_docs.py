"""The public README must not link to excluded local files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_local_markdown_links_resolve() -> None:
    document = ROOT / "README.md"
    for destination in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
        if "://" in destination or destination.startswith("#"):
            continue
        path = (document.parent / destination.split("#", 1)[0]).resolve()
        assert path.exists(), f"{document}: {destination}"
