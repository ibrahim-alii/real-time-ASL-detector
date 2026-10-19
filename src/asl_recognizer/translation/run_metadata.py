from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


@dataclass(frozen=True, slots=True)
class RunMetadata:
    git_revision: str
    seed: int
    manifest_hashes: dict[str, str]
    container: str
    container_sha256: str | None
    python: str
    torch: str
    cuda: str | None
    dependencies: dict[str, str]


def collect_run_metadata(
    *, seed: int, manifest_hashes: dict[str, str], container: str
) -> RunMetadata:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    packages = (
        "mmpose", "mmcv", "mmengine", "mmdet", "numpy", "opencv-python",
        "transformers", "sacrebleu", "bleurt-pytorch"
    )
    dependencies: dict[str, str] = {}
    for package in packages:
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "unavailable"
    return RunMetadata(
        revision,
        seed,
        dict(sorted(manifest_hashes.items())),
        container,
        os.environ.get("ASL_CONTAINER_SHA256"),
        platform.python_version(),
        torch.__version__,
        torch.version.cuda,
        dependencies,
    )


def write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
