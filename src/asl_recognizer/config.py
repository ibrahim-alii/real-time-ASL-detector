from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset_root: Path
    artifacts_root: Path
    seed: int = 42
    vocabulary_size: int = 25
    sequence_length: int = 32


def load_config(path: Path) -> AppConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    project = raw.get("project", {})
    required = {"dataset_root", "artifacts_root"}
    missing = required - project.keys()
    if missing:
        raise ValueError(f"missing config values: {', '.join(sorted(missing))}")
    config = AppConfig(
        dataset_root=Path(project["dataset_root"]),
        artifacts_root=Path(project["artifacts_root"]),
        seed=int(project.get("seed", 42)),
        vocabulary_size=int(project.get("vocabulary_size", 25)),
        sequence_length=int(project.get("sequence_length", 32)),
    )
    if config.vocabulary_size < 1 or config.sequence_length < 1:
        raise ValueError("vocabulary_size and sequence_length must be positive")
    return config

