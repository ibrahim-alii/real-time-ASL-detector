"""Standard-library-only persistence and resource guards for the Sol controller."""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .youtube import file_hash

GIB = 1024**3


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def tree_bytes(root: Path) -> int:
    for attempt in range(3):
        try:
            return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        except FileNotFoundError:
            # Atomic publishers may rename a temporary file during this scan.
            if attempt == 2:
                raise
    raise RuntimeError("storage scan did not complete")


def capacity(
    path: Path, required: int, *, remaining_quota: int, margin: float = 0.3
) -> dict[str, int]:
    if required < 0 or remaining_quota < 0 or not math.isfinite(margin) or margin < 0:
        raise ValueError("capacity arguments must be nonnegative")
    available = min(shutil.disk_usage(path).free, remaining_quota)
    needed = math.ceil(required * (1 + margin))
    if needed > available:
        raise ValueError(f"insufficient capacity: need {needed} bytes, available {available}")
    return {
        "required_bytes": required,
        "required_with_headroom": needed,
        "available_bytes": available,
    }


@contextmanager
def writer_lock(path: Path) -> Iterator[None]:
    """Kernel-owned lock: survives stale files, releases on process death."""
    fcntl = importlib.import_module("fcntl")  # Linux controller only

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another controller or stage writer is active") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def seal(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)}


def verify_seal(value: dict[str, Any]) -> bool:
    path = Path(value["path"])
    return (
        path.is_file()
        and path.stat().st_size == value["bytes"]
        and file_hash(path) == value["sha256"]
    )
