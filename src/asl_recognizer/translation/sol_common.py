"""Standard-library-only persistence and resource guards for the Sol controller."""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .youtube import file_hash

GIB = 1024**3


def atomic_replace(temporary: Path, destination: Path) -> None:
    """Windows readers/scanners can briefly deny replacement of a closed file."""
    for attempt in range(6):
        try:
            temporary.replace(destination)
            return
        except PermissionError as error:
            if (sys.platform != "win32" or getattr(error, "winerror", None) not in {5, 32, 33}
                    or attempt == 5):
                raise
            time.sleep(0.02 * 2**attempt)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)


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


def scan_tree_bytes(root: Path) -> int:
    """Count the same file bytes with cached directory-entry metadata and no dir symlinks."""
    def size(directory: Path) -> int:
        total = 0
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    total += size(Path(entry.path))
                elif entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except FileNotFoundError:
                        if not entry.name.endswith(".video-only.mp4"):
                            raise
                        # Remux publication renames this file over the MP4; double counting is safe.
                        published = Path(entry.path).with_name(
                            entry.name.removesuffix(".video-only.mp4") + ".mp4"
                        )
                        total += published.stat().st_size
        return total

    for attempt in range(3):
        try:
            return size(root)
        except FileNotFoundError:
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


def _seal_stat(path: Path) -> list[int]:
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def seal(path: Path) -> dict[str, object]:
    before = _seal_stat(path)
    stable = sys.platform == "linux" and time.time_ns() - before[4] >= 2_000_000_000
    digest = file_hash(path)
    if _seal_stat(path) != before:
        raise ValueError("file changed while sealing")
    result: dict[str, object] = {"path": str(path), "bytes": before[2], "sha256": digest}
    if stable:
        result["verified_stat"] = {"identity": before, "sha256": digest, "version": 2}
    return result


def verify_seal(value: dict[str, Any], *, cached: bool = False) -> bool:
    path = Path(value["path"])
    try:
        if not path.is_file():
            return False
        for attempt in range(2):
            before = _seal_stat(path)
            if before[2] != value["bytes"]:
                return False
            signature = {"identity": before, "sha256": value["sha256"], "version": 2}
            # BeeGFS timestamps are whole seconds: certify only after that interval has settled.
            stable = sys.platform == "linux" and time.time_ns() - before[4] >= 2_000_000_000
            if attempt == 0 and cached and stable and value.get("verified_stat") == signature:
                return True
            if file_hash(path) != value["sha256"]:
                return False
            if _seal_stat(path) != before:
                if not cached:
                    return False
                # A first read can update BeeGFS metadata; rehash once against the new identity.
                continue
            if stable:
                value["verified_stat"] = signature
            else:
                value.pop("verified_stat", None)
            return True
        return False
    except FileNotFoundError:
        return False
