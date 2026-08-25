from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from .models import (
    AnnotatedInterval,
    CalibrationKind,
    CalibrationSession,
    CalibrationSplit,
)

_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CalibrationRecorder:
    """Collect local annotation metadata without retaining frames or landmarks."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory
        self._active: tuple[str, CalibrationSplit, CalibrationKind, int] | None = None
        self._annotations: list[AnnotatedInterval] = []

    def start(
        self,
        session_id: str,
        *,
        split: CalibrationSplit,
        kind: CalibrationKind,
        started_at_ms: int,
    ) -> None:
        if self._active is not None:
            raise RuntimeError("a calibration session is already active")
        if not _SAFE_SESSION_ID.fullmatch(session_id):
            raise ValueError("session ID contains unsafe characters")
        if split not in ("calibration", "evaluation"):
            raise ValueError("invalid calibration split")
        if kind not in ("idle", "unrelated_motion", "transitions"):
            raise ValueError("invalid calibration recording kind")
        if started_at_ms < 0:
            raise ValueError("session start must be non-negative")
        self._active = (session_id, split, kind, started_at_ms)
        self._annotations = []

    def annotate(self, label: str, *, start_ms: int, end_ms: int) -> None:
        if self._active is None:
            raise RuntimeError("no calibration session is active")
        annotation = AnnotatedInterval(label, start_ms, end_ms)
        started_at_ms = self._active[3]
        if annotation.start_ms < started_at_ms:
            raise ValueError("annotation begins before the session")
        if self._annotations and annotation.start_ms < self._annotations[-1].end_ms:
            raise ValueError("annotations cannot overlap")
        self._annotations.append(annotation)

    def finish(self, *, media_filename: str, ended_at_ms: int) -> Path:
        if self._active is None:
            raise RuntimeError("no calibration session is active")
        media = Path(media_filename)
        if media.name != media_filename or media.is_absolute():
            raise ValueError("media filename must be a plain filename")
        session_id, split, kind, started_at_ms = self._active
        session = CalibrationSession(
            session_id=session_id,
            split=cast(CalibrationSplit, split),
            kind=cast(CalibrationKind, kind),
            started_at_ms=started_at_ms,
            ended_at_ms=ended_at_ms,
            media_filename=media_filename,
            annotations=tuple(self._annotations),
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        destination = self.output_directory / f"{session_id}.json"
        session.to_json(destination)
        self._active = None
        self._annotations = []
        return destination

    def cancel(self) -> None:
        self._active = None
        self._annotations = []
