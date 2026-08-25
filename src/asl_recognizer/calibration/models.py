from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

CalibrationSplit = Literal["calibration", "evaluation"]
CalibrationKind = Literal["idle", "unrelated_motion", "transitions"]


@dataclass(frozen=True, slots=True)
class AnnotatedInterval:
    label: str
    start_ms: int
    end_ms: int

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("annotation label cannot be blank")
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("annotation interval must have a non-negative start before its end")


@dataclass(frozen=True, slots=True)
class CalibrationSession:
    session_id: str
    split: CalibrationSplit
    kind: CalibrationKind
    started_at_ms: int
    ended_at_ms: int
    media_filename: str
    annotations: tuple[AnnotatedInterval, ...] = ()

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session ID cannot be blank")
        if self.split not in ("calibration", "evaluation"):
            raise ValueError("invalid calibration split")
        if self.kind not in ("idle", "unrelated_motion", "transitions"):
            raise ValueError("invalid calibration recording kind")
        if self.started_at_ms < 0 or self.ended_at_ms <= self.started_at_ms:
            raise ValueError("session must have a non-negative start before its end")
        media = Path(self.media_filename)
        if media.name != self.media_filename or media.is_absolute():
            raise ValueError("media filename must be a plain filename")
        previous_end = self.started_at_ms
        for annotation in self.annotations:
            if annotation.start_ms < self.started_at_ms or annotation.end_ms > self.ended_at_ms:
                raise ValueError("annotation must fall within the recording session")
            if annotation.start_ms < previous_end:
                raise ValueError("annotations cannot overlap")
            previous_end = annotation.end_ms

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> CalibrationSession:
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotations = tuple(AnnotatedInterval(**item) for item in payload.pop("annotations", []))
        return cls(annotations=annotations, **payload)

