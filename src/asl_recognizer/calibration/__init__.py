"""Local-only calibration annotations and continuous-recognition evaluation."""

from .evaluation import EventMetrics, evaluate_events
from .models import AnnotatedInterval, CalibrationKind, CalibrationSession, CalibrationSplit
from .recorder import CalibrationRecorder

__all__ = [
    "AnnotatedInterval",
    "CalibrationKind",
    "CalibrationRecorder",
    "CalibrationSession",
    "CalibrationSplit",
    "EventMetrics",
    "evaluate_events",
]

