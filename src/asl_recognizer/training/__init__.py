"""Training data adapters and deterministic optimization helpers."""

from .data import SequenceBatch, SequenceExample, collate_examples, discover_cache_index
from .engine import EvaluationResult, TrainingConfig, TrainingHistory, evaluate, train_model

__all__ = [
    "EvaluationResult",
    "SequenceBatch",
    "SequenceExample",
    "TrainingConfig",
    "TrainingHistory",
    "collate_examples",
    "discover_cache_index",
    "evaluate",
    "train_model",
]
