from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..features.schema import FEATURE_SCHEMA_VERSION
from .gru import GRUConfig, LandmarkGRU

BUNDLE_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    """Raised when a model bundle does not match the runtime contract."""


@dataclass(frozen=True, slots=True)
class BundleMetadata:
    labels: tuple[str, ...]
    feature_schema_version: str
    preprocessing_hash: str
    dataset_hash: str
    model_config: GRUConfig
    bundle_version: int = BUNDLE_VERSION


def preprocessing_fingerprint(configuration: dict[str, object]) -> str:
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def save_bundle(
    path: Path,
    model: LandmarkGRU,
    *,
    labels: tuple[str, ...],
    preprocessing_hash: str,
    dataset_hash: str,
) -> None:
    if len(labels) != model.config.num_classes or len(labels) != len(set(labels)):
        raise ValueError("labels must be unique and match the model class count")
    if not preprocessing_hash or not dataset_hash:
        raise ValueError("preprocessing_hash and dataset_hash cannot be blank")
    payload: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "labels": list(labels),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "preprocessing_hash": preprocessing_hash,
        "dataset_hash": dataset_hash,
        "model_config": model.config.to_dict(),
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_bundle(
    path: Path,
    *,
    expected_labels: tuple[str, ...] | None = None,
    expected_preprocessing_hash: str | None = None,
    expected_dataset_hash: str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[LandmarkGRU, BundleMetadata]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise CheckpointCompatibilityError(f"unable to load safe model bundle: {error}") from error
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("model bundle must contain an object")
    required = {
        "bundle_version",
        "labels",
        "feature_schema_version",
        "preprocessing_hash",
        "dataset_hash",
        "model_config",
        "state_dict",
    }
    if set(payload) != required:
        raise CheckpointCompatibilityError("model bundle fields are invalid")
    if payload["bundle_version"] != BUNDLE_VERSION:
        raise CheckpointCompatibilityError("unsupported model bundle version")
    if payload["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise CheckpointCompatibilityError("feature schema is incompatible")

    raw_labels = payload["labels"]
    if not isinstance(raw_labels, list):
        raise CheckpointCompatibilityError("model labels are invalid")
    labels = tuple(raw_labels)
    if not labels or not all(isinstance(label, str) and label for label in labels):
        raise CheckpointCompatibilityError("model labels are invalid")
    if len(labels) != len(set(labels)):
        raise CheckpointCompatibilityError("model labels must be unique")
    if expected_labels is not None and labels != expected_labels:
        raise CheckpointCompatibilityError("model labels do not match expected labels")
    preprocessing_hash = payload["preprocessing_hash"]
    if not isinstance(preprocessing_hash, str) or not preprocessing_hash:
        raise CheckpointCompatibilityError("preprocessing hash is invalid")
    if (
        expected_preprocessing_hash is not None
        and preprocessing_hash != expected_preprocessing_hash
    ):
        raise CheckpointCompatibilityError("preprocessing hash does not match")

    try:
        config = GRUConfig.from_dict(payload["model_config"])
    except (TypeError, ValueError) as error:
        raise CheckpointCompatibilityError(f"model configuration is invalid: {error}") from error
    if config.num_classes != len(labels):
        raise CheckpointCompatibilityError("model class count does not match labels")
    if not isinstance(payload["dataset_hash"], str) or not payload["dataset_hash"]:
        raise CheckpointCompatibilityError("dataset hash is invalid")
    if expected_dataset_hash is not None and payload["dataset_hash"] != expected_dataset_hash:
        raise CheckpointCompatibilityError("dataset hash does not match")
    if not isinstance(payload["state_dict"], dict):
        raise CheckpointCompatibilityError("model state dictionary is invalid")

    model = LandmarkGRU(config)
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except RuntimeError as error:
        raise CheckpointCompatibilityError(f"model weights are incompatible: {error}") from error
    model.to(map_location)
    model.eval()
    metadata = BundleMetadata(
        labels=labels,
        feature_schema_version=payload["feature_schema_version"],
        preprocessing_hash=preprocessing_hash,
        dataset_hash=payload["dataset_hash"],
        model_config=config,
    )
    return model, metadata
