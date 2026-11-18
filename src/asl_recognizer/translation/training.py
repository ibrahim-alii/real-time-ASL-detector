"""Validated landmark examples and encoded model inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import TranslationFeatureCache
from .contracts import TranslationRequest
from .features import TRANSLATION_FEATURE_SCHEMA
from .manifest import ClipManifest, read_manifest
from .schema import MMPOSE_SCHEMA, translation_schema


def encoded_values(request: TranslationRequest, encoding: str) -> torch.Tensor:
    values = request.values.copy()
    if encoding == "mask-v1":
        schema = translation_schema(request.feature_schema)
        coordinates = values.reshape(len(values), schema.landmarks, -1)
        coordinates[~request.landmark_mask] = 0
        values = np.concatenate(
            (coordinates.reshape(len(values), -1), request.landmark_mask.astype(np.float32)), axis=1
        )
    elif encoding != "sentinel-v1":
        raise ValueError("unsupported translation input encoding")
    return torch.from_numpy(values)



def _examples(
    path: Path, source: str, features: Path, feature_schema: str = TRANSLATION_FEATURE_SCHEMA
) -> tuple[ClipManifest, list[tuple[str, str]], dict[str, Any] | None]:
    manifest = read_manifest(path)
    split = "train" if source == "how2sign" else None
    cache = TranslationFeatureCache(features / source)
    examples: list[tuple[str, str]] = []
    extractor_identity: dict[str, Any] | None = None
    for record in manifest.accepted(split):
        if source == "youtube-asl" and record.split not in {"train", "unassigned"}:
            continue
        expected = {
            "manifest_hash": manifest.fingerprint,
            "feature_schema": feature_schema,
            "source": source,
            "caption": record.caption,
        }
        artifact = cache.find_valid(record.clip_id, expected_metadata=expected)
        if artifact is None:
            raise ValueError(f"missing or mismatched {source} feature: {record.clip_id}")
        if feature_schema == MMPOSE_SCHEMA:
            identity = artifact[1].get("extractor")
            if not isinstance(identity, dict) or not identity:
                raise ValueError("MMPose training requires recorded extractor identity")
            if extractor_identity is not None and identity != extractor_identity:
                raise ValueError("training features use different MMPose extractors")
            extractor_identity = identity
        examples.append((record.clip_id, record.caption))
    if not examples:
        raise ValueError(f"no accepted {source} training examples")
    return manifest, examples, extractor_identity
