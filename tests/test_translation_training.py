from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from asl_recognizer.translation import evaluation, training
from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import TranslationRequest
from asl_recognizer.translation.manifest import build_manifest, write_manifest
from asl_recognizer.translation.model import DirectTranslationConfig, LandmarkT5
from asl_recognizer.translation.run_metadata import collect_run_metadata
from asl_recognizer.translation.schema import MMPOSE_SCHEMA, translation_schema


def _source(tmp_path: Path, name: str, schema_name: str = MMPOSE_SCHEMA) -> Path:
    rows = [
        {
            "id": f"{name}-clip",
            "video_id": name,
            "start": 0,
            "end": 1,
            "caption": "hello",
            "split": "train",
        }
    ]
    if name == "how2sign":
        for split in ("dev", "test"):
            rows.append(
                {
                    "id": f"how2sign-{split}", "video_id": name,
                    "start": 0, "end": 1, "caption": "hello", "split": split,
                }
            )
    manifest = build_manifest(rows, source=name)
    path = tmp_path / f"{name}.csv"
    write_manifest(path, manifest)
    cache = TranslationFeatureCache(tmp_path / "features" / name)
    schema = translation_schema(schema_name)
    for row in rows:
        cache.store(
            str(row["id"]),
            TranslationRequest(
                np.ones((2, schema.dimension), dtype=np.float32),
                np.ones((2, schema.landmarks), dtype=np.bool_), schema_name,
            ),
            metadata={
                "source": name,
                "caption": "hello",
                "split": row["split"],
                "manifest_hash": manifest.fingerprint,
                "feature_schema": schema_name,
                "extractor": {"backend": "mmpose", "assets": {"pose_weights": "fixture"}},
            },
        )
    return path








def test_bleurt_uses_frozen_model_revision(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[str, str]] = []

    class Scorer:
        @classmethod
        def from_pretrained(cls, identity: str, *, revision: str):
            calls.append((identity, revision))
            return cls()

        def to(self, _device: str):
            return self

        def eval(self):
            return self

        def __call__(self, **_kwargs: object):
            return SimpleNamespace(logits=torch.tensor([0.5]))

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, identity: str, *, revision: str):
            calls.append((identity, revision))
            return cls()

        def __call__(self, _references: list[str], _candidates: list[str], **_kwargs: object):
            return {"input_ids": torch.ones((1, 1), dtype=torch.long)}

    module = ModuleType("bleurt_pytorch")
    module.BleurtForSequenceClassification = Scorer  # type: ignore[attr-defined]
    module.BleurtTokenizer = Tokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bleurt_pytorch", module)
    assert evaluation._bleurt_scores(["hello"], ["hello"], "cpu") == [0.5]
    assert calls == [(evaluation.BLEURT_MODEL_ID, evaluation.BLEURT_REVISION)] * 2


def test_run_metadata_records_dependency_versions(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ASL_CONTAINER_SHA256", "abc123")
    metadata = collect_run_metadata(
        seed=42, manifest_hashes={"how2sign": "digest"}, container="local"
    )
    assert metadata.dependencies["transformers"] != "unavailable"
    assert metadata.dependencies["numpy"] != "unavailable"
    assert "bleurt-pytorch" in metadata.dependencies
    assert metadata.container_sha256 == "abc123"




@pytest.mark.parametrize("schema_name", [MMPOSE_SCHEMA])
def test_landmark_t5_forward_and_beam_decode_without_model_download(  # type: ignore[no-untyped-def]
    monkeypatch, schema_name: str
) -> None:
    from transformers import T5Config, T5ForConditionalGeneration

    config = T5Config(
        vocab_size=32,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        pad_token_id=0,
        decoder_start_token_id=0,
        eos_token_id=1,
    )
    config._name_or_path = "tiny-test"
    monkeypatch.setattr(
        T5ForConditionalGeneration,
        "from_pretrained",
        lambda *_args, **_kwargs: T5ForConditionalGeneration(config),
    )
    schema = translation_schema(schema_name)
    model = LandmarkT5(DirectTranslationConfig(
        decoder_max_tokens=4, beam_width=2, feature_schema=schema_name
    ))
    values = torch.ones((1, 2, schema.dimension), dtype=torch.float32)
    labels = torch.tensor([[3, 1]])
    assert torch.isfinite(model(values, torch.ones((1, 2), dtype=torch.long), labels))
    request = TranslationRequest(
        values[0].numpy(), np.ones((2, schema.landmarks), dtype=np.bool_), schema_name
    )
    tokenizer = type("Tokenizer", (), {"decode": lambda self, *_args, **_kwargs: "hello"})()
    result = model.translate(request, tokenizer)
    assert result.english == "hello"
    assert result.beam_width == 2



def test_mmpose_training_rejects_mixed_extractor_assets(tmp_path: Path) -> None:
    manifest_path = _source(tmp_path, "youtube-asl", MMPOSE_SCHEMA)
    cache = TranslationFeatureCache(tmp_path / "features" / "youtube-asl")
    request, metadata = cache.load("youtube-asl-clip")
    # Rebuild a manifest with two train examples from different extraction assets.
    manifest = build_manifest([
        {"id": name, "video_id": name, "caption": "hello", "split": "train",
         "start": 0, "end": 1} for name in ("first", "second")
    ], source="youtube-asl")
    write_manifest(manifest_path, manifest)
    for name in ("first", "second"):
        cache.store(name, request, metadata={
            **metadata, "manifest_hash": manifest.fingerprint,
            "extractor": {"pose_weights": name},
        })
    with pytest.raises(ValueError, match="different MMPose"):
        training._examples(manifest_path, "youtube-asl", tmp_path / "features", MMPOSE_SCHEMA)
