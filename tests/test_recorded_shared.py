import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from asl_recognizer.translation import recorded
from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import DirectTranslationResult, TranslationRequest


def test_cold_recorded_path_preserves_raw_cache_and_contract(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "recorded_fixture", Path("tests/test_local_translation.py").resolve()
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_recorded_extraction_interval_mask_and_overlay(tmp_path, monkeypatch)
    assert list((tmp_path / "features/raw-pose").glob("*.npz"))
    _, metadata = TranslationFeatureCache(tmp_path / "features").load("recorded")
    assert metadata["preprocessing"]["frame_cap"] == 256


def test_cold_translation_rejects_wrong_preprocessing(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": {
                "feature_schema": "mmpose-coco-wholebody-133-xy-v1",
                "extractor": {"fixture": True, "confidence_threshold": 0.5},
                "preprocessing": {"required": "contract"},
            },
            "model": {},
        },
        checkpoint,
    )
    output = tmp_path / "result.json"

    def extract(*args, **kwargs):
        root = tmp_path / "result-features"
        root.mkdir(exist_ok=True)
        (root / "extraction.json").write_text(
            json.dumps({"extraction_seconds": 1, "processing_fps": 1})
        )
        request = TranslationRequest(np.ones((2, 266), np.float32), np.ones((2, 133), np.bool_))
        TranslationFeatureCache(root).store(
            "recorded",
            request,
            metadata={
                "extractor": {"fixture": True, "confidence_threshold": 0.5},
                "preprocessing": {"wrong": True},
            },
        )

    monkeypatch.setattr(recorded.subprocess, "run", extract)
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.translate.return_value = DirectTranslationResult("fixture", None, "tiny", "v1", 1.0)
    monkeypatch.setattr(recorded, "LandmarkT5", lambda *a: model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    with pytest.raises(ValueError, match="preprocessing"):
        recorded.infer_video(
            checkpoint=checkpoint, video=tmp_path / "video", assets={}, output=output, device="cpu"
        )
