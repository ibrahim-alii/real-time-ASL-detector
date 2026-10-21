import importlib.util
from pathlib import Path

import pytest
import torch

from asl_recognizer.translation import evaluation
from asl_recognizer.translation.manifest import read_manifest


def test_evaluation_refuses_preprocessing_mismatch(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "evaluation_fixture", Path("tests/test_local_translation.py").resolve()
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.tiny.__wrapped__(monkeypatch)
    path = helper.source(tmp_path, "how2sign")
    manifest = read_manifest(path)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "mode": "experiment",
            "config": {
                "feature_schema": "mmpose-coco-wholebody-133-xy-v1",
                "extractor": {"backend": "mmpose", "assets": {"fixture": "asset"}},
                "preprocessing": {"required": "v1"},
            },
            "manifest_hashes": {"how2sign": manifest.fingerprint},
            "model": helper.Model(None).state_dict(),
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match=r"mismatched|preprocessing"):
        evaluation.evaluate_checkpoint(
            checkpoint=checkpoint,
            manifest_path=path,
            features=tmp_path / "features",
            output=tmp_path / "evaluation",
            split="dev",
            with_bleurt=False,
        )
