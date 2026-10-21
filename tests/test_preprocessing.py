import importlib.util
from pathlib import Path

import pytest

from asl_recognizer.translation import experiment
from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.manifest import read_manifest


def fixture_module():
    path = Path("tests/test_local_translation.py").resolve()
    spec = importlib.util.spec_from_file_location("local_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_ignores_test_and_rejects_mixed_train_dev(tmp_path):
    helper = fixture_module()
    path = helper.source(tmp_path, "how2sign")
    cache = TranslationFeatureCache(tmp_path / "features/how2sign")
    manifest = read_manifest(path)
    contract = {"sampling": "bounded-v1", "normalization": "bbox-v1"}
    for row in manifest.accepted():
        request, metadata = cache.load(row.clip_id)
        cache.store(
            row.clip_id,
            request,
            metadata={
                **metadata,
                "preprocessing": {"sealed": "test"} if row.split == "test" else contract,
            },
        )
    assert experiment.preprocessing_contract({"how2sign": path}, tmp_path / "features") == contract
    dev = manifest.accepted("dev")[0]
    request, metadata = cache.load(dev.clip_id)
    cache.store(dev.clip_id, request, metadata={**metadata, "preprocessing": {"changed": True}})
    with pytest.raises(ValueError, match="preprocessing"):
        experiment.preprocessing_contract({"how2sign": path}, tmp_path / "features")


def test_legacy_contract_is_explicit_and_mixing_is_rejected(tmp_path):
    helper = fixture_module()
    path = helper.source(tmp_path, "how2sign")
    assert experiment.preprocessing_contract({"how2sign": path}, tmp_path / "features") is None
    cache = TranslationFeatureCache(tmp_path / "features/how2sign")
    row = read_manifest(path).accepted("train")[0]
    request, metadata = cache.load(row.clip_id)
    cache.store(row.clip_id, request, metadata={**metadata, "preprocessing": {"version": 1}})
    with pytest.raises(ValueError, match="preprocessing"):
        experiment.preprocessing_contract({"how2sign": path}, tmp_path / "features")


def test_checkpoint_records_versioned_contract(tmp_path, monkeypatch):
    from dataclasses import replace

    import torch

    helper = fixture_module()
    helper.tiny.__wrapped__(monkeypatch)
    path = helper.source(tmp_path, "how2sign")
    contract = {"sampling": "fixture-v1"}
    cache = TranslationFeatureCache(tmp_path / "features/how2sign")
    for row in read_manifest(path).accepted():
        if row.split not in {"train", "dev"}:
            continue
        request, metadata = cache.load(row.clip_id)
        cache.store(row.clip_id, request, metadata={**metadata, "preprocessing": contract})
    experiment.run_experiment(
        features=tmp_path / "features",
        how2sign_manifest=path,
        youtube_manifest=None,
        output=tmp_path / "run",
        config=replace(helper.cpu_config(), youtube_steps=0),
        job_steps=1,
        device_name="cpu",
    )
    saved = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    assert saved["config"]["preprocessing"] == contract
    assert saved["config"]["preprocessing_status"] == "versioned"
    exposure = saved["exposure"]["how2sign"]
    assert exposure["examples_seen"] == 2 and exposure["unique_examples"] == 1
    assert exposure["target_tokens"] == 2 and exposure["effective_epochs"] == 2
