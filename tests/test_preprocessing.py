import importlib.util
from pathlib import Path

import pytest

from asl_recognizer.translation.cache import TranslationFeatureCache, preprocessing_contract
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
    assert preprocessing_contract({"how2sign": path}, tmp_path / "features") == contract
    dev = manifest.accepted("dev")[0]
    request, metadata = cache.load(dev.clip_id)
    cache.store(dev.clip_id, request, metadata={**metadata, "preprocessing": {"changed": True}})
    with pytest.raises(ValueError, match="preprocessing"):
        preprocessing_contract({"how2sign": path}, tmp_path / "features")


def test_legacy_contract_is_explicit_and_mixing_is_rejected(tmp_path):
    helper = fixture_module()
    path = helper.source(tmp_path, "how2sign")
    assert preprocessing_contract({"how2sign": path}, tmp_path / "features") is None
    cache = TranslationFeatureCache(tmp_path / "features/how2sign")
    row = read_manifest(path).accepted("train")[0]
    request, metadata = cache.load(row.clip_id)
    cache.store(row.clip_id, request, metadata={**metadata, "preprocessing": {"version": 1}})
    with pytest.raises(ValueError, match="preprocessing"):
        preprocessing_contract({"how2sign": path}, tmp_path / "features")

