import importlib.util
from pathlib import Path

from asl_recognizer.translation.cache import TranslationFeatureCache


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

