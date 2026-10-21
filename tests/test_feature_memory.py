from pathlib import Path

import numpy as np
import pytest

from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import TranslationRequest


def request():
    return TranslationRequest(np.zeros((4, 266), np.float32), np.ones((4, 133), np.bool_))


def test_feature_memory_cache_is_bounded_and_invalidates(tmp_path, monkeypatch):
    from asl_recognizer.translation import cache as module

    cache = TranslationFeatureCache(tmp_path, max_memory_bytes=8000)
    cache.store("one", request(), metadata={"caption": "first"})
    cache.store("two", request(), metadata={"caption": "second"})
    monkeypatch.setattr(
        module, "_cache_version", lambda p: (p.stat().st_mtime_ns, p.stat().st_size), raising=False
    )
    original = module.file_hash
    calls = []

    def hashed(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, "file_hash", hashed)
    first, _ = cache.load("one")
    first.values[:] = 10
    again, meta = cache.load("one")
    assert not again.values.any() and meta["caption"] == "first"
    assert len(calls) == 1
    cache.load("two")
    assert cache.memory_bytes <= 8000
    cache.load("one")
    assert len(calls) == 3
    cache.store("one", request(), metadata={"caption": "changed"})
    assert cache.load("one")[1]["caption"] == "changed"
    path = cache._path("one")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        cache.load("one")


def test_unsafe_file_versions_always_hash(tmp_path, monkeypatch):
    from asl_recognizer.translation import cache as module

    cache = TranslationFeatureCache(tmp_path, max_memory_bytes=8000)
    cache.store("one", request(), metadata={})
    monkeypatch.setattr(module, "_cache_version", lambda path: None, raising=False)
    original = module.file_hash
    calls = []

    def hashed(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, "file_hash", hashed)
    cache.load("one")
    cache.load("one")
    assert len(calls) == 2 and cache.memory_bytes == 0


def test_only_settled_linux_versions_can_skip_hashing(monkeypatch):
    from types import SimpleNamespace

    from asl_recognizer.translation import cache as module

    value = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=1_000_000_000)
    path = SimpleNamespace(stat=lambda: value)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(module, "time", SimpleNamespace(time_ns=lambda: 1_500_000_000))
    assert module._cache_version(path) is None
    monkeypatch.setattr(module, "time", SimpleNamespace(time_ns=lambda: 4_000_000_000))
    assert module._cache_version(path) == (1, 2, 3, 4, 1_000_000_000)
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="win32"))
    assert module._cache_version(path) is None
    with pytest.raises(ValueError, match="negative"):
        TranslationFeatureCache(Path("unused"), max_memory_bytes=-1)
