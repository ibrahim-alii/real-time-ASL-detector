"""Cached integrity checks must invalidate on file or expected-digest changes."""

import time
from unittest.mock import Mock

import pytest

from asl_recognizer.translation import sol_common as s


@pytest.fixture(autouse=True)
def mature_file_clock(monkeypatch):
    monkeypatch.setattr(time, "time_ns", lambda: 10**30)


def test_cached_seal_avoids_reread_and_invalidates_expected_hash(tmp_path, monkeypatch):
    p = tmp_path / "a"
    p.write_bytes(b"good")
    v = s.seal(p)
    monkeypatch.setattr(s.sys, "platform", "linux")
    h = Mock(wraps=s.file_hash)
    monkeypatch.setattr(s, "file_hash", h)
    assert s.verify_seal(v, cached=True) and h.call_count == 0
    v["sha256"] = "changed"
    assert not s.verify_seal(v, cached=True) and h.call_count == 1


def test_legacy_verified_once_and_changed_identity_rehashed(tmp_path, monkeypatch):
    p = tmp_path / "a"
    p.write_bytes(b"good")
    v = s.seal(p)
    v.pop("verified_stat")
    monkeypatch.setattr(s.sys, "platform", "linux")
    h = Mock(wraps=s.file_hash)
    monkeypatch.setattr(s, "file_hash", h)
    assert s.verify_seal(v, cached=True) and h.call_count == 1
    assert s.verify_seal(v, cached=True) and h.call_count == 1
    old = s._seal_stat(p)
    monkeypatch.setattr(s, "_seal_stat", lambda _: [*old[:-1], old[-1] + 1])
    assert s.verify_seal(v, cached=True) and h.call_count == 2
    p.write_bytes(b"evil")
    monkeypatch.setattr(s, "_seal_stat", lambda _: [*old[:-1], old[-1] + 2])
    assert not s.verify_seal(v, cached=True)


def test_full_verification_and_missing_file(tmp_path, monkeypatch):
    p = tmp_path / "a"
    p.write_bytes(b"good")
    v = s.seal(p)
    h = Mock(wraps=s.file_hash)
    monkeypatch.setattr(s, "file_hash", h)
    assert s.verify_seal(v) and h.call_count == 1
    p.unlink()
    assert not s.verify_seal(v, cached=True)


def test_changed_during_seal_rejected(tmp_path, monkeypatch):
    p = tmp_path / "a"
    p.write_bytes(b"good")
    old = s._seal_stat(p)
    values = iter([old, [*old[:-1], old[-1] + 1]])
    monkeypatch.setattr(s, "_seal_stat", lambda _: next(values))
    with pytest.raises(ValueError, match="changed while sealing"):
        s.seal(p)


def test_non_linux_cache_still_hashes(tmp_path, monkeypatch):
    p = tmp_path / "a"
    p.write_bytes(b"good")
    value = s.seal(p)
    monkeypatch.setattr(s.sys, "platform", "win32")
    digest = Mock(wraps=s.file_hash)
    monkeypatch.setattr(s, "file_hash", digest)
    assert s.verify_seal(value, cached=True)
    assert digest.call_count == 1


def test_changed_while_verifying_does_not_cache(tmp_path, monkeypatch):
    p = tmp_path / "a"
    p.write_bytes(b"good")
    value = s.seal(p)
    value.pop("verified_stat")
    before = s._seal_stat(p)
    stats = iter([before, [*before[:-1], before[-1] + 1]])
    monkeypatch.setattr(s, "_seal_stat", lambda _: next(stats))
    assert not s.verify_seal(value)
    assert "verified_stat" not in value
