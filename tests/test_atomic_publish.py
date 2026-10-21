"""Reproduce observed Windows atomic-publish locking without changing permissions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from asl_recognizer.translation import acquisition, sol_common


def lock_error(code):
    error = PermissionError("fixture Windows file lock")
    error.winerror = code
    return error


@pytest.mark.parametrize("kind", ["json", "csv"])
def test_atomic_publish_retries_only_transient_windows_locks(tmp_path, monkeypatch, kind):
    target = tmp_path / ("state." + kind)
    target.write_text("previous")
    original = Path.replace
    attempts, sleeps = [], []
    monkeypatch.setattr(sol_common, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(sol_common.time, "sleep", sleeps.append)

    def replace(source, destination):
        if Path(destination) == target:
            attempts.append(source)
            if len(attempts) <= 2:
                assert target.read_text() == "previous"
                raise lock_error(5 if len(attempts) == 1 else 32)
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", replace)
    if kind == "json":
        sol_common.write_json(target, {"value": "next"})
        assert json.loads(target.read_text()) == {"value": "next"}
    else:
        acquisition.write_csv(target, [{"value": "next"}])
        assert target.read_text().splitlines() == ["value", "next"]
    assert len(attempts) == 3 and sleeps == [0.02, 0.04]
    assert not target.with_suffix(target.suffix + ".tmp").exists()


@pytest.mark.parametrize(
    "platform,code,count", [("linux", 5, 1), ("win32", 123, 1), ("win32", 32, 6)]
)
def test_atomic_publish_keeps_old_file_and_propagates_persistent_failure(
    tmp_path, monkeypatch, platform, code, count
):
    target = tmp_path / "state.json"
    target.write_text("previous")
    attempts = []
    monkeypatch.setattr(sol_common, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(sol_common.time, "sleep", lambda seconds: None)

    def replace(source, destination):
        attempts.append(source)
        raise lock_error(code)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError):
        sol_common.write_json(target, {"value": "next"})
    assert target.read_text() == "previous" and len(attempts) == count
