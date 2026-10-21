"""Synthetic auth checks: cached media cannot prove current credentials work."""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "probe_fixture", Path("scripts/sol_youtube_batch.py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("kind", ["legacy", "changed_cookie", "changed_sources", "stale", "cached"])
def test_batch_refuses_unproven_current_authentication(tmp_path, monkeypatch, kind):
    batch = module()
    run, scratch = tmp_path / "run", tmp_path / "scratch"
    active = scratch / "youtube-asl"
    active.mkdir(parents=True)
    official = active / "official-ids.txt"
    official.write_text("P-D0kEmncGw\n")
    cookie = tmp_path / "current.txt"
    cookie.write_text("synthetic credential")
    # Permission metadata is simulated on Windows, without reading real credentials.
    batch.COOKIE = SimpleNamespace(
        is_file=lambda: True, stat=lambda: SimpleNamespace(st_mode=0o100600)
    )
    monkeypatch.setattr(batch, "file_hash", lambda path: "current", raising=False)
    batch.RUN, batch.SCRATCH = run, scratch
    report = {
        "status": "completed",
        "mode": "probe",
        "available": 1,
        "completed_at_unix": 10000,
        "probe_identity": {
            "version": 1,
            "cookie_sha256": "current",
            "official_ids_sha256": "current",
        },
        "batch_progress": {"acquired_pairs": 1},
    }
    if kind == "legacy":
        report.pop("probe_identity")
    if kind == "changed_cookie":
        report["probe_identity"]["cookie_sha256"] = "old"
    if kind == "changed_sources":
        report["probe_identity"]["official_ids_sha256"] = "old"
    if kind == "stale":
        report["completed_at_unix"] = 0
    if kind == "cached":
        report["batch_progress"]["acquired_pairs"] = 0
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(report))
    monkeypatch.setattr(batch.time, "time", lambda: 10010)
    monkeypatch.setattr(batch, "project_headroom", lambda path: 1024**4)
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(batch, "writer_lock", lambda path: nullcontext())
    calls = []

    def acquire(**kwargs):
        calls.append(kwargs)
        return {
            "videos": {
                "P-D0kEmncGw": {"availability": "available", "split": "train", "attempts": 1}
            }
        }

    monkeypatch.setattr(batch, "acquire_youtube", acquire)
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch", "batch", "--round-dir", str(run / "reports/round"), "--probe-report", str(probe)],
    )
    with pytest.raises(RuntimeError, match="probe"):
        batch.main()
    assert not calls


def test_probe_downloads_fresh_pair_each_run_and_binds_identity(tmp_path, monkeypatch):
    batch = module()
    run, scratch = tmp_path / "run", tmp_path / "scratch"
    active = scratch / "youtube-asl"
    active.mkdir(parents=True)
    (active / "official-ids.txt").write_text("P-D0kEmncGw\n")
    batch.RUN, batch.SCRATCH = run, scratch
    batch.COOKIE = SimpleNamespace(
        is_file=lambda: True, stat=lambda: SimpleNamespace(st_mode=0o100600)
    )
    monkeypatch.setattr(batch, "file_hash", lambda path: "current", raising=False)
    monkeypatch.setattr(batch, "project_headroom", lambda path: 1024**4)
    monkeypatch.setattr(batch.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**4))
    locks = []

    def lock(path):
        locks.append(path)
        return nullcontext()

    monkeypatch.setattr(batch, "writer_lock", lock)
    roots = []

    def acquire(**kwargs):
        roots.append(kwargs["root"])
        return {
            "videos": {
                "P-D0kEmncGw": {"availability": "available", "split": "train", "attempts": 1}
            }
        }

    monkeypatch.setattr(batch, "acquire_youtube", acquire)
    for index in range(2):
        directory = run / "reports" / str(index)
        monkeypatch.setattr(sys, "argv", ["batch", "probe", "--round-dir", str(directory)])
        batch.main()
        report = json.loads((directory / "probe-report.json").read_text())
        assert report["probe_identity"] == {
            "version": 1,
            "cookie_sha256": "current",
            "official_ids_sha256": "current",
        }
        assert report["completed_at_unix"] > 0
    assert roots[0] != roots[1]
    assert all(path != active for path in roots)
    assert locks == [active / "acquisition.lock"] * 2
