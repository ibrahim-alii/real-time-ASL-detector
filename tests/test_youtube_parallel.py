"""Concurrency checks use synthetic files, not evidence of dataset quality."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from asl_recognizer.translation import acquisition
from asl_recognizer.translation.sol_common import read_json, write_json


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    root = tmp_path / "youtube"
    root.mkdir()
    ids = root / "ids.txt"
    ids.write_text("abcdefghijk\nlmnopqrstuv\n12345678901\n")
    monkeypatch.setattr(acquisition.time, "sleep", lambda _seconds: None)

    def runner(args: list[str]) -> str:
        if "--dump-single-json" in args:
            return json.dumps({"channel_id": "channel", "subtitles": {"en": []},
                               "formats": [{"format_id": "137", "height": 720, "ext": "mp4",
                                            "vcodec": "h264", "acodec": "none", "filesize": 100}]})
        if "--write-subs" in args:
            video_id = args[-1].split("v=")[-1]
            raw = root / "raw-captions"
            raw.mkdir(exist_ok=True)
            write_json(raw / f"{video_id}.en.json3", {"events": [
                {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello"}]}]})
            write_json(raw / f"{video_id}.info.json", {"id": video_id})
        elif "-f" in args:
            assert int(args[args.index("--max-filesize") + 1]) == 130
            Path(args[args.index("-o") + 1]).write_bytes(b"video")
        elif args[0] == "ffprobe":
            return json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "2"}})
        return ""

    return root, dict(ids_path=ids, root=root, byte_limit=100000,
                      scratch_remaining=lambda: 100000, workers=2), runner


def test_two_sources_overlap_and_only_coordinator_writes_ledger(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, kwargs, original = fixture(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    coordinator = threading.get_ident()
    save = acquisition._save_youtube_acquisition
    calls: list[str] = []

    def writer(*args):  # type: ignore[no-untyped-def]
        assert threading.get_ident() == coordinator
        save(*args)

    def runner(args: list[str]) -> str:
        if "--dump-single-json" in args:
            calls.append(args[-1])
            if len(calls) <= 2:
                barrier.wait(timeout=5)
        return original(args)

    monkeypatch.setattr(acquisition, "_save_youtube_acquisition", writer)
    result = acquisition.acquire_youtube(**kwargs, runner=runner)
    assert len(result["videos"]) == 3
    assert all(row["availability"] == "available" for row in result["videos"].values())
    assert len((root / "index.csv").read_text().splitlines()) == 4
    calls.clear()
    acquisition.acquire_youtube(**kwargs, runner=runner)
    assert not calls


def test_access_block_drains_workers_and_does_not_schedule_more(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, kwargs, original = fixture(tmp_path, monkeypatch)
    barrier = threading.Barrier(2)
    committed_failure = threading.Event()
    lock = threading.Lock()
    metadata: list[str] = []
    calls: list[list[str]] = []
    save = acquisition._save_youtube_acquisition

    def writer(path, ledger):  # type: ignore[no-untyped-def]
        save(path, ledger)
        if any(row.get("failure_kind") == "access_blocked" for row in ledger["videos"].values()):
            committed_failure.set()

    def runner(args: list[str]) -> str:
        calls.append(args)
        if "--dump-single-json" in args:
            with lock:
                metadata.append(args[-1])
                number = len(metadata)
            barrier.wait(timeout=5)
            if number == 1:
                raise RuntimeError("HTTP 429 Too Many Requests")
            assert committed_failure.wait(timeout=5)
        return original(args)

    monkeypatch.setattr(acquisition, "_save_youtube_acquisition", writer)
    with pytest.raises(RuntimeError, match="investigation") as failure:
        acquisition.acquire_youtube(**kwargs, runner=runner)
    assert acquisition.acquisition_failure_kind(str(failure.value)) == "access_blocked"
    ledger = read_json(root / "acquisition.json")
    assert len(metadata) == 2
    # Metadata-only cancellation refunds the source attempt completely.
    assert len(ledger["videos"]) == 1
    assert all(row["availability"] != "available" for row in ledger["videos"].values())
    assert not any("-f" in args for args in calls)


def test_reservations_are_aggregate_and_released(tmp_path):  # type: ignore[no-untyped-def]
    reservations = acquisition._Reservations(tmp_path, 100000, lambda: 1000)
    first = reservations.reserve(500, False)
    assert first == 650
    with pytest.raises(ValueError, match="insufficient capacity"):
        reservations.reserve(500, False)
    reservations.release(first, False)
    with pytest.raises(ValueError, match="insufficient capacity"):
        reservations.reserve(500, True)
    assert reservations.reserve(500, False) == 650


def test_unexpected_worker_failure_commits_other_completed_sources(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, kwargs, runner = fixture(tmp_path, monkeypatch)
    original = acquisition._acquire_youtube_source
    barrier = threading.Barrier(2)
    finished = threading.Event()
    lock = threading.Lock()
    started = 0

    def source(row, **options):  # type: ignore[no-untyped-def]
        nonlocal started
        with lock:
            started += 1
            number = started
        barrier.wait(timeout=5)
        if number == 1:
            assert finished.wait(timeout=5)
            options["stop"].set()
            raise TypeError("unexpected fixture error")
        result = original(row, **options)
        finished.set()
        return result

    monkeypatch.setattr(acquisition, "_acquire_youtube_source", source)
    with pytest.raises(TypeError, match="unexpected fixture"):
        acquisition.acquire_youtube(**kwargs, runner=runner)
    rows = list(read_json(root / "acquisition.json")["videos"].values())
    assert len(rows) == 2
    assert sum(row["availability"] == "available" for row in rows) == 1
    assert any(row.get("reason") == "unexpected acquisition error: TypeError" for row in rows)


def test_parallel_batch_limit_counts_only_new_attempts_on_resume(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    _root, kwargs, original = fixture(tmp_path, monkeypatch)
    kwargs["workers"] = 3
    metadata: list[str] = []

    def runner(args: list[str]) -> str:
        if "--dump-single-json" in args:
            metadata.append(args[-1])
        return original(args)

    result = acquisition.acquire_youtube(**kwargs, runner=runner, video_limit=2)
    assert len(metadata) == len(result["videos"]) == 2
    metadata.clear()
    result = acquisition.acquire_youtube(**kwargs, runner=runner, video_limit=2)
    assert len(metadata) == 1
    assert len(result["videos"]) == 3
