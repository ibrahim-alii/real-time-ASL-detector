"""Concurrency checks use synthetic files, not evidence of dataset quality."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from asl_recognizer.translation import acquisition
from asl_recognizer.translation.sol_common import read_json, tree_bytes, write_json


@pytest.mark.parametrize(('message', 'expected'), [
    ('downloader or video probe failed (exit 1)', 'retryable'),
    ('source unavailable', 'permanent_unavailable'),
    ('caption boundaries exceed video duration', 'invalid_data'),
    ('403 YouTube access denied', 'access_blocked'),
    ('sign in / cookies rejected', 'access_blocked'),
])
def test_diagnostic_field_names_do_not_change_failure_category(
    message: str, expected: str,
) -> None:
    flags = {'http429': False, 'http403': False, 'cookie_invalid': False,
             'sign_in': False, 'confirm_you': False}
    reason = message + '; diagnostic_flags=' + json.dumps(flags)
    assert acquisition.acquisition_failure_kind(reason) == expected


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


@pytest.mark.parametrize('workers', [4, 10, 20])
def test_parallel_sources_overlap_with_one_ledger_writer(tmp_path, monkeypatch, workers):  # type: ignore[no-untyped-def]
    _root, kwargs, original = fixture(tmp_path, monkeypatch)
    kwargs['ids_path'].write_text('\n'.join(str(10000000000 + i) for i in range(workers)) + '\n')
    kwargs['workers'] = workers
    barrier = threading.Barrier(workers)
    coordinator = threading.get_ident()
    save = acquisition._save_youtube_acquisition

    def writer(*args):  # type: ignore[no-untyped-def]
        assert threading.get_ident() == coordinator
        save(*args)

    def runner(args: list[str]) -> str:
        if '--dump-single-json' in args:
            barrier.wait(timeout=5)
        return original(args)

    monkeypatch.setattr(acquisition, '_save_youtube_acquisition', writer)
    result = acquisition.acquire_youtube(**kwargs, runner=runner)
    assert len(result['videos']) == workers
    assert all(row['availability'] == 'available' for row in result['videos'].values())


@pytest.mark.parametrize('workers', [0, 5, 9, 11, 19, 21])
def test_unsupported_worker_count_is_rejected_before_acquisition(
    tmp_path: Path, workers: int,
) -> None:
    with pytest.raises(ValueError, match='workers'):
        acquisition.acquire_youtube(ids_path=tmp_path / 'ids', root=tmp_path, byte_limit=100,
                                    scratch_remaining=lambda: 100, workers=workers)


def test_storage_scan_restarts_after_atomic_rename() -> None:
    entry = Mock()
    entry.is_file.return_value = True
    entry.stat.side_effect = [FileNotFoundError('renamed'), SimpleNamespace(st_size=9)]
    root = Mock()
    root.rglob.return_value = [entry]
    assert tree_bytes(root) == 9
    assert root.rglob.call_count == 2


@pytest.mark.parametrize(('error', 'attempts'), [
    (FileNotFoundError('continuous churn'), 3), (PermissionError('unreadable'), 1),
])
def test_storage_scan_never_accepts_incomplete_usage(error: OSError, attempts: int) -> None:
    entry = Mock()
    entry.is_file.return_value = True
    entry.stat.side_effect = error
    root = Mock()
    root.rglob.return_value = [entry]
    with pytest.raises(type(error)):
        tree_bytes(root)
    assert root.rglob.call_count == attempts


@pytest.mark.parametrize('workers', [2, 10, 20])
def test_access_block_drains_workers_and_does_not_schedule_more(tmp_path, monkeypatch, workers):  # type: ignore[no-untyped-def]
    root, kwargs, original = fixture(tmp_path, monkeypatch)
    kwargs['ids_path'].write_text(
        '\n'.join(str(10000000000 + i) for i in range(workers + 1)) + '\n')
    kwargs['workers'] = workers
    barrier = threading.Barrier(workers)
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
    assert len(metadata) == workers
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


@pytest.mark.parametrize("approximate_size", [100, 10000000])
@pytest.mark.parametrize("headroom", [100000, 100000000])
def test_approximate_sizes_reserve_conservatively_before_download(
    tmp_path, monkeypatch, approximate_size, headroom,
):  # type: ignore[no-untyped-def]
    _root, kwargs, original = fixture(tmp_path, monkeypatch)
    downloads = []

    def runner(args: list[str]) -> str:
        if "--dump-single-json" in args:
            info = json.loads(original(args))
            info["duration"] = 2
            selected = info["formats"][0]
            selected.pop("filesize")
            selected.update(filesize_approx=approximate_size, tbr=0.4)
            return json.dumps(info)
        if "-f" in args:
            downloads.append(args)
            allocation = int(args[args.index("--max-filesize") + 1])
            assert allocation >= max(500, approximate_size)
            Path(args[args.index("-o") + 1]).write_bytes(b"x" * 500)
            return ""
        return original(args)

    kwargs.update(byte_limit=100000000, scratch_remaining=lambda: headroom, video_limit=1)
    if headroom == 100000:
        with pytest.raises(ValueError, match="insufficient capacity"):
            acquisition.acquire_youtube(**kwargs, runner=runner)
        assert not downloads
    else:
        ledger = acquisition.acquire_youtube(**kwargs, runner=runner)
        assert len(downloads) == 1
        assert all(row["availability"] == "available" for row in ledger["videos"].values())


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


def test_resume_reuses_only_verified_unchanged_artifacts(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    import time

    from asl_recognizer.translation import sol_common

    root, kwargs, runner = fixture(tmp_path, monkeypatch)
    acquisition.acquire_youtube(**kwargs, runner=runner)
    monkeypatch.setattr(sol_common.sys, 'platform', 'linux')
    monkeypatch.setattr(time, 'time_ns', lambda: 10**30)
    ledger = read_json(root / 'acquisition.json')
    for row in ledger['videos'].values():
        for saved in row['files']:
            assert sol_common.verify_seal(saved)
    write_json(root / 'acquisition.json', ledger)
    digest = Mock(side_effect=AssertionError('unchanged artifact must not be re-read'))
    monkeypatch.setattr(sol_common, 'file_hash', digest)
    calls = Mock(side_effect=AssertionError('completed source must not be requested'))
    ledger = acquisition.acquire_youtube(**kwargs, runner=calls)
    assert len(ledger['videos']) == 3
    assert all(row['availability'] == 'available' for row in ledger['videos'].values())
    assert all('verified_stat' in saved
               for row in read_json(root / 'acquisition.json')['videos'].values()
               for saved in row['files'])
    digest.assert_not_called()
    calls.assert_not_called()
