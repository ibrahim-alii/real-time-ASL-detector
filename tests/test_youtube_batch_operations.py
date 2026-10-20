"""Operational batch reporting must not mistake an exhausted retry pass for coverage."""
from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def load_batch():  # type: ignore[no-untyped-def]
    path = Path(__file__).resolve().parents[1] / 'scripts/sol_youtube_batch.py'
    spec = importlib.util.spec_from_file_location('sol_youtube_batch', path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exhausted_failures_are_not_downloaded_coverage() -> None:
    batch = load_batch()
    ledger = {'videos': {
        'one': {'availability': 'available', 'split': 'train'},
        'two': {'availability': 'unavailable', 'attempts': 3},
    }}
    report = batch.coverage(ledger, ['one', 'two'], 3)
    assert report['pass_exhausted'] is True
    assert report['all_sources_downloaded'] is False
    assert report['available'] == 1 and report['exhausted_unavailable'] == 1
    report = batch.coverage(ledger, ['one', 'two', 'three'], 3)
    assert report['pending_sources'] == 1 and report['pass_exhausted'] is False


def test_only_official_sources_count_toward_completion() -> None:
    batch = load_batch()
    ledger = {'videos': {
        'one': {'availability': 'available', 'split': 'dev'},
        'extra': {'availability': 'available', 'split': 'train'},
    }}
    report = batch.coverage(ledger, ['one'], 3)
    assert report['all_sources_downloaded'] is True
    assert report['available'] == 1 and report['available_by_split'] == {'dev': 1}


def test_project_headroom_tracks_parallel_download_growth() -> None:
    batch = load_batch()
    with patch.object(batch.subprocess, 'run', side_effect=[
        SimpleNamespace(stdout='100 scratch'), SimpleNamespace(stdout='750 scratch'),
    ]):
        assert batch.project_headroom(Path('scratch'), budget=1000) == 900
        assert batch.project_headroom(Path('scratch'), budget=1000) == 250


def test_headroom_retries_disappearing_entries_without_using_partial_size() -> None:
    batch = load_batch()
    missing = batch.subprocess.CalledProcessError(
        1, ['du'], output='100 scratch', stderr='du: temporary: No such file or directory',
    )
    with patch.object(batch.subprocess, 'run', side_effect=[
        missing, SimpleNamespace(stdout='750 scratch'),
    ]) as run:
        assert batch.project_headroom(Path('scratch'), budget=1000) == 250
    assert run.call_count == 2


@pytest.mark.parametrize(('diagnostic', 'attempts'), [
    ('No such file or directory', 3), ('Permission denied', 1),
    ('No such file or directory\nPermission denied', 1), ('', 1),
])
def test_headroom_stops_after_bounded_or_nontransient_scan_failure(
    diagnostic: str, attempts: int,
) -> None:
    batch = load_batch()
    error = batch.subprocess.CalledProcessError(1, ['du'], output='1 scratch', stderr=diagnostic)
    with (patch.object(batch.subprocess, 'run', side_effect=error) as run,
          pytest.raises(batch.subprocess.CalledProcessError)):
        batch.project_headroom(Path('scratch'), budget=1000)
    assert run.call_count == attempts


def test_permanent_and_exhausted_counts_are_disjoint() -> None:
    batch = load_batch()
    report = batch.coverage({'videos': {
        'one': {'availability': 'unavailable', 'attempts': 3,
                'reason': 'manual-English-captions-unavailable'},
    }}, ['one'], 3)
    assert report['permanent_or_invalid_sources'] == 1
    assert report['exhausted_unavailable'] == 0
    assert report['pass_exhausted'] is True and report['all_sources_downloaded'] is False


@pytest.mark.parametrize('diagnostic', ['HTTP 429', 'Please sign in', 'HTTP 403'])
def test_authenticated_errors_stop_without_exposing_response(
    diagnostic: str, tmp_path: Path,
) -> None:
    batch = load_batch()
    batch.COOKIE = tmp_path / 'cookies.txt'
    batch.COOKIE.write_text('private fixture')
    response = SimpleNamespace(returncode=1, stdout='', stderr=diagnostic + ' SECRET_VALUE')
    with (patch.object(batch.subprocess, 'run', return_value=response),
          pytest.raises(RuntimeError) as error):
        batch.runner(['python', '-m', 'yt_dlp', 'https://www.youtube.com/watch?v=one'])
    assert 'SECRET_VALUE' not in str(error.value)


def test_cookie_argument_applies_only_to_youtube_commands(tmp_path: Path) -> None:
    batch = load_batch()
    batch.COOKIE = tmp_path / 'cookies.txt'
    batch.COOKIE.write_text('private fixture')
    response = SimpleNamespace(returncode=0, stdout='{}', stderr='')
    with patch.object(batch.subprocess, 'run', return_value=response) as call:
        batch.runner(['python', '-m', 'yt_dlp', 'https://www.youtube.com/watch?v=one'])
        args = call.call_args.args[0]
        assert args[-1].startswith('https://www.youtube.com/')
        copied = Path(args[args.index('--cookies') + 1])
        assert copied != batch.COOKIE and not copied.exists()
        batch.runner(['ffprobe', 'video.mp4'])
        assert '--cookies' not in call.call_args.args[0]


@pytest.mark.parametrize(('diagnostic', 'expected', 'flag'), [
    ('Private video. Sign in if granted access', 'source unavailable', 'private_video'),
    ('HTTP 403; cookies FAQ', '403 YouTube access denied', 'http403'),
    ('Private video; cookies are no longer valid', 'sign in / cookies rejected', 'cookie_invalid'),
    ('Sign in to confirm your age', 'sign in / cookies rejected', 'age_required'),
    ('Sign in to confirm you are not a bot', 'sign in / cookies rejected', 'human_verification'),
    ('HTTP 429; Private video', '429 YouTube rate limit', 'http429'),
    ('HTTP 403; Private video', '403 YouTube access denied', 'http403'),
])
def test_failure_flags_distinguish_source_and_session_without_raw_response(
    diagnostic: str, expected: str, flag: str, tmp_path: Path,
) -> None:
    batch = load_batch()
    batch.COOKIE = tmp_path / 'cookies.txt'
    batch.COOKIE.write_text('synthetic')
    response = SimpleNamespace(returncode=1, stdout='', stderr=diagnostic + ' SECRET_VALUE')
    with (patch.object(batch.subprocess, 'run', return_value=response),
          pytest.raises(RuntimeError, match=expected) as error):
        batch.runner(['python', '-m', 'yt_dlp', 'synthetic-source'])
    detail = str(error.value)
    assert 'SECRET_VALUE' not in detail
    flags = json.loads(detail.split('; diagnostic_flags=', 1)[1])
    assert flags[flag] is True
    assert all(type(value) is bool for value in flags.values())


@pytest.mark.parametrize(('diagnostic', 'expected'), [
    ('HTTP Error 403; see FAQ#how-do-i-pass-cookies', '403 YouTube access denied'),
    ('Video unavailable; see FAQ#how-do-i-pass-cookies', 'source unavailable'),
    ('WARNING: cookies are no longer valid', 'sign in / cookies rejected'),
    ('Sign in to confirm your age; see cookies FAQ', 'sign in / cookies rejected'),
    ('HTTP Error 429; see cookies FAQ', '429 YouTube rate limit'),
])
def test_cookie_help_link_does_not_override_actual_failure(
    diagnostic: str, expected: str, tmp_path: Path,
) -> None:
    batch = load_batch()
    batch.COOKIE = tmp_path / 'cookies.txt'
    batch.COOKIE.write_text('private fixture')
    response = SimpleNamespace(returncode=1, stdout='', stderr=diagnostic + ' SECRET_VALUE')
    with (patch.object(batch.subprocess, 'run', return_value=response),
          pytest.raises(RuntimeError, match=expected) as failure):
        batch.runner(['python', '-m', 'yt_dlp', 'source'])
    assert 'SECRET_VALUE' not in str(failure.value)


def test_concurrent_commands_isolate_and_remove_cookie_jars(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    batch = load_batch()
    batch.COOKIE = tmp_path / 'cookies.txt'
    batch.COOKIE.write_text('private fixture')
    barrier = Barrier(3)
    copies: list[Path] = []

    def download(args, **kwargs):  # type: ignore[no-untyped-def]
        cookie = Path(args[args.index('--cookies') + 1])
        copies.append(cookie)
        assert cookie.read_text() == 'private fixture'
        cookie.write_text('updated worker jar')
        barrier.wait(timeout=5)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    with (patch.object(batch.subprocess, 'run', side_effect=download),
          ThreadPoolExecutor(max_workers=3) as pool):
        results = [pool.submit(batch.runner, ['python', '-m', 'yt_dlp', 'source'])
                   for _ in range(3)]
        for result in results:
            result.result()
    assert len(set(copies)) == 3
    assert all(not path.exists() for path in copies)
    assert batch.COOKIE.read_text() == 'private fixture'


@pytest.mark.parametrize('previous_status', ['completed', 'failed'])
def test_large_batch_parameters_and_failed_predecessor_guard(
    tmp_path, monkeypatch, previous_status
):  # type: ignore[no-untyped-def]
    batch = load_batch()
    run, scratch = tmp_path / 'remote', tmp_path / 'scratch'
    staging = scratch / 'youtube-asl'
    staging.mkdir(parents=True)
    (staging / 'official-ids.txt').write_text('one\ntwo\n')
    directory = run / 'reports' / 'large-round'
    monkeypatch.setattr(batch, 'RUN', run)
    monkeypatch.setattr(batch, 'SCRATCH', scratch)
    monkeypatch.setattr(batch, 'COOKIE', SimpleNamespace(
        is_file=lambda: True, stat=lambda: SimpleNamespace(st_mode=0o100600)))
    monkeypatch.setattr(batch, 'read_json', lambda path: (
        {'status': previous_status} if path.name == 'previous.json'
        else {'status': 'completed', 'available': 1}))
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='0'))
    monkeypatch.setattr(batch.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(batch, 'writer_lock', lambda path: nullcontext())
    acquire = Mock(return_value={
        'videos': {'one': {'availability': 'available', 'split': 'train', 'attempts': 1}}})
    monkeypatch.setattr(batch, 'acquire_youtube', acquire)
    monkeypatch.setattr(sys, 'argv', ['batch', 'batch', '--round-dir', str(directory),
                                    '--workers', '3',
                                    '--previous-report', str(run / 'previous.json')])
    if previous_status == 'failed':
        with pytest.raises(RuntimeError, match='predecessor'):
            batch.main()
        acquire.assert_not_called()
    else:
        batch.main()
        assert acquire.call_args.kwargs['video_limit'] == 500
        assert acquire.call_args.kwargs['byte_limit'] == 256 * 1024**3
        assert acquire.call_args.kwargs['workers'] == 3
        report = json.loads((directory / 'batch-report.json').read_text())
        assert report['batch_progress']['acquired_pairs'] == 1


def test_batch_progress_excludes_cached_and_cancelled_sources() -> None:
    batch = load_batch()
    cached = {'availability': 'available', 'attempts': 1, 'files': [{'bytes': 1000}]}
    before = {'videos': {'cached': cached}}
    after = {'videos': {
        'cached': cached,
        'new': {'availability': 'available', 'attempts': 1, 'files': [{'bytes': 200}]},
        'blocked': {'availability': 'unavailable', 'attempts': 1, 'reason': 'HTTP 403'},
        'no-caption': {'availability': 'unavailable', 'attempts': 1,
                       'reason': 'manual-English-captions-unavailable'},
    }}
    report = batch.batch_progress(before, after, 120)
    assert report == {
        'attempted_sources': 3, 'acquired_pairs': 1, 'paired_artifact_bytes': 200,
        'failure_categories': {'access_blocked': 1, 'permanent_unavailable': 1},
        'seconds': 120, 'pairs_per_hour': 30,
    }
    assert batch.batch_progress(before, before, 0)['pairs_per_hour'] is None


def test_batch_progress_counts_successful_retry_and_repair() -> None:
    batch = load_batch()
    before = {'videos': {
        'retry': {'availability': 'unavailable', 'attempts': 1},
        'repair': {'availability': 'available', 'attempts': 1},
    }}
    after = {'videos': {key: {'availability': 'available', 'attempts': 2}
                        for key in before['videos']}}
    assert batch.batch_progress(before, after, 3600)['acquired_pairs'] == 2


def test_failed_batch_reports_persisted_partial_progress(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    batch = load_batch()
    run, scratch = tmp_path / 'run', tmp_path / 'scratch'
    staging = scratch / 'youtube-asl'
    staging.mkdir(parents=True)
    (staging / 'official-ids.txt').write_text('one\ntwo\n')
    directory = run / 'reports' / 'failed-round'
    probe = tmp_path / 'probe.json'
    probe.write_text(json.dumps({'status': 'completed', 'available': 1}))
    monkeypatch.setattr(batch, 'RUN', run)
    monkeypatch.setattr(batch, 'SCRATCH', scratch)
    monkeypatch.setattr(batch, 'COOKIE', SimpleNamespace(
        is_file=lambda: True, stat=lambda: SimpleNamespace(st_mode=0o100600)))
    monkeypatch.setattr(batch.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='0'))
    monkeypatch.setattr(batch.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1024**4))
    monkeypatch.setattr(batch, 'writer_lock', lambda path: nullcontext())

    def fail(**kwargs):  # type: ignore[no-untyped-def]
        batch.write_json(staging / 'acquisition.json', {'videos': {
            'one': {'availability': 'available', 'attempts': 1},
            'two': {'availability': 'unavailable', 'attempts': 1, 'reason': 'HTTP 429'},
        }})
        raise RuntimeError('authentication requires investigation')

    monkeypatch.setattr(batch, 'acquire_youtube', fail)
    monkeypatch.setattr(sys, 'argv', ['batch', 'batch', '--round-dir', str(directory),
                                    '--probe-report', str(probe)])
    with pytest.raises(RuntimeError, match='investigation'):
        batch.main()
    report = json.loads((directory / 'batch-report.json').read_text())
    assert report['status'] == 'failed'
    assert report['batch_progress']['attempted_sources'] == 2
    assert report['batch_progress']['acquired_pairs'] == 1
    assert report['batch_progress']['failure_categories'] == {'access_blocked': 1}
