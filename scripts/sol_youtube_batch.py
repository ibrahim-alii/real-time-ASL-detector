"""Bounded authenticated acquisition on Sol, using the existing tested downloader."""
from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

from asl_recognizer.translation.acquisition import acquire_youtube, acquisition_failure_kind
from asl_recognizer.translation.sol_common import read_json, write_json, writer_lock

RUN = Path(os.environ.get('ASL_RUN_ROOT', str(Path.home() / 'asl-translation/run')))
SCRATCH = Path(os.environ.get('ASL_SCRATCH_ROOT',
                             f'/scratch/{getpass.getuser()}/asl-translation'))
COOKIE = Path(os.environ.get('ASL_COOKIE_FILE',
                            str(RUN.parent / 'private/youtube-cookies.txt')))
ROUND = RUN / 'reports/youtube-probe'



def quota_headroom(scratch: Path, budget: int, host: str) -> int:
    """User-wide usage conservatively bounds this project's bytes; enforce group quota too."""
    headrooms = []
    for flag, identity in (("--uid", os.getuid()), ("--gid", scratch.stat().st_gid)):
        result = subprocess.run(
            ['beegfs-ctl', '--getquota', flag, str(identity),
             '--sysMgmtdHost=' + host, '--csv'],
            capture_output=True, text=True, check=True, timeout=15,
        )
        rows = [row for row in csv.reader(result.stdout.splitlines())
                if len(row) == 6 and row[1].strip() == str(identity)]
        if len(rows) != 1:
            raise RuntimeError('quota response lacks exactly one matching identity')
        used, files = int(rows[0][2]), int(rows[0][4])
        hard, file_hard = (0 if value.strip().lower() == "unlimited" else int(value)
                           for value in (rows[0][3], rows[0][5]))
        if min(used, hard, files, file_hard) < 0 or (file_hard and files >= file_hard):
            raise RuntimeError('invalid quota response or inode quota exhausted')
        if flag == '--uid':
            headrooms.append(budget - used)
        if hard:
            headrooms.append(hard - used)
    return max(0, min(headrooms))


def project_headroom(scratch: Path, budget: int = 1024**4) -> int:
    """Recheck space consumed by this batch and parallel corpus downloads."""
    if host := os.environ.get("ASL_BEEGFS_QUOTA_HOST"):
        return quota_headroom(scratch, budget, host)
    for attempt in range(3):
        try:
            usage = subprocess.run(['du', '-sb', str(scratch)], capture_output=True,
                                   text=True, check=True, timeout=120)
            break
        except subprocess.CalledProcessError as error:
            # Concurrent writers can remove temporary entries during a tree scan.
            lines = (error.stderr or '').splitlines()
            if attempt == 2 or not lines or any(
                'No such file or directory' not in line for line in lines
            ):
                raise
    return max(0, budget - int(usage.stdout.split()[0]))


def coverage(ledger: dict, ids: list[str], attempts: int) -> dict:
    """Separate downloaded coverage from exhausted failures and unattempted sources."""
    videos = ledger['videos']
    official = set(ids)
    available = [v for key, v in videos.items()
                 if key in official and v['availability'] == 'available']
    permanent = [key for key in ids if key in videos
                 and videos[key]['availability'] != 'available'
                 and acquisition_failure_kind(str(videos[key].get('reason', '')))
                 in {'permanent_unavailable', 'invalid_data'}]
    permanent_ids = set(permanent)
    exhausted = [key for key in ids if key in videos
                 and key not in permanent_ids
                 and videos[key]['availability'] != 'available'
                 and videos[key].get('attempts', 0) >= attempts]
    pending = [key for key in ids if key not in permanent_ids and (key not in videos or (
        videos[key]['availability'] != 'available'
        and videos[key].get('attempts', 0) < attempts))]
    return {'official_sources': len(ids), 'available': len(available),
            'available_by_split': dict(Counter(v['split'] for v in available)),
            'exhausted_unavailable': len(exhausted), 'pending_sources': len(pending),
            'permanent_or_invalid_sources': len(permanent),
            'unvisited_sources': sum(key not in videos for key in ids),
            'all_sources_downloaded': len(available) == len(ids),
            'pass_exhausted': not pending}


def runner(args: list[str], **kwargs: object) -> str:
    with ExitStack() as cleanup:
        if 'yt_dlp' in args:
            # yt-dlp saves its jar on exit; parallel commands need private copies.
            directory = cleanup.enter_context(TemporaryDirectory(
                prefix='youtube-worker-', dir=COOKIE.parent))
            cookie = Path(directory) / 'cookies.txt'
            shutil.copyfile(COOKIE, cookie)
            cookie.chmod(0o600)
            args = [*args[:-1], '--cookies', str(cookie), '--sleep-requests', '2', args[-1]]
        result = subprocess.run(args, capture_output=True, text=True, timeout=600, check=False)
    if result.returncode:
        diagnostic = result.stderr.lower()
        flags = {
            'http429': '429' in diagnostic or 'too many requests' in diagnostic,
            'http403': '403' in diagnostic,
            'cookie_invalid': 'cookies are no longer valid' in diagnostic,
            'sign_in': 'sign in' in diagnostic,
            'confirm_you': 'confirm you' in diagnostic,
            'private_video': 'private video' in diagnostic,
            'age_required': 'confirm your age' in diagnostic,
            'human_verification': 'not a bot' in diagnostic,
        }

        def failure(message: str) -> RuntimeError:
            # Persist only known boolean markers, never raw downloader credentials/URLs.
            return RuntimeError(message + '; diagnostic_flags=' + json.dumps(flags))

        if '429' in diagnostic or 'too many requests' in diagnostic:
            raise failure('429 YouTube rate limit; stop authenticated acquisition')
        if flags['cookie_invalid']:
            raise failure('sign in / cookies rejected; stop authenticated acquisition')
        if flags['private_video'] and not (
            flags['http403'] or flags['age_required'] or flags['human_verification']
        ):
            raise failure('source unavailable')
        if any(word in diagnostic for word in (
            'sign in', 'cookies are no longer valid', 'confirm you',
        )):
            raise failure('sign in / cookies rejected; stop authenticated acquisition')
        if '403' in diagnostic:
            raise failure('403 YouTube access denied; stop authenticated acquisition')
        if 'unavailable' in diagnostic or 'private video' in diagnostic:
            raise failure('source unavailable')
        raise failure(f'downloader or video probe failed (exit {result.returncode})')
    return result.stdout


def batch_progress(before: dict, after: dict, seconds: float) -> dict:
    """Measure this invocation's paired-data yield, excluding cached prior successes."""
    attempted = [row for key, row in after['videos'].items()
                 if row.get('attempts', 0) > before['videos'].get(key, {}).get('attempts', 0)]
    available = [row for row in attempted if row['availability'] == 'available']
    failures = [row for row in attempted if row['availability'] != 'available']
    return {
        'attempted_sources': len(attempted),
        'acquired_pairs': len(available),
        'paired_artifact_bytes': sum(saved['bytes'] for row in available
                                     for saved in row.get('files', [])),
        'failure_categories': dict(Counter(
            acquisition_failure_kind(str(row.get('reason', ''))) for row in failures)),
        'seconds': seconds,
        'pairs_per_hour': len(available) * 3600 / seconds if seconds > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('probe', 'batch'))
    parser.add_argument('--round-dir', type=Path, required=True)
    parser.add_argument('--video-limit', type=int, default=500)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4, 10, 20), default=1)
    parser.add_argument('--byte-limit', type=int, default=256 * 1024**3)
    parser.add_argument('--previous-report', type=Path)
    parser.add_argument('--probe-report', type=Path,
                        default=ROUND / 'probe-report.json')
    args = parser.parse_args()
    directory = args.round_dir.resolve()
    if not directory.is_relative_to(RUN / 'reports'):
        raise ValueError('report directory must be under the fixed Sol ASL project')
    if args.video_limit < 1 or not 0 < args.byte_limit <= 512 * 1024**3:
        raise ValueError('invalid acquisition limits')
    directory.mkdir(parents=True, exist_ok=True)
    official = SCRATCH / 'youtube-asl/official-ids.txt'
    if not COOKIE.is_file() or COOKIE.stat().st_mode & 0o077:
        raise RuntimeError('cookie file must exist with owner-only permissions')
    if args.mode == 'probe':
        video_id = 'P-D0kEmncGw'
        if video_id not in official.read_text().splitlines():
            raise ValueError('probe is not in official dataset')
        root = SCRATCH / 'youtube-auth-mp4-probe-20261018'
        root.mkdir(parents=True, exist_ok=True)
        ids = root / 'official-ids.txt'
        ids.write_text(video_id + '\n')
        byte_limit, video_limit, attempts = 512 * 1024**2, 1, 1
    else:
        probe = read_json(args.probe_report)
        if probe['status'] != 'completed' or probe.get('available', 0) != 1:
            raise RuntimeError('successful one-video probe is required')
        root = SCRATCH / 'youtube-asl'
        ids = official
        byte_limit, video_limit, attempts = args.byte_limit, args.video_limit, 3
    report = {'status': 'running', 'mode': args.mode, 'video_limit': video_limit,
              'byte_limit': byte_limit, 'workers': args.workers,
              'root': str(root), 'training_started': False}
    destination = directory / f'{args.mode}-report.json'
    write_json(destination, report)
    started = time.monotonic()
    try:
        if args.previous_report:
            previous = read_json(args.previous_report)
            if previous['status'] != 'completed':
                raise RuntimeError('predecessor did not complete; investigate before retrying')
        remaining = project_headroom(SCRATCH)
        # This cap is total corpus staging, not a reservation for each invocation.
        # Reserve a bounded increment; every source is also checked by acquire_youtube.
        reservation = min(byte_limit, 2 * 1024**3)
        if remaining < reservation or shutil.disk_usage(SCRATCH).free < reservation:
            raise RuntimeError('insufficient capacity for bounded acquisition')
        report.update(scratch_project_bytes=1024**4 - remaining,
                      project_budget_remaining_bytes=remaining,
                      space_accounting=('conservative-user-and-group-quota'
                                        if os.environ.get('ASL_BEEGFS_QUOTA_HOST')
                                        else 'project-du'))
        write_json(destination, report)
        with writer_lock(root / 'acquisition.lock'):
            ledger_path = root / 'acquisition.json'
            before = read_json(ledger_path) if ledger_path.exists() else {'videos': {}}
            batch_started = time.monotonic()
            ledger = None
            try:
                ledger = acquire_youtube(ids_path=ids, root=root, byte_limit=byte_limit,
                                         scratch_remaining=lambda: project_headroom(SCRATCH),
                                         attempts=attempts, seed=42, runner=runner,
                                         video_limit=video_limit, workers=args.workers)
            finally:
                # The downloader persists completed in-flight work before raising.
                after = ledger if ledger is not None else (
                    read_json(ledger_path) if ledger_path.exists() else before)
                report['batch_progress'] = batch_progress(
                    before, after, time.monotonic() - batch_started)
        report.update(status='completed',
                      **coverage(ledger, ids.read_text().splitlines(), attempts),
                      outcomes=dict(Counter(v['availability'] for v in ledger['videos'].values())))
        if args.mode == 'probe' and report['available'] != 1:
            raise RuntimeError('one-video probe did not acquire a usable pair')
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic() - started
        write_json(destination, report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
