"""Bounded, resumable acquisition; unavailable sources never become substitutions."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from threading import Event, Lock
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit
from zipfile import ZipFile

from .sol_common import (
    atomic_replace,
    capacity,
    read_json,
    scan_tree_bytes,
    seal,
    verify_seal,
    write_json,
)
from .youtube import channel_split, file_hash

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def google_drive_download_url(url: str) -> str:
    """Preserve explicit download mode through Google's large-file confirmation."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "drive.google.com":
        raise ValueError("expected an official Google Drive file URL")
    match = re.fullmatch(r"/file/d/([A-Za-z0-9_-]+)/view", parsed.path)
    file_id = match[1] if match else parse_qs(parsed.query).get("id", [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError("Google Drive URL lacks a valid file ID")
    return "https://drive.google.com/uc?" + urlencode({"export": "download", "id": file_id})


def command(args: list[str], *, timeout: int = 3600) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        # Downloader stderr may contain expiring URLs; keep only the final diagnostic.
        raise RuntimeError(result.stderr.strip().splitlines()[-1:] or ["command failed"])
    return result.stdout


def json3_captions(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in raw.get("events", []):
        text = html.unescape("".join(str(s.get("utf8", "")) for s in event.get("segs", [])))
        text = " ".join(text.split())
        if not text:
            continue
        start, duration = event.get("tStartMs"), event.get("dDurationMs")
        if (
            isinstance(start, bool)
            or isinstance(duration, bool)
            or not isinstance(start, (int, float))
            or not isinstance(duration, (int, float))
        ):
            raise ValueError("caption timing is malformed")
        if start < 0 or duration <= 0:
            raise ValueError("caption interval must be positive")
        rows.append({"start_ms": round(start), "end_ms": round(start + duration), "caption": text})
    if not rows:
        raise ValueError("manual English caption track is empty")
    return rows


def english_track(info: dict[str, Any]) -> str:
    tracks = info.get("subtitles") or {}
    choices = sorted(
        k
        for k in tracks
        if (k == "en" or k.startswith("en-"))
        and not any(
            "tlang=" in f.get("url", "") or "kind=asr" in f.get("url", "") for f in tracks[k]
        )
    )
    if not choices:
        raise ValueError("manual-English-captions-unavailable")
    return "en" if "en" in choices else choices[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    atomic_replace(temporary, path)


def fetch_ids(url: str, output: Path) -> None:
    if output.exists():
        ids = output.read_text(encoding="utf-8").splitlines()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            ids = response.read().decode("utf-8").splitlines()
    if not ids or any(not VIDEO_ID.fullmatch(item.strip()) for item in ids):
        raise ValueError("official ID list has invalid video IDs")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(dict.fromkeys(item.strip() for item in ids)) + "\n", encoding="utf-8"
        )


def acquisition_failure_kind(reason: str) -> str:
    """Avoid repeatedly requesting sources that cannot supply the required paired data."""
    reason = reason.lower().split("; diagnostic_flags=", 1)[0]
    if any(word in reason for word in (
        "manual-english-captions-unavailable", "source unavailable", "video is unavailable",
        "private video", "video has been removed", "video has been deleted",
    )):
        return "permanent_unavailable"
    if any(word in reason for word in (
        "caption boundaries exceed", "caption timing is malformed", "caption interval",
        "manual english caption track is empty",
    )):
        return "invalid_data"
    if any(word in reason for word in (
        "429", "too many requests", "sign in", "confirm you", "cookies", "403",
        "throttling/authentication requires investigation",
    )):
        return "access_blocked"
    return "retryable"


def _save_youtube_acquisition(root: Path, ledger: dict[str, Any]) -> None:
    """Publish the ledger and its index together, including on interrupted batches."""
    write_json(root / "acquisition.json", ledger)
    fields = [
        "video_id", "channel_id", "video_path", "caption_path", "downloaded_at",
        "availability", "split", "signer_id",
    ]
    # Unknown channels cannot be split safely; retain those failures in the ledger.
    index = [v for v in ledger["videos"].values() if v["channel_id"]]
    write_csv(root / "index.csv", index, fields)


class _AcquisitionStopped(Exception):
    """An in-flight source stopped before its next network operation."""


class _Reservations:
    """Conservative aggregate reservations for concurrent source downloads."""

    def __init__(self, root: Path, limit: int, remaining: Callable[[], int]) -> None:
        self.root, self.limit, self.remaining = root, limit, remaining
        self.lock = Lock()
        self.outstanding = 0

    def reserve(self, estimated: int, muxed: bool) -> int:
        # Include the existing 30% capacity margin in the hard per-source budget.
        allocation = math.ceil(estimated * 1.3)
        required = allocation * (2 if muxed else 1)
        with self.lock:
            headroom = min(self.remaining(), self.limit - scan_tree_bytes(self.root),
                           shutil.disk_usage(self.root).free)
            capacity(self.root, required, remaining_quota=max(0, headroom - self.outstanding),
                     margin=0)
            self.outstanding += required
        return allocation

    def release(self, allocation: int, muxed: bool) -> None:
        with self.lock:
            self.outstanding -= allocation * (2 if muxed else 1)


def _acquire_youtube_source(
    row: dict[str, Any], *, root: Path, runner: Callable[..., str],
    reservations: _Reservations, stop: Event, seed: int,
) -> tuple[dict[str, Any], Exception | None, bool]:
    """Write only this source's files; the coordinator owns all ledger writes."""
    video_id = str(row["video_id"])
    url = f"https://www.youtube.com/watch?v={video_id}"
    base = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-plugin-dirs",
            "--no-playlist", "--js-runtimes", "node", "--retries", "2",
            "--fragment-retries", "2", "--socket-timeout", "30"]
    allocation = 0
    remove_audio = False

    def checked_runner(args: list[str]) -> str:
        if "yt_dlp" in args and stop.is_set():
            raise _AcquisitionStopped()
        return runner(args)

    try:
        info = json.loads(checked_runner([*base, "--skip-download", "--dump-single-json", url]))
        channel = info.get("channel_id")
        if not isinstance(channel, str) or not channel:
            raise ValueError("channel identity unavailable")
        row.update(channel_id=channel, split=channel_split(channel, seed))
        track = english_track(info)
        row["caption_track"] = track
        formats = [
            f
            for f in info.get("formats", [])
            if f.get("ext") == "mp4"
            and f.get("vcodec") not in (None, "none")
            and 0 < (f.get("height") or 0) <= 720
        ]
        if not formats:
            raise ValueError("no suitable mp4 video format")
        video_only = [f for f in formats if f.get("acodec") == "none"]
        formats = video_only or formats
        selected = max(formats, key=lambda f: (f.get("height") or 0, f.get("tbr") or 0))
        remove_audio = selected.get("acodec") != "none"
        estimated = selected.get("filesize")
        if not estimated:
            # Approximate sizes/bitrates can understate muxed files severalfold.
            estimated = max(
                selected.get("filesize_approx") or 0,
                max(selected.get("tbr") or 0, 10000) * 1000 / 8 * info.get("duration", 0),
            )
        if estimated <= 0:
            raise ValueError("video size cannot be estimated")
        allocation = reservations.reserve(int(estimated), remove_audio)
        raw_root = root / "raw-captions"
        raw_root.mkdir(exist_ok=True)
        checked_runner(
            [
                *base,
                "--skip-download",
                "--write-subs",
                "--no-write-auto-subs",
                "--sub-langs",
                track,
                "--sub-format",
                "json3",
                "--write-info-json",
                "-o",
                str(raw_root / "%(id)s.%(ext)s"),
                url,
            ]
        )
        raw_caption = raw_root / f"{video_id}.{track}.json3"
        rows = json3_captions(read_json(raw_caption))
        video = root / str(row["video_path"])
        video.parent.mkdir(exist_ok=True)
        checked_runner(
            [
                *base,
                "-f",
                str(selected["format_id"]),
                "--max-filesize",
                str(allocation),
                "-o",
                str(video),
                url,
            ]
        )
        if video.stat().st_size > allocation:
            raise ValueError("insufficient capacity: source exceeded reservation")
        if remove_audio:
            # The reservation includes both the original and video-only MP4.
            stripped = video.with_suffix(".video-only.mp4")
            checked_runner([
                "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(video),
                "-map", "0:v:0", "-c:v", "copy", "-an", str(stripped),
            ])
            stripped.replace(video)
        probe = json.loads(
            checked_runner(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )
        if not any(s.get("codec_type") == "video" for s in probe.get("streams", [])):
            raise ValueError("downloaded file has no video stream")
        if any(s.get("codec_type") == "audio" for s in probe.get("streams", [])):
            raise ValueError("training video must not contain audio")
        duration_ms = float(probe["format"]["duration"]) * 1000
        if any(r["end_ms"] > duration_ms + 100 for r in rows):
            raise ValueError("caption boundaries exceed video duration")
        caption = root / str(row["caption_path"])
        write_csv(caption, rows)
        row.update(
            availability="available",
            format_id=selected["format_id"],
            source_audio_removed=remove_audio,
            duration_ms=duration_ms,
            files=[
                seal(video),
                seal(caption),
                seal(raw_caption),
                seal(raw_root / f"{video_id}.info.json"),
            ],
        )
    except _AcquisitionStopped:
        # An interrupted source retains its old retry allowance on resume.
        return row, None, True
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        row["reason"] = str(error)
        row["failure_kind"] = acquisition_failure_kind(str(error))
        if "insufficient capacity" in str(error):
            stop.set()
            return row, error, False
        if row["failure_kind"] == "access_blocked":
            stop.set()
            error = RuntimeError("YouTube throttling/authentication requires investigation")
            return row, error, False
    finally:
        if allocation:
            reservations.release(allocation, remove_audio)
    return row, None, False


def acquire_youtube(
    *,
    ids_path: Path,
    root: Path,
    byte_limit: int,
    scratch_remaining: Callable[[], int],
    attempts: int = 3,
    seed: int = 42,
    runner: Callable[..., str] = command,
    video_limit: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Acquire another deterministic batch; resume verifies both video and caption files."""
    if workers not in {1, 2, 3, 4, 10, 20}:
        raise ValueError("workers must be one of 1, 2, 3, 4, 10, 20")
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "acquisition.json"
    identity = {
        "official_ids_sha256": file_hash(ids_path),
        "seed": seed,
        "format_policy": "video-only-mp4-at-most-720p-manual-English-json3-v1",
    }
    ledger: dict[str, Any] = (
        read_json(ledger_path) if ledger_path.exists() else {"identity": identity, "videos": {}}
    )
    if ledger["identity"] != identity:
        raise ValueError("acquisition identity changed; use a new run")
    ids = list(dict.fromkeys(ids_path.read_text(encoding="utf-8").splitlines()))
    if attempts < 1 or byte_limit < 1 or any(not VIDEO_ID.fullmatch(i) for i in ids):
        raise ValueError("invalid acquisition limits or IDs")
    ids.sort(key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    def candidates() -> Iterator[tuple[str, dict[str, Any] | None]]:
        for video_id in ids:
            previous = ledger["videos"].get(video_id)
            if previous and previous["availability"] != "available":
                kind = acquisition_failure_kind(str(previous.get("reason", "")))
                previous["failure_kind"] = kind
                if kind in {"permanent_unavailable", "invalid_data"}:
                    continue
            if (
                previous
                and previous["availability"] == "available"
                and all(verify_seal(s, cached=True) for s in previous["files"])
            ):
                continue
            if previous and previous["availability"] == "available":
                # A completed but corrupt file must not be accepted by yt-dlp's exists shortcut.
                for saved in previous["files"]:
                    path = Path(saved["path"])
                    if path.is_file() and not verify_seal(saved):
                        path.replace(path.with_name(path.name + f".corrupt-{previous['attempts']}"))
                previous["availability"] = "changed"
                previous["reason"] = "completed-file-integrity-check-failed"
                _save_youtube_acquisition(root, ledger)
            if previous and previous.get("attempts", 0) >= attempts:
                continue
            yield video_id, previous

    stop = Event()
    reservations = _Reservations(root, byte_limit, scratch_remaining)
    pending: dict[Future[tuple[dict[str, Any], Exception | None, bool]],
                  tuple[str, dict[str, Any] | None]] = {}
    fatal: Exception | None = None
    processed = 0
    source = iter(candidates())
    exhausted = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or not exhausted:
            while not stop.is_set() and not exhausted and len(pending) < workers:
                if video_limit is not None and processed >= video_limit:
                    exhausted = True
                    break
                try:
                    video_id, previous = next(source)
                except StopIteration:
                    exhausted = True
                    break
                if scan_tree_bytes(root) >= byte_limit:
                    stop.set()
                    fatal = ValueError(
                        "YouTube staging byte limit reached before enough accepted pairs"
                    )
                    break
                row: dict[str, Any] = {
                    "video_id": video_id, "channel_id": "",
                    "video_path": f"videos/{video_id}.mp4",
                    "caption_path": f"captions/{video_id}.csv", "split": "",
                    "downloaded_at": datetime.now(UTC).isoformat(),
                    "availability": "unavailable",
                    "attempts": (previous or {}).get("attempts", 0) + 1,
                }
                ledger["videos"][video_id] = dict(row)
                _save_youtube_acquisition(root, ledger)
                future = pool.submit(_acquire_youtube_source, row, root=root, runner=runner,
                                     reservations=reservations, stop=stop, seed=seed)
                pending[future] = (video_id, previous)
                processed += 1
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                video_id, previous = pending.pop(future)
                try:
                    row, error, untouched = future.result()
                except Exception as unexpected:
                    # Drain other workers and save their outcomes before propagating.
                    row = ledger["videos"][video_id]
                    row.update(reason=f"unexpected acquisition error: {type(unexpected).__name__}",
                               failure_kind="retryable")
                    error, untouched = unexpected, False
                    stop.set()
                if untouched:
                    if previous is None:
                        ledger["videos"].pop(video_id, None)
                    else:
                        ledger["videos"][video_id] = previous
                else:
                    ledger["videos"][video_id] = row
                if error is not None:
                    fatal = fatal or error
                    stop.set()
                _save_youtube_acquisition(root, ledger)
            if scan_tree_bytes(root) > byte_limit:
                fatal = fatal or ValueError("YouTube staging exceeded byte limit; stop acquisition")
                stop.set()
            if not stop.is_set():
                time.sleep(2)
    _save_youtube_acquisition(root, ledger)
    if fatal is not None:
        raise fatal
    return ledger


def extract_recordings(
    archive: Path,
    wanted: set[str],
    output: Path,
    *,
    remaining_quota: int,
) -> set[str]:
    """Extract only selected full recordings, with ZIP CRC verification and path safety."""
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        selected = [
            i
            for i in bundle.infolist()
            if Path(i.filename).stem in wanted and i.filename.lower().endswith(".mp4")
        ]
        stems = [Path(i.filename).stem for i in selected]
        if len(stems) != len(set(stems)):
            raise ValueError("duplicate recording names in archive")
        capacity(output, sum(i.file_size for i in selected), remaining_quota=remaining_quota)
        for member in selected:
            target = output / (Path(member.filename).stem + ".mp4")
            marker = target.with_suffix(".json")
            if marker.exists() and verify_seal(read_json(marker)):
                continue
            temporary = target.with_suffix(".mp4.part")
            with bundle.open(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            if temporary.stat().st_size != member.file_size:
                raise ValueError("recording extraction size mismatch")
            temporary.replace(target)
            write_json(marker, seal(target))
    return set(stems)


def recording_archive_groups(
    entries: list[tuple[str, Path]],
) -> list[list[tuple[str, Path]]]:
    """Group numbered split ZIP volumes with their final central-directory file."""
    files = {path.name: (url, path) for url, path in entries}
    if len(files) != len(entries):
        raise ValueError("duplicate recording archive filenames")
    groups: dict[str, list[tuple[int, tuple[str, Path]]]] = {}
    finals: dict[str, tuple[str, Path]] = {}
    for name, entry in files.items():
        path = entry[1]
        if path.suffix == ".zip":
            finals[path.stem] = entry
        elif match := re.fullmatch(r"(.+)\.z([0-9]{2,})", name):
            groups.setdefault(match[1], []).append((int(match[2]), entry))
        else:
            raise ValueError(f"unexpected official recording archive: {name}")
    if groups.keys() - finals.keys():
        raise ValueError("split recording archive lacks final ZIP volume")
    result = []
    for stem, final in sorted(finals.items()):
        parts = sorted(groups.get(stem, []))
        if [number for number, _ in parts] != list(range(1, len(parts) + 1)):
            raise ValueError("split recording archive has missing or duplicate volumes")
        result.append([*(entry for _, entry in parts), final])
    return result



def extract_split_recordings(
    parts: list[Path], wanted: set[str], output: Path, *,
    tool: Path, remaining_quota: int, timeout: int = 21600,
) -> set[str]:
    """Stream selected split-ZIP members through 7-Zip with independent CRC checks."""
    import zlib

    if len(parts) < 2 or parts[-1].suffix != ".zip":
        raise ValueError("split extraction requires numbered volumes and a final ZIP")
    sources = []
    for part in parts:
        source = read_json(part.with_suffix(part.suffix + ".json"))
        if Path(source["path"]) != part or not verify_seal(source):
            raise ValueError("split recording archive part failed integrity check")
        sources.append(source)
    listing = subprocess.run(
        [str(tool), "l", "-slt", "-ba", str(parts[-1].resolve())],
        capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL, timeout=120,
    ).stdout
    selected: dict[str, dict[str, str]] = {}
    for block in listing.strip().split("\n\n"):
        fields = dict(line.split(" = ", 1) for line in block.splitlines() if " = " in line)
        name = fields.get("Path", "")
        member = PurePosixPath(name.replace("\\", "/"))
        if member.suffix.lower() != ".mp4" or member.stem not in wanted:
            continue
        if (member.is_absolute() or ".." in member.parts or ":" in name
                or fields.get("Folder") != "-" or fields.get("Encrypted") != "-"
                or (fields.get("Attributes", "").split() or [""])[-1].startswith("l")):
            raise ValueError("unsafe or encrypted recording archive member")
        if member.stem in selected:
            raise ValueError("duplicate recording names in archive")
        if (not fields.get("Size", "").isdigit()
                or not re.fullmatch(r"[0-9A-Fa-f]{8}", fields.get("CRC", ""))):
            raise ValueError("recording archive member lacks size or CRC")
        selected[member.stem] = fields
    output.mkdir(parents=True, exist_ok=True)
    pending: dict[str, dict[str, Any]] = {}
    for stem, fields in selected.items():
        target = output / (stem + ".mp4")
        marker = target.with_suffix(".json")
        identity = {"archive_parts": sources, "member": fields["Path"],
                    "member_crc": fields["CRC"], "member_bytes": int(fields["Size"])}
        if marker.exists():
            saved = read_json(marker)
            if all(saved.get(k) == v for k, v in identity.items()) and verify_seal(saved):
                continue
        pending[stem] = identity
    capacity(output, sum(int(entry["member_bytes"]) for entry in pending.values()),
             remaining_quota=remaining_quota)
    for stem, identity in sorted(pending.items()):
        target = output / (stem + ".mp4")
        temporary = target.with_suffix(".mp4.part")
        try:
            with temporary.open("wb") as writer:
                subprocess.run(
                    [str(tool), "x", "-so", "-spd", str(parts[-1].resolve()),
                     str(identity["member"])],
                    stdout=writer, stderr=subprocess.PIPE, check=True,
                    stdin=subprocess.DEVNULL, timeout=timeout,
                )
            checksum = 0
            with temporary.open("rb") as reader:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    checksum = zlib.crc32(chunk, checksum)
            if (temporary.stat().st_size != identity["member_bytes"]
                    or checksum != int(str(identity["member_crc"]), 16)):
                raise ValueError("recording extraction size or CRC mismatch")
            temporary.replace(target)
            write_json(target.with_suffix(".json"), {**seal(target), **identity})
        finally:
            temporary.unlink(missing_ok=True)
    return set(selected)
