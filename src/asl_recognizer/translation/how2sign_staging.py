"""Select verified full How2Sign recordings without expanding the complete corpus."""
from __future__ import annotations

import hashlib
import re
import stat
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from zipfile import BadZipFile, ZipFile

from .manifest import read_manifest
from .sol_common import capacity, read_json, write_json
from .youtube import file_hash

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ARCHIVE_SPLITS = {"train_raw_videos": "train", "val_raw_videos": "dev",
                   "test_raw_videos": "test"}


class _Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class _Source:
    split: str
    name: str
    path: Path
    member: str | None
    size: int
    crc: int
    sha256: str | None
    identity: dict[str, Any]


def _inside(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or "\\" in relative or any(p in {".", ".."} for p in parts.parts):
        raise ValueError("unsafe source path")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("unsafe source path")
    return path


def _recording_name(name: str) -> str:
    if (PurePosixPath(name).name != name or "\\" in name
            or not name.endswith("-rgb_front.mp4") or ":" in name):
        raise ValueError("unsafe or non-recording filename")
    return name


def _hash(value: object) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("invalid source SHA256 receipt")
    return value


def _matches(path: Path, source: _Source) -> bool:
    marker = path.with_suffix(".verified.json")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        receipt = read_json(marker)
        saved = receipt["seal"]
        return bool(receipt["source"] == source.identity
                    and saved["path"] == str(path) and saved["bytes"] == source.size
                    and path.stat().st_size == source.size
                    and file_hash(path) == saved["sha256"]
                    and (source.sha256 is None or saved["sha256"] == source.sha256))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _copy_checked(handle: _Readable, temporary: Path, source: _Source) -> dict[str, object]:
    count, crc = 0, 0
    sha = hashlib.sha256()
    try:
        with temporary.open("wb") as target:
            while block := handle.read(8 * 1024**2):
                count += len(block)
                if count > source.size:
                    raise ValueError("source size exceeds verified receipt")
                crc = zlib.crc32(block, crc)
                sha.update(block)
                target.write(block)
        digest = sha.hexdigest()
        if count != source.size or crc != source.crc:
            raise ValueError("source size or CRC does not match verified receipt")
        if source.sha256 is not None and digest != source.sha256:
            raise ValueError("source SHA256 does not match verified receipt")
        return {"bytes": count, "sha256": digest}
    except BadZipFile as error:
        raise ValueError("source ZIP member CRC check failed") from error


def stage_recordings(
    *, manifest_path: Path, catalogue_path: Path, mirror_root: Path,
    recovery_root: Path, recovery_status: Path, output: Path, remaining_quota: int,
) -> dict[str, Any]:
    """Stage accepted train/dev full recordings and seal each output atomically.

    Selected mirror archives receive a fresh SHA256 check against their acquisition
    receipts. Newly staged members receive size/CRC checks; recovered recordings
    also receive SHA256 checks. Reused outputs receive fresh SHA256 checks.
    Test metadata is preserved, never staged.
    """
    manifest = read_manifest(manifest_path)
    wanted: dict[str, str] = {}
    all_splits: dict[str, str] = {}
    for record in manifest.accepted():
        if record.source != "how2sign":
            raise ValueError("staging requires a How2Sign manifest")
        name = _recording_name(record.video_id + ".mp4")
        if name in all_splits and all_splits[name] != record.split:
            raise ValueError("recording spans manifest splits")
        all_splits[name] = record.split
        if record.split in {"train", "dev"}:
            wanted[name] = record.split
        elif record.split != "test":
            raise ValueError("staging requires assigned official splits")
    if not wanted:
        raise ValueError("manifest has no accepted train/dev recordings")

    catalogue = read_json(catalogue_path)
    locations: dict[str, tuple[str, Path]] = {}
    for entry in catalogue["archives"]:
        relative = str(entry["path"])
        path = _inside(mirror_root, relative)
        parts = PurePosixPath(relative).parts
        if len(parts) != 2 or parts[0] not in _ARCHIVE_SPLITS or path.suffix != ".zip":
            raise ValueError("invalid mirror archive layout")
        split = _ARCHIVE_SPLITS[parts[0]]
        for value in entry["recordings"]:
            name = _recording_name(str(value))
            if name in locations:
                raise ValueError("duplicate recording basename in catalogue")
            locations[name] = (split, path)

    recovered: dict[str, dict[str, Any]] = {}
    for row in read_json(recovery_status)["completed"]:
        name = _recording_name(str(row["filename"]))
        if name in recovered or name in locations:
            raise ValueError("duplicate mirror/recovery recording basename")
        if row["split"] != "train":
            raise ValueError("unexpected recovered recording split")
        recovered[name] = row
    if set(wanted) - (locations.keys() | recovered.keys()):
        raise ValueError("selected recording missing from verified sources")

    sources: list[_Source] = []
    checked_archives: dict[Path, tuple[str, dict[str, tuple[int, int]]]] = {}
    for name, split in sorted(wanted.items()):
        if name in locations:
            source_split, archive = locations[name]
            if split != source_split:
                raise ValueError("source recording does not match official split")
            if archive not in checked_archives:
                receipt = read_json(archive.with_suffix(".verified.json"))
                digest = _hash(receipt["sha256"])
                if archive.stat().st_size != receipt["bytes"] or file_hash(archive) != digest:
                    raise ValueError("mirror archive size or SHA256 mismatch")
                with ZipFile(archive) as bundle:
                    members: dict[str, tuple[int, int]] = {}
                    for member in bundle.infolist():
                        member_name = _recording_name(member.filename)
                        if (member_name in members or member.is_dir()
                                or stat.S_ISLNK(member.external_attr >> 16)):
                            raise ValueError("duplicate or unsafe ZIP member")
                        members[member_name] = (member.file_size, member.CRC)
                if set(members) != set(receipt["recordings"]):
                    raise ValueError("archive members differ from acquisition receipt")
                expected = {n for n, (_, p) in locations.items() if p == archive}
                if expected != set(members):
                    raise ValueError("archive members differ from catalogue")
                checked_archives[archive] = (digest, members)
            digest, members = checked_archives[archive]
            size, crc = members[name]
            identity = {"kind": "mirror", "archive": str(archive), "sha256": digest,
                        "member": name, "bytes": size, "crc32": crc, "split": split}
            sources.append(_Source(split, name, archive, name, size, crc, None, identity))
        else:
            row = recovered[name]
            source_path = _inside(recovery_root, "train/" + name)
            receipt = read_json(source_path.with_suffix(".verified.json"))
            keys = ("filename", "bytes", "sha256", "crc32", "split")
            if any(receipt.get(key) != row[key] for key in keys):
                raise ValueError("recovery receipt differs from completion status")
            digest = _hash(row["sha256"])
            size, crc = int(row["bytes"]), int(row["crc32"])
            if source_path.stat().st_size != size:
                raise ValueError("recovered source size mismatch")
            identity = {"kind": "recovery", "filename": name, "sha256": digest,
                        "bytes": size, "crc32": crc, "split": split}
            sources.append(_Source(split, name, source_path, None, size, crc, digest, identity))

    output.mkdir(parents=True, exist_ok=True)
    targets = [(source, _inside(output, source.split + "/" + source.name)) for source in sources]
    needed = sum(source.size for source, path in targets if not _matches(path, source))
    capacity(output, needed, remaining_quota=remaining_quota)
    recordings: list[dict[str, Any]] = []
    reused = 0
    for source, destination in targets:
        if _matches(destination, source):
            reused += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".mp4.part")
            try:
                if source.member is None:
                    with source.path.open("rb") as handle:
                        seal = _copy_checked(handle, temporary, source)
                else:
                    with ZipFile(source.path) as bundle, bundle.open(source.member) as handle:
                        seal = _copy_checked(handle, temporary, source)
                temporary.replace(destination)
                write_json(destination.with_suffix(".verified.json"), {
                    "source": source.identity, "seal": {**seal, "path": str(destination)}})
            finally:
                temporary.unlink(missing_ok=True)
        recordings.append({"path": str(destination), "source": source.identity})
    return {"manifest_hash": manifest.fingerprint, "recordings": recordings, "reused": reused,
            "archive_hashes_recomputed": len(checked_archives), "test_recordings_staged": 0,
            "newly_staged_size_crc_checks": len(recordings) - reused,
            "reused_output_sha256_checks": reused, "training_started": False}
