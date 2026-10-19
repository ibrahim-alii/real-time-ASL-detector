"""Pinned official 7-Zip binary for reading multipart recording archives."""

from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

from .sol_common import read_json, seal, verify_seal, write_json

URL = "https://github.com/ip7z/7zip/releases/download/26.04/7z2604-linux-x64.tar.xz"
SHA256 = "fc0327ba27e89bd086cf426dff17d77de582953cdbbc10a6576540a06853ffcd"


def ensure_archive_tool(root: Path) -> Path:
    """Verify the release digest before installing only its console executable."""
    binary = root / "7zz"
    marker = root / "7zz.json"
    if marker.exists():
        saved = read_json(marker)
        if (saved.get("release_sha256") == SHA256
                and saved.get("url") == URL and Path(saved["path"]) == binary
                and verify_seal(saved)):
            return binary
        raise ValueError("archive tool identity or checksum changed")
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read(20 * 1024**2 + 1)
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("official 7-Zip release checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:xz") as archive:
        member = archive.getmember("7zz")
        if not member.isfile():
            raise ValueError("archive tool executable must be a regular file")
        reader = archive.extractfile(member)
        if reader is None:
            raise ValueError("archive tool executable missing")
        with reader:
            executable = reader.read()
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "7zz.part"
    temporary.write_bytes(executable)
    temporary.chmod(0o755)
    temporary.replace(binary)
    write_json(marker, {**seal(binary), "url": URL, "release_sha256": SHA256,
                        "version": "26.04"})
    return binary
