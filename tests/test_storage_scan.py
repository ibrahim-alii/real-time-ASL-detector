"""Resource scans must retain exact byte counts and fail closed on incomplete walks."""

from unittest.mock import Mock

import pytest

from asl_recognizer.translation import sol_common as s


def test_fast_scan_matches_nested_tree(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "a").write_bytes(b"abc")
    (tmp_path / "d/b").write_bytes(b"12345")
    assert s.scan_tree_bytes(tmp_path) == s.tree_bytes(tmp_path) == 8


def test_fast_scan_retries_complete_scan(tmp_path, monkeypatch):
    (tmp_path / "a").write_bytes(b"123")
    real = s.os.scandir
    scan = Mock(side_effect=[FileNotFoundError("race"), real(tmp_path)])
    monkeypatch.setattr(s.os, "scandir", scan)
    assert s.scan_tree_bytes(tmp_path) == 3 and scan.call_count == 2


@pytest.mark.parametrize(
    "error,calls", [(FileNotFoundError("race"), 3), (PermissionError("denied"), 1)]
)
def test_fast_scan_fails_closed(tmp_path, monkeypatch, error, calls):
    scan = Mock(side_effect=error)
    monkeypatch.setattr(s.os, "scandir", scan)
    with pytest.raises(type(error)):
        s.scan_tree_bytes(tmp_path)
    assert scan.call_count == calls


def test_scan_does_not_follow_directory_symlinks(tmp_path):
    import os

    root = tmp_path / "root"
    root.mkdir()
    (root / "data").write_bytes(b"123")
    other = tmp_path / "other"
    other.mkdir()
    (other / "data").write_bytes(b"123456")
    try:
        os.symlink(other, root / "alias", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink privileges unavailable")
    assert s.scan_tree_bytes(root) == s.tree_bytes(root) == 3


@pytest.mark.parametrize("staging_name", [
    "clip.video-only.mp4", "clip.mp4.part", "clip.mp4.ytdl", "clip.mp4.tmp",
])
def test_publication_rename_counts_destination_conservatively(tmp_path, monkeypatch, staging_name):
    from contextlib import nullcontext

    published = tmp_path / "clip.mp4"
    published.write_bytes(b"old!")
    temporary = tmp_path / staging_name
    temporary.write_bytes(b"new!!!")
    old = Mock(name="published")
    old.name, old.path = published.name, str(published)
    old.is_dir.return_value = False
    old.is_file.return_value = True
    old.stat.side_effect = published.stat
    pending = Mock(name="temporary")
    pending.name, pending.path = temporary.name, str(temporary)
    pending.is_dir.return_value = False
    pending.is_file.return_value = True

    def rename_then_stat():
        temporary.replace(published)
        raise FileNotFoundError(str(temporary))

    pending.stat.side_effect = rename_then_stat
    monkeypatch.setattr(s.os, "scandir", lambda _: nullcontext([old, pending]))
    assert s.scan_tree_bytes(tmp_path) == 10 >= published.stat().st_size


@pytest.mark.parametrize("staging_name", [
    "clip.video-only.mp4", "clip.mp4.part", "clip.mp4.ytdl", "clip.mp4.tmp",
])
def test_missing_staging_and_destination_fails_closed(tmp_path, monkeypatch, staging_name):
    from contextlib import nullcontext

    entry = Mock()
    entry.name, entry.path = staging_name, str(tmp_path / staging_name)
    entry.is_dir.return_value = False
    entry.is_file.return_value = True
    entry.stat.side_effect = FileNotFoundError("removed")
    monkeypatch.setattr(s.os, "scandir", lambda _: nullcontext([entry]))
    with pytest.raises(FileNotFoundError):
        s.scan_tree_bytes(tmp_path)
