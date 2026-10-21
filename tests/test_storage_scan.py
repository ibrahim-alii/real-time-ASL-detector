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
