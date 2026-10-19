"""Git guards inspect staged blobs and outgoing history, not ignored credentials."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/git_safety.py'


def load_guard():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location('git_safety', SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('path', [
    'PRD.md', 'docs/plan.txt', 'sub/README.md', '.env', '.aws/config',
    '.asl-sol-access/known_hosts', 'config/sol_translation.json', 'youtube-cookies.txt',
    'models/model.pt', 'data/raw.json', 'artifacts/report.json', '.codex/state.json',
    'reports/audit.json', 'plans/next.txt',
])
def test_protected_paths_rejected(path: str) -> None:
    assert load_guard().path_problem(path)


@pytest.mark.parametrize('path', [
    'README.md', 'LICENSE', 'src/asl_recognizer/models/model.py',
    'src/asl_recognizer/data/index.py', 'config/sol_translation.example.json',
])
def test_public_source_and_license_allowed(path: str) -> None:
    assert load_guard().path_problem(path) is None


def test_historical_license_is_legal_metadata_not_a_secret() -> None:
    guard = load_guard()
    assert guard.path_problem('LICENSE.md')
    assert guard.path_problem('LICENSE.md', historical=True) is None


@pytest.mark.parametrize('content', [
    ('-----BEGIN ' + 'PRIVATE KEY-----').encode(),
    ('ghp_' + 'a' * 36).encode(),
    ('# Netscape ' + 'HTTP Cookie File').encode(),
    ('https://user:' + 'fixture-password@example.org').encode(),
])
def test_secret_categories_detected_without_printing_content(content: bytes) -> None:
    problems = load_guard().blob_problems('config.py', content)
    assert problems and all(content.decode() not in reason for reason in problems)


def test_staged_version_scanned_even_if_worktree_version_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    guard = load_guard()
    path = tmp_path / 'config.py'
    path.write_text('ghp_' + 'a' * 36)
    guard.git('add', 'config.py')
    path.write_text('safe working copy')
    failures = guard.check(guard.staged_entries())
    assert failures == ['config.py: GitHub token']
    guard.git('add', 'config.py')
    assert guard.check(guard.staged_entries()) == []


def test_internal_refs_cannot_be_pushed() -> None:
    guard = load_guard()
    with pytest.raises(ValueError, match='internal'):
        guard.outgoing_commits('refs/codex/private ' + '1' * 40 + ' refs/codex/private ' + '0' * 40)


def test_outgoing_history_checks_deleted_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    guard = load_guard()
    (tmp_path / 'private.md').write_text('local documentation')
    guard.git('add', 'private.md')
    for action in ('add', 'remove'):
        if action == 'remove':
            guard.git('rm', 'private.md')
        guard.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                  'commit', '-q', '-m', action)
    oid = guard.git('rev-parse', 'HEAD').decode().strip()
    commits = guard.outgoing_commits(f'refs/heads/main {oid} refs/heads/main ' + '0' * 40)
    assert len(commits) == 2
    failures = [x for commit in commits for x in guard.check(
        guard.tree_entries(commit), historical=True)]
    assert any('private.md:' in failure for failure in failures)


def test_hook_cli_blocks_protected_staged_file(tmp_path: Path) -> None:
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'PRD.md').write_text('private planning')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'PRD.md'], check=True)
    result = subprocess.run([sys.executable, str(SCRIPT), '--staged'], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert 'PRD.md:' in result.stderr and 'private planning' not in result.stderr
