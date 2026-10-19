"""Check staged Git blobs or outgoing history; never print matching secret text."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import PurePosixPath

MAX_BYTES = 5 * 1024**2
PRIVATE_PARTS = {
    '.git', '.codex', '.agents', '.aws', '.ssh', '.asl-sol-access',
    '.venv', '.venv-mmpose', 'venv', 'node_modules', '__pycache__',
    '.pytest_cache', '.mypy_cache', '.ruff_cache',
}
PRIVATE_ROOTS = {'docs', 'data', 'artifacts', 'models', 'notes', 'plans', 'reports'}
PRIVATE_SUFFIXES = {
    '.md', '.markdown', '.rst', '.pdf', '.docx', '.pptx', '.pem', '.key', '.p12', '.pfx',
    '.pt', '.pth', '.ckpt', '.onnx', '.gguf', '.safetensors', '.npz', '.npy', '.bin',
    '.mp4', '.mov', '.mkv', '.avi', '.wav', '.mp3', '.zip', '.tar', '.gz', '.tgz',
    '.exe', '.dll', '.pyd', '.pyc', '.csv', '.tsv', '.log',
}
SECRETS = {
    'private key': re.compile(
        rb'-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----'),
    'GitHub token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'AWS access key': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'Google API key': re.compile(rb'\bAIza[A-Za-z0-9_-]{35}\b'),
    'API token': re.compile(rb'\bsk-(?:proj-|svcacct-|ant-)?[A-Za-z0-9_-]{32,}\b'),
    'cookie export': re.compile(rb'(?im)^# (?:Netscape HTTP Cookie File|HTTP Cookie File)'),
    'credential URL': re.compile(rb'https?://[^\s/:@]+:[^\s/@]+@'),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-c', 'core.excludesFile=.gitignore', *args])


def path_problem(path: str, *, historical: bool = False) -> str | None:
    name = PurePosixPath(path)
    parts = [p.casefold() for p in name.parts]
    if path == 'README.md' or (historical and path == 'LICENSE.md'):
        return None
    if parts[0] in PRIVATE_ROOTS or any(
        p in PRIVATE_PARTS or p.startswith(('.test-tmp-', '.codex-test-')) for p in parts
    ):
        return 'local/protected directory'
    if name.suffix.casefold() in PRIVATE_SUFFIXES:
        return 'documentation, credential, data or generated file'
    if any(p.startswith('.env') or p in {'id_rsa', 'id_ed25519'} for p in parts):
        return 'credential filename'
    if any('cookie' in p or p.startswith(('credentials', 'secrets')) for p in parts):
        return 'credential filename'
    if path.casefold() == 'config/sol_translation.json':
        return 'personal Sol configuration; commit the example instead'
    return None


def blob_problems(path: str, content: bytes, *, historical: bool = False) -> list[str]:
    problems = []
    problem = path_problem(path, historical=historical)
    if problem:
        problems.append(problem)
    if len(content) > MAX_BYTES:
        problems.append('file exceeds 5 MiB source limit')
    for category, pattern in SECRETS.items():
        if pattern.search(content):
            problems.append(category)
    return problems


def staged_entries() -> list[tuple[str, str, str]]:
    entries = []
    for item in git('ls-files', '--stage', '-z').split(b'\0'):
        if not item:
            continue
        metadata, path = item.split(b'\t', 1)
        mode, oid, stage = metadata.decode().split()
        if stage != '0':
            raise ValueError('resolve index conflicts before committing')
        entries.append((path.decode('utf-8'), mode, oid))
    return entries


def tree_entries(commit: str) -> list[tuple[str, str, str]]:
    entries = []
    for item in git('ls-tree', '-r', '-z', commit).split(b'\0'):
        if item:
            metadata, path = item.split(b'\t', 1)
            mode, _kind, oid = metadata.decode().split()
            entries.append((path.decode('utf-8'), mode, oid))
    return entries


def check(entries: list[tuple[str, str, str]], *, historical: bool = False) -> list[str]:
    failures = []
    for path, mode, oid in entries:
        if mode not in {'100644', '100755'}:
            failures.append(f'{path}: symlink/submodule entries require explicit review')
            continue
        problem = path_problem(path, historical=historical)
        size = int(git('cat-file', '-s', oid))
        if problem or size > MAX_BYTES:
            failures.append(f'{path}: {problem or "file exceeds 5 MiB source limit"}')
            continue  # Never read a protected file merely to scan it.
        for reason in blob_problems(path, git('cat-file', 'blob', oid), historical=historical):
            failures.append(f'{path}: {reason}')
    return failures


def outgoing_commits(updates: str) -> list[str]:
    commits = set()
    zero = '0' * 40
    for line in updates.splitlines():
        local_ref, local_oid, _remote_ref, remote_oid = line.split()
        if local_oid == zero:
            continue
        if not local_ref.startswith(('refs/heads/', 'refs/tags/')):
            raise ValueError('internal/non-public refs must not be pushed')
        args = [local_oid]
        if remote_oid != zero:
            known = subprocess.run(['git', 'cat-file', '-e', remote_oid],
                                   capture_output=True, check=False)
            if known.returncode == 0:
                args.append('^' + remote_oid)
        commits.update(git('rev-list', *args).decode().splitlines())
    return sorted(commits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--staged', action='store_true')
    mode.add_argument('--pre-push', action='store_true')
    args = parser.parse_args()
    try:
        failures = check(staged_entries()) if args.staged else []
        if args.pre_push:
            for commit in outgoing_commits(sys.stdin.read()):
                failures.extend(f'{commit[:12]}: {x}' for x in
                                check(tree_entries(commit), historical=True))
        if failures:
            print('Git safety check rejected these paths/categories:', file=sys.stderr)
            print('\n'.join(sorted(set(failures))), file=sys.stderr)
            return 1
    except (ValueError, subprocess.CalledProcessError, UnicodeError):
        print('Git safety check could not complete; commit/push blocked.', file=sys.stderr)
        return 1
    print('Git safety check passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
