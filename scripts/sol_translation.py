"""Local SSH entry point: content-hashed deployment and one-platform Sol control."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    "src",
    "hpc",
    "requirements",
    "config",
    "scripts",
    "pyproject.toml",
    "tests",
    "README.md",
)


def bundle_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE:
        entry = root / name
        for path in entry.rglob("*") if entry.is_dir() else [entry]:
            if path.is_symlink():
                raise ValueError("deployment must not contain symlinks")
            if (path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
                    and path.relative_to(root).as_posix() != "config/sol_translation.json"):
                files.append(path)
    return sorted(set(files))


def create_bundle(root: Path, output: Path) -> str:
    files = bundle_files(root)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    identity = digest.hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    return identity


def invoke(args: list[str], *, timeout: int = 180) -> str:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(
            "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
            or "remote command failed"
        )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "prepare", "run", "status", "resume"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/sol_translation.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Import the stdlib-only configuration validator without installing the project.
    sys.path.insert(0, str(ROOT / "src"))
    from asl_recognizer.translation.sol import validate_config

    validate_config(config)
    if config["host"] != "sol.asu.edu":
        raise ValueError("this entry point only connects to the configured Sol host")
    access = ROOT / ".asl-sol-access"
    options = [
        "-i",
        str(access / "sol_asl_ed25519"),
        "-o",
        f'UserKnownHostsFile="{(access / "known_hosts").as_posix()}"',
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "HostKeyAlgorithms=rsa-sha2-512,rsa-sha2-256",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    target = f"{config['user']}@{config['host']}"
    durable = config["durable_root"]
    if args.action == "check":
        print(
            invoke(
                [
                    "ssh",
                    *options,
                    target,
                    shlex.join(
                        [
                            "bash",
                            "-lc",
                            "set -e; whoami; myaccounts; "
                            "if command -v quota >/dev/null; then "
                            "timeout -k 2 15 quota -s || echo 'quota query unavailable'; fi; "
                            f"df -h /home/{config['user']} /scratch/{config['user']}; "
                            f"timeout -k 2 20 du -sh /home/{config['user']} "
                            f"/scratch/{config['user']} "
                            "|| echo 'recursive usage scan timed out'; "
                            "sinfo -o '%P %G %l %a'",
                        ]
                    ),
                ]
            )
        )
        return 0
    deployment = ROOT / "artifacts/sol/deployment.json"
    if args.action == "prepare":
        bundle = ROOT / "artifacts/sol/code.tar.gz"
        source_identity = create_bundle(ROOT, bundle)
        config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
        identity = hashlib.sha256((source_identity + config_hash).encode()).hexdigest()
        remote_code = f"{durable}/releases/{identity}"
        uploaded = f"{durable}/code-{identity}.tar.gz"
        invoke(["ssh", *options, target, shlex.join(["mkdir", "-p", remote_code])])
        invoke(["scp", *options, str(bundle), f"{target}:{uploaded}"], timeout=600)
        archive_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
        verify = (
            "test "
            + shlex.quote(archive_hash)
            + ' = "$(sha256sum '
            + shlex.quote(uploaded)
            + ' | cut -d" " -f1)"'
        )
        invoke(["ssh", *options, target, verify])
        # Bundles are generated locally from an explicit allowlist, with links rejected.
        invoke(
            [
                "ssh",
                *options,
                target,
                shlex.join(
                    [
                        "tar",
                        "--no-same-owner",
                        "--no-same-permissions",
                        "-xzf",
                        uploaded,
                        "-C",
                        remote_code,
                    ]
                ),
            ]
        )
        remote_config = f"{remote_code}/config/sol_translation.json"
        invoke(["scp", *options, str(args.config.resolve()), f"{target}:{remote_config}"])
        marker = {
            "bundle_sha256": identity,
            "source_sha256": source_identity,
            "archive_sha256": archive_hash,
            "config_sha256": config_hash,
            "code": remote_code,
            "config": remote_config,
        }
        deployment.parent.mkdir(parents=True, exist_ok=True)
        deployment.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        marker_path = deployment.parent / "remote-deployment.json"
        marker_path.write_text(json.dumps(marker) + "\n", encoding="utf-8")
        invoke(["scp", *options, str(marker_path), f"{target}:{remote_code}/deployment.json"])
    if not deployment.exists():
        raise ValueError("run prepare first to deploy the exact working tree")
    marker = json.loads(deployment.read_text(encoding="utf-8"))
    remote = (
        "export PYTHONPATH="
        + shlex.quote(marker["code"] + "/src")
        + "; "
        + shlex.join(
            [
                "python3",
                "-m",
                "asl_recognizer.translation.sol",
                args.action,
                "--config",
                marker["config"],
            ]
        )
    )
    print(invoke(["ssh", *options, target, remote]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"Sol command failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
