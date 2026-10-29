"""Linux Sol job controller. State and Slurm, rather than exit status alone, drive resume."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from .sol_common import capacity, read_json, write_json, writer_lock

STAGES = (
    "bootstrap",
    "runtime",
    "annotations",
    "how2sign",
    "youtube-download",
    "youtube-validate",
    "manifests",
    "extract-how2sign",
    "extract-youtube",
    "smoke",
    "benchmark",
    "train",
    "finalize",
    "diagnostic",
    "ablation",
    "ablation-finalize",
    "recorded-inference",
)
RETIRED_STAGES = {"smoke", "benchmark", "train", "finalize", "diagnostic",
                  "ablation", "ablation-finalize", "recorded-inference"}
CPU_STAGES = {"bootstrap", "annotations", "how2sign", "youtube-download", "manifests"}
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}


def validate_config(config: dict[str, Any]) -> None:
    for key in ("user", "account", "partition", "qos", "gpu"):
        value = config[key]
        if (
            not isinstance(value, str)
            or not value
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-:"
                for c in value
            )
        ):
            raise ValueError(f"invalid Slurm/user field: {key}")
    for key, parent in (("durable_root", "/home"), ("scratch_root", "/scratch")):
        path = PurePosixPath(config[key])
        prefix = PurePosixPath(parent) / config["user"]
        if (
            not path.is_absolute()
            or not path.is_relative_to(prefix)
            or ".." in path.parts
            or path == prefix
        ):
            raise ValueError(f"{key} must be a project directory in personal {parent}")
    for key in (
        "youtube_limit_bytes",
        "scratch_limit_bytes",
        "home_limit_bytes",
        "train_limit",
        "dev_limit",
        "how2sign_train_limit",
        "extraction_shards",
        "allocation_seconds",
        "max_attempts",
    ):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"positive integer required: {key}")
    if config["extraction_shards"] > 2:
        raise ValueError("bounded setup allows at most two extraction shards")


def run_command(args: list[str]) -> str:
    result = subprocess.run(args, text=True, capture_output=True, timeout=120, check=False)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def job_state(job_id: str) -> str:
    try:
        queued = run_command(["squeue", "-h", "-j", job_id, "-o", "%T"])
    except RuntimeError as error:
        if "Invalid job id specified" not in str(error):
            raise
        queued = ""  # Slurm can purge completed jobs from its live controller.
    if queued:
        return queued.splitlines()[0].strip()
    accounting = run_command(["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State"])
    for line in accounting.splitlines():
        parts = line.split("|")
        if parts[0] == job_id:
            return parts[1].split()[0].rstrip("+")
    return "UNKNOWN"  # accounting lag must never cause duplicate submission


def find_submission(token: str, user: str) -> str | None:
    name = f"asl-sol-{token}"
    queued = run_command(["squeue", "-h", "-u", user, "-n", name, "-o", "%i"])
    if queued:
        return queued.splitlines()[0]
    rows = run_command(
        [
            "sacct",
            "-n",
            "-P",
            "-u",
            user,
            "--starttime",
            "now-7days",
            "--format=JobIDRaw,JobName%64,State",
        ]
    )
    for line in rows.splitlines():
        fields = line.split("|")
        if len(fields) >= 3 and fields[1] == name and fields[0].isdigit():
            return fields[0]
    return None


def remaining(config: dict[str, Any], kind: str) -> int:
    # Home's statvfs/df view on Sol already exposes the user's 100 GiB quota.
    # Scratch has no enforced per-user quota. Bound this run's footprint instead
    # of repeatedly walking unrelated corpora and caches in the user's directory.
    project = Path(config["durable_root" if kind == "home" else "scratch_root"])
    used = int(run_command(["du", "-sb", str(project)]).split()[0]) if project.exists() else 0
    return max(0, int(config[f"{kind}_limit_bytes"]) - used)


def resource_check(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    if getpass.getuser() != config["user"]:
        raise ValueError("unexpected Sol login identity")
    accounts = run_command(["myaccounts"])
    if config["account"] not in accounts:
        raise ValueError("configured research account is not available")
    report: dict[str, Any] = {
        "accounts": accounts,
        "quota": run_command([
            "bash", "-lc",
            "if command -v quota >/dev/null; then timeout -k 2 15 quota -s || "
            "echo 'quota query unavailable; configured ASU policy ceilings apply'; "
            "else echo 'quota tool unavailable; configured ASU policy ceilings apply'; fi",
        ]),
        "partitions": run_command(["sinfo", "-o", "%P %G %l %a"]),
    }
    for kind in ("scratch", "home"):
        root = Path(config["durable_root" if kind == "home" else "scratch_root"])
        existing = root
        while not existing.exists():
            existing = existing.parent
        report[kind] = capacity(existing, 0, remaining_quota=remaining(config, kind))
    return report


def submit(
    stage: str, config_path: Path, config: dict[str, Any], code: Path, token: str, run_root: Path
) -> str:
    seconds = config["allocation_seconds"]
    command = [
        "python3",
        "-m",
        "asl_recognizer.translation.sol_pipeline",
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--token",
        token,
    ]
    wrapped = f"export PYTHONPATH={shlex.quote(str(code / 'src'))}; " + shlex.join(command)
    args = [
        "sbatch",
        "--parsable",
        "--account",
        config["account"],
        "--partition",
        config["partition"],
        "--qos",
        config["qos"],
        "--job-name",
        f"asl-sol-{token}",
        "--time",
        str(max(1, seconds // 60)),
        "--cpus-per-task",
        "8",
        "--mem",
        "32G",
        "--chdir",
        str(code),
        "--output",
        str(run_root / "slurm" / "%j.out"),
    ]
    if stage not in CPU_STAGES:
        args += ["--gpus", config["gpu"]]
    args += ["--wrap", wrapped]
    run_command([*args[:1], "--test-only", *args[1:]])
    job = run_command(args).split(";")[0]
    if not job.isdigit():
        raise ValueError("Slurm returned an invalid job ID")
    return job


def advance(config_path: Path, action: str) -> dict[str, Any]:
    config = read_json(config_path)
    validate_config(config)
    root = Path(config["durable_root"])
    code = Path(__file__).resolve().parents[3]
    deployment = read_json(code / "deployment.json")
    identity = {"bundle_sha256": deployment["bundle_sha256"], "config": config}
    state_path = root / "controller.json"
    if action == "check":
        return resource_check(config)
    if action == "status":
        status_state: dict[str, Any] = (
            read_json(state_path) if state_path.exists() else {"status": "not-started"}
        )
        if status_state.get("job_id"):
            status_state["slurm_state"] = job_state(status_state["job_id"])
        return status_state
    resource_check(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "slurm").mkdir(exist_ok=True)
    with writer_lock(root / "controller.lock"):
        state: dict[str, Any] = (
            read_json(state_path)
            if state_path.exists()
            else {
                "identity": identity,
                "stage_index": 0,
                "status": "ready",
                "history": [],
                "auto_continue": action != "prepare",
            }
        )
        if state["identity"] != identity:
            raise ValueError("run code or configuration changed; create a new run root")
        if action in {"run", "resume"}:
            state["auto_continue"] = True
            write_json(state_path, state)
        if state["status"] in {"submitting", "submission-uncertain"} and not state.get("job_id"):
            recovered = find_submission(state["token"], config["user"])
            if recovered is None:
                # Neither an empty queue nor delayed accounting proves non-submission.
                # Preserve the original intent and require investigation, never guess.
                state["status"] = "submission-uncertain"
                write_json(state_path, state)
                return state
            state.update(job_id=recovered, status="submitted")
            write_json(state_path, state)
        if state.get("job_id"):
            slurm = job_state(state["job_id"])
            if slurm in ACTIVE or slurm == "UNKNOWN":
                return {**state, "slurm_state": slurm}
            receipt = root / "receipts" / f"{state['token']}.json"
            if slurm != "COMPLETED" or not receipt.exists():
                state.update(status="failed", slurm_state=slurm)
                write_json(state_path, state)
                if action != "resume":
                    return state
                state.pop("job_id")
            else:
                result = read_json(receipt)
                if result.get("status") != "completed" or result.get("token") != state["token"]:
                    raise ValueError("job completion receipt is invalid")
                state["history"].append(
                    {"stage": state["stage"], "job_id": state.pop("job_id"), "result": result}
                )
                if result.get("next_stage"):
                    state["stage_index"] = STAGES.index(result["next_stage"])
                else:
                    state["stage_index"] += 1
        if state["stage_index"] >= len(STAGES):
            state["status"] = "completed"
            write_json(state_path, state)
            return state
        if state["stage_index"] > 0 and not state.get("auto_continue", True):
            state["status"] = "prepared"
            write_json(state_path, state)
            return state
        stage = STAGES[state["stage_index"]]
        if stage in RETIRED_STAGES:
            state.update(stage=stage, status="retired", auto_continue=False)
            write_json(state_path, state)
            return state
        sequence = int(state.get("sequence", 0)) + 1
        token = hashlib.sha256(
            f"{deployment['bundle_sha256']}:{config_path}:{stage}:{sequence}".encode()
        ).hexdigest()[:20]
        state.update(stage=stage, token=token, sequence=sequence, status="submitting")
        # Journal the intent before sbatch. An uncertain submission is reconciled by job name.
        write_json(state_path, state)
        candidates = run_command(
            ["squeue", "-h", "-u", config["user"], "-n", f"asl-sol-{token}", "-o", "%i"]
        ).splitlines()
        try:
            state["job_id"] = (
                candidates[0]
                if candidates
                else submit(stage, config_path, config, code, token, root)
            )
        except (RuntimeError, subprocess.SubprocessError):
            state["status"] = "submission-uncertain"
            write_json(state_path, state)
            raise
        state["status"] = "submitted"
        write_json(state_path, state)
        return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("check", "prepare", "run", "status", "resume", "continue")
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(advance(args.config.resolve(), args.action), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
