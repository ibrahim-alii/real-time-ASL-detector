"""One resumable Sol stage per allocation; successful stages schedule the controller."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from zipfile import ZipFile

from .acquisition import (
    acquire_youtube,
    extract_recordings,
    extract_split_recordings,
    fetch_ids,
    google_drive_download_url,
    recording_archive_groups,
    write_csv,
)
from .sol import RETIRED_STAGES, STAGES, remaining, run_command, validate_config
from .sol_common import (
    GIB,
    capacity,
    read_json,
    seal,
    tree_bytes,
    verify_seal,
    write_json,
    writer_lock,
)
from .youtube import file_hash, freeze_snapshot, verify_snapshot


class Pipeline:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = read_json(config_path)
        validate_config(self.config)
        self.code = Path(__file__).resolve().parents[3]
        self.root = Path(self.config["durable_root"])
        self.scratch = Path(self.config["scratch_root"])
        self.runtime = self.scratch / "runtime"
        self.train = self.runtime / "train/bin/python"
        self.extract = self.runtime / "extract/bin/python"
        self.manifests = self.root / "manifests"
        self.features = self.scratch / "features"
        self.reports = self.root / "reports"
        for path in (self.scratch, self.manifests, self.reports):
            path.mkdir(parents=True, exist_ok=True)

    def execute(self, args: list[str], *, env: dict[str, str] | None = None) -> None:
        subprocess.run(
            args,
            check=True,
            env={**os.environ, "PYTHONPATH": str(self.code / "src"), **(env or {})},
        )

    def worker(self, interpreter: Path, *args: object) -> None:
        self.execute(
            [str(interpreter), "-m", "asl_recognizer.translation.worker", *(str(a) for a in args)]
        )

    def assets(self) -> list[str]:
        document = read_json(self.scratch / "models/mmpose/assets.json")
        for name, path in document["assets"].items():
            if file_hash(Path(path)) != document["sha256"][name]:
                raise ValueError("MMPose asset changed")
        return [
            arg
            for name, path in document["assets"].items()
            for arg in ("--" + name.replace("_", "-"), path)
        ]

    def drive(
        self, url: str, destination: Path, reserve: int, *, validate_zip: bool = True,
    ) -> None:
        import gdown

        marker = destination.with_suffix(destination.suffix + ".json")
        if marker.exists() and verify_seal(read_json(marker)):
            return
        capacity(self.scratch, reserve, remaining_quota=remaining(self.config, "scratch"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        # gdown's own .part files support HTTP range resume; completed files get sealed.
        result = gdown.download(
            url=google_drive_download_url(url), output=str(destination),
            fuzzy=False, resume=True, use_cookies=False,
        )
        if not result or not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError("official Google Drive download did not complete")
        if destination.suffix == ".zip" and validate_zip:
            with ZipFile(destination) as archive:
                if not archive.infolist():
                    raise ValueError("empty recording archive")
        write_json(marker, {**seal(destination), "url": url})

    def bootstrap(self) -> None:
        capacity(self.scratch, 20 * GIB, remaining_quota=remaining(self.config, "scratch"))
        self.execute(
            ["bash", str(self.code / "hpc/sol-bootstrap.sh")],
            env={"ASL_CODE_ROOT": str(self.code), "ASL_RUNTIME_ROOT": str(self.runtime)},
        )

        self.execute(
            [
                str(self.extract),
                str(self.code / "scripts/fetch_mmpose_assets.py"),
                "--output",
                str(self.scratch / "models/mmpose"),
            ]
        )

    def runtime_stage(self) -> None:
        self.worker(
            self.extract,
            "runtime-check",
            *self.assets(),
            "--device",
            "cuda:0",
            "--output",
            self.reports / "runtime.json",
        )

    def annotations(self) -> None:
        from .manifest import read_manifest

        fetch_ids(self.config["official_ids_url"], self.scratch / "youtube-asl/official-ids.txt")
        annotations = self.scratch / "how2sign/annotations"
        for split in ("train", "dev", "test"):
            self.drive(
                self.config["how2sign"][f"{split}_annotations"],
                annotations / f"{split}.csv",
                10 * 1024**2,
            )
        full = self.manifests / "how2sign-full.csv"
        if not full.exists():
            self.worker(
                self.train,
                "how2sign-manifest",
                "--train",
                annotations / "train.csv",
                "--dev",
                annotations / "dev.csv",
                "--test",
                annotations / "test.csv",
                "--alignment",
                "realigned",
                "--video-layout",
                "recordings",
                "--output",
                full,
            )
        manifest = read_manifest(full)
        for split, count in (
            ("train", self.config["how2sign_train_limit"]),
            ("dev", self.config["dev_limit"]),
        ):
            if len(manifest.accepted(split)) < count:
                raise ValueError("How2Sign annotation yield below target")
        bounded = self.manifests / "how2sign.csv"
        if not bounded.exists():
            self.worker(
                self.train,
                "subset-manifest",
                "--manifest",
                full,
                "--output",
                bounded,
                "--train-limit",
                self.config["how2sign_train_limit"],
                "--dev-limit",
                self.config["dev_limit"],
                "--seed",
                self.config["seed"],
            )

    def how2sign(self) -> None:
        import gdown

        from .manifest import read_manifest

        manifest = read_manifest(self.manifests / "how2sign.csv")
        for split, reserve in (("train", 290 * GIB), ("dev", 16 * GIB)):
            wanted = {r.video_id for r in manifest.accepted(split)}
            output = self.scratch / "how2sign/recordings" / split
            source = self.config["how2sign"][f"{split}_recordings"]
            archives = self.scratch / "how2sign/archives" / split
            archives.mkdir(parents=True, exist_ok=True)
            if "/folders/" in source:
                listing = gdown.download_folder(
                    url=source,
                    output=str(archives),
                    skip_download=True,
                    remaining_ok=False,
                    use_cookies=False,
                )
                if not listing:
                    raise RuntimeError("official How2Sign recording folder unavailable")
                entries = [
                    (
                        f"https://drive.google.com/file/d/{item.id}/view",
                        archives / Path(item.path).name,
                    )
                    for item in listing
                ]
            else:
                entries = [(source, archives / "recordings.zip")]
            found: set[str] = set()
            for group in recording_archive_groups(entries):
                multipart = len(group) > 1
                if multipart:
                    from .archive_tool import ensure_archive_tool

                    tool = ensure_archive_tool(self.runtime / "archive-tools")
                    write_json(self.reports / "archive-tool.json",
                               read_json(tool.with_suffix(".json")))
                for url, path in group:
                    self.drive(url, path, max(20 * GIB, reserve - tree_bytes(archives)),
                               validate_zip=not multipart)
                path = group[-1][1]
                if multipart:
                    found |= extract_split_recordings(
                        [part for _, part in group], wanted, output, tool=tool,
                        remaining_quota=remaining(self.config, "scratch"),
                        timeout=int(self.config["allocation_seconds"] * 0.8),
                    )
                else:
                    found |= extract_recordings(
                        path, wanted, output, remaining_quota=remaining(self.config, "scratch")
                    )
                if wanted <= found:
                    break
            if not wanted <= found:
                raise ValueError(
                    f"official archives lack {len(wanted - found)} selected recordings"
                )

    def youtube_download(self) -> None:
        root = self.scratch / "youtube-asl"
        acquire_youtube(
            ids_path=root / "official-ids.txt",
            root=root,
            byte_limit=self.config["youtube_limit_bytes"],
            scratch_remaining=lambda: remaining(self.config, "scratch"),
            attempts=self.config["max_attempts"],
            seed=self.config["seed"],
            video_limit=50,
        )

    def youtube_validate(self) -> str | None:
        from .manifest import read_manifest

        root = self.scratch / "youtube-asl"
        with (root / "index.csv").open(newline="", encoding="utf-8") as handle:
            if not list(csv.DictReader(handle)):
                raise ValueError(
                    "no YouTube channel metadata retrieved; inspect acquisition failures"
                )
        digest = file_hash(root / "index.csv")
        snapshot = self.root / "snapshots" / digest
        if not snapshot.exists():
            freeze_snapshot(
                index=root / "index.csv",
                root=root,
                official_ids=root / "official-ids.txt",
                output=snapshot,
            )
        verify_snapshot(snapshot / "snapshot.json", root)
        people = snapshot / "people.csv"
        if not people.exists():
            self.worker(
                self.extract,
                "validate-people-mmpose",
                *self.assets(),
                "--device",
                "cuda:0",
                "--input",
                snapshot / "captions.csv",
                "--video-root",
                root / "videos",
                "--output",
                people,
                "--resume-root",
                self.scratch / "youtube-asl/person-evidence",
            )
        full = snapshot / "manifest.csv"
        if not full.exists():
            self.worker(
                self.train,
                "manifest",
                "--input",
                people,
                "--source",
                "youtube-asl",
                "--output",
                full,
            )
        manifest = read_manifest(full)
        counts = {s: len(manifest.accepted(s)) for s in ("train", "dev")}
        write_json(
            self.reports / "youtube-yield.json", {"snapshot": str(snapshot), "counts": counts}
        )
        if counts["train"] < self.config["train_limit"] or counts["dev"] < self.config["dev_limit"]:
            ledger = read_json(root / "acquisition.json")
            if len(ledger["videos"]) >= len((root / "official-ids.txt").read_text().splitlines()):
                raise ValueError("official ID list exhausted before bounded train/dev targets")
            return "youtube-download"
        destination = self.manifests / "youtube.csv"
        if not destination.exists():
            self.worker(
                self.train,
                "subset-manifest",
                "--manifest",
                full,
                "--output",
                destination,
                "--train-limit",
                self.config["train_limit"],
                "--dev-limit",
                self.config["dev_limit"],
                "--seed",
                self.config["seed"],
            )
        write_json(
            self.root / "selected-snapshot.json",
            {"path": str(snapshot), "sha256": file_hash(snapshot / "snapshot.json")},
        )
        return None

    def manifests_stage(self) -> None:
        from .data_audit import audit_manifest
        from .manifest import read_manifest
        from .review import export_review

        for source, name in (("youtube-asl", "youtube"), ("how2sign", "how2sign")):
            path = self.manifests / f"{name}.csv"
            report = audit_manifest(read_manifest(path))
            write_json(self.reports / f"{name}-audit.json", report)
            if not report["source_split_isolation_passed"]:
                raise ValueError("source overlap audit failed")
            review = self.root / "reviews" / f"{source}.csv"
            review.parent.mkdir(exist_ok=True)
            if not review.exists():
                export_review(path, review)
        write_json(
            self.root / "quality-status.json",
            {
                "semantic_review": "pending",
                "purpose": "unreviewed-bounded-engineering-experiment",
                "test_sealed": True,
                "full_corpus_scaling_allowed": False,
            },
        )

    def extraction(self, source: str) -> str | None:
        from .cache import TranslationFeatureCache
        from .manifest import read_manifest

        name = "youtube" if source == "youtube-asl" else source
        manifest_path = self.manifests / f"{name}.csv"
        manifest = read_manifest(manifest_path)
        video_root = (
            self.scratch / "youtube-asl/videos"
            if source == "youtube-asl"
            else self.scratch / "how2sign/recordings"
        )
        started = time.monotonic()
        stage = "extract-youtube" if source == "youtube-asl" else "extract-how2sign"
        for shard in range(self.config["extraction_shards"]):
            budget = self.config["allocation_seconds"] * 0.8 - (time.monotonic() - started)
            if budget <= 0:
                return stage
            self.worker(
                self.extract,
                "extract",
                *self.assets(),
                "--backend",
                "mmpose",
                "--device",
                "cuda:0",
                "--manifest",
                manifest_path,
                "--video-root",
                video_root,
                "--output",
                self.features / source,
                "--report-root",
                self.reports / source,
                "--shard-count",
                self.config["extraction_shards"],
                "--shard-index",
                shard,
                "--splits",
                "train",
                "dev",
                "--overlay-root",
                self.scratch / "overlays" / source,
                "--overlay-review",
                self.root / "reviews" / f"{source}.csv",
                "--wall-time-seconds",
                budget,
            )
            report = read_json(self.reports / source / f"extract-shard-{shard:05d}.json")
            if report["failures"]:
                raise ValueError(
                    f"extraction failures in {source} shard {shard}: {report['failures']}"
                )
            if report["paused"]:
                return stage
        cache = TranslationFeatureCache(self.features / source)
        for row in manifest.accepted():
            if row.split not in {"train", "dev"}:
                continue
            if (
                cache.find_valid(
                    row.clip_id,
                    expected_metadata={
                        "manifest_hash": manifest.fingerprint,
                        "source": source,
                        "caption": row.caption,
                        "split": row.split,
                        "feature_schema": "mmpose-coco-wholebody-133-xy-v1",
                    },
                )
                is None
            ):
                raise ValueError(f"extraction incomplete: {row.clip_id}")
        with (self.root / "reviews" / f"{source}.csv").open(newline="", encoding="utf-8") as handle:
            review_rows = list(csv.DictReader(handle))
        write_csv(
            self.root / "reviews" / f"{source}-media.csv",
            [
                {
                    "clip_id": row["clip_id"],
                    "video_id": row["video_id"],
                    "start_ms": row["start_ms"],
                    "end_ms": row["end_ms"],
                    "overlay_path": str(
                        self.scratch
                        / "overlays"
                        / source
                        / (hashlib.sha256(row["clip_id"].encode()).hexdigest() + ".mp4")
                    ),
                    "video_root": str(video_root),
                }
                for row in review_rows
            ],
        )
        return None

    def stage(self, name: str) -> str | None:
        if name in RETIRED_STAGES:
            raise ValueError("legacy training stage is retired")
        actions: dict[str, Callable[[], str | None]] = {
            "bootstrap": self.bootstrap,
            "runtime": self.runtime_stage,
            "annotations": self.annotations,
            "how2sign": self.how2sign,
            "youtube-download": self.youtube_download,
            "youtube-validate": self.youtube_validate,
            "manifests": self.manifests_stage,
            "extract-how2sign": lambda: self.extraction("how2sign"),
            "extract-youtube": lambda: self.extraction("youtube-asl"),
        }
        return actions[name]()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args(argv)
    pipeline = Pipeline(args.config.resolve())
    os.environ["PATH"] = str(pipeline.runtime / "python/bin") + os.pathsep + os.environ["PATH"]
    if args.stage != "bootstrap" and Path(sys.prefix) != pipeline.train.parent.parent:
        os.execv(
            str(pipeline.train),
            [
                str(pipeline.train),
                "-m",
                "asl_recognizer.translation.sol_pipeline",
                *(argv or sys.argv[1:]),
            ],
        )
    receipt = pipeline.root / "receipts" / f"{args.token}.json"
    started = time.monotonic()
    with writer_lock(pipeline.root / "stage.lock"):
        try:
            next_stage = pipeline.stage(args.stage)
            write_json(
                receipt,
                {
                    "token": args.token,
                    "stage": args.stage,
                    "status": "completed",
                    "seconds": time.monotonic() - started,
                    "next_stage": next_stage,
                },
            )
        except Exception as error:
            write_json(
                receipt,
                {
                    "token": args.token,
                    "stage": args.stage,
                    "status": "failed",
                    "error": str(error),
                    "seconds": time.monotonic() - started,
                },
            )
            raise
    # A tiny CPU continuation reconciles this allocation after successful termination.
    if os.environ.get("SLURM_JOB_ID"):
        command = f"export PYTHONPATH={shlex.quote(str(pipeline.code / 'src'))}; " + shlex.join(
            [
                "python3",
                "-m",
                "asl_recognizer.translation.sol",
                "continue",
                "--config",
                str(args.config),
            ]
        )
        run_command(
            [
                "sbatch",
                "--parsable",
                "--account",
                pipeline.config["account"],
                "--partition",
                pipeline.config["partition"],
                "--qos",
                pipeline.config["qos"],
                "--time",
                "00:20:00",
                "--cpus-per-task",
                "1",
                "--mem",
                "1G",
                "--dependency",
                "afterok:" + os.environ["SLURM_JOB_ID"],
                "--output",
                str(pipeline.root / "slurm/controller-%j.out"),
                "--wrap",
                command,
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
