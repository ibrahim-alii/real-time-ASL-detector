"""Resumable command surface used by ASU Slurm scripts.

Download and video decoding stay outside the repository because corpus access is licensed;
the manifest is the complete source of work, including failures that must not be replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections.abc import Generator
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

from ..features.mmpose_backend import MMPoseWholeBodyBackend
from ..features.video import iter_every_other_video_frame
from .cache import TranslationFeatureCache
from .data_audit import audit_manifest, read_signer_map
from .features import mmpose_translation_request
from .how2sign import build_how2sign_manifest
from .local_runtime import OverlayWriter, extractor_identity
from .manifest import build_manifest, read_manifest, write_manifest
from .pose_cache import PoseCache, pose_quality
from .runtime import preprocessing_identity
from .schema import MMPOSE_SCHEMA
from .sol_common import write_json
from .youtube import file_hash


def _pose_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("pose-config", "pose-weights", "detector-config", "detector-weights"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keypoint-threshold", type=float, default=0.5)


def _assets(args: argparse.Namespace) -> dict[str, Path]:
    return {name: getattr(args, name) for name in (
        "pose_config", "pose_weights", "detector_config", "detector_weights")}



def _video_index(root: Path) -> dict[tuple[str, str], Path]:
    videos: dict[tuple[str, str], Path] = {}
    for path in root.rglob("*.mp4"):
        directories = set(path.relative_to(root).parts[:-1])
        split = next((name for name in ("train", "val", "dev", "test") if name in directories), "*")
        key = ("dev" if split == "val" else split, path.stem)
        if key in videos:
            raise ValueError(f"duplicate video filename in source split: {path.stem}")
        videos[key] = path
    return videos


def _shard_for_video(video_id: str, shard_count: int) -> int:
    return int(hashlib.sha256(video_id.encode()).hexdigest(), 16) % shard_count


def _extraction_backend(
    args: argparse.Namespace,
) -> tuple[MMPoseWholeBodyBackend, dict[str, object]]:
    assets = {
        "pose_config": args.pose_config, "pose_weights": args.pose_weights,
        "detector_config": args.detector_config, "detector_weights": args.detector_weights,
    }
    if any(path is None for path in assets.values()):
        raise ValueError("MMPose requires pose and detector config/weights paths")
    detector = MMPoseWholeBodyBackend(
        **assets, device=args.device, confidence_threshold=args.keypoint_threshold
    )
    identity: dict[str, object] = {
        "feature_schema": MMPOSE_SCHEMA,
        "extractor": extractor_identity(assets, args.keypoint_threshold),
    }
    return detector, identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m asl_recognizer.translation.worker")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("youtube-snapshot")
    snapshot.add_argument("--index", type=Path, required=True)
    snapshot.add_argument("--root", type=Path, required=True)
    snapshot.add_argument("--official-ids", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--exclude-hashes", type=Path)
    snapshot.add_argument("--seed", type=int, default=42)
    verify = commands.add_parser("verify-snapshot")
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--root", type=Path, required=True)
    review = commands.add_parser("export-review")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--count", type=int, default=100)
    review.add_argument("--seed", type=int, default=42)
    subset = commands.add_parser("subset-manifest")
    subset.add_argument("--manifest", type=Path, required=True)
    subset.add_argument("--output", type=Path, required=True)
    subset.add_argument("--train-limit", type=int, required=True)
    subset.add_argument("--dev-limit", type=int, default=100)
    subset.add_argument("--seed", type=int, default=42)
    gate = commands.add_parser("review-gate")
    gate.add_argument("--review", type=Path, required=True)
    gate.add_argument("--manifest", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    runtime = commands.add_parser("runtime-check")
    _pose_arguments(runtime)
    runtime.add_argument("--output", type=Path, required=True)
    mmpose_people = commands.add_parser("validate-people-mmpose")
    _pose_arguments(mmpose_people)
    mmpose_people.add_argument("--input", type=Path, required=True)
    mmpose_people.add_argument("--video-root", type=Path, required=True)
    mmpose_people.add_argument("--output", type=Path, required=True)
    mmpose_people.add_argument("--resume-root", type=Path)
    recorded = commands.add_parser("extract-video")
    _pose_arguments(recorded)
    recorded.add_argument("--video", type=Path, required=True)
    recorded.add_argument("--output", type=Path, required=True)
    recorded.add_argument("--start-ms", type=int, default=0)
    recorded.add_argument("--end-ms", type=int)
    recorded.add_argument("--overlay", type=Path)
    estimate = commands.add_parser("estimate-local")
    estimate.add_argument("--progress", type=Path, required=True)
    estimate.add_argument("--extraction-report", type=Path, required=True)
    estimate.add_argument("--feature-root", type=Path, required=True)
    estimate.add_argument("--target-clips", type=int, required=True)
    estimate.add_argument("--target-frames", type=int, required=True)
    estimate.add_argument("--target-steps", type=int, default=205000)
    estimate.add_argument("--output", type=Path, required=True)
    ingest = commands.add_parser("manifest")
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--source", choices=("youtube-asl", "how2sign"), required=True)
    ingest.add_argument("--output", type=Path, required=True)
    how2sign = commands.add_parser("how2sign-manifest")
    how2sign.add_argument("--train", type=Path, required=True)
    how2sign.add_argument("--dev", type=Path, required=True)
    how2sign.add_argument("--test", type=Path, required=True)
    how2sign.add_argument("--output", type=Path, required=True)
    how2sign.add_argument("--pilot-first-video", action="store_true")
    how2sign.add_argument("--pilot-clips", type=int, default=32)
    how2sign.add_argument("--alignment", choices=("realigned",), default="realigned")
    how2sign.add_argument("--video-layout", choices=("recordings",),
                         default="recordings")
    audit = commands.add_parser("audit-data")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--signers", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--report-root", type=Path)
    extract.add_argument("--video-root", type=Path, required=True)
    extract.add_argument("--backend", choices=("mmpose",), default="mmpose")
    extract.add_argument("--pose-config", type=Path)
    extract.add_argument("--pose-weights", type=Path)
    extract.add_argument("--detector-config", type=Path)
    extract.add_argument("--detector-weights", type=Path)
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--keypoint-threshold", type=float, default=0.5)
    extract.add_argument("--overlay-root", type=Path)
    extract.add_argument("--overlay-review", type=Path)
    extract.add_argument("--wall-time-seconds", type=float)
    extract.add_argument("--splits", nargs="+", choices=("train", "dev", "test"))
    extract.add_argument("--shard-count", type=int, required=True)
    extract.add_argument("--shard-index", type=int, required=True)
    return parser


def _local_command(args: argparse.Namespace) -> int | None:
    """Handle the recorded-video experiment commands."""
    if args.command == "youtube-snapshot":
        from .youtube import freeze_snapshot
        freeze_snapshot(index=args.index, root=args.root, official_ids=args.official_ids,
                        output=args.output, seed=args.seed,
                        exclude_hashes=set(args.exclude_hashes.read_text().splitlines())
                        if args.exclude_hashes else None)
    elif args.command == "verify-snapshot":
        from .youtube import verify_snapshot
        verify_snapshot(args.snapshot, args.root)
    elif args.command == "export-review":
        from .review import export_review
        export_review(args.manifest, args.output, count=args.count, seed=args.seed)
    elif args.command == "subset-manifest":
        from .subsets import subset_manifest
        subset_manifest(args.manifest, args.output, train_limit=args.train_limit,
                        dev_limit=args.dev_limit, seed=args.seed)
    elif args.command == "review-gate":
        from .review import review_gate
        report = review_gate(args.review, manifest_path=args.manifest)
        write_json(args.output, report)
        return 0 if report["passed"] else 1
    elif args.command == "runtime-check":
        from .local_runtime import runtime_check
        write_json(args.output, runtime_check(_assets(args), device=args.device,
                                              threshold=args.keypoint_threshold))
    elif args.command == "validate-people-mmpose":
        from .person_validation import EVIDENCE_FIELDS, annotate_caption_rows
        if args.output.exists():
            raise FileExistsError(args.output)
        with args.input.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        identity = extractor_identity(_assets(args), args.keypoint_threshold)
        with MMPoseWholeBodyBackend(**_assets(args), device=args.device,
                                   confidence_threshold=args.keypoint_threshold) as detector:
            annotated, counts = annotate_caption_rows(
                rows, video_root=args.video_root,
                model_sha256=str(identity["assets"]["detector_weights"]),
                detect_people=detector.count_people, validator_name="mmpose-person-1hz-v1",
                resume_root=args.resume_root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            fields = list(dict.fromkeys([*(rows[0] if rows else []), *EVIDENCE_FIELDS,
                                         "availability"]))
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(annotated)
        write_json(args.output.with_suffix(".json"), {"extractor": identity, "outcomes": counts})
    elif args.command == "extract-video":
        from .recorded import extract_video
        extract_video(video=args.video, assets=_assets(args), output=args.output,
                      device=args.device, start_ms=args.start_ms, end_ms=args.end_ms,
                      overlay=args.overlay, threshold=args.keypoint_threshold)
    elif args.command == "estimate-local":
        from .resources import resource_estimate
        write_json(args.output, resource_estimate(
            progress=args.progress, extraction_report=args.extraction_report,
            feature_root=args.feature_root, target_clips=args.target_clips,
            target_frames=args.target_frames, target_steps=args.target_steps))
    else:
        return None
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    local_result = _local_command(args)
    if local_result is not None:
        return local_result
    if args.command == "audit-data":
        report = audit_manifest(
            read_manifest(args.manifest),
            signer_map=read_signer_map(args.signers) if args.signers else None,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    if args.command == "how2sign-manifest":
        manifest = build_how2sign_manifest(
            train=args.train, dev=args.dev, test=args.test,
            pilot_first_video=args.pilot_first_video, pilot_clips=args.pilot_clips,
            alignment=args.alignment, video_layout=args.video_layout,
        )
        write_manifest(args.output, manifest)
        return 0
    if args.command == "manifest":
        with args.input.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if args.source == "how2sign" and any(
            row.get("split") not in {"train", "dev", "test"} for row in rows
        ):
            raise ValueError("How2Sign rows require official train/dev/test split labels")
        manifest = build_manifest(
            rows, source=args.source, require_person_count=args.source == "youtube-asl"
        )
        write_manifest(args.output, manifest)
        return 0
    if args.command == "extract":
        started = time.monotonic()
        budget = args.wall_time_seconds
        if budget is not None and (not math.isfinite(budget) or budget <= 0):
            raise ValueError("wall time must be finite and positive")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard index must be within shard count")
        manifest = read_manifest(args.manifest)
        work = [item for item in manifest.accepted()
                if _shard_for_video(item.video_id, args.shard_count) == args.shard_index
                and (args.splits is None or item.split in args.splits)]
        args.output.mkdir(parents=True, exist_ok=True)
        cache = TranslationFeatureCache(args.output)
        raw_cache = PoseCache(args.output / "raw-pose")
        preprocessing = preprocessing_identity()
        videos = _video_index(args.video_root)
        video_hashes: dict[Path, str] = {}
        completed: list[str] = []
        failures: dict[str, str] = {}
        paused = False
        backend, extractor_identity = _extraction_backend(args)
        overlay_ids: set[str] | None = None
        if args.overlay_review is not None:
            with args.overlay_review.open(newline="", encoding="utf-8") as handle:
                overlay_ids = {row["clip_id"] for row in csv.DictReader(handle)}
        with backend as detector:
            for item in work:
                if budget is not None and time.monotonic() - started >= budget:
                    paused = True
                    break
                video_path = videos.get((item.split, item.video_id)) or videos.get(
                    ("*", item.video_id)
                )
                if video_path is None:
                    failures[item.clip_id] = "source-video-unavailable"
                    continue
                if video_path not in video_hashes:
                    video_hashes[video_path] = file_hash(video_path)
                identity = {
                    "source": item.source,
                    "split": item.split,
                    "caption": item.caption,
                    "manifest_hash": manifest.fingerprint,
                    **extractor_identity,
                    "source_video_hash": video_hashes[video_path],
                    "start_ms": item.start_ms,
                    "end_ms": item.end_ms,
                    "preprocessing": preprocessing,
                    "raw_pose_sampling": "every-second-native-frame-v1",
                }
                overlay_requested = args.overlay_root is not None \
                    and (overlay_ids is None or item.clip_id in overlay_ids)
                overlay_path = (args.overlay_root / (hashlib.sha256(
                    item.clip_id.encode()).hexdigest() + ".mp4")) if overlay_requested else None
                try:
                    raw_cache.load(item.clip_id, expected_metadata=identity)
                    raw_valid = True
                except (OSError, RuntimeError, ValueError, KeyError):
                    raw_valid = False
                if raw_valid and cache.find_valid(
                        item.clip_id, expected_metadata=identity) is not None \
                        and (overlay_path is None or overlay_path.is_file()):
                    completed.append(item.clip_id)
                    continue
                try:
                    detector.reset()
                    wholebody_frames = []
                    with ExitStack() as stack:
                        writer = None
                        if overlay_path is not None:
                            import cv2
                            capture = cv2.VideoCapture(str(video_path))
                            try:
                                fps = capture.get(cv2.CAP_PROP_FPS)
                                if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                                    raise ValueError("unreadable overlay video frame rate")
                            finally:
                                capture.release()
                            writer = stack.enter_context(OverlayWriter(
                                video_path, overlay_path, fps=fps / 2))
                        images = cast(Generator[tuple[Any, int], None, None],
                                      iter_every_other_video_frame(
                                          video_path, start_ms=item.start_ms, end_ms=item.end_ms))
                        stack.callback(images.close)
                        for image, timestamp in images:
                            frame = detector.detect_translation(image, timestamp)
                            wholebody_frames.append(frame)
                            if writer is not None:
                                writer.append(image, frame)
                    request = mmpose_translation_request(wholebody_frames)
                    metadata = {**identity, "quality": pose_quality(wholebody_frames)}
                    cache.store(item.clip_id, request, metadata=metadata)
                    raw_cache.store(item.clip_id, wholebody_frames, metadata=metadata)
                    completed.append(item.clip_id)
                except (OSError, RuntimeError, ValueError) as error:
                    failures[item.clip_id] = str(error)
        report_root = args.report_root or args.output
        report_root.mkdir(parents=True, exist_ok=True)
        (report_root / f"extract-shard-{args.shard_index:05d}.json").write_text(
            json.dumps(
                {
                    "manifest_hash": manifest.fingerprint,
                    "completed": completed,
                    "failures": failures,
                    "paused": paused,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0
    raise ValueError("unsupported command")


if __name__ == "__main__":
    raise SystemExit(main())
