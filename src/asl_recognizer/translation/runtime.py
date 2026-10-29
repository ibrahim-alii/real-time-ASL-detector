"""Reusable Sol extraction engine with a bounded JSON-line protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Generator
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from typing import Any, TextIO, cast

import numpy as np

from ..features.mmpose_backend import MMPoseWholeBodyBackend, WholeBodyFrame
from ..features.video import iter_video_frames
from .cache import TranslationFeatureCache
from .features import mmpose_translation_request
from .local_runtime import OverlayWriter, extractor_identity
from .pose_cache import PREPROCESSING, PoseCache, pose_quality
from .schema import MMPOSE_SCHEMA
from .sol_common import write_json
from .youtube import file_hash


def preprocessing_identity() -> dict[str, Any]:
    paths = (Path(__file__).with_name("features.py"),
             Path(__file__).parents[1] / "features/video.py")
    digest = hashlib.sha256("".join(file_hash(path) for path in paths).encode()).hexdigest()
    return {**PREPROCESSING, "implementation_sha256": digest}


class ExtractionEngine:
    def __init__(self, assets: dict[str, Path], *, device: str = "cuda:0",
                 threshold: float = 0.5) -> None:
        started = time.perf_counter()
        self.device = device
        self.identity = extractor_identity(assets, threshold)
        self.backend = MMPoseWholeBodyBackend(**assets, device=device,
                                             confidence_threshold=threshold)
        self.initialization_seconds = time.perf_counter() - started

    def extract(self, *, video: Path, output: Path, start_ms: int = 0,
                end_ms: int | None = None, overlay: Path | None = None,
                native_pose: bool = False) -> dict[str, Any]:
        import cv2

        if output.exists():
            raise FileExistsError(output)
        started = time.perf_counter()
        capture = cv2.VideoCapture(str(video))
        try:
            fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not capture.isOpened() or not np.isfinite(fps) or fps <= 0 or count <= 0:
                raise ValueError("unreadable recorded video")
            duration = round(count * 1000 / fps)
        finally:
            capture.release()
        end_ms = duration if end_ms is None else end_ms
        if end_ms > duration:
            raise ValueError("sentence interval exceeds recording duration")
        frames: list[WholeBodyFrame] = []
        timings = {"decode_seconds": 0.0, "pose_seconds": 0.0, "overlay_seconds": 0.0}
        self.backend.reset()
        with ExitStack() as stack:
            stride = 1 if native_pose else 2
            writer = stack.enter_context(OverlayWriter(video, overlay, fps=fps / stride)) \
                if overlay is not None else None
            images = cast(Generator[tuple[Any, int], None, None],
                          iter_video_frames(video, start_ms=start_ms, end_ms=end_ms,
                                            frame_stride=stride))
            stack.callback(images.close)
            while True:
                tick = time.perf_counter()
                try:
                    image, timestamp = next(images)
                except StopIteration:
                    break
                timings["decode_seconds"] += time.perf_counter() - tick
                tick = time.perf_counter()
                frame = self.backend.detect_translation(image, timestamp)
                frames.append(frame)
                timings["pose_seconds"] += time.perf_counter() - tick
                if writer is not None:
                    tick = time.perf_counter()
                    writer.append(image, frame)
                    timings["overlay_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        request = mmpose_translation_request(frames[::2] if native_pose else frames)
        metadata = {"feature_schema": MMPOSE_SCHEMA, "extractor": self.identity,
                    "preprocessing": preprocessing_identity(),
                    "source_video_hash": file_hash(video), "start_ms": start_ms,
                    "raw_pose_sampling": "every-native-frame-v1" if native_pose
                    else "every-second-native-frame-v1",
                    "end_ms": end_ms, "fps": fps, "width": int(image.shape[1]),
                    "height": int(image.shape[0])}
        TranslationFeatureCache(output).store("recorded", request, metadata=metadata)
        PoseCache(output / "raw-pose").store("recorded", frames, metadata=metadata)
        timings["preparation_seconds"] = time.perf_counter() - tick
        elapsed = time.perf_counter() - started
        report = {**metadata, **timings, "sampled_frames": len(frames),
                  "raw_pose_frames": len(frames),
                  "encoder_frames": len(request.values), "extraction_seconds": elapsed,
                  "processing_fps": len(frames) / elapsed,
                  "initialization_seconds": self.initialization_seconds,
                  "quality": pose_quality(frames)}
        write_json(output / "extraction.json", report)
        return report

    def close(self) -> None:
        self.backend.close()


def request_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("worker path must be absolute within the session root")
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("worker path escapes the session root")
    return path


def serve(handler: Callable[[dict[str, Any]], dict[str, Any]], *,
          stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    def send(value: object) -> None:
        stdout.write(json.dumps(value, allow_nan=False) + "\n")
        stdout.flush()

    send({"ready": True, "protocol": 1})
    while line := stdin.readline(65537):
        identifier = None
        try:
            if len(line) > 65536 or not line.endswith("\n"):
                raise ValueError("worker request exceeds the protocol limit")
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise ValueError("worker request requires an ID")
            identifier = item["id"]
            with redirect_stdout(sys.stderr):
                result = handler(item)
            send({"id": identifier, "ok": True, "result": result})
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            send({"id": identifier, "ok": False, "error": type(error).__name__})


def require_sol(root: Path) -> None:
    if (sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID")
            or not root.resolve().is_relative_to(Path("/scratch"))):
        raise ValueError("real runtime workers require a Sol Slurm session in scratch")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("extract",), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--assets-json", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require_sol(args.work_root)
    if args.role == "extract":
        if args.assets_json is None:
            parser.error("extraction worker requires --assets-json")
        assets = json.loads(args.assets_json.read_text(encoding="utf-8"))["assets"]
        with redirect_stdout(sys.stderr):
            engine = ExtractionEngine({key: Path(value) for key, value in assets.items()},
                                      device=args.device, threshold=args.threshold)

        def extract(item: dict[str, Any]) -> dict[str, Any]:
            return engine.extract(video=request_path(args.work_root, item["video"]),
                output=request_path(args.work_root, item["output"]),
                start_ms=int(item.get("start_ms", 0)), end_ms=item.get("end_ms"),
                native_pose=bool(item.get("native_pose", False)),
                overlay=request_path(args.work_root, item["overlay"]) if item.get("overlay")
                else None)
        try:
            serve(extract, stdin=sys.stdin, stdout=sys.stdout)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
