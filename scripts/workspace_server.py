"""Loopback camera workspace; real inference runs only inside a Sol GPU job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from asl_recognizer.translation.runtime_client import PersistentWorker

MAX_UPLOAD = 16 * 1024 * 1024
ASSET_NAMES = ("pose_config", "pose_weights", "detector_config", "detector_weights")


class WarmRuntime:
    """Own one extractor and one translator; retain their separate Python environments."""

    def __init__(self, config: dict[str, Any]) -> None:
        started = time.perf_counter()
        self.stack = ExitStack()
        try:
            name = self.stack.enter_context(
                tempfile.TemporaryDirectory(prefix="asl-workers-", dir=config["work_root"])
            )
            root = Path(name)
            assets = root / "assets.json"
            assets.write_text(
                json.dumps({"assets": {key: config[key] for key in ASSET_NAMES}}), encoding="utf-8"
            )
            log = self.stack.enter_context((root / "workers.log").open("w", encoding="utf-8"))
            src = str(Path(__file__).resolve().parents[1] / "src")
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            common = [
                "-m",
                "asl_recognizer.translation.runtime",
                "--work-root",
                config["work_root"],
            ]
            self.extractor = self.stack.enter_context(
                PersistentWorker(
                    [
                        config["extraction_python"],
                        *common,
                        "--role",
                        "extract",
                        "--assets-json",
                        str(assets),
                        "--threshold",
                        str(config.get("keypoint_threshold", 0.5)),
                    ],
                    env={**env, "PYTHONPATH": src},
                    stderr=log,
                )
            )
            self.translator = self.stack.enter_context(
                PersistentWorker(
                    [
                        config["python"],
                        *common,
                        "--role",
                        "translate",
                        "--checkpoint",
                        config["checkpoint"],
                    ],
                    env={**env, "PYTHONPATH": src},
                    stderr=log,
                    timeout=600,
                )
            )
            # Cold checkpoint startup has a separate budget from per-clip processing.
            self.translator.timeout = 300
        except BaseException:
            self.stack.close()
            raise
        self.initialization_ms = (time.perf_counter() - started) * 1000

    def infer(self, root: Path, video: Path, overlay: Path) -> dict[str, Any]:
        started = time.perf_counter()
        features = root / "features"
        extracted = self.extractor.request(
            {"video": str(video), "output": str(features), "overlay": str(overlay)}
        )
        tick = time.perf_counter()
        result = self.translator.request({"features": str(features)})
        return {
            **result,
            "quality_status": "unverified",
            "timings": {
                "extraction_ms": extracted["extraction_seconds"] * 1000,
                "translation_request_ms": (time.perf_counter() - tick) * 1000,
                "initialization_ms": self.initialization_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
                "warm_workers": True,
            },
        }

    def close(self) -> None:
        self.stack.close()


def load_runtime(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("inference requires a Sol Slurm GPU allocation; omit config for preview")
    config: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for key in ("python", "extraction_python", "checkpoint", *ASSET_NAMES):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_file():
            raise ValueError(f"missing runtime file: {key}")
    root = Path(config["work_root"]).resolve()
    if not root.is_relative_to(Path("/scratch")) or not root.is_dir():
        raise ValueError("work_root must be an existing Sol scratch directory")
    if shutil.which("ffmpeg") is None:
        raise ValueError("ffmpeg must be available in the Sol job")
    config.setdefault("warm_workers", True)
    return config


def translate_clip(data: bytes, config: dict[str, Any],
                   warm: WarmRuntime | None = None) -> dict[str, Any]:
    """Bound decoding, reuse checkpoint identity checks, then remove session files."""
    import cv2

    started = time.perf_counter()
    if shutil.disk_usage(config["work_root"]).free < 2 * 1024**3:
        raise ValueError("Sol scratch needs at least 2 GiB free for a clip")
    with tempfile.TemporaryDirectory(prefix="asl-workspace-", dir=config["work_root"]) as name:
        root = Path(name)
        source, video = root / "browser.webm", root / "clip.mp4"
        source.write_bytes(data)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(source), "-an", "-t", "21",
             "-vf", "scale=640:480:force_original_aspect_ratio=decrease",
             "-r", "30", "-c:v", "libx264", "-preset", "fast",
             "-pix_fmt", "yuv420p", str(video)],
            check=True, capture_output=True, timeout=45,
        )
        capture = cv2.VideoCapture(str(video))
        try:
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not capture.isOpened() or not 1 <= count <= 603:
                raise ValueError("record a readable clip of at most 20 seconds")
        finally:
            capture.release()
        report, overlay = root / "result.json", root / "overlay.mp4"
        if warm is not None:
            transcode_ms = (time.perf_counter() - started) * 1000
            result = warm.infer(root, video, overlay)
            landmarks = json.loads(overlay.with_suffix(".json").read_text(encoding="utf-8"))
            total_ms = (time.perf_counter() - started) * 1000
            return {
                "english": result["english"],
                "model_id": result["model_id"],
                "model_version": result["model_version"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "latency_ms": total_ms,
                "quality_status": result["quality_status"],
                "timings": {
                    **result["timings"],
                    "transcode_ms": transcode_ms,
                    "total_ms": total_ms,
                },
                "landmarks": {key: landmarks[key] for key in ("width", "height", "frames")},
            }
        command = [config["python"], "-m", "asl_recognizer.translation.worker", "infer-video",
                   "--video", str(video), "--checkpoint", config["checkpoint"],
                   "--output", str(report), "--overlay", str(overlay),
                   "--extraction-python", config["extraction_python"], "--device", "cuda:0"]
        for key in ASSET_NAMES:
            command.extend(["--" + key.replace("_", "-"), config[key]])
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              start_new_session=True, env=env) as process:
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                if sys.platform == "linux":
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)
        result = json.loads(report.read_text(encoding="utf-8"))
        landmarks = json.loads(overlay.with_suffix(".json").read_text(encoding="utf-8"))
        # Return display data only: no remote filenames, model weights, or cache files.
        total_ms = (time.perf_counter() - started) * 1000
        return {"english": result["english"], "model_id": result["model_id"],
                "model_version": result["model_version"], "latency_ms": total_ms,
                "checkpoint_sha256": result.get("checkpoint_sha256"),
                "timings": {"total_ms": total_ms, "decode_ms": result["latency_ms"],
                            "warm_workers": False},
                "quality_status": result["quality_status"],
                "landmarks": {key: landmarks[key] for key in ("width", "height", "frames")}}


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port: int, runtime: dict[str, Any] | None = None) -> None:
        self.runtime = runtime
        self.warm: WarmRuntime | None = None
        self.inference_lock = threading.Lock()
        super().__init__(("127.0.0.1", port), WorkspaceHandler)
        try:
            self.ensure_warm()
        except BaseException:
            self.server_close()
            raise

    def server_close(self) -> None:
        self.reset_warm()
        super().server_close()

    def reset_warm(self) -> None:
        if self.warm is not None:
            self.warm.close()
            self.warm = None

    def ensure_warm(self) -> WarmRuntime | None:
        if (
            self.runtime is not None
            and self.runtime.get("warm_workers", False)
            and self.warm is None
        ):
            self.warm = WarmRuntime(self.runtime)
        return self.warm


class WorkspaceHandler(BaseHTTPRequestHandler):
    server: WorkspaceServer

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Camera requests and operational details stay out of access logs.

    def reply(self, status: int, value: object, *, html: bool = False) -> None:
        body = str(value).encode() if html else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8" if html else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(self), microphone=()")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; connect-src 'self'; media-src blob:; "
                         "img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def trusted(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.reply(403, {"error": "Open the workspace through its localhost URL."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            self.reply(403, {"error": "Cross-origin requests are not allowed."})
            return False
        return True

    def do_GET(self) -> None:
        if not self.trusted():
            return
        if self.path == "/api/status":
            self.reply(200, {"ready": self.server.runtime is not None, "max_seconds": 20})
        elif self.path == "/":
            page = Path(__file__).with_name("asl_workspace.html")
            if page.is_file():
                self.reply(200, page.read_text(encoding="utf-8"), html=True)
            else:
                self.reply(503, {"error": "Workspace page is missing from this source checkout."})
        else:
            self.reply(404, {"error": "Not found"})

    def do_POST(self) -> None:
        self.close_connection = True
        self.connection.settimeout(30)
        if not self.trusted():
            return
        if self.path != "/api/translate":
            self.reply(404, {"error": "Not found"})
            return
        if self.headers.get("X-ASL-Request") != "clip":
            self.reply(403, {"error": "Use the workspace recording controls."})
            return
        if self.server.runtime is None:
            self.reply(503, {"error": "Preview only. Connect to a configured Sol GPU workspace."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_UPLOAD:
                raise ValueError("Record a shorter clip (maximum upload: 16 MiB).")
            if self.headers.get("Content-Type", "").split(";")[0] not in {
                "video/webm", "video/mp4",
            }:
                raise ValueError("Use a browser that records WebM or MP4 video.")
        except ValueError as error:
            self.reply(400, {"error": str(error)})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.reply(409, {"error": "Sol is busy with another clip. Try again when it finishes."})
            return
        try:
            data = self.rfile.read(size)
            if len(data) != size:
                raise ValueError("Upload was interrupted. Record and submit again.")
            self.reply(200, translate_clip(data, self.server.runtime, self.server.ensure_warm()))
        except ValueError as error:
            self.reply(422, {"error": str(error)})
        except subprocess.TimeoutExpired:
            self.server.reset_warm()
            self.reply(504, {"error": "Sol processing timed out. Try a shorter clip."})
        except (OSError, subprocess.CalledProcessError, KeyError, RuntimeError):
            self.server.reset_warm()
            self.reply(502, {"error": "Sol could not process this clip. Check the GPU runtime, "
                                     "checkpoint and signer visibility; then retry."})
        finally:
            self.server.inference_lock.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runtime-config", type=Path)
    args = parser.parse_args()
    runtime = load_runtime(args.runtime_config)
    with WorkspaceServer(args.port, runtime) as server:
        print(f"ASL workspace: http://127.0.0.1:{server.server_port}", flush=True)
        print("Sol inference configured" if runtime else "Camera preview only; Sol not connected")
        with suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
