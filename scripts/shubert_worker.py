"""Sol-only, pinned SHuBERT video inference worker in its isolated environment.

The model assets and official source remain on Sol; this module does not download them.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


def request_path(root, value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("worker path must be absolute within the session root")
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("worker path escapes the session root")
    return path


def require_sol(root):
    if (sys.platform != "linux" or not os.environ.get("SLURM_JOB_ID")
            or not root.resolve().is_relative_to(Path("/scratch"))):
        raise ValueError("real runtime workers require a Sol Slurm session in scratch")


def serve(handler, *, stdin=sys.stdin, stdout=sys.stdout):
    def send(value):
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


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError("unsafe asset path")
    result = root.joinpath(path)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("asset path escapes root")
    return result


def verify_asset(path, asset):
    size = path.stat().st_size
    if size != asset["bytes"]:
        raise ValueError("asset size mismatch")
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            blob.update(block)
    valid = (sha.hexdigest() == asset["sha256"] if asset.get("sha256")
             else blob.hexdigest() == asset["git_blob_oid"])
    if not valid:
        raise ValueError("asset hash mismatch")


def make_handler(root: Path, engine: Any, checkpoint_sha256: str):
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ValueError("checkpoint SHA256 must be lowercase hexadecimal")

    def translate(item: dict[str, Any]) -> dict[str, Any]:
        video = request_path(root, item['video'])
        if not video.is_file():
            raise ValueError('video must be an existing session file')
        probe_video(video)
        started = time.perf_counter()
        english = engine.translate(video)
        if not isinstance(english, str):
            raise ValueError('SHuBERT returned invalid text')
        return {
            'english': english,
            'model_id': 'SHuBERT',
            'model_version': '578a0233e770c8ce4dc75d859b91fdea7c34f5aa',
            'checkpoint_sha256': checkpoint_sha256,
            'latency_ms': (time.perf_counter() - started) * 1000,
            'quality_status': 'unverified',
        }

    return translate


def probe_video(video: Path) -> None:
    try:
        output = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
             "-show_entries", "stream=width,height,nb_read_frames,duration", "-of", "json",
             str(video)], timeout=30, stderr=subprocess.DEVNULL
        )
    except subprocess.SubprocessError as error:
        raise ValueError("Video probe failed") from error
    streams = json.loads(output)["streams"]
    if len(streams) != 1:
        raise ValueError("Expected one video stream")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    frames, duration = int(stream["nb_read_frames"]), float(stream["duration"])
    if not (0 < width * height <= 3840 * 2160 and width > 0 and height > 0
            and 0 < frames <= 600 and 0 < duration <= 20):
        raise ValueError("Video exceeds the SHuBERT pilot bounds")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(work: Path) -> str:
    manifest_path = work / "source-and-asset-manifest.json"
    if sha_file(manifest_path) != (
        "d182267e9747826f0479337f7610cf0f342b3b7d34d3ea8eb8a42eea034e0d99"
    ):
        raise ValueError("SHuBERT manifest identity changed")
    manifest = json.loads(manifest_path.read_bytes())
    checkpoint = None
    for asset in manifest["model_assets"]:
        path = safe_path(work / "assets", asset["path"])
        verify_asset(path, asset)
        if asset["path"] == "models/checkpoint-11625/pytorch_model.bin":
            checkpoint = asset["sha256"]
    if not checkpoint:
        raise ValueError("SHuBERT checkpoint identity missing")
    for item in manifest["source_files"]:
        if sha_file(safe_path(work / "source", item["path"])) != item["sha256"]:
            raise ValueError("SHuBERT source identity changed")
    git = ["git", "-C", str(work / "dinov2-source")]
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != (
        "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
    ):
        raise ValueError("DINO source identity changed")
    if subprocess.check_output([*git, "status", "--porcelain"], text=True).strip():
        raise ValueError("DINO source is not clean")
    return checkpoint


def check_gpu(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    free, total = torch.cuda.mem_get_info()
    if total <= 0 or free < total * 0.75:
        raise RuntimeError("GPU reserve unavailable")
    torch.cuda.set_per_process_memory_fraction(0.75)




def load_checked(model_class, checkpoint):
    model, info = model_class.from_pretrained(
        checkpoint, local_files_only=True, output_loading_info=True
    )
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if not isinstance(info, dict) or any(key not in info for key in keys):
        raise ValueError("Checkpoint loading diagnostics missing")
    errors = {key: info[key] for key in keys if info[key]}
    if errors:
        raise ValueError(f"Checkpoint tensor incompatibility: {errors}")
    return model


def validate_features(face, left, right, pose):
    import numpy as np

    arrays = tuple(np.asarray(value) for value in (face, left, right, pose))
    if any(array.ndim != 2 for array in arrays):
        raise ValueError("Features must be two-dimensional")
    length = arrays[0].shape[0]
    if length == 0 or any(array.shape[0] != length for array in arrays):
        raise ValueError("Feature streams must have equal nonzero frame counts")
    if tuple(array.shape[1] for array in arrays) != (384, 384, 384, 14):
        raise ValueError("Unexpected feature dimensions")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonfinite feature values")
    return arrays


class Runtime:
    def __init__(self, work, torch, features, inference, tokenizer_class):
        self.torch = torch
        self.features = features
        self.work = work
        self.model = load_checked(
            inference.SignLanguageByT5ForConditionalGeneration,
            str(work / "assets/models/checkpoint-11625"),
        )
        names = (name for name, _ in self.model.named_parameters())
        if not any("signhubert_adapter.signhubert" in name for name in names):
            raise ValueError("Checkpoint model lacks the SHuBERT encoder")
        self.tokenizer = tokenizer_class.from_pretrained(
            str(work / "assets/models/byt5_base"), local_files_only=True
        )
        self.model.to("cuda").eval()
        self.last_generation_seconds = None
        self.last_detection_counts = None

    def generate(self, face, left, right, pose, *_args):
        arrays = validate_features(face, left, right, pose)
        tensors = [
            self.torch.tensor(a, dtype=self.torch.float32).unsqueeze(0).to("cuda") for a in arrays
        ]
        start = time.monotonic()
        with self.torch.no_grad():
            ids = self.model.generate(
                face_features=tensors[0],
                left_hand_features=tensors[1],
                right_hand_features=tensors[2],
                pose_features=tensors[3],
                max_length=2048,
                num_beams=5,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        self.torch.cuda.synchronize()
        self.last_generation_seconds = time.monotonic() - start
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    def translate(self, video):
        torch = self.torch
        features = self.features
        original_hub = torch.hub.load
        original_test = features.test
        original_detection = features.video_holistic

        def local_hub(repository, name, **kwargs):
            if repository != "facebookresearch/dinov2" or name != "dinov2_vits14_reg":
                raise ValueError("Unexpected DINO source request")
            return original_hub(str(self.work / "dinov2-source"), name, source="local", **kwargs)

        def detection(*args, **kwargs):
            result = original_detection(*args, **kwargs)
            self.last_detection_counts = {
                "frames": len(result),
                "failed_frames": sum(value is None for value in result.values()),
                "missing_face": sum(
                    not value or not value.get("face_landmarks") for value in result.values()
                ),
                "missing_hands": sum(
                    not value or not value.get("hand_landmarks") for value in result.values()
                ),
                "missing_pose": sum(
                    not value or not value.get("pose_landmarks") for value in result.values()
                ),
            }
            return result

        models = self.work / "assets/models"
        config = {
            "dino_face_model_path": str(models / "dinov2face.pth"),
            "dino_hands_model_path": str(models / "dinov2hand.pth"),
            "mediapipe_face_model_path": str(models / "face_landmarker_v2_with_blendshapes.task"),
            "mediapipe_hands_model_path": str(models / "hand_landmarker.task"),
            "slt_model_config": str(models / "byt5_base/config.json"),
            "slt_model_checkpoint": str(models / "checkpoint-11625"),
            "slt_tokenizer_checkpoint": str(models / "byt5_base"),
            "temp_dir": str(self.work),
        }
        self.last_generation_seconds = self.last_detection_counts = None
        try:
            torch.hub.load, features.test, features.video_holistic = (
                local_hub,
                self.generate,
                detection,
            )
            return features.SHuBERTProcessor(config).process_video(str(video))
        finally:
            torch.hub.load, features.test, features.video_holistic = (
                original_hub,
                original_test,
                original_detection,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    require_sol(args.work_root)
    require_sol(args.model_root)
    if not args.work_root.is_dir() or not args.model_root.is_dir():
        raise ValueError("Sol roots must already exist")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with redirect_stdout(sys.stderr):
        checkpoint = verify_assets(args.model_root)
        sys.path.insert(0, str(args.model_root / "source"))
        import features
        import inference
        import torch
        from transformers import ByT5Tokenizer

        check_gpu(torch)
        engine = Runtime(args.model_root, torch, features, inference, ByT5Tokenizer)
    serve(make_handler(args.work_root, engine, checkpoint))


if __name__ == "__main__":
    main()
