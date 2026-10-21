"""Pinned official Uni-Sign pose-only inference; assets and real work stay on Sol."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch

from ..features.mmpose_backend import WholeBodyFrame
from .pose_cache import PoseCache
from .runtime import request_path, require_sol, serve
from .youtube import file_hash

SOURCE_URL = "https://github.com/ZechengLi19/Uni-Sign"
SOURCE_REVISION = "eed438bcb49e30405cd6ccdfcccca330c134e830"
WEIGHTS_REPOSITORY = "ZechengLi19/Uni-Sign"
WEIGHTS_REVISION = "eab251b7fe7e8521afc0e67be98add670ea40a0d"
WEIGHTS_FILENAME = "how2sign_pose_only_slt.pth"
WEIGHTS_SHA256 = "1bfd5f3312f04e4736f0a52f4ef9535916e6de9676a2a0d00c708748683fb00d"
TOKENIZER_REVISION = "2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f"


def verify_source(root: Path) -> None:
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        changes = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                           "--untracked-files=no"], text=True)
    except subprocess.CalledProcessError as error:
        raise ValueError("official Uni-Sign source checkout is required") from error
    if revision != SOURCE_REVISION or changes:
        raise ValueError("official Uni-Sign source revision or tracked content differs")


def verify_checkpoint(path: Path) -> None:
    if file_hash(path) != WEIGHTS_SHA256:
        raise ValueError("official Uni-Sign checkpoint checksum differs")


def official_preprocessor(root: Path) -> Callable[..., Any]:
    """Use the pinned functions verbatim without importing training-only dependencies."""
    verify_source(root)
    path = root / "datasets.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    functions: list[ast.stmt] = [node for node in parsed.body if isinstance(node, ast.FunctionDef)
                                 and node.name in {"crop_scale", "load_part_kp"}]
    if len(functions) != 2:
        raise ValueError("official preprocessing functions are missing")
    namespace: dict[str, Any] = {"np": np, "torch": torch, "copy": copy}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return cast(Callable[..., Any], namespace["load_part_kp"])


def prepare_input(frames: Sequence[WholeBodyFrame], *, width: int, height: int,
                  preprocess: Callable[..., Any], max_length: int = 256, seed: int = 42
                  ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if not frames or min(width, height, max_length) <= 0:
        raise ValueError("Uni-Sign requires frames, dimensions and a positive frame cap")
    if any(frame.raw_coordinates is None or frame.scores is None for frame in frames):
        raise ValueError("Uni-Sign requires raw coordinates and continuous scores")
    indices = sorted(random.Random(seed).sample(range(len(frames)), max_length)) \
        if len(frames) > max_length else list(range(len(frames)))
    skeletons, scores = [], []
    for index in indices:
        coordinates, confidence = frames[index].raw_coordinates, frames[index].scores
        assert coordinates is not None and confidence is not None
        skeletons.append(coordinates[None] / np.array([width, height])[None, None])
        scores.append(confidence[None])
    parts = preprocess(skeletons, scores, force_ok=True)
    batch = {part: value.to(torch.float32).unsqueeze(0) for part, value in parts.items()}
    batch["attention_mask"] = torch.ones((1, len(indices)), dtype=torch.long)
    return batch, {"indices": indices, "seed": seed, "native_frames": len(frames),
                   "max_length": max_length, "preprocessing": "official-online-pose-only"}


@torch.inference_mode()
def generate_text(model: Any, batch: dict[str, torch.Tensor], *, beam_width: int = 5,
                  max_tokens: int = 128) -> str:
    # The official forward requires a target for an unused loss. Never pass dev references.
    stack = model(batch, {"gt_sentence": [""]})
    tokens = model.generate(stack, num_beams=beam_width, max_new_tokens=max_tokens)
    return str(model.mt5_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]).strip()


class UniSignEngine:
    def __init__(self, source: Path, checkpoint: Path, tokenizer: Path, *,
                 device: str = "cuda:0") -> None:
        from transformers import MT5Config, MT5ForConditionalGeneration

        started = time.perf_counter()
        verify_source(source)
        verify_checkpoint(checkpoint)
        self.preprocess = official_preprocessor(source)
        self.device = device
        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        spec = importlib.util.spec_from_file_location("asl_official_unisign", source / "models.py")
        if spec is None or spec.loader is None:
            raise ValueError("official Uni-Sign model is missing")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(source))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        module.__dict__["mt5_path"] = str(tokenizer)
        config = MT5Config.from_pretrained(tokenizer, local_files_only=True)
        # The verified full checkpoint supplies every weight; avoid fetching duplicate base weights.
        with patch.object(MT5ForConditionalGeneration, "from_pretrained",
                          side_effect=lambda *_a, **_k: MT5ForConditionalGeneration(config)):
            model = module.Uni_Sign(SimpleNamespace(hidden_dim=256, dataset="How2Sign",
                                                    rgb_support=False, label_smoothing=0.2))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        self.model = model.to(device=device, dtype=self.dtype).eval()
        self.initialization_seconds = time.perf_counter() - started
        self.identity = {"source_revision": SOURCE_REVISION, "weights_revision": WEIGHTS_REVISION,
                         "checkpoint_sha256": WEIGHTS_SHA256,
                         "tokenizer_revision": TOKENIZER_REVISION,
                         "tokenizer_files": {p.name: file_hash(p) for p in tokenizer.iterdir()
                                             if p.is_file()}, "strict_state_dict": True}

    def translate(self, frames: Sequence[WholeBodyFrame], metadata: dict[str, Any]
                  ) -> dict[str, Any]:
        if metadata.get("raw_pose_sampling") != "every-native-frame-v1":
            raise ValueError("Uni-Sign comparison requires native raw pose frames")
        started = time.perf_counter()
        batch, receipt = prepare_input(frames, width=metadata["width"], height=metadata["height"],
                                        preprocess=self.preprocess)
        inputs = {key: value.to(device=self.device, dtype=self.dtype)
                  for key, value in batch.items()}
        english = generate_text(self.model, inputs)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        return {"english": english, "latency_ms": (time.perf_counter() - started) * 1000,
                "input": receipt, "identity": self.identity, "beam_width": 5,
                "decoder_max_tokens": 128, "initialization_seconds": self.initialization_seconds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "tokenizer", "work-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    require_sol(args.work_root)
    with redirect_stdout(sys.stderr):
        engine = UniSignEngine(args.source, args.checkpoint, args.tokenizer, device=args.device)

    def translate(item: dict[str, Any]) -> dict[str, Any]:
        root = request_path(args.work_root, item["features"])
        frames, metadata = PoseCache(root / "raw-pose").load("recorded")
        return engine.translate(frames, metadata)
    serve(translate, stdin=sys.stdin, stdout=sys.stdout)


if __name__ == "__main__":
    main()
