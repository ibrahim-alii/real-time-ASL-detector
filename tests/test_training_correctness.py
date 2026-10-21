"""Numerical, recovery, deadline, and bounded video regression tests."""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from asl_recognizer.translation import experiment
from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import TranslationRequest
from asl_recognizer.translation.manifest import build_manifest, write_manifest


@pytest.fixture
def fixture(monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location(
        "fixture", Path(__file__).with_name("test_local_translation.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from transformers import AutoTokenizer, optimization
    monkeypatch.setattr(experiment, "LandmarkT5", module.Model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: module.Tokenizer())
    monkeypatch.setattr(optimization, "Adafactor", lambda params, **k:
                        torch.optim.SGD(params, lr=0.001))
    monkeypatch.setattr(experiment, "_dev", lambda *a, **k:
                        {"bleu": 10.0, "dev_loss": 1.0, "examples": []})
    return module


def test_unequal_token_accumulation(tmp_path: Path, fixture: Any,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoTokenizer

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
            tokens = [[1] if text == "short" else [1, 2, 3, 4] for text in texts]
            size = max(map(len, tokens))
            return {"input_ids": torch.tensor([t + [0] * (size - len(t)) for t in tokens])}

    class Model(torch.nn.Module):
        def __init__(self, config: Any) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(266, 6)

        def forward(self, values: Any, mask: Any, labels: Any) -> torch.Tensor:
            logits = self.layer(values.mean(1)).unsqueeze(1).expand(-1, labels.shape[1], -1)
            return torch.nn.functional.cross_entropy(logits.reshape(-1, 6), labels.flatten())

    monkeypatch.setattr(experiment, "LandmarkT5", Model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    rows = [dict(id=str(i), video_id=str(i), split="train" if i < 2 else "dev",
                 caption="short" if i == 0 else "long", start=0, end=1) for i in range(3)]
    manifest = build_manifest(rows, source="how2sign")
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    cache = TranslationFeatureCache(tmp_path / "features/how2sign")
    for index, row in enumerate(manifest.records):
        request = TranslationRequest(np.full((2, 266), 0.1 * (index + 1), np.float32),
                                     np.ones((2, 133), np.bool_))
        cache.store(row.clip_id, request, metadata={"manifest_hash": manifest.fingerprint,
                    "feature_schema": request.feature_schema, "source": "how2sign",
                    "caption": row.caption, "split": row.split, "extractor": {"tiny": True}})
    config = replace(fixture.cpu_config(), youtube_steps=0, how2sign_steps=3,
                     validation_interval=2, batch_size=4)
    for size in (1, 4):
        experiment.run_experiment(features=tmp_path / "features", how2sign_manifest=path,
            youtube_manifest=None, output=tmp_path / str(size),
            config=replace(config, microbatch_size=size), job_steps=1, device_name="cpu")
    micro = torch.load(tmp_path / "1/latest.pt", weights_only=True)
    full = torch.load(tmp_path / "4/latest.pt", weights_only=True)
    for name, value in full["model"].items():
        torch.testing.assert_close(value, micro["model"][name], rtol=1e-6, atol=1e-7)
    assert micro["train_loss"] == pytest.approx(full["train_loss"], rel=1e-6)


@pytest.mark.parametrize("fail_step", [0, 1])
def test_selected_publication_interruption(tmp_path: Path, fixture: Any,
        monkeypatch: pytest.MonkeyPatch, fail_step: int) -> None:
    path = fixture.source(tmp_path, "how2sign")
    output = tmp_path / "run"
    config = replace(fixture.cpu_config(), youtube_steps=0, how2sign_steps=2)
    args = dict(features=tmp_path / "features", how2sign_manifest=path,
                youtube_manifest=None, output=output, config=config, device_name="cpu")
    original = experiment.atomic_checkpoint
    fired = False

    def interrupt(payload: dict[str, Any], destination: Path) -> None:
        nonlocal fired
        if (not fired and payload["global_step"] == fail_step
                and destination.name == "best-how2sign.pt"):
            fired = True
            raise OSError("simulated interruption publishing selection")
        original(payload, destination)

    evaluations = [0]
    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        evaluations[0] += 1
        return {"bleu": 10.0 + int(fail_step == 1 and evaluations[0] > 1),
                "dev_loss": 1.0, "examples": []}
    monkeypatch.setattr(experiment, "_dev", evaluate)
    monkeypatch.setattr(experiment, "atomic_checkpoint", interrupt)
    with pytest.raises(OSError, match="simulated interruption"):
        experiment.run_experiment(**args)
    monkeypatch.setattr(experiment, "atomic_checkpoint", original)
    experiment.run_experiment(**args)
    experiment.finalize_experiment(output, tmp_path / "selected.pt")
    latest = torch.load(output / "latest.pt", weights_only=True)
    selected = torch.load(output / "best-how2sign.pt", weights_only=True)
    assert latest["status"] == "completed"
    assert selected["best_bleu"] == latest["selected_bleu"] == 10.0 + fail_step


def test_initial_deadline_prevents_new_work(tmp_path: Path, fixture: Any,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    path = fixture.source(tmp_path, "how2sign")
    clock = [0.0]
    monkeypatch.setattr(experiment.time, "monotonic", lambda: clock[0])

    def evaluate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        clock[0] = 100
        return {"bleu": 1.0, "dev_loss": 1.0, "examples": []}

    monkeypatch.setattr(experiment, "_dev", evaluate)
    experiment.run_experiment(features=tmp_path / "features", how2sign_manifest=path,
        youtube_manifest=None, output=tmp_path / "run",
        config=replace(fixture.cpu_config(), youtube_steps=0), device_name="cpu",
        wall_time_seconds=5, strict_deadline=True)
    state = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    assert state["global_step"] == 0


def test_sampling_stays_inside_caption(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    from asl_recognizer.features.video import sampled_video_segment

    class Capture:
        position = 0.0
        def isOpened(self) -> bool:
            return True
        def get(self, prop: int) -> float:
            return {cv2.CAP_PROP_FRAME_COUNT: 100, cv2.CAP_PROP_FPS: 30,
                    cv2.CAP_PROP_POS_MSEC: self.position}.get(prop, 0)
        def set(self, prop: int, position: float) -> None:
            self.position = position
        def read(self) -> tuple[bool, Any]:
            return True, np.zeros((2, 2, 3), np.uint8)
        def release(self) -> None:
            pass
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: Capture())
    sampled = sampled_video_segment(Path("synthetic"), start_ms=0, end_ms=600,
                                    sample_rate_hz=1)
    assert [timestamp for _, timestamp in sampled.frames] == [0]


def test_video_iterator_is_bounded_and_closes(monkeypatch):
    import cv2

    from asl_recognizer.features import video
    class Capture:
        reads = 0
        released = False
        def isOpened(self): return True
        def get(self, prop):
            return {cv2.CAP_PROP_FRAME_COUNT: 100, cv2.CAP_PROP_FPS: 30}[prop]
        def set(self, prop, value): pass
        def read(self):
            self.reads += 1
            return True, np.zeros((2, 2, 3), np.uint8)
        def release(self): self.released = True
    capture = Capture()
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a: capture)
    iterator = video.iter_every_other_video_frame(Path('synthetic'), start_ms=0, end_ms=2000)
    assert next(iterator)[1] == 0
    assert capture.reads == 1
    iterator.close()
    assert capture.released
