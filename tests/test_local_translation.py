from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from asl_recognizer.translation import evaluation, experiment, recorded
from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import DirectTranslationResult, TranslationRequest
from asl_recognizer.translation.experiment import ExperimentConfig
from asl_recognizer.translation.manifest import build_manifest, write_manifest
from asl_recognizer.translation.review import export_review, review_gate
from asl_recognizer.translation.schema import MMPOSE_SCHEMA
from asl_recognizer.translation.worker import main
from asl_recognizer.translation.youtube import channel_split, freeze_snapshot, verify_snapshot


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def snapshot_input(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "one.mp4").write_bytes(b"video-one")
    write_csv(tmp_path / "one.csv", [{"start_ms": 0, "end_ms": 1000, "caption": "hello"}])
    index = tmp_path / "index.csv"
    write_csv(
        index,
        [
            {
                "video_id": "one",
                "channel_id": "channel",
                "video_path": "one.mp4",
                "caption_path": "one.csv",
                "downloaded_at": "2026-10-06",
                "availability": "available",
                "split": "train",
            },
            {
                "video_id": "missing",
                "channel_id": "other",
                "video_path": "missing.mp4",
                "caption_path": "missing.csv",
                "downloaded_at": "2026-10-06",
                "availability": "unavailable",
                "split": "dev",
            },
        ],
    )
    ids = tmp_path / "ids.txt"
    ids.write_text("one\nmissing\n", encoding="utf-8")
    return index, ids


def test_snapshot_freezes_captions_and_preserves_unavailable(tmp_path: Path) -> None:
    index, ids = snapshot_input(tmp_path)
    output = tmp_path / "snapshot"
    report = freeze_snapshot(index=index, root=tmp_path, official_ids=ids, output=output)
    assert report["videos"][0]["availability"] == "unavailable"
    verify_snapshot(output / "snapshot.json", tmp_path)
    (tmp_path / "one.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="caption changed"):
        verify_snapshot(output / "snapshot.json", tmp_path)
    with pytest.raises(FileExistsError):
        freeze_snapshot(index=index, root=tmp_path, official_ids=ids, output=output)


def test_snapshot_rejects_channel_leakage_and_path_escape(tmp_path: Path) -> None:
    index, ids = snapshot_input(tmp_path)
    rows = list(csv.DictReader(index.open()))
    rows[1]["channel_id"] = "channel"
    write_csv(index, rows)
    with pytest.raises(ValueError, match="spans"):
        freeze_snapshot(index=index, root=tmp_path, official_ids=ids, output=tmp_path / "a")
    rows[1]["channel_id"] = "other"
    rows[0]["video_path"] = "../one.mp4"
    write_csv(index, rows)
    with pytest.raises(ValueError, match="inside"):
        freeze_snapshot(index=index, root=tmp_path, official_ids=ids, output=tmp_path / "b")
    assert channel_split("same") == channel_split("same")


def test_snapshot_excludes_evaluation_hash(tmp_path: Path) -> None:
    from asl_recognizer.translation.youtube import file_hash

    index, ids = snapshot_input(tmp_path)
    report = freeze_snapshot(
        index=index,
        root=tmp_path,
        official_ids=ids,
        output=tmp_path / "snapshot",
        exclude_hashes={file_hash(tmp_path / "one.mp4")},
    )
    assert report["videos"][1]["reason"] == "duplicate-or-evaluation-video"


def test_review_requires_complete_identified_reviews(tmp_path: Path) -> None:
    manifest = build_manifest(
        [
            {
                "id": str(i),
                "video_id": str(i),
                "start": 0,
                "end": 1,
                "caption": "hello",
                "split": "train",
            }
            for i in range(100)
        ],
        source="how2sign",
    )
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    review = tmp_path / "review.csv"
    export_review(path, review)
    assert not review_gate(review, manifest_path=path)["passed"]
    with review.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update(
            meaning_ok="yes",
            timing_ok="yes",
            visibility_ok="yes",
            extraction_ok="yes",
            reviewer="ASL reviewer",
        )
    write_csv(review, rows)
    assert review_gate(review, manifest_path=path)["passed"]
    rows[0]["meaning_ok"] = ""
    write_csv(review, rows)
    assert not review_gate(review, manifest_path=path)["passed"]
    rows[0]["caption"] = "different caption"
    write_csv(review, rows)
    with pytest.raises(ValueError, match="differs"):
        review_gate(review, manifest_path=path)


class Tokenizer:
    pad_token_id = 0

    def __call__(self, texts: list[str], **_kwargs: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1] for _ in texts])}


class Model(torch.nn.Module):
    def __init__(self, _config: object) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(266, 2)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        pooled = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return torch.nn.functional.cross_entropy(self.layer(pooled), labels[:, 0])

    def translate(self, _request: object, _tokenizer: object, *, device: str):  # type: ignore[no-untyped-def]
        return DirectTranslationResult("this is a complete sentence", None, "tiny", "test", 1.0)


def source(tmp_path: Path, name: str) -> Path:
    manifest = build_manifest(
        [
            {
                "id": f"{name}-{split}",
                "video_id": f"{name}-{split}",
                "start": 0,
                "end": 1,
                "caption": "this is a complete sentence",
                "split": split,
            }
            for split in ("train", "dev", "test")
        ],
        source=name,
    )
    path = tmp_path / f"{name}.csv"
    write_manifest(path, manifest)
    cache = TranslationFeatureCache(tmp_path / "features" / name)
    for row in manifest.records:
        cache.store(
            row.clip_id,
            TranslationRequest(
                np.ones((2, 266), np.float32), np.ones((2, 133), np.bool_), MMPOSE_SCHEMA
            ),
            metadata={
                "manifest_hash": manifest.fingerprint,
                "source": name,
                "split": row.split,
                "caption": row.caption,
                "feature_schema": MMPOSE_SCHEMA,
                "extractor": {"backend": "mmpose", "assets": {"fixture": "asset"}},
            },
        )
    return path


@pytest.fixture
def tiny(monkeypatch):  # type: ignore[no-untyped-def]
    from transformers import AutoTokenizer, optimization

    monkeypatch.setattr(experiment, "LandmarkT5", Model)
    monkeypatch.setattr(evaluation, "LandmarkT5", Model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_k: Tokenizer())
    monkeypatch.setattr(
        optimization, "Adafactor", lambda params, **_k: torch.optim.SGD(params, lr=0.001)
    )


def cpu_config(**kwargs: object) -> ExperimentConfig:
    return replace(
        ExperimentConfig(),
        batch_size=2,
        microbatch_size=1,
        precision="fp32",
        gradient_checkpointing=False,
        youtube_steps=2,
        how2sign_steps=2,
        validation_interval=1,
        checkpoint_interval=1,
        **kwargs,
    )


def test_accumulation_resume_selection_and_final_test(tmp_path: Path, tiny) -> None:  # type: ignore[no-untyped-def]
    yt, h2s = source(tmp_path, "youtube-asl"), source(tmp_path, "how2sign")
    kwargs = dict(
        features=tmp_path / "features",
        how2sign_manifest=h2s,
        youtube_manifest=yt,
        config=cpu_config(),
        device_name="cpu",
    )
    experiment.run_experiment(**kwargs, output=tmp_path / "full")
    for _ in range(4):
        experiment.run_experiment(**kwargs, output=tmp_path / "resumed", job_steps=1)
    full = torch.load(tmp_path / "full/latest.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=True)
    assert full["status"] == resumed["status"] == "completed"
    assert full["global_step"] == resumed["global_step"] == 4
    for state in (full, resumed):
        assert set(state["source_timings"]) == {"youtube-asl", "how2sign"}
        for timing in state["source_timings"].values():
            assert timing["optimizer_updates"] == 2
            assert timing["optimizer_seconds"] > 0
            assert timing["validation_seconds_max"] > 0
        assert state["checkpoint_seconds_max"] > 0
    for key, value in full["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="finalized experiment"):
        evaluation.evaluate_checkpoint(
            checkpoint=tmp_path / "full/latest.pt",
            manifest_path=h2s,
            features=tmp_path / "features",
            output=tmp_path / "too-early",
            split="test",
            with_bleurt=False,
        )
    selected = tmp_path / "final.pt"
    experiment.finalize_experiment(tmp_path / "full", selected)
    evaluation.evaluate_checkpoint(
        checkpoint=selected,
        manifest_path=h2s,
        features=tmp_path / "features",
        output=tmp_path / "test-report",
        split="test",
        with_bleurt=False,
    )
    report = json.loads((tmp_path / "test-report/report.json").read_text())
    assert report["bleu"] == pytest.approx(100)


def test_accumulation_matches_effective_batch(tmp_path: Path, tiny) -> None:  # type: ignore[no-untyped-def]
    yt, h2s = source(tmp_path, "youtube-asl"), source(tmp_path, "how2sign")
    kwargs = dict(
        features=tmp_path / "features",
        how2sign_manifest=h2s,
        youtube_manifest=yt,
        device_name="cpu",
        job_steps=1,
    )
    config = cpu_config()
    experiment.run_experiment(**kwargs, config=config, output=tmp_path / "micro")
    experiment.run_experiment(
        **kwargs, config=replace(config, microbatch_size=2), output=tmp_path / "batch"
    )
    a = torch.load(tmp_path / "micro/latest.pt", weights_only=True)
    b = torch.load(tmp_path / "batch/latest.pt", weights_only=True)
    for key, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][key])


def test_early_stop_finishes_each_phase_and_rejects_changed_resume(tmp_path: Path, tiny) -> None:  # type: ignore[no-untyped-def]
    yt, h2s = source(tmp_path, "youtube-asl"), source(tmp_path, "how2sign")
    config = replace(cpu_config(), youtube_steps=10, how2sign_steps=10, early_stopping_patience=1)
    kwargs = dict(
        features=tmp_path / "features",
        how2sign_manifest=h2s,
        youtube_manifest=yt,
        device_name="cpu",
        output=tmp_path / "run",
    )
    experiment.run_experiment(**kwargs, config=config)
    saved = torch.load(tmp_path / "run/latest.pt", weights_only=True)
    # Initial model is already best; one non-improvement per phase.
    assert saved["global_step"] == 2
    with pytest.raises(ValueError, match="identity"):
        experiment.run_experiment(**kwargs, config=replace(config, seed=43))


def test_regressing_training_selects_initial_checkpoint(tmp_path: Path, tiny, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    yt, h2s = source(tmp_path, "youtube-asl"), source(tmp_path, "how2sign")
    evaluations = iter((10.0, 9.0, 8.0, 10.0, 9.0, 8.0))
    monkeypatch.setattr(
        experiment,
        "_dev",
        lambda *_a, **_k: {"bleu": next(evaluations), "dev_loss": 1.0, "examples": []},
    )
    run = tmp_path / "regression"
    experiment.run_experiment(
        features=tmp_path / "features",
        how2sign_manifest=h2s,
        youtube_manifest=yt,
        output=run,
        config=cpu_config(),
        device_name="cpu",
    )
    selected = torch.load(run / "best-how2sign.pt", weights_only=True)
    assert selected["phase_step"] == 0 and selected["best_bleu"] == 10.0
    initial = torch.load(run / "best-youtube-asl.pt", weights_only=True)
    for key, value in selected["model"].items():
        torch.testing.assert_close(value, initial["model"][key], rtol=0, atol=0)


def test_allocation_wall_budget_saves_resumable_update(tmp_path: Path, tiny) -> None:  # type: ignore[no-untyped-def]
    yt, h2s = source(tmp_path, "youtube-asl"), source(tmp_path, "how2sign")
    run = tmp_path / "wall-budget"
    kwargs = dict(
        features=tmp_path / "features",
        how2sign_manifest=h2s,
        youtube_manifest=yt,
        output=run,
        config=cpu_config(),
        device_name="cpu",
    )
    experiment.run_experiment(**kwargs, wall_time_seconds=0.000001)
    first = torch.load(run / "latest.pt", weights_only=True)
    assert first["global_step"] == 1 and first["status"] == "running"
    experiment.run_experiment(**kwargs)
    assert torch.load(run / "latest.pt", weights_only=True)["status"] == "completed"


def test_recorded_extraction_interval_mask_and_overlay(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cv2

    from asl_recognizer.features.mmpose_backend import WholeBodyFrame
    from asl_recognizer.translation import runtime

    video = tmp_path / "recording.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()

    class Backend:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self):  # type: ignore[no-untyped-def]
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def reset(self) -> None:
            pass

        def close(self) -> None:
            pass

        def detect_translation(self, _image: object, timestamp: int) -> WholeBodyFrame:
            return WholeBodyFrame(np.ones((133, 2), np.float32), np.ones(133, np.bool_), timestamp,
                                  np.ones(133, np.float32), np.ones((133, 2), np.float32))

    monkeypatch.setattr(recorded, "MMPoseWholeBodyBackend", Backend, raising=False)
    monkeypatch.setattr(recorded, "extractor_identity",
                        lambda *_a: {"fixture": "asset"}, raising=False)
    monkeypatch.setattr(runtime, "MMPoseWholeBodyBackend", Backend)
    monkeypatch.setattr(runtime, "extractor_identity", lambda *_a: {"fixture": "asset"})
    report = recorded.extract_video(
        video=video,
        assets={},
        output=tmp_path / "features",
        device="cpu",
        start_ms=200,
        end_ms=600,
        overlay=tmp_path / "overlay.mp4",
    )
    assert report["sampled_frames"] == 2
    assert (tmp_path / "overlay.mp4").is_file()
    with pytest.raises(ValueError, match="exceeds"):
        recorded.extract_video(video=video, assets={}, output=tmp_path / "invalid", end_ms=2000)


def test_cli_snapshot_and_pending_review_gate(tmp_path: Path) -> None:
    index, ids = snapshot_input(tmp_path)
    assert (
        main(
            [
                "youtube-snapshot",
                "--index",
                str(index),
                "--root",
                str(tmp_path),
                "--official-ids",
                str(ids),
                "--output",
                str(tmp_path / "snapshot"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "verify-snapshot",
                "--snapshot",
                str(tmp_path / "snapshot/snapshot.json"),
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )


def test_subset_preserves_test_and_split_assignments(tmp_path: Path) -> None:
    from asl_recognizer.translation.manifest import read_manifest
    from asl_recognizer.translation.subsets import subset_manifest

    manifest = build_manifest(
        [
            {
                "id": f"{split}-{i}",
                "video_id": f"{split}-{i}",
                "start": 0,
                "end": 1,
                "caption": "hello",
                "split": split,
            }
            for split in ("train", "dev", "test")
            for i in range(3)
        ],
        source="how2sign",
    )
    original, subset = tmp_path / "original.csv", tmp_path / "subset.csv"
    write_manifest(original, manifest)
    subset_manifest(original, subset, train_limit=1, dev_limit=1)
    selected = read_manifest(subset)
    assert len(selected.accepted("train")) == len(selected.accepted("dev")) == 1
    assert len(selected.accepted("test")) == 3
    assert [(row.clip_id, row.split) for row in selected.records] == [
        (row.clip_id, row.split) for row in manifest.records
    ]


def test_resource_estimate_counts_optimizer_updates_and_cache_sizes(tmp_path: Path) -> None:
    from asl_recognizer.translation.resources import resource_estimate

    progress, extraction = tmp_path / "progress.json", tmp_path / "extraction.json"
    progress.write_text(
        json.dumps({"global_step": 10, "optimizer_seconds": 20, "peak_vram_bytes": 100})
    )
    extraction.write_text(json.dumps({"processing_fps": 10}))
    (tmp_path / "feature.npz").write_bytes(b"1234567890")
    report = resource_estimate(
        progress=progress,
        extraction_report=extraction,
        feature_root=tmp_path,
        target_clips=100,
        target_frames=36000,
        target_steps=1800,
    )
    assert report["training_gpu_hours"] == 1
    assert report["extraction_gpu_hours"] == 1
    assert report["estimated_feature_bytes"] == 1000


def test_recorded_inference_refuses_mismatched_extractor(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": {"feature_schema": MMPOSE_SCHEMA, "extractor": {"confidence_threshold": 0.5}},
            "model": {},
        },
        checkpoint,
    )

    def extract(command: list[str], **_kwargs: object) -> None:
        destination = Path(command[command.index("--output") + 1])
        TranslationFeatureCache(destination).store(
            "recorded",
            TranslationRequest(
                np.ones((2, 266), np.float32), np.ones((2, 133), np.bool_), MMPOSE_SCHEMA
            ),
            metadata={"extractor": {"different": "assets"}},
        )

    monkeypatch.setattr(recorded.subprocess, "run", extract)
    monkeypatch.setattr(
        recorded, "LandmarkT5", lambda *_a: pytest.fail("must reject before loading the model")
    )
    with pytest.raises(ValueError, match="differs"):
        recorded.infer_video(
            checkpoint=checkpoint,
            video=tmp_path / "video.mp4",
            assets={},
            output=tmp_path / "result.json",
            device="cpu",
        )
