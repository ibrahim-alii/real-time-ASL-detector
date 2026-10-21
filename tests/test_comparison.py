import pytest

from asl_recognizer.translation.manifest import ClipManifest, ManifestRecord


def test_comparison_scores_aligned_unique_dev_outputs():
    from asl_recognizer.translation.comparison import comparison_report

    rows = [
        {
            "clip_id": str(i),
            "video_id": str(i),
            "reference": "this is a clear reference sentence with eight words",
            "baseline": {"english": "wrong", "latency_ms": 10 + i},
            "unisign": {
                "english": "this is a clear reference sentence with eight words",
                "latency_ms": 20 + i,
            },
        }
        for i in range(3)
    ]
    report = comparison_report(rows, bootstrap_samples=20)
    assert report["models"]["unisign"]["bleu"] == pytest.approx(100)
    assert report["models"]["unisign"]["chrf"] == 100
    assert report["models"]["baseline"]["latency_ms"]["p50"] == 11
    assert report["paired_difference"]["bleu"]["lower"] > 0
    assert report["test_examples_used"] == 0
    assert report["semantic_review"] == "pending"
    with pytest.raises(ValueError, match="unique"):
        comparison_report([*rows, rows[0]])


def test_development_selection_never_uses_test():
    from asl_recognizer.translation.comparison import development_rows

    manifest = ClipManifest(
        tuple(
            ManifestRecord(s, "how2sign", s, s, 0, 100, "reference")
            for s in ("train", "dev", "test")
        )
    )
    assert [r.clip_id for r in development_rows(manifest, 8)] == ["dev"]
    with pytest.raises(ValueError):
        development_rows(manifest, 0)
    leaking = ClipManifest(
        (
            ManifestRecord("train", "how2sign", "train", "same", 0, 100, "ref"),
            ManifestRecord("dev", "how2sign", "dev", "same", 0, 100, "ref"),
        )
    )
    with pytest.raises(ValueError, match="recording"):
        development_rows(leaking, 8)


def test_compare_clients_receive_features_without_reference(tmp_path):
    from asl_recognizer.translation.comparison import predict

    class Worker:
        def __init__(self):
            self.requests = []

        def request(self, item, **kwargs):
            self.requests.append(item)
            return {"english": "result", "latency_ms": 1}

    worker = Worker()
    result = predict(worker, tmp_path)
    assert result["english"] == "result"
    assert set(worker.requests[0]) == {"features"}
    assert result["request_ms"] >= 0


def test_cli_runs_bounded_dev_pipeline_and_reuses_workers(tmp_path, monkeypatch):
    import json

    from asl_recognizer.translation import comparison, runtime
    from asl_recognizer.translation.manifest import write_manifest

    manifest = ClipManifest(
        tuple(
            ManifestRecord(s, "how2sign", s, s, 0, 100, "this is the reference with several words")
            for s in ("train", "dev", "test")
        )
    )
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    monkeypatch.setattr(runtime, "require_sol", lambda root: None)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    calls = []
    workers = []

    class Worker:
        def __init__(self, command, **kwargs):
            self.command = command
            self.closed = False
            workers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def request(self, item, **kwargs):
            calls.append(item)
            if "video" in item:
                return {"raw_pose_frames": 4}
            return {"english": "this is the reference with several words", "latency_ms": 2}

    monkeypatch.setattr(comparison, "PersistentWorker", Worker)
    monkeypatch.setattr(comparison.subprocess, "check_output", lambda *a, **k: "synthetic GPU")
    arguments = []
    for name in (
        "work-root",
        "recordings",
        "assets-json",
        "baseline-checkpoint",
        "extraction-python",
        "translation-python",
        "unisign-source",
        "unisign-checkpoint",
        "tokenizer",
    ):
        arguments.extend(["--" + name, str(tmp_path)])
    output = tmp_path / "result"
    comparison.main([*arguments, "--manifest", str(path), "--output", str(output)])
    report = json.loads((output / "comparison.json").read_text())
    assert report["status"] == "completed" and report["examples"] == 1
    assert report["initialization"]["warm_prediction_parity"] is True
    assert all(worker.closed for worker in workers) and len(workers) == 3
    assert all("reference" not in item and "caption" not in item for item in calls)
    assert sum("video" in item for item in calls) == 1
    assert "train.mp4" not in str(calls) and "test.mp4" not in str(calls)
