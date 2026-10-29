import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import asl_recognizer


def workspace():
    root = Path(asl_recognizer.__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "workspace_warm", root / "scripts/workspace_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_warm_runtime_reuses_isolated_workers_and_closes(tmp_path, monkeypatch):
    module = workspace()
    workers = []

    class Worker:
        def __init__(self, command, **kwargs):
            self.command = command
            self.calls = []
            self.closed = False
            workers.append(self)
            self.env = kwargs["env"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def request(self, item):
            self.calls.append(item)
            if "video" in item:
                return {"extraction_seconds": 0.1}
            return {
                "english": "fixture",
                "model_id": "tiny",
                "model_version": "v1",
                "latency_ms": 1,
                "checkpoint_sha256": "a" * 64,
            }

    monkeypatch.setattr(module, "PersistentWorker", Worker)
    config = dict.fromkeys(
        ("python", "extraction_python", "checkpoint", *module.ASSET_NAMES), "fixture"
    )
    config["work_root"] = str(tmp_path)
    runtime = module.WarmRuntime(config)
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        result = runtime.infer(root, root / "video", root / "overlay")
        assert result["checkpoint_sha256"] == "a" * 64 and result["timings"]["total_ms"] >= 0
    assert len(workers) == 2 and all(len(worker.calls) == 2 for worker in workers)
    assert "dependencies" not in workers[0].env["PYTHONPATH"]
    runtime.close()
    assert all(worker.closed for worker in workers)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0", "1"]


def test_failed_worker_start_cleans_session(tmp_path, monkeypatch):
    module = workspace()

    def fail(*a, **k):
        raise RuntimeError("failed")

    monkeypatch.setattr(module, "PersistentWorker", fail)
    config = dict.fromkeys(
        ("python", "extraction_python", "checkpoint", *module.ASSET_NAMES), "fixture"
    )
    config["work_root"] = str(tmp_path)
    with pytest.raises(RuntimeError):
        module.WarmRuntime(config)
    assert not list(tmp_path.iterdir())


def test_warm_clip_returns_total_timing_and_cleans_files(tmp_path, monkeypatch):
    module = workspace()
    capture = MagicMock()
    capture.get.return_value = 30
    monkeypatch.setattr("cv2.VideoCapture", lambda *args: capture)
    monkeypatch.setattr(module.shutil, "disk_usage", lambda root: MagicMock(free=3 * 1024**3))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: None)
    warm = MagicMock()

    def infer(root, video, overlay):
        overlay.with_suffix(".json").write_text(
            json.dumps({"width": 32, "height": 24, "frames": [], "source": "private"})
        )
        return {
            "english": "result",
            "model_id": "tiny",
            "model_version": "v1",
            "checkpoint_sha256": "a" * 64,
            "quality_status": "unverified",
            "timings": {"warm_workers": True},
        }

    warm.infer.side_effect = infer
    result = module.translate_clip(b"synthetic", {"work_root": str(tmp_path)}, warm)
    assert result["checkpoint_sha256"] == "a" * 64
    assert result["latency_ms"] == result["timings"]["total_ms"]
    assert result["timings"]["warm_workers"] is True and "private" not in json.dumps(result)
    assert not list(tmp_path.iterdir())


def test_server_owns_and_closes_warm_runtime(tmp_path, monkeypatch):
    module = workspace()
    warm = MagicMock()
    monkeypatch.setattr(module, "WarmRuntime", lambda config: warm)
    server = module.WorkspaceServer(0, {"work_root": str(tmp_path), "warm_workers": True})
    assert server.warm is warm
    server.server_close()
    assert warm.close.called


def test_server_can_replace_failed_workers(tmp_path, monkeypatch):
    module = workspace()
    instances = []

    def create(config):
        value = MagicMock()
        instances.append(value)
        return value

    monkeypatch.setattr(module, "WarmRuntime", create)
    server = module.WorkspaceServer(0, {"work_root": str(tmp_path), "warm_workers": True})
    first = server.warm
    server.reset_warm()
    assert first.close.called and server.warm is None
    assert server.ensure_warm() is instances[1] and len(instances) == 2
    server.server_close()


def test_unisign_worker_receives_native_pose_and_reports_identity(tmp_path, monkeypatch):
    module = workspace()
    workers = []

    class Worker:
        def __init__(self, command, **kwargs):
            self.command, self.calls = command, []
            workers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, item):
            self.calls.append(item)
            if "video" in item:
                return {"extraction_seconds": 0.1}
            return {
                "english": "hello",
                "identity": {"checkpoint_sha256": "a" * 64, "source_revision": "rev"},
            }

    monkeypatch.setattr(module, "PersistentWorker", Worker)
    config = dict.fromkeys(
        ("python", "extraction_python", "checkpoint", *module.ASSET_NAMES), "fixture"
    )
    config.update(
        work_root=str(tmp_path),
        model="unisign",
        unisign_source="source",
        unisign_tokenizer="tokenizer",
    )
    runtime = module.WarmRuntime(config)
    try:
        result = runtime.infer(tmp_path, tmp_path / "video", tmp_path / "overlay")
        assert "asl_recognizer.translation.unisign" in workers[1].command
        assert "--source" in workers[1].command and "--tokenizer" in workers[1].command
        assert workers[0].calls[0]["native_pose"] is True
        assert result["model_id"] == "uni-sign" and result["model_version"] == "rev"
        assert result["checkpoint_sha256"] == "a" * 64
    finally:
        runtime.close()


def test_unisign_dependencies_only_apply_to_translator(tmp_path, monkeypatch):
    import os

    module = workspace()
    workers = []

    class Worker:
        def __init__(self, command, **kwargs):
            self.env = kwargs["env"]
            workers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(module, "PersistentWorker", Worker)
    config = dict.fromkeys(
        ("python", "extraction_python", "checkpoint", *module.ASSET_NAMES), "fixture"
    )
    paths = [str(tmp_path / "deps"), str(tmp_path / "protobuf")]
    config.update(work_root=str(tmp_path), model="unisign", unisign_source="source",
                  unisign_tokenizer="tokenizer", unisign_pythonpath=paths)
    runtime = module.WarmRuntime(config)
    try:
        assert workers[1].env["PYTHONPATH"].split(os.pathsep)[1:] == paths
        assert workers[0].env["PYTHONPATH"].split(os.pathsep)[1:] == []
    finally:
        runtime.close()


def test_unisign_rejects_invalid_dependency_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    module = workspace()
    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    config = tmp_path / "config.json"
    for value in ("/tmp", ["relative"], [str(tmp_path / "missing")]):
        config.write_text(json.dumps({"model": "unisign", "unisign_pythonpath": value}))
        with pytest.raises(ValueError, match="dependency paths"):
            module.load_runtime(config)
