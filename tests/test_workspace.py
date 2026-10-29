"""Synthetic workspace contracts: no camera, network service, or model assets."""

from __future__ import annotations

import http.client
import importlib.util
import json
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from asl_recognizer.features.mmpose_backend import WholeBodyFrame
from asl_recognizer.translation.local_runtime import write_overlay

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "workspace_server", ROOT / "scripts/workspace_server.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def server(workspace: ModuleType) -> Iterator[Any]:
    instance = workspace.WorkspaceServer(0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=2)


def request(
    server: Any, method: str, path: str, *, body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    with closing(http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)) as client:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        return response.status, dict(response.getheaders()), response.read()


def test_preview_serves_ui_and_refuses_inference(server: Any) -> None:
    code, headers, body = request(server, "GET", "/")
    assert code == 200 and b"Camera &amp; landmarks" in body
    assert headers["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    code, _, body = request(server, "GET", "/api/status")
    assert code == 200 and json.loads(body)["ready"] is False
    code, _, _ = request(server, "POST", "/api/translate", body=b"synthetic",
                         headers={"X-ASL-Request": "clip"})
    assert code == 503
    assert request(server, "GET", "/../config/sol_translation.json")[0] == 404


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example"}, {"Origin": "https://evil.example"},
])
def test_host_and_origin_guards(server: Any, headers: dict[str, str]) -> None:
    assert request(server, "GET", "/api/status", headers=headers)[0] == 403


def test_inference_boundaries_and_single_writer(server: Any, workspace: ModuleType,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    server.runtime = {"synthetic": True}
    headers = {"X-ASL-Request": "clip", "Content-Type": "video/webm"}
    assert request(server, "POST", "/api/translate", body=b"test")[0] == 403
    assert request(server, "POST", "/api/translate", headers=headers)[0] == 400
    oversized = {**headers, "Content-Length": str(workspace.MAX_UPLOAD + 1)}
    assert request(server, "POST", "/api/translate", headers=oversized)[0] == 400
    wrong_type = {**headers, "Content-Type": "application/json"}
    assert request(server, "POST", "/api/translate", body=b"test", headers=wrong_type)[0] == 400
    server.inference_lock.acquire()
    try:
        assert request(server, "POST", "/api/translate", body=b"test", headers=headers)[0] == 409
    finally:
        server.inference_lock.release()
    monkeypatch.setattr(workspace, "translate_clip", lambda *_: {"english": "Synthetic result"})
    code, _, body = request(server, "POST", "/api/translate", body=b"test", headers=headers)
    assert code == 200 and json.loads(body)["english"] == "Synthetic result"
    assert server.inference_lock.acquire(timeout=2)
    server.inference_lock.release()


def test_wrong_content_type_drains_bounded_body(workspace: ModuleType) -> None:
    from io import BytesIO
    from types import SimpleNamespace

    handler = object.__new__(workspace.WorkspaceHandler)
    handler.connection = MagicMock()
    handler.server = SimpleNamespace(runtime={"synthetic": True})
    handler.trusted = lambda: True
    handler.path = "/api/translate"
    handler.headers = {"X-ASL-Request": "clip", "Content-Length": "4",
                       "Content-Type": "application/json"}
    handler.rfile = BytesIO(b"test")
    handler.reply = MagicMock()
    handler.do_POST()
    assert handler.rfile.tell() == 4
    assert handler.reply.call_args.args[0] == 400


def test_backend_failures_release_writer(server: Any, workspace: ModuleType,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    server.runtime = {"synthetic": True}

    def fail(*_: object) -> None:
        raise subprocess.CalledProcessError(1, ["secret-remote-path"])

    monkeypatch.setattr(workspace, "translate_clip", fail)
    code, _, body = request(server, "POST", "/api/translate", body=b"test",
                            headers={"X-ASL-Request": "clip", "Content-Type": "video/webm"})
    assert code == 502 and b"secret-remote-path" not in body
    assert server.inference_lock.acquire(timeout=2)
    server.inference_lock.release()


def test_runtime_requires_slurm(workspace: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    assert workspace.load_runtime(None) is None
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm"):
        workspace.load_runtime(Path("does-not-exist.json"))


def test_workspace_rejects_legacy_model_and_cold_inference(
    workspace: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(workspace.sys, "platform", "linux")
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps({"model": "landmarkt5"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported runtime model"):
        workspace.load_runtime(config)
    with pytest.raises(ValueError, match="requires warm workers"):
        workspace.translate_clip(b"synthetic", {"work_root": str(tmp_path)})


def test_overlay_sidecar_preserves_pixel_coordinates_and_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("cv2.VideoWriter", MagicMock())
    image = np.zeros((24, 32, 3), np.uint8)
    coordinates = np.zeros((133, 2), np.float32)
    coordinates[91] = [12, 15]
    presence = np.zeros(133, np.bool_)
    presence[91] = True
    frames = [(image, WholeBodyFrame(coordinates, presence, timestamp)) for timestamp in (0, 67)]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "overlay.mp4"
        write_overlay(Path("synthetic.webm"), frames, output)
        value = json.loads(output.with_suffix(".json").read_text())
        assert (value["width"], value["height"]) == (32, 24)
        assert [frame["timestamp_ms"] for frame in value["frames"]] == [0, 67]
        assert value["frames"][0]["coordinates"][91] == [12, 15]
        assert value["frames"][0]["presence"][91] is True
        assert value["frames"][0]["presence"][90] is False
        assert not image.any()


def test_translation_returns_display_data_and_cleans_session(
    workspace: ModuleType, monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = MagicMock()
    capture.get.return_value = 30
    monkeypatch.setattr("cv2.VideoCapture", lambda *_: capture)
    monkeypatch.setattr(workspace.shutil, "disk_usage", lambda _: MagicMock(free=3 * 1024**3))
    transcodes: list[list[str]] = []
    monkeypatch.setattr(
        workspace.subprocess, "run", lambda command, **_: transcodes.append(command)
    )
    warm = MagicMock()

    def infer(root: Path, video: Path, overlay: Path) -> dict[str, object]:
        overlay.with_suffix(".json").write_text(json.dumps({
            "width": 32, "height": 24, "frames": [], "source": "private-path"
        }))
        return {"english": "Synthetic result", "model_id": "fixture", "model_version": "v1",
                "checkpoint_sha256": "a" * 64, "quality_status": "unverified",
                "timings": {"warm_workers": True}}

    warm.infer.side_effect = infer
    with tempfile.TemporaryDirectory() as temporary:
        config = {"work_root": temporary, "model": "shubert"}
        result = workspace.translate_clip(b"synthetic", config, warm)
        assert result["english"] == "Synthetic result"
        assert "private-path" not in json.dumps(result)
        assert not list(Path(temporary).iterdir())
        assert warm.infer.call_count == 1
        filters = transcodes[0][transcodes[0].index("-vf") + 1]
        assert filters == "scale=640:480:force_original_aspect_ratio=decrease:force_divisible_by=2"
        assert transcodes[0][transcodes[0].index("-pix_fmt") + 1] == "yuv420p"


def test_frontend_result_validation_and_syntax() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node required for browser script contract check")
    page = (ROOT / "scripts/asl_workspace.html").read_text(encoding="utf-8")
    assert "A clear view helps." not in page
    assert "Enable your camera to record a new clip." not in page
    script = page.split("<script>", 1)[1].split("</script>", 1)[0]
    # Execute the actual validator with a tiny DOM; no browser camera or model is touched.
    prelude = """
const assert = require('node:assert/strict');
const elements = new Map();
global.document = {getElementById(id) { if(!elements.has(id)) elements.set(id, {
 getContext:()=>({clearRect(){}}), addEventListener(){}, setAttribute(){},
 querySelector:()=>({textContent:''}), classList:{toggle(){},remove(){}}, checked:true
 }); return elements.get(id); }};
global.window={addEventListener(){}};
global.requestAnimationFrame=()=>{};
global.fetch=async()=>({ok:true,json:async()=>({ready:false})});
"""
    checks = """
const result = {english:'Fixture',model_id:'test',model_version:'v1',latency_ms:1,
 landmarks:{width:640,height:480,frames:[{timestamp_ms:0,
 coordinates:Array.from({length:133},()=>[10,20]),presence:Array(133).fill(true)}]}};
assert.equal(validateResult(result).english,'Fixture');
const bad=structuredClone(result);bad.landmarks.frames[0].coordinates=[];
assert.throws(()=>validateResult(bad),/invalid landmark/);
const unordered=structuredClone(result);unordered.landmarks.frames.push(result.landmarks.frames[0]);
assert.throws(()=>validateResult(unordered),/invalid landmark/);
message('');assert.equal(document.getElementById('status-row').hidden,true);
message('Failure',true);assert.equal(document.getElementById('status-row').hidden,false);
"""
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "workspace.cjs"
        path.write_text(prelude + script + checks, encoding="utf-8")
        subprocess.run([node, "--check", str(path)], check=True, capture_output=True, timeout=10)
        subprocess.run([node, str(path)], check=True, capture_output=True, timeout=10)
