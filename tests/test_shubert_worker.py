import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.shubert_worker import (
    check_gpu,
    load_checked,
    make_handler,
    probe_video,
    safe_path,
    serve,
    validate_features,
    verify_asset,
    verify_assets,
)


@patch("scripts.shubert_worker.probe_video")
class WorkerContractTest(unittest.TestCase):
    def test_video_request_and_rejection_keep_worker_alive(self, _probe):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root = Path(name).resolve()
            video = root / 'clip.mp4'
            video.write_bytes(b'synthetic')
            engine = Mock()
            engine.translate.return_value = 'hello'
            request = '\n'.join(json.dumps(row) for row in [
                {'id': '1', 'video': str(video)},
                {'id': '2', 'video': str(root.parent / 'outside.mp4')},
                {'id': '3', 'video': str(video)},
            ]) + '\n'
            out = io.StringIO()
            serve(make_handler(root, engine, 'a' * 64), stdin=io.StringIO(request), stdout=out)
            rows = [json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual(rows[0], {'ready': True, 'protocol': 1})
            self.assertEqual(rows[1]['result']['english'], 'hello')
            self.assertEqual(rows[1]['result']['checkpoint_sha256'], 'a' * 64)
            self.assertFalse(rows[2]['ok'])
            self.assertTrue(rows[3]['ok'])
            self.assertEqual(engine.translate.call_count, 2)

    def test_missing_video_and_invalid_output_rejected(self, _probe):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root = Path(name).resolve()
            engine = Mock()
            handler = make_handler(root, engine, 'a' * 64)
            with self.assertRaises(ValueError):
                handler({'video': str(root / 'missing')})
            video = root / 'clip.mp4'
            video.write_bytes(b'synthetic')
            engine.translate.return_value = None
            with self.assertRaises(ValueError):
                handler({'video': str(video)})
            with self.assertRaises(ValueError):
                make_handler(root, engine, 'bad digest')

    def test_rejects_wrong_manifest_before_reading_assets(self, _probe):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root = Path(name)
            (root / "source-and-asset-manifest.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify_assets(root)

    def test_gpu_checks_before_memory_cap(self, _probe):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        with self.assertRaises(RuntimeError):
            check_gpu(torch)
        torch.cuda.set_per_process_memory_fraction.assert_not_called()
        torch.cuda.is_available.return_value = True
        torch.cuda.mem_get_info.return_value = (74, 100)
        with self.assertRaises(RuntimeError):
            check_gpu(torch)
        torch.cuda.set_per_process_memory_fraction.assert_not_called()
        torch.cuda.mem_get_info.return_value = (75, 100)
        check_gpu(torch)
        torch.cuda.set_per_process_memory_fraction.assert_called_once_with(0.75)

    def test_engine_error_does_not_leak_details(self, _probe):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root = Path(name).resolve()
            video = root / "clip.mp4"
            video.write_bytes(b"synthetic")
            engine = Mock()
            engine.translate.side_effect = RuntimeError("private data")
            out = io.StringIO()
            serve(make_handler(root, engine, "a" * 64),
                  stdin=io.StringIO(json.dumps({"id": "1", "video": str(video)}) + "\n"),
                  stdout=out)
            self.assertEqual(json.loads(out.getvalue().splitlines()[1]),
                             {"id": "1", "ok": False, "error": "RuntimeError"})

    def test_probe_enforces_decoding_limits(self, _probe):
        with patch("scripts.shubert_worker.subprocess.check_output") as command:
            command.return_value = json.dumps({"streams": [{"width": 640, "height": 480,
                "nb_read_frames": "600", "duration": "20"}]}).encode()
            probe_video(Path("clip.mp4"))
            for key, value in (("nb_read_frames", "601"), ("width", 20000),
                               ("duration", "21")):
                row = {"width": 640, "height": 480, "nb_read_frames": "600", "duration": "20"}
                row[key] = value
                command.return_value = json.dumps({"streams": [row]}).encode()
                with self.assertRaises(ValueError):
                    probe_video(Path("clip.mp4"))

    def test_probe_timeout_is_sanitized(self, _probe):
        with patch("scripts.shubert_worker.subprocess.check_output") as command:
            command.side_effect = subprocess.TimeoutExpired("ffprobe", 30)
            with self.assertRaisesRegex(ValueError, "Video probe failed"):
                probe_video(Path("clip.mp4"))

    def test_probe_rejects_nan_duration(self, _probe):
        with patch("scripts.shubert_worker.subprocess.check_output") as command:
            command.return_value = json.dumps({"streams": [{"width": 640, "height": 480,
                "nb_read_frames": "1", "duration": "nan"}]}).encode()
            with self.assertRaises(ValueError):
                probe_video(Path("clip.mp4"))

    def test_protocol_rejects_invalid_lines_without_exposing_details(self, _probe):
        requests = ["{}\n", json.dumps({"id": "a", "video": "x" * 65536}) + "\n",
                    json.dumps({"id": "b"}) + "\n"]
        out = io.StringIO()
        serve(lambda _: {"english": "ok"}, stdin=io.StringIO("".join(requests)), stdout=out)
        replies = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertFalse(replies[1]["ok"])
        self.assertFalse(replies[2]["ok"])
        self.assertTrue(replies[-1]["ok"])

    def test_asset_identity_and_path_guard(self, _probe):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root = Path(name).resolve()
            file = root / "sample"
            file.write_bytes(b"abc")
            verify_asset(file, {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()})
            with self.assertRaises(ValueError):
                verify_asset(file, {"bytes": 3, "sha256": "0" * 64})
            with self.assertRaises(ValueError):
                safe_path(root, "../escape")

    def test_loading_and_feature_validation_fail_closed(self, _probe):
        import numpy as np

        model_class = Mock()
        model_class.from_pretrained.return_value = (Mock(), {"missing_keys": ["x"],
            "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})
        with self.assertRaises(ValueError):
            load_checked(model_class, "checkpoint")
        arrays = [np.zeros((2, n)) for n in (384, 384, 384, 14)]
        validate_features(*arrays)
        arrays[0][0, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_features(*arrays)


if __name__ == '__main__':
    unittest.main()
