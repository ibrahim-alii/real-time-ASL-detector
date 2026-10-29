import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workspace", ROOT / "scripts/workspace_server.py")
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


class IntegrationTest(unittest.TestCase):
    def test_shubert_routes_video_and_closes_workers(self):
        workers = []

        class Worker:
            def __init__(self, command, **kwargs):
                self.command, self.calls, self.closed = command, [], False
                workers.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def request(self, item):
                self.calls.append(item)
                if "output" in item:
                    return {"extraction_seconds": 0.1}
                return {"english": "hi", "model_id": "SHuBERT", "model_version": "v",
                        "checkpoint_sha256": "a" * 64}

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root = Path(name).resolve()
            config = dict.fromkeys(("python", "extraction_python", *workspace.ASSET_NAMES), "x")
            config.update(work_root=str(root), model="shubert", shubert_model_root="model")
            with patch.object(workspace, "PersistentWorker", Worker):
                runtime = workspace.WarmRuntime(config)
                try:
                    result = runtime.infer(root, root / "video", root / "overlay")
                    self.assertEqual(workers[1].calls, [{"video": str(root / "video")}])
                    self.assertFalse(workers[0].calls[0]["native_pose"])
                    self.assertEqual(result["model_id"], "SHuBERT")
                    self.assertIn("shubert_worker.py", str(workers[1].command))
                    self.assertIn("--model-root", workers[1].command)
                finally:
                    runtime.close()
            self.assertTrue(all(worker.closed for worker in workers))

    def test_shubert_requires_warm_mode_and_model_root(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            root = Path(name).resolve()
            config = dict.fromkeys(("python", "extraction_python", *workspace.ASSET_NAMES),
                                   str(root / "file"))
            (root / "file").touch()
            config.update(work_root=str(root), model="shubert", shubert_model_root=str(root))
            path = root / "config.json"
            with patch.object(workspace.sys, "platform", "linux"), \
                    patch.dict(os.environ, {"SLURM_JOB_ID": "1"}), \
                    patch.object(workspace.shutil, "which", return_value="/bin/ffmpeg"), \
                    patch.object(Path, "is_relative_to", return_value=True):
                path.write_text(json.dumps(config))
                self.assertEqual(workspace.load_runtime(path)["model"], "shubert")
                config["warm_workers"] = False
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    workspace.load_runtime(path)


if __name__ == "__main__":
    unittest.main()
