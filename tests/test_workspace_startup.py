import importlib.util
from pathlib import Path


def test_workspace_allows_cold_translation_startup_without_extending_requests(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "startup_workspace", Path("scripts/workspace_server.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    workers = []

    class Worker:
        def __init__(self, command, **kwargs):
            self.startup_timeout = kwargs.get("timeout", 300)
            self.timeout = self.startup_timeout
            workers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(module, "PersistentWorker", Worker)
    config = dict.fromkeys(
        ("python", "extraction_python", "checkpoint", *module.ASSET_NAMES), "fixture"
    )
    config.update(work_root=str(tmp_path), model="shubert", shubert_model_root="fixture")
    runtime = module.WarmRuntime(config)
    try:
        assert workers[1].startup_timeout == 600
        assert workers[0].timeout == workers[1].timeout == 300
    finally:
        runtime.close()
