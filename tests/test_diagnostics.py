from types import SimpleNamespace

import numpy as np
import pytest
import torch

from asl_recognizer.translation.contracts import DirectTranslationResult, TranslationRequest


class Tokenizer:
    pad_token_id = 9

    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.tensor([[0 if text == "zero" else 1] for text in texts])}


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(input_encoding="sentinel-v1")
        self.input_projection = torch.nn.Linear(266, 2, bias=False)
        self.t5 = torch.nn.Module()
        self.t5.encoder = torch.nn.Linear(2, 2, bias=False)
        self.t5.decoder = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.input_projection.weight.zero_()
            self.input_projection.weight[0, 0] = 1
            self.input_projection.weight[1, 0] = -1
            self.t5.encoder.weight.copy_(torch.eye(2))
            self.t5.decoder.weight.copy_(torch.eye(2))

    def forward(self, values, attention, labels):
        logits = self.t5.decoder(self.t5.encoder(self.input_projection(values))).mean(1)
        return torch.nn.functional.cross_entropy(logits, labels[:, 0])

    def translate(self, request, tokenizer, **kwargs):
        return DirectTranslationResult(
            "zero" if request.values[0, 0] > 0 else "one", None, "tiny", "v1", 1.0
        )


def examples():
    return [
        (
            TranslationRequest(np.full((3, 266), value, np.float32), np.ones((3, 133), np.bool_)),
            caption,
        )
        for value, caption in ((1, "zero"), (-1, "one"))
    ]


def test_conditioning_reports_gradients_controls_and_restores_model():
    from asl_recognizer.translation.diagnostics import conditioning_report

    model = Model()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    report = conditioning_report(model, Tokenizer(), examples(), device="cpu")
    assert (
        report["controls"]["real"]["target_nll"]
        < report["controls"]["permutation-17"]["target_nll"]
    )
    assert report["gradient_norms"]["input_projection"] > 0
    assert report["gradient_norms"]["encoder"] > 0
    assert report["gradient_norms"]["decoder"] > 0
    assert report["projection_activation"]["std"] > 0
    assert report["semantic_review"] == "pending" and report["test_examples_used"] == 0
    assert model.training and all(param.grad is None for param in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_derangements_and_size_guards():
    from asl_recognizer.translation.diagnostics import conditioning_report, derangement

    order = derangement(8, 42)
    assert sorted(order) == list(range(8)) and all(i != value for i, value in enumerate(order))
    assert derangement(8, 42) == order
    with pytest.raises(ValueError):
        conditioning_report(Model(), Tokenizer(), examples()[:1], device="cpu")


def test_cli_profiles_validated_dev_without_test_features(tmp_path, monkeypatch):
    import json

    from asl_recognizer.translation import diagnostics
    from asl_recognizer.translation.cache import TranslationFeatureCache
    from asl_recognizer.translation.manifest import ClipManifest, ManifestRecord, write_manifest

    rows = tuple(
        ManifestRecord(str(i), "how2sign", "dev", str(i), 0, 100, caption)
        for i, (_, caption) in enumerate(examples())
    )
    manifest = ClipManifest(rows)
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    cache = TranslationFeatureCache(tmp_path / "features/how2sign")
    for row, (request, _) in zip(rows, examples(), strict=True):
        cache.store(
            row.clip_id,
            request,
            metadata={
                "manifest_hash": manifest.fingerprint,
                "caption": row.caption,
                "split": "dev",
                "source": "how2sign",
                "extractor": {"tiny": True},
            },
        )
    engine = SimpleNamespace(
        model=Model(),
        tokenizer=Tokenizer(),
        device="cpu",
        extractor={"tiny": True},
        preprocessing=None,
        config=SimpleNamespace(input_encoding="sentinel-v1"),
        checkpoint_sha256="a" * 64,
    )
    monkeypatch.setattr(diagnostics, "require_sol", lambda root: None)
    monkeypatch.setattr(diagnostics, "TranslationEngine", lambda path: engine)
    monkeypatch.setattr(
        diagnostics.torch,
        "load",
        lambda *a, **k: {
            "manifest_hashes": {"how2sign": manifest.fingerprint},
            "global_step": 3,
            "config": {},
        },
    )
    output = tmp_path / "diagnostic.json"
    diagnostics.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--manifest",
            str(path),
            "--features",
            str(tmp_path / "features"),
            "--work-root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )
    value = json.loads(output.read_text())
    assert value["checkpoint_step"] == 3 and value["input_profile"]["iterations"] == 20
    assert value["parameter_updates"] == 0 and value["test_examples_used"] == 0
    assert value["preprocessing_status"] == "legacy-unversioned"


def test_nonfinite_budgets_are_rejected():
    from asl_recognizer.translation.diagnostics import conditioning_report

    with pytest.raises(ValueError):
        conditioning_report(
            Model(), Tokenizer(), examples(), device="cpu", wall_time_seconds=float("nan")
        )


def test_derangement_fallback_and_timeout_restore(monkeypatch):
    from asl_recognizer.translation import diagnostics

    monkeypatch.setattr(
        diagnostics.np.random,
        "default_rng",
        lambda seed: SimpleNamespace(permutation=lambda n: np.arange(n)),
    )
    assert diagnostics.derangement(3, 1) == [1, 2, 0]
    with pytest.raises(ValueError):
        diagnostics.derangement(1, 1)
    times = iter([0, 10000])
    monkeypatch.setattr(diagnostics, "time", SimpleNamespace(monotonic=lambda: next(times)))
    model = Model()
    with pytest.raises(TimeoutError):
        diagnostics.conditioning_report(model, Tokenizer(), examples(), device="cpu")
    assert model.training and all(param.grad is None for param in model.parameters())
