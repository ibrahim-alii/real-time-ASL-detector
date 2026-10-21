import numpy as np
import pytest

from asl_recognizer.features.mmpose_backend import WholeBodyFrame


def raw_frames(count=5):
    coordinates = np.arange(266, dtype=np.float32).reshape(133, 2)
    return [
        WholeBodyFrame(
            coordinates.copy(),
            np.ones(133, np.bool_),
            time * 10,
            np.full(133, 0.8, np.float32),
            coordinates.copy(),
        )
        for time in range(count)
    ]


def test_preprocessing_keeps_scores_and_native_sampling():
    from asl_recognizer.translation.unisign import prepare_input

    calls = []

    def official(skeletons, scores, force_ok):
        import torch

        calls.append((skeletons, scores, force_ok))
        return {
            part: torch.zeros((len(scores), joints, 3))
            for part, joints in [("body", 9), ("left", 21), ("right", 21), ("face_all", 18)]
        }

    batch, receipt = prepare_input(
        raw_frames(8), width=200, height=100, preprocess=official, max_length=4, seed=42
    )
    assert batch["body"].shape == (1, 4, 9, 3)
    assert batch["attention_mask"].tolist() == [[1] * 4]
    assert receipt["indices"] == sorted(receipt["indices"])
    assert len(set(receipt["indices"])) == 4
    assert calls[0][0][0].shape == (1, 133, 2)
    np.testing.assert_allclose(calls[0][0][0][0, 0], [0, 0.01])
    np.testing.assert_allclose(calls[0][1][0], 0.8)
    assert calls[0][2] is True
    assert (
        prepare_input(
            raw_frames(8), width=200, height=100, preprocess=official, max_length=4, seed=42
        )[1]
        == receipt
    )


def test_preprocessing_rejects_legacy_thresholded_pose():
    from asl_recognizer.translation.unisign import prepare_input

    frames = [WholeBodyFrame(np.ones((133, 2), np.float32), np.ones(133, np.bool_), 0)]
    with pytest.raises(ValueError, match="raw"):
        prepare_input(frames, width=200, height=100, preprocess=lambda *a: None)


def test_unisign_adapter_never_passes_references():
    from unittest.mock import MagicMock

    import torch

    from asl_recognizer.translation.unisign import generate_text

    model = MagicMock()
    model.return_value = {"inputs_embeds": torch.zeros(1, 2, 4)}
    model.generate.return_value = torch.tensor([[1, 2]])
    model.mt5_tokenizer.batch_decode.return_value = [" result "]
    english = generate_text(
        model, {"attention_mask": torch.ones(1, 2)}, beam_width=5, max_tokens=128
    )
    assert english == "result"
    assert model.call_args.args[1] == {"gt_sentence": [""]}
    assert model.generate.call_args.kwargs == {"num_beams": 5, "max_new_tokens": 128}


def test_source_pin_and_weight_hash_are_mandatory(tmp_path):
    from asl_recognizer.translation.unisign import verify_checkpoint, verify_source

    with pytest.raises(ValueError, match="source"):
        verify_source(tmp_path)
    weight = tmp_path / "weights"
    weight.write_bytes(b"tiny")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(weight)


def test_engine_loads_strict_weights_and_runs_preprocessor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import torch
    import transformers

    from asl_recognizer.translation import unisign

    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    module = SimpleNamespace(Uni_Sign=lambda args: model)
    monkeypatch.setattr(unisign, "verify_source", lambda root: None)
    monkeypatch.setattr(unisign, "verify_checkpoint", lambda path: None)
    monkeypatch.setattr(
        unisign,
        "official_preprocessor",
        lambda root: lambda *a, **k: {"body": torch.zeros(3, 9, 3)},
    )
    monkeypatch.setattr(
        unisign,
        "importlib",
        SimpleNamespace(
            util=SimpleNamespace(
                spec_from_file_location=lambda *a: SimpleNamespace(
                    loader=SimpleNamespace(exec_module=lambda module: None)
                ),
                module_from_spec=lambda spec: module,
            )
        ),
    )
    monkeypatch.setattr(transformers.MT5Config, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"model": {"weight": "verified"}})
    (tmp_path / "config.json").write_text("{}")
    engine = unisign.UniSignEngine(tmp_path, tmp_path / "checkpoint", tmp_path, device="cpu")
    model.load_state_dict.assert_called_once_with({"weight": "verified"}, strict=True)
    monkeypatch.setattr(unisign, "generate_text", lambda *a, **k: "result")
    result = engine.translate(
        raw_frames(3), {"raw_pose_sampling": "every-native-frame-v1", "width": 200, "height": 100}
    )
    assert result["english"] == "result" and result["input"]["native_frames"] == 3
    assert result["identity"]["strict_state_dict"] is True
    with pytest.raises(ValueError, match="native"):
        engine.translate(raw_frames(3), {})


def test_official_functions_loaded_without_training_imports(tmp_path, monkeypatch):
    from asl_recognizer.translation import unisign

    monkeypatch.setattr(unisign, "verify_source", lambda root: None)
    (tmp_path / "datasets.py").write_text(
        "import missing_training_dependency\ndef crop_scale(x):\n return x+1\n"
        "def load_part_kp(x):\n return crop_scale(x)\n"
    )
    assert unisign.official_preprocessor(tmp_path)(2) == 3
    (tmp_path / "datasets.py").write_text("def unrelated():pass")
    with pytest.raises(ValueError, match="functions"):
        unisign.official_preprocessor(tmp_path)


def test_source_verification_rejects_dirty_checkout(tmp_path, monkeypatch):
    from asl_recognizer.translation import unisign

    answers = iter([unisign.SOURCE_REVISION, "", unisign.SOURCE_REVISION, " M models.py"])
    monkeypatch.setattr(unisign.subprocess, "check_output", lambda *a, **k: next(answers))
    unisign.verify_source(tmp_path)
    with pytest.raises(ValueError, match="tracked"):
        unisign.verify_source(tmp_path)
    monkeypatch.setattr(unisign, "file_hash", lambda path: unisign.WEIGHTS_SHA256)
    unisign.verify_checkpoint(tmp_path / "weight")


def test_cli_keeps_library_output_off_protocol(tmp_path, monkeypatch):
    import io
    import json
    import sys
    from unittest.mock import MagicMock

    from asl_recognizer.translation import unisign

    monkeypatch.setattr(unisign, "require_sol", lambda root: None)

    def constructor(*args, **kwargs):
        print("library initialization")
        engine = MagicMock()
        engine.translate.return_value = {"english": "result"}
        return engine

    monkeypatch.setattr(unisign, "UniSignEngine", constructor)
    monkeypatch.setattr(unisign.PoseCache, "load", lambda *a: ([], {}))
    incoming = io.StringIO(json.dumps({"id": "one", "features": str(tmp_path / "features")}) + "\n")
    outgoing = io.StringIO()
    monkeypatch.setattr(sys, "stdin", incoming)
    monkeypatch.setattr(sys, "stdout", outgoing)
    unisign.main(
        [
            "--source",
            str(tmp_path),
            "--checkpoint",
            str(tmp_path / "weights"),
            "--tokenizer",
            str(tmp_path),
            "--work-root",
            str(tmp_path),
        ]
    )
    rows = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert rows[0]["ready"] is True and rows[1]["result"]["english"] == "result"
