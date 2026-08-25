from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import cast

from asl_recognizer.app.webcam import WebcamRuntime
from asl_recognizer.cli import main
from asl_recognizer.data.index import manifest_fingerprint, write_manifest, write_vocabulary
from asl_recognizer.data.schema import SampleRecord, Split
from asl_recognizer.features.cache import FeatureCache
from asl_recognizer.features.normalize import PREPROCESSING_VERSION
from asl_recognizer.features.schema import (
    FEATURE_DIMENSION,
    LANDMARK_COUNT,
    FeatureSequence,
    LandmarkFrame,
)
from asl_recognizer.models.checkpoint import load_bundle, preprocessing_fingerprint
from asl_recognizer.models.gru import GRUConfig, LandmarkGRU
from asl_recognizer.models.predictor import GRUPredictor
from asl_recognizer.streaming import StreamingConfig, StreamingRecognizer


def _landmarks(timestamp_ms: int) -> LandmarkFrame:
    coordinates = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    presence = np.zeros(LANDMARK_COUNT, dtype=np.bool_)
    coordinates[0] = [0.5, 0.4, 0.0]
    coordinates[43] = [0.4, 0.5, 0.0]
    coordinates[44] = [0.6, 0.5, 0.0]
    presence[[0, 43, 44]] = True
    return LandmarkFrame(coordinates, presence, timestamp_ms)


class _Capture:
    def __init__(self) -> None:
        self.frames = iter([np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(3)])
        self.released = False

    def is_opened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self.released = True


class _Display:
    def __init__(self) -> None:
        self.keys = iter([32, -1, 27])
        self.closed = False

    def show(self, _frame: object, _status: object, _landmarks: object) -> None:
        return None

    def poll_key(self, _delay_ms: int) -> int:
        return next(self.keys)

    def is_open(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


class _Detector:
    def detect(self, _frame: object, timestamp_ms: int) -> LandmarkFrame:
        return _landmarks(timestamp_ms)


class _Clock:
    def __init__(self) -> None:
        self.value = 1.0

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def test_landmarks_to_gru_to_streaming_to_webcam_runtime() -> None:
    model = LandmarkGRU(GRUConfig(num_classes=2, hidden_size=4, dropout=0.0))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[:] = torch.tensor([8.0, -8.0])
    predictor = GRUPredictor(model, ("HELLO", "THANKS"))
    recognizer = StreamingRecognizer(
        predictor,
        StreamingConfig(
            inference_stride=1,
            stable_windows=1,
            minimum_frames=1,
            minimum_hand_rate=0.0,
            minimum_motion=0.0,
            cooldown_ms=0,
        ),
    )
    capture = _Capture()
    display = _Display()

    status = WebcamRuntime(
        capture,
        display,
        _Detector(),
        recognizer,
        clock=_Clock(),
        process_clock=_Clock(),
    ).run()

    assert status.history == ("HELLO",)
    assert capture.released and display.closed


def _feature_sequence(value: float) -> FeatureSequence:
    return FeatureSequence(
        values=np.full((32, FEATURE_DIMENSION), value, dtype=np.float32),
        landmark_mask=np.ones((32, LANDMARK_COUNT), dtype=np.bool_),
        temporal_mask=np.ones(32, dtype=np.bool_),
        timestamps_ms=np.arange(32, dtype=np.int64),
    )


def test_train_cli_produces_reloadable_runtime_bundle(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    cache_root = tmp_path / "cache"
    output = tmp_path / "model"
    video_root.mkdir()
    records: list[SampleRecord] = []
    identity: dict[str, object] = {
        "holistic_model_sha256": "a" * 64,
        "sequence_length": 32,
        "preprocessing_version": PREPROCESSING_VERSION,
        "minimum_decode_rate": 0.95,
    }
    cache = FeatureCache(cache_root)
    for split_index, split in enumerate(("train", "val", "test")):
        for class_id, gloss in enumerate(("HELLO", "THANKS")):
            video_id = f"{split}-{gloss}"
            path = video_root / f"{video_id}.mp4"
            path.touch()
            records.append(
                SampleRecord(
                    video_id,
                    path,
                    gloss,
                    f"signer-{split_index}-{class_id}",
                    cast(Split, split),
                    class_id,
                )
            )
            cache.store(
                f"{split_index * 2 + class_id + 1:064x}",
                _feature_sequence(float(class_id * 2 - 1)),
                metadata={
                    "video_id": video_id,
                    "preprocessing_identity": identity,
                },
            )
    fingerprint = manifest_fingerprint(records)
    manifest = tmp_path / "manifest.csv"
    vocabulary = tmp_path / "vocabulary.json"
    provenance = {"train.csv": "train", "val.csv": "val", "test.csv": "test"}
    write_manifest(records, manifest, video_root=video_root, source_hashes=provenance)
    write_vocabulary(
        ("HELLO", "THANKS"),
        vocabulary,
        dataset_version="test",
        manifest_fingerprint_value=fingerprint,
        source_hashes=provenance,
    )

    exit_code = main(
        [
            "train",
            "--manifest",
            str(manifest),
            "--vocabulary",
            str(vocabulary),
            "--video-root",
            str(video_root),
            "--cache-root",
            str(cache_root),
            "--output-dir",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
        ]
    )

    assert exit_code == 0
    model, metadata = load_bundle(
        output / "model.pt",
        expected_labels=("HELLO", "THANKS"),
        expected_preprocessing_hash=preprocessing_fingerprint(identity),
        expected_dataset_hash=fingerprint,
    )
    assert model.config.num_classes == 2
    assert metadata.dataset_hash == fingerprint
    assert (output / "training-report.json").is_file()
