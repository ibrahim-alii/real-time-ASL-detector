from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from asl_recognizer.features.video import every_other_video_frame
from asl_recognizer.translation.cache import TranslationFeatureCache
from asl_recognizer.translation.contracts import DirectTranslationResult, TranslationRequest
from asl_recognizer.translation.evaluation import evaluate_translations
from asl_recognizer.translation.features import (
    TRANSLATION_FEATURE_SCHEMA,
)
from asl_recognizer.translation.how2sign import build_how2sign_manifest
from asl_recognizer.translation.manifest import build_manifest, read_manifest, write_manifest
from asl_recognizer.translation.worker import _video_index
from asl_recognizer.translation.worker import main as worker_main


def test_manifest_is_deterministic_and_retains_unavailable_clips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rows = [
        {"id": "good", "video_id": "a", "start": 1, "end": 2, "caption": "hello"},
        {
            "id": "gone",
            "video_id": "b",
            "start": 1,
            "end": 2,
            "caption": "bye",
            "availability": "unavailable",
        },
        {"id": "bad", "video_id": "c", "start": 1, "end": 1, "caption": "x"},
    ]
    manifest = build_manifest(rows, source="youtube-asl")
    assert [item.status for item in manifest.records] == ["accepted", "unavailable", "malformed"]
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    assert read_manifest(path).fingerprint == manifest.fingerprint


def test_manifest_uses_explicit_time_units_and_checks_digest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    manifest = build_manifest(
        [
            {"id": "seconds", "video_id": "v", "start": 12000, "end": 12001, "caption": "hello"},
            {
                "id": "milliseconds",
                "video_id": "w",
                "start_ms": 12000,
                "end_ms": 13000,
                "caption": "bye",
            },
        ],
        source="youtube-asl",
    )
    assert [item.start_ms for item in manifest.records] == [12000000, 12000]
    path = tmp_path / "manifest.csv"
    write_manifest(path, manifest)
    path.write_text(path.read_text(encoding="utf-8").replace("hello", "changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        read_manifest(path)


def test_youtube_caption_filters_are_audited() -> None:
    rows = [
        {"id": "long", "video_id": "a", "start_ms": 0, "end_ms": 1000, "caption": "x" * 301},
        {"id": "short", "video_id": "b", "start_ms": 0, "end_ms": 199, "caption": "x"},
        {"id": "duration", "video_id": "c", "start_ms": 0, "end_ms": 60001, "caption": "x"},
        {
            "id": "people",
            "video_id": "d",
            "start_ms": 0,
            "end_ms": 1000,
            "caption": "x",
            "person_count": 2,
        },
    ]
    manifest = build_manifest(rows, source="youtube-asl")
    assert [item.reason for item in manifest.records] == [
        "caption-over-300-characters",
        "caption-under-200ms",
        "caption-over-60s",
        "person-count-not-one",
    ]


def test_manifest_command_keeps_unverified_person_count_out_of_training(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "captions.csv"
    source.write_text(
        "clip_id,video_id,start_ms,end_ms,caption\nc1,v1,0,1000,hello\n", encoding="utf-8"
    )
    output = tmp_path / "manifest.csv"
    assert (
        worker_main(
            ["manifest", "--input", str(source), "--source", "youtube-asl", "--output", str(output)]
        )
        == 0
    )
    record = read_manifest(output).records[0]
    assert record.status == "rejected"
    assert record.reason == "person-count-unverified"


def test_manifest_records_invalid_person_count_without_aborting_import() -> None:
    manifest = build_manifest(
        [
            {
                "id": "bad-count",
                "video_id": "v1",
                "start_ms": 0,
                "end_ms": 1000,
                "caption": "hello",
                "person_count": "unknown",
            }
        ],
        source="youtube-asl",
        require_person_count=True,
    )
    assert manifest.records[0].status == "rejected"
    assert manifest.records[0].reason == "person-count-invalid"


def test_official_how2sign_tables_use_clip_relative_times(tmp_path) -> None:  # type: ignore[no-untyped-def]
    header = "VIDEO_ID\tSENTENCE_NAME\tSTART\tEND\tSENTENCE\n"
    train = tmp_path / "train.csv"
    dev = tmp_path / "dev.csv"
    test = tmp_path / "test.csv"
    train.write_text(header + "a\ta_1-rgb_front\t94.95\t106.38\thello\n"
                     + "b\tb_1-rgb_front\t0\t2\tbye\n", encoding="utf-8")
    dev.write_text(header + "c\tc_1-rgb_front\t1\t3\tdev\n", encoding="utf-8")
    test.write_text(header + "d\td_1-rgb_front\t4\t5\ttest\n", encoding="utf-8")
    manifest = build_how2sign_manifest(train=train, dev=dev, test=test)
    assert len(manifest.records) == 4
    assert (manifest.records[0].video_id, manifest.records[0].start_ms,
            manifest.records[0].end_ms) == ("a_1-rgb_front", 0, 11430)
    assert [item.split for item in manifest.records] == ["train", "train", "dev", "test"]
    pilot = build_how2sign_manifest(train=train, dev=dev, test=test, pilot_first_video=True)
    assert [item.clip_id for item in pilot.records] == ["train:a_1-rgb_front"]


def test_video_index_keeps_official_splits_apart_and_shards_by_source_video(tmp_path) -> None:  # type: ignore[no-untyped-def]
    train = tmp_path / "train" / "rgb_front"
    val = tmp_path / "val" / "rgb_front"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    (train / "clip.mp4").write_bytes(b"train")
    (val / "clip.mp4").write_bytes(b"dev")
    index = _video_index(tmp_path)
    assert index[("train", "clip")].read_bytes() == b"train"
    assert index[("dev", "clip")].read_bytes() == b"dev"








def test_native_half_frame_sampling_uses_caption_boundary(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class Capture:
        position = 0
        released = False
        seeks = 0

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> int:
            return {1: 10, 2: 10}[property_id]

        def set(self, _property_id: int, position: int) -> None:
            self.position = position
            self.seeks += 1

        def read(self) -> tuple[bool, np.ndarray]:
            frame = np.full((1, 1, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = Capture()
    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            CAP_PROP_FRAME_COUNT=1,
            CAP_PROP_FPS=2,
            CAP_PROP_POS_FRAMES=3,
            VideoCapture=lambda _path: capture,
        ),
    )
    sampled = every_other_video_frame(SimpleNamespace(), start_ms=200, end_ms=700)  # type: ignore[arg-type]
    assert [timestamp for _, timestamp in sampled.frames] == [200, 400, 600]
    assert [int(frame[0, 0, 0]) for frame, _ in sampled.frames] == [2, 4, 6]
    assert capture.released
    assert capture.seeks == 1


def test_translation_cache_detects_corruption(tmp_path) -> None:  # type: ignore[no-untyped-def]
    request = TranslationRequest(
        np.ones((2, 266), dtype=np.float32), np.ones((2, 133), dtype=np.bool_)
    )
    cache = TranslationFeatureCache(tmp_path)
    path = cache.store("clip", request, metadata={"split": "train"})
    assert cache.find_valid("clip") is not None
    path.write_bytes(b"corrupt")
    assert cache.find_valid("clip") is None


def test_translation_cache_rejects_stale_processing_identity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    request = TranslationRequest(
        np.ones((2, 266), dtype=np.float32), np.ones((2, 133), dtype=np.bool_)
    )
    cache = TranslationFeatureCache(tmp_path)
    cache.store("clip", request, metadata={"feature_schema": TRANSLATION_FEATURE_SCHEMA})
    assert cache.find_valid("clip", expected_metadata={"feature_schema": "older"}) is None




def test_direct_result_keeps_empty_decodes_as_evaluation_failures() -> None:
    result = DirectTranslationResult("", None, "t5", "v", 1)
    assert result.english == ""
    with pytest.raises(ValueError):
        DirectTranslationResult("", None, "", "v", 1)


def test_translation_metrics_use_paper_bleurt_scale() -> None:
    predictions = (DirectTranslationResult("hello world", -1.0, "t5", "v", 5),)
    result = evaluate_translations(
        predictions, ("hello world",), bleurt_scorer=lambda *_args: [0.4663]
    )
    assert result.bleurt == pytest.approx(46.63)
    assert result.exact_match == 1
    assert result.bleu == pytest.approx(0)
