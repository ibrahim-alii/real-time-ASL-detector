from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from asl_recognizer.translation import acquisition, sol
from asl_recognizer.translation.contracts import TranslationRequest
from asl_recognizer.translation.model import encoded_values
from asl_recognizer.translation.sol_common import capacity, read_json, write_json


def test_benchmark_budgets_slower_corpus_and_measured_checkpoint(tmp_path, monkeypatch):
    from asl_recognizer.translation import sol_training

    pipeline = SimpleNamespace(
        scratch=tmp_path / "scratch", root=tmp_path / "home",
        reports=tmp_path / "home/reports", features=tmp_path / "features",
        manifests=tmp_path / "manifests", config={"allocation_seconds": 1000},
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kw: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _i: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(sol_training, "collect", lambda: None)

    def run(**kwargs):
        size = kwargs["config"].microbatch_size
        if size == 8:
            raise torch.cuda.OutOfMemoryError("out of memory")
        write_json(kwargs["output"] / "progress.json", {
            "source_timings": {
                "youtube-asl": {"optimizer_seconds": 50, "optimizer_updates": 25,
                                "validation_seconds_max": 10},
                "how2sign": {"optimizer_seconds": 250, "optimizer_updates": 25,
                             "validation_seconds_max": 30},
            },
            "checkpoint_seconds_max": 4,
            "peak_vram_bytes": 800 if size == 4 else 700,
        })

    monkeypatch.setattr(sol_training, "run_experiment", run)
    sol_training.benchmark(pipeline)
    budget = read_json(pipeline.reports / "training-budget.json")
    assert budget["microbatch"] == 2
    assert budget["seconds_per_update"] == 10
    assert budget["seconds_per_validation"] == 30
    assert budget["seconds_per_checkpoint"] == 4
    assert budget["job_steps"] == int((800 - 60 - 20) / (10 + 4 / 50))
    assert budget["scratch_bytes"] > 0
    assert len(read_json(pipeline.reports / "benchmark.json")["candidates"]) == 4


def test_benchmark_selects_fastest_safe_microbatch_not_largest(tmp_path, monkeypatch):
    from asl_recognizer.translation import sol_training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kw: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _i: SimpleNamespace(total_memory=1000))
    monkeypatch.setattr(sol_training, "collect", lambda: None)

    def run(**kwargs):
        size = kwargs["config"].microbatch_size
        seconds = {1: 2.0, 2: 0.5, 4: 3.0, 8: 1.0}[size]
        write_json(kwargs["output"] / "progress.json", {
            "source_timings": {corpus: {
                "optimizer_seconds": seconds * 25, "optimizer_updates": 25,
                "validation_seconds_max": 1,
            } for corpus in ("youtube-asl", "how2sign")},
            "checkpoint_seconds_max": 1,
            "peak_vram_bytes": 800 if size == 2 else 500,
        })

    monkeypatch.setattr(sol_training, "run_experiment", run)
    pipeline = SimpleNamespace(
        scratch=tmp_path / "scratch", root=tmp_path / "home",
        reports=tmp_path / "home/reports", features=tmp_path / "features",
        manifests=tmp_path / "manifests", config={"allocation_seconds": 1000},
    )
    sol_training.benchmark(pipeline)
    budget = read_json(pipeline.reports / "training-budget.json")
    # Size 2 is fastest but fails VRAM headroom; size 8 beats safe sizes 1 and 4.
    assert budget["microbatch"] == 8
    assert budget["seconds_per_update"] == 1.0

    # With size 8 slower, select size 1 even though the larger size 4 also fits.
    def slower_large(**kwargs):
        run(**kwargs)
        path = kwargs["output"] / "progress.json"
        progress = read_json(path)
        if kwargs["config"].microbatch_size == 8:
            for timing in progress["source_timings"].values():
                timing["optimizer_seconds"] = 100
            write_json(path, progress)

    monkeypatch.setattr(sol_training, "run_experiment", slower_large)
    sol_training.benchmark(pipeline)
    assert read_json(pipeline.reports / "training-budget.json")["microbatch"] == 1


@pytest.mark.parametrize("message", ["non-finite training loss", "unexpected CUDA failure"])
def test_benchmark_stops_on_unexpected_runtime_failure(tmp_path, monkeypatch, message):
    from asl_recognizer.translation import sol_training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda **_kw: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda _i: SimpleNamespace(total_memory=1000))
    attempted = []

    def run(**kwargs):
        attempted.append(kwargs["config"].microbatch_size)
        raise RuntimeError(message)

    monkeypatch.setattr(sol_training, "run_experiment", run)
    pipeline = SimpleNamespace(
        scratch=tmp_path, features=tmp_path, manifests=tmp_path, reports=tmp_path,
    )
    with pytest.raises(RuntimeError, match=message):
        sol_training.benchmark(pipeline)
    assert attempted == [1]
    assert not (tmp_path / "training-config.json").exists()


def test_archive_tool_verifies_publisher_digest_and_cached_binary(tmp_path, monkeypatch):
    import hashlib

    from asl_recognizer.translation import archive_tool

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:xz") as archive:
        member = tarfile.TarInfo("7zz")
        member.size = 6
        archive.addfile(member, io.BytesIO(b"binary"))
    data = payload.getvalue()
    monkeypatch.setattr(archive_tool, "SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(archive_tool.urllib.request, "urlopen", lambda *_a, **_kw:
                        io.BytesIO(data))
    binary = archive_tool.ensure_archive_tool(tmp_path / "tool")
    assert binary.read_bytes() == b"binary"
    assert archive_tool.ensure_archive_tool(tmp_path / "tool") == binary
    binary.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity or checksum"):
        archive_tool.ensure_archive_tool(tmp_path / "tool")
    monkeypatch.setattr(archive_tool, "SHA256", "wrong")
    with pytest.raises(ValueError, match="release checksum"):
        archive_tool.ensure_archive_tool(tmp_path / "other")
    assert not (tmp_path / "other/7zz").exists()


def test_scheduler_imports_without_scientific_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "-c",
         "import asl_recognizer.translation.sol; import asl_recognizer.translation.sol_pipeline"],
        env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_drive_preserves_explicit_download_mode_without_fuzzy_rewriting(tmp_path, monkeypatch):
    from asl_recognizer.translation import sol_pipeline

    pipeline = object.__new__(sol_pipeline.Pipeline)
    pipeline.scratch = tmp_path
    pipeline.config = {}
    monkeypatch.setattr(sol_pipeline, "remaining", lambda *_a: 10000)
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output"]).write_bytes(b"volume")
        return kwargs["output"]

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))
    pipeline.drive("https://drive.google.com/file/d/official-id/view", tmp_path / "test.z01", 1)
    assert calls[0]["url"] == "https://drive.google.com/uc?export=download&id=official-id"
    assert calls[0]["fuzzy"] is False
    assert calls[0]["use_cookies"] is False
    assert read_json(tmp_path / "test.z01.json")["url"].endswith("/official-id/view")
    assert acquisition.google_drive_download_url(
        "https://drive.google.com/uc?id=official-id"
    ) == calls[0]["url"]
    with pytest.raises(ValueError):
        acquisition.google_drive_download_url("https://example.com/file/d/official-id/view")


def test_completed_job_missing_from_live_queue_uses_accounting(monkeypatch):  # type: ignore[no-untyped-def]
    def command(args):  # type: ignore[no-untyped-def]
        if args[0] == "squeue":
            raise RuntimeError("squeue failed: slurm_load_jobs error: Invalid job id specified")
        return "123|FAILED|\n123.batch|FAILED|"

    monkeypatch.setattr(sol, "run_command", command)
    assert sol.job_state("123") == "FAILED"
    monkeypatch.setattr(sol, "run_command", lambda _args: "")
    assert sol.job_state("123") == "UNKNOWN"
    monkeypatch.setattr(sol, "run_command", lambda _args: (_ for _ in ()).throw(
        RuntimeError("Slurm authentication failed")
    ))
    with pytest.raises(RuntimeError, match="authentication"):
        sol.job_state("123")


def test_recording_archive_grouping_orders_parts_and_keeps_independent_zips(tmp_path):  # type: ignore[no-untyped-def]
    entries = [(name, tmp_path / name) for name in
               ("train.zip", "train.z02", "other.zip", "train.z01")]
    groups = acquisition.recording_archive_groups(entries)
    assert [[path.name for _, path in group] for group in groups] == [
        ["other.zip"], ["train.z01", "train.z02", "train.zip"]
    ]


@pytest.mark.parametrize("names", [
    ["train.z01"], ["train.z02", "train.zip"],
    ["train.z01", "train.z03", "train.zip"],
    ["train.z00", "train.zip"], ["train.zip", "train.zip"], ["train.rar"],
])
def test_recording_archive_grouping_rejects_incomplete_or_ambiguous_inputs(tmp_path, names):  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        acquisition.recording_archive_groups([(name, tmp_path / name) for name in names])


def split_fixture(tmp_path):  # type: ignore[no-untyped-def]
    from asl_recognizer.translation.sol_common import seal

    parts = [tmp_path / "train.z01", tmp_path / "train.zip"]
    for part in parts:
        part.write_bytes(b"part")
        write_json(part.with_suffix(part.suffix + ".json"),
                   {**seal(part), "url": f"official/{part.name}"})
    return parts



def split_listing(name="nested/sample.mp4", payload=b"recording"):
    return (f"Path = {name}\nFolder = -\nEncrypted = -\nAttributes = -rw-r-----\n"
            f"Size = {len(payload)}\nCRC = {zlib.crc32(payload):08X}\n\n")


def test_split_extraction_reuses_verified_output_and_checks_capacity(
    tmp_path, monkeypatch,
):  # type: ignore[no-untyped-def]
    parts = split_fixture(tmp_path)
    output = tmp_path / "videos"
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["stdin"] == subprocess.DEVNULL
        if args[1] == "l":
            return SimpleNamespace(stdout=split_listing()+split_listing("other.mp4"))
        assert "-spd" in args
        kwargs["stdout"].write(b"recording")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(acquisition.subprocess, "run", run)
    kwargs = dict(tool=Path("7zz"), remaining_quota=1000)
    assert acquisition.extract_split_recordings(parts, {"sample"}, output, **kwargs) == {"sample"}
    assert (output / "sample.mp4").read_bytes() == b"recording"
    assert not (output / "other.mp4").exists()
    kwargs["remaining_quota"] = 0
    acquisition.extract_split_recordings(parts, {"sample"}, output, **kwargs)
    assert len([args for args in calls if args[1]=="x"]) == 1
    (output / "sample.mp4").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="insufficient capacity"):
        acquisition.extract_split_recordings(parts, {"sample"}, output, **kwargs)
    kwargs["remaining_quota"] = 1000
    acquisition.extract_split_recordings(parts, {"sample"}, output, **kwargs)
    assert len([args for args in calls if args[1]=="x"]) == 2
    parts[0].write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        acquisition.extract_split_recordings(parts, {"sample"}, output, **kwargs)


@pytest.mark.parametrize("failure", ["extractor", "crc"])
def test_split_extraction_failure_is_not_sealed(tmp_path, monkeypatch, failure):
    parts = split_fixture(tmp_path)
    output = tmp_path / "videos"

    def run(args, **kwargs):
        if args[1] == "l":
            return SimpleNamespace(stdout=split_listing())
        kwargs["stdout"].write(b"truncated")
        if failure == "extractor":
            raise subprocess.CalledProcessError(2, args)

    monkeypatch.setattr(acquisition.subprocess, "run", run)
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        acquisition.extract_split_recordings(parts, {"sample"}, output,
                                             tool=Path("7zz"), remaining_quota=1000)
    assert not (output / "sample.mp4").exists()
    assert not (output / "sample.json").exists()
    assert not (output / "sample.mp4.part").exists()
    assert all(part.read_bytes() == b"part" for part in parts)


@pytest.mark.parametrize("listing", [split_listing("../sample.mp4"),
                                    split_listing("/sample.mp4"),
                                    split_listing()+split_listing("other/sample.mp4"),
                                    split_listing().replace("CRC =", "Missing =")])
def test_split_extraction_rejects_unsafe_or_ambiguous_listing(tmp_path, monkeypatch, listing):
    monkeypatch.setattr(acquisition.subprocess, "run", lambda *_a, **_kw:
                        SimpleNamespace(stdout=listing))
    with pytest.raises(ValueError):
        acquisition.extract_split_recordings(split_fixture(tmp_path), {"sample"},
            tmp_path / "out", tool=Path("7zz"), remaining_quota=1000)


@pytest.mark.skipif(shutil.which("zip") is None or not os.environ.get("ASL_ARCHIVE_TOOL"),
                    reason="requires split-ZIP fixture generator and verified 7-Zip binary")
def test_real_split_zip_round_trip(tmp_path):
    from asl_recognizer.translation.sol_common import seal

    video = tmp_path / "sample.mp4"
    contents = os.urandom(160_000)
    video.write_bytes(contents)
    archive = tmp_path / "train.zip"
    subprocess.run(["zip", "-q", "-0", "-s", "64k", "-j", str(archive), str(video)],
                   check=True)
    parts = [*sorted(tmp_path.glob("train.z[0-9]*")), archive]
    assert len(parts) > 1
    for part in parts:
        write_json(part.with_suffix(part.suffix + ".json"), seal(part))
    output=tmp_path / "out"
    acquisition.extract_split_recordings(parts, {"sample"}, output,
        tool=Path(os.environ["ASL_ARCHIVE_TOOL"]), remaining_quota=300000)
    assert (output / "sample.mp4").read_bytes() == contents


def test_pipeline_downloads_all_split_volumes_before_assembly(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from asl_recognizer.translation import sol_pipeline
    from asl_recognizer.translation.manifest import ClipManifest, ManifestRecord, write_manifest

    pipeline = object.__new__(sol_pipeline.Pipeline)
    pipeline.config = {"how2sign": {"train_recordings": "official/folders/train",
                                    "dev_recordings": "official/dev"},
                       "allocation_seconds": 28800}
    pipeline.scratch = tmp_path / "scratch"
    pipeline.manifests = tmp_path / "manifests"
    write_manifest(pipeline.manifests / "how2sign.csv", ClipManifest(tuple(
        ManifestRecord(split, "how2sign", split, split, 0, 1000, "reference")
        for split in ("train", "dev")
    )))
    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download_folder=lambda **_kw: [
        SimpleNamespace(id=name, path=name) for name in ("train.zip", "train.z01")
    ]))
    monkeypatch.setattr(sol_pipeline, "remaining", lambda *_args: 10**12)
    from asl_recognizer.translation import archive_tool
    pipeline.runtime = tmp_path / "runtime"
    pipeline.reports = tmp_path / "reports"
    tool = pipeline.runtime / "archive-tools/7zz"
    tool.parent.mkdir(parents=True)
    write_json(tool.with_suffix(".json"), {"version": "26.04"})
    monkeypatch.setattr(archive_tool, "ensure_archive_tool", lambda _root: tool)
    events = []
    monkeypatch.setattr(pipeline, "drive", lambda _url, path, _reserve, **kw:
                        events.append(("download", path.name, kw["validate_zip"])))

    def extract_split(parts, wanted, output, **_kw):  # type: ignore[no-untyped-def]
        events.append(("extract-split", [part.name for part in parts]))
        return wanted

    monkeypatch.setattr(sol_pipeline, "extract_split_recordings", extract_split)
    monkeypatch.setattr(sol_pipeline, "extract_recordings", lambda path, wanted, *_args, **_kw:
                        events.append(("extract", path.parent.name)) or wanted)
    pipeline.how2sign()
    assert events == [("download", "train.z01", False), ("download", "train.zip", False),
                      ("extract-split", ["train.z01", "train.zip"]),
                      ("download", "recordings.zip", True), ("extract", "dev")]


@pytest.mark.parametrize("mismatch", [False, True])
def test_recorded_stage_selects_five_dev_clips_and_checks_cache(
    tmp_path, monkeypatch, mismatch,
):  # type: ignore[no-untyped-def]
    from asl_recognizer.translation import sol_pipeline
    from asl_recognizer.translation.cache import TranslationFeatureCache
    from asl_recognizer.translation.manifest import ClipManifest, ManifestRecord, write_manifest
    from asl_recognizer.translation.schema import MMPOSE_SCHEMA

    if os.name == "nt":
        tmp_path = Path("\\\\?\\" + str(tmp_path))
    pipeline = object.__new__(sol_pipeline.Pipeline)
    pipeline.root = tmp_path / "home"
    pipeline.scratch = tmp_path / "scratch"
    pipeline.manifests = pipeline.root / "manifests"
    pipeline.reports = pipeline.root / "reports"
    pipeline.features = pipeline.scratch / "features"
    pipeline.config = {"seed": 42}
    pipeline.train = tmp_path / "train"
    pipeline.extract = tmp_path / "extract"
    monkeypatch.setattr(pipeline, "assets", lambda: [])
    checkpoint = pipeline.root / "runs/baseline/selected.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    rows = tuple(ManifestRecord(f"clip-{i}", "how2sign", split, f"video-{i}",
                                10, 100, "reference")
                 for i, split in enumerate(["dev"] * 7 + ["train", "test"]))
    write_manifest(pipeline.manifests / "how2sign.csv", ClipManifest(rows))
    request = TranslationRequest(np.zeros((1, 266), dtype=np.float32),
                                 np.ones((1, 133), dtype=np.bool_), MMPOSE_SCHEMA)
    metadata = {"extractor": {"identity": "pose"}}
    calls = []
    for row in rows:
        video = pipeline.scratch / f"how2sign/recordings/{row.split}/{row.video_id}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        TranslationFeatureCache(pipeline.features / "how2sign").store(
            row.clip_id, request, metadata=metadata
        )

    def worker(_python, *args):  # type: ignore[no-untyped-def]
        calls.append(args)
        output = args[args.index("--output") + 1]
        video = args[args.index("--video") + 1]
        overlay = args[args.index("--overlay") + 1]
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_bytes(b"overlay")
        write_json(output, {"checkpoint_sha256": sol_pipeline.file_hash(checkpoint),
                            "extraction": {"source_video_hash": sol_pipeline.file_hash(video)}})
        actual = TranslationRequest(request.values + int(mismatch), request.landmark_mask,
                                    MMPOSE_SCHEMA)
        TranslationFeatureCache(output.parent / (output.stem + "-features")).store(
            "recorded", actual, metadata=metadata
        )

    monkeypatch.setattr(pipeline, "worker", worker)
    if mismatch:
        with pytest.raises(ValueError, match="preprocessing differs"):
            pipeline.recorded_inference()
    else:
        pipeline.recorded_inference()
        report = read_json(pipeline.reports / "recorded-inference.json")
        assert len(report["clips"]) == 5
        assert all("/dev/" in str(args[args.index("--video") + 1]).replace("\\", "/")
                   for args in calls)
        pipeline.recorded_inference()
        assert len(calls) == 5


def test_storage_budget_scans_only_project_directory(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    project = tmp_path / "bounded-run"
    config = {"scratch_root": str(project), "scratch_limit_bytes": 1000}
    calls = []
    monkeypatch.setattr(sol, "run_command", lambda args: calls.append(args) or "250\tproject")
    assert sol.remaining(config, "scratch") == 1000
    assert not calls
    project.mkdir()
    assert sol.remaining(config, "scratch") == 750
    assert calls == [["du", "-sb", str(project)]]
    config.update(durable_root=str(project), home_limit_bytes=2000)
    assert sol.remaining(config, "home") == 1750


def test_live_resource_preflight_uses_durable_home_root(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config = json.loads(Path("config/sol_translation.example.json").read_text())
    config.update(durable_root=str(tmp_path), scratch_root=str(tmp_path))
    monkeypatch.setattr(sol, "validate_config", lambda _c: None)
    monkeypatch.setattr(sol.getpass, "getuser", lambda: config["user"])
    monkeypatch.setattr(sol, "run_command", lambda args: (
        config["account"] if args[0] == "myaccounts" else "0 project"
    ))
    report = sol.resource_check(config)
    assert report["home"]["available_bytes"] > 0
    assert report["scratch"]["available_bytes"] > 0


def test_capacity_checks_quota_and_disk_with_headroom(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr("shutil.disk_usage", lambda _p: SimpleNamespace(free=2000))
    assert capacity(tmp_path, 1000, remaining_quota=1500)["required_with_headroom"] == 1300
    with pytest.raises(ValueError, match="insufficient"):
        capacity(tmp_path, 1000, remaining_quota=1200)
    with pytest.raises(ValueError, match="insufficient"):
        capacity(tmp_path, 2000, remaining_quota=10000)


def test_captions_keep_original_cue_times_and_require_manual_english():
    rows = acquisition.json3_captions(
        {
            "events": [
                {
                    "tStartMs": 123,
                    "dDurationMs": 456,
                    "segs": [{"utf8": "hello &amp; "}, {"utf8": "world\n"}],
                }
            ]
        }
    )
    assert rows == [{"start_ms": 123, "end_ms": 579, "caption": "hello & world"}]
    assert acquisition.english_track({"subtitles": {"en-US": [], "en": []}}) == "en"
    with pytest.raises(ValueError, match="unavailable"):
        acquisition.english_track({"automatic_captions": {"en": []}})
    with pytest.raises(ValueError, match="unavailable"):
        acquisition.english_track({"subtitles": {"en": [{"url": "captions?tlang=en"}]}})
    with pytest.raises(ValueError, match="positive"):
        acquisition.json3_captions(
            {"events": [{"tStartMs": 0, "dDurationMs": -1, "segs": [{"utf8": "hello"}]}]}
        )


def youtube_fixture(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(acquisition.time, "sleep", lambda _s: None)
    root = tmp_path / "staging"
    root.mkdir()
    ids = root / "ids.txt"
    ids.write_text("abcdefghijk\n", encoding="utf-8")
    calls = []

    def runner(args, **_kwargs):  # type: ignore[no-untyped-def]
        calls.append(args)
        if "--dump-single-json" in args:
            return json.dumps(
                {
                    "channel_id": "channel",
                    "subtitles": {"en": []},
                    "formats": [
                        {
                            "format_id": "137",
                            "height": 720,
                            "ext": "mp4",
                            "vcodec": "h264",
                            "acodec": "none",
                            "filesize": 10,
                        }
                    ],
                }
            )
        if "--write-subs" in args:
            raw = root / "raw-captions"
            raw.mkdir(exist_ok=True)
            write_json(
                raw / "abcdefghijk.en.json3",
                {"events": [{"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "hello"}]}]},
            )
            write_json(raw / "abcdefghijk.info.json", {"id": "abcdefghijk"})
        elif args[0] == "ffprobe":
            return json.dumps({"streams": [{"codec_type": "video"}], "format": {"duration": "2"}})
        elif "-f" in args:
            (root / "videos/abcdefghijk.mp4").write_bytes(b"video")
        return ""

    return root, ids, calls, runner


def test_youtube_resume_verifies_all_provenance_and_does_not_redownload(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, ids, calls, runner = youtube_fixture(tmp_path, monkeypatch)
    kwargs = dict(
        ids_path=ids, root=root, byte_limit=100000, scratch_remaining=lambda: 100000, runner=runner
    )
    first = acquisition.acquire_youtube(**kwargs)
    assert first["videos"]["abcdefghijk"]["availability"] == "available"
    assert len(first["videos"]["abcdefghijk"]["files"]) == 4
    # An interrupted older batch may have left a stale/missing validation index.
    (root / "index.csv").unlink()
    calls.clear()
    acquisition.acquire_youtube(**kwargs)
    assert calls == []
    assert "abcdefghijk" in (root / "index.csv").read_text()
    (root / "raw-captions/abcdefghijk.info.json").write_text("changed")
    acquisition.acquire_youtube(**kwargs)
    assert calls


def test_youtube_throttle_stops_and_persists_failure(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, ids, _calls, _runner = youtube_fixture(tmp_path, monkeypatch)

    def fail(_args):  # type: ignore[no-untyped-def]
        raise RuntimeError("HTTP 429 Too Many Requests")

    with pytest.raises(RuntimeError, match="investigation"):
        acquisition.acquire_youtube(
            ids_path=ids, root=root, byte_limit=10000, scratch_remaining=lambda: 10000, runner=fail
        )
    row = read_json(root / "acquisition.json")["videos"]["abcdefghijk"]
    assert row["availability"] == "unavailable" and row["attempts"] == 1
    assert (root / "index.csv").is_file()


@pytest.mark.parametrize("strip_success", [True, False])
def test_youtube_muxed_mp4_fallback_strips_audio_and_checks_result(
    tmp_path, monkeypatch, strip_success
):  # type: ignore[no-untyped-def]
    root, ids, calls, original = youtube_fixture(tmp_path, monkeypatch)

    def runner(args):  # type: ignore[no-untyped-def]
        if "--dump-single-json" in args:
            info = json.loads(original(args))
            info["formats"][0]["acodec"] = "aac"
            return json.dumps(info)
        if args[0] == "ffmpeg":
            calls.append(args)
            assert args[args.index("-c:v") + 1] == "copy"
            assert "-an" in args
            Path(args[-1]).write_bytes(b"stripped-video")
            return ""
        if args[0] == "ffprobe" and not strip_success:
            return json.dumps({"streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
                               "format": {"duration": "2"}})
        return original(args)

    ledger = acquisition.acquire_youtube(
        ids_path=ids, root=root, byte_limit=100000, scratch_remaining=lambda: 100000,
        runner=runner,
    )
    row = ledger["videos"]["abcdefghijk"]
    assert any(args[0] == "ffmpeg" for args in calls)
    if strip_success:
        assert row["availability"] == "available"
        assert row["source_audio_removed"] is True
        assert (root / row["video_path"]).read_bytes() == b"stripped-video"
        assert len(row["files"]) == 4
    else:
        assert row["availability"] == "unavailable"
        assert row["reason"] == "training video must not contain audio"


def test_youtube_prefers_video_only_without_audio_remux(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, ids, calls, original = youtube_fixture(tmp_path, monkeypatch)

    def runner(args):  # type: ignore[no-untyped-def]
        if "--dump-single-json" in args:
            info = json.loads(original(args))
            info["formats"][0]["height"] = 360
            info["formats"].append({**info["formats"][0], "format_id": "22",
                                    "height": 720, "acodec": "aac"})
            return json.dumps(info)
        return original(args)

    row = acquisition.acquire_youtube(
        ids_path=ids, root=root, byte_limit=100000, scratch_remaining=lambda: 100000,
        runner=runner,
    )["videos"]["abcdefghijk"]
    assert row["availability"] == "available" and row["format_id"] == "137"
    assert row["source_audio_removed"] is False
    assert not any(args[0] == "ffmpeg" for args in calls)


def test_youtube_missing_caption_is_not_substituted_or_repeated(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, ids, calls, _runner = youtube_fixture(tmp_path, monkeypatch)

    def runner(args):  # type: ignore[no-untyped-def]
        calls.append(args)
        return json.dumps({"channel_id": "channel", "automatic_captions": {"en": []}})

    kwargs = dict(
        ids_path=ids,
        root=root,
        byte_limit=10000,
        scratch_remaining=lambda: 10000,
        runner=runner,
        attempts=2,
    )
    for _ in range(3):
        acquisition.acquire_youtube(**kwargs)
    assert len(calls) == 1
    assert (
        read_json(root / "acquisition.json")["videos"]["abcdefghijk"]["availability"]
        == "unavailable"
    )


def test_youtube_transient_failures_retain_bounded_retries(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    root, ids, calls, _runner = youtube_fixture(tmp_path, monkeypatch)

    def runner(args):  # type: ignore[no-untyped-def]
        calls.append(args)
        raise RuntimeError("temporary network failure")

    for _ in range(3):
        acquisition.acquire_youtube(
            ids_path=ids, root=root, byte_limit=100000,
            scratch_remaining=lambda: 100000, runner=runner, attempts=2,
        )
    assert len(calls) == 2
    row = read_json(root / 'acquisition.json')['videos']['abcdefghijk']
    assert row['attempts'] == 2 and row['failure_kind'] == 'retryable'


def test_recording_zip_extracts_only_wanted_content_and_recomputes_corruption(tmp_path):  # type: ignore[no-untyped-def]
    archive = tmp_path / "videos.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/one.mp4", b"one")
        bundle.writestr("nested/two.mp4", b"two")
    output = tmp_path / "recordings"
    assert acquisition.extract_recordings(archive, {"one"}, output, remaining_quota=1000) == {"one"}
    assert not (output / "two.mp4").exists()
    (output / "one.mp4").write_bytes(b"bad")
    acquisition.extract_recordings(archive, {"one"}, output, remaining_quota=1000)
    assert (output / "one.mp4").read_bytes() == b"one"
    with pytest.raises(ValueError, match="insufficient"):
        acquisition.extract_recordings(archive, {"two"}, output, remaining_quota=1)


def test_mask_encoding_removes_sentinel_preserves_missingness_and_cache_input():
    values = np.ones((2, 266), dtype=np.float32)
    mask = np.ones((2, 133), dtype=np.bool_)
    values[:, :2] = -1000
    mask[:, 0] = False
    request = TranslationRequest(values, mask, "mmpose-coco-wholebody-133-xy-v1")
    encoded = encoded_values(request, "mask-v1")
    assert encoded.shape == (2, 399)
    assert encoded[:, :2].eq(0).all() and encoded[:, 266].eq(0).all()
    assert encoded[:, 267:].eq(1).all()
    assert request.values[0, 0] == -1000
    torch.testing.assert_close(encoded_values(request, "sentinel-v1"), torch.from_numpy(values))


def controller_fixture(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config = json.loads(Path("config/sol_translation.example.json").read_text())
    config["durable_root"] = str(tmp_path / "run")
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    real_read = read_json
    monkeypatch.setattr(
        sol,
        "read_json",
        lambda p: {"bundle_sha256": "bundle"} if p.name == "deployment.json" else real_read(p),
    )
    monkeypatch.setattr(sol, "validate_config", lambda _c: None)
    monkeypatch.setattr(sol, "resource_check", lambda _c: {})
    monkeypatch.setattr(sol, "writer_lock", lambda _p: contextlib.nullcontext())
    monkeypatch.setattr(sol, "run_command", lambda _a: "")
    submissions = []

    def submit(*args):  # type: ignore[no-untyped-def]
        submissions.append(args[0])
        return str(100 + len(submissions))

    monkeypatch.setattr(sol, "submit", submit)
    return config_path, Path(config["durable_root"]), submissions


@pytest.mark.parametrize("stage", sol.STAGES[9:])
def test_controller_does_not_submit_retired_stages(tmp_path, monkeypatch, stage):
    config, root, submissions = controller_fixture(tmp_path, monkeypatch)
    state = sol.advance(config, "run")
    state.pop("job_id")
    state.update(stage_index=sol.STAGES.index(stage), status="ready")
    write_json(root / "controller.json", state)
    result = sol.advance(config, "resume")
    assert result["status"] == "retired"
    assert result["stage"] == stage
    assert read_json(root / "controller.json")["stage_index"] == sol.STAGES.index(stage)
    assert submissions == ["bootstrap"]


def test_controller_running_unknown_and_failed_jobs_never_duplicate(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config, _root, submissions = controller_fixture(tmp_path, monkeypatch)
    first = sol.advance(config, "run")
    assert first["job_id"] == "101"
    for status in ("RUNNING", "UNKNOWN", "FAILED"):
        monkeypatch.setattr(sol, "job_state", lambda _j, s=status: s)
        sol.advance(config, "run")
    assert submissions == ["bootstrap"]
    sol.advance(config, "resume")
    assert submissions == ["bootstrap", "bootstrap"]


def test_controller_completed_job_requires_receipt_and_advances_once(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config, root, submissions = controller_fixture(tmp_path, monkeypatch)
    first = sol.advance(config, "run")
    monkeypatch.setattr(sol, "job_state", lambda _j: "COMPLETED")
    assert sol.advance(config, "run")["status"] == "failed"
    write_json(
        root / "receipts" / f"{first['token']}.json",
        {"token": first["token"], "status": "completed", "next_stage": None},
    )
    second = sol.advance(config, "run")
    assert second["stage"] == "runtime"
    assert submissions == ["bootstrap", "runtime"]


def test_uncertain_submission_keeps_original_intent_until_reconciled(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config, root, submissions = controller_fixture(tmp_path, monkeypatch)
    first = sol.advance(config, "run")
    first.pop("job_id")
    first["status"] = "submission-uncertain"
    write_json(root / "controller.json", first)
    monkeypatch.setattr(sol, "find_submission", lambda _t, _u: None)
    assert sol.advance(config, "resume")["token"] == first["token"]
    assert submissions == ["bootstrap"]
    monkeypatch.setattr(sol, "find_submission", lambda _t, _u: "101")
    monkeypatch.setattr(sol, "job_state", lambda _j: "RUNNING")
    assert sol.advance(config, "resume")["job_id"] == "101"
    assert submissions == ["bootstrap"]


def test_controller_rejects_config_changes(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config, _root, _submissions = controller_fixture(tmp_path, monkeypatch)
    sol.advance(config, "run")
    changed = read_json(config)
    changed["seed"] = 100
    write_json(config, changed)
    with pytest.raises(ValueError, match="configuration changed"):
        sol.advance(config, "resume")


def test_prepare_stops_after_bootstrap_until_run(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config, root, submissions = controller_fixture(tmp_path, monkeypatch)
    first = sol.advance(config, "prepare")
    write_json(
        root / "receipts" / f"{first['token']}.json",
        {"token": first["token"], "status": "completed", "next_stage": None},
    )
    monkeypatch.setattr(sol, "job_state", lambda _j: "COMPLETED")
    assert sol.advance(config, "continue")["status"] == "prepared"
    assert submissions == ["bootstrap"]
    assert sol.advance(config, "run")["stage"] == "runtime"


def test_run_persists_continuation_while_preparation_is_running(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config, root, submissions = controller_fixture(tmp_path, monkeypatch)
    sol.advance(config, "prepare")
    monkeypatch.setattr(sol, "job_state", lambda _j: "RUNNING")
    sol.advance(config, "run")
    assert read_json(root / "controller.json")["auto_continue"] is True
    assert submissions == ["bootstrap"]


def test_slurm_submission_uses_one_gpu_only_for_gpu_stages(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    config = json.loads(Path("config/sol_translation.example.json").read_text())
    calls = []

    def command(args):  # type: ignore[no-untyped-def]
        calls.append(args)
        return "123" if "--test-only" not in args else ""

    monkeypatch.setattr(sol, "run_command", command)
    for stage in ("bootstrap", "extract-youtube"):
        sol.submit(stage, tmp_path / "config.json", config, tmp_path, "token", tmp_path)
    assert "--gpus" not in calls[1]
    assert calls[3][calls[3].index("--gpus") + 1] == "a100:1"
    assert "--test-only" in calls[0] and "--test-only" in calls[2]


def test_configuration_rejects_paths_escaping_personal_storage():
    config = json.loads(Path("config/sol_translation.example.json").read_text())
    sol.validate_config(config)
    for path in ("/scratch/other/project", "/scratch/syedibra/../../other", "/scratch/syedibra"):
        config["scratch_root"] = path
        with pytest.raises(ValueError, match="personal"):
            sol.validate_config(config)


def test_bundle_captures_untracked_runtime_and_excludes_credentials(tmp_path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("sol_entry", "scripts/sol_translation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "src").mkdir()
    source = tmp_path / "src/new.py"
    source.write_text("first")
    (tmp_path / ".asl-sol-access").mkdir()
    (tmp_path / ".asl-sol-access/private").write_text("secret")
    (tmp_path / "config").mkdir()
    (tmp_path / "config/sol_translation.json").write_text('{"user": "private-user"}')
    (tmp_path / "config/sol_translation.example.json").write_text('{"user": "your_user"}')
    (tmp_path / "PRD.md").write_text("local requirements")
    (tmp_path / "AGENTS.md").write_text("local instructions")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/SOL_JOB_GUIDE.md").write_text("local operations")
    first = module.create_bundle(tmp_path, tmp_path / "bundle.tar.gz")
    source.write_text("second")
    second = module.create_bundle(tmp_path, tmp_path / "bundle.tar.gz")
    assert first != second
    import tarfile

    with tarfile.open(tmp_path / "bundle.tar.gz") as archive:
        assert archive.getnames() == ["config/sol_translation.example.json", "src/new.py"]
