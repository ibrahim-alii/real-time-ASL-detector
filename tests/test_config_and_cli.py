from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from asl_recognizer.cli import main
from asl_recognizer.config import load_config
from asl_recognizer.data.index import manifest_fingerprint
from asl_recognizer.data.schema import SampleRecord


def test_load_config_reads_defaults_and_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[project]\ndataset_root = "dataset"\nartifacts_root = "outputs"\n',
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.dataset_root == Path("dataset")
    assert config.seed == 42
    assert config.sequence_length == 32


def test_load_config_rejects_missing_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[project]\nseed = 7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifacts_root, dataset_root"):
        load_config(path)


def test_manifest_fingerprint_is_order_independent() -> None:
    first = SampleRecord("a", Path("a.mp4"), "A", "p1", "train")
    second = SampleRecord("b", Path("b.mp4"), "B", "p2", "val")

    assert manifest_fingerprint([first, second]) == manifest_fingerprint([second, first])


def test_data_index_cli_writes_reproducible_artifacts(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    split_paths: dict[str, Path] = {}
    for split, signer in (("train", "p1"), ("val", "p2"), ("test", "p3")):
        filename = f"{split}-HELLO.mp4"
        (videos / filename).touch()
        split_path = tmp_path / f"{split}.csv"
        with split_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["Participant ID", "Video file", "Gloss"]
            )
            writer.writeheader()
            writer.writerow(
                {"Participant ID": signer, "Video file": filename, "Gloss": "HELLO"}
            )
        split_paths[split] = split_path
    output = tmp_path / "output"

    exit_code = main(
        [
            "data-index",
            "--train-csv",
            str(split_paths["train"]),
            "--val-csv",
            str(split_paths["val"]),
            "--test-csv",
            str(split_paths["test"]),
            "--video-root",
            str(videos),
            "--output-dir",
            str(output),
            "--vocabulary-size",
            "1",
            "--skip-official-count-check",
        ]
    )

    assert exit_code == 0
    assert (output / "manifest.csv").is_file()
    vocabulary = json.loads((output / "vocabulary.json").read_text(encoding="utf-8"))
    assert vocabulary["labels"] == [{"class_id": 0, "gloss": "HELLO"}]
    assert len(vocabulary["content_hash"]) == 64


def test_load_config_rejects_nonpositive_sizes(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[project]\ndataset_root="data"\nartifacts_root="out"\nsequence_length=0\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be positive"):
        load_config(path)
