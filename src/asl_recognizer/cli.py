from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data.index import (
    assign_class_ids,
    build_manifest,
    file_sha256,
    manifest_fingerprint,
    OFFICIAL_SIGNER_COUNTS,
    read_manifest,
    read_vocabulary,
    select_vocabulary,
    write_manifest,
    write_vocabulary,
)
from .evaluation.metrics import classification_report
from .features.cache import FeatureCache
from .features.extract import extract_dataset
from .features.mediapipe_backend import MediaPipeHolisticBackend
from .features.normalize import PREPROCESSING_VERSION
from .features.video import sampled_video_frames
from .features.visualize import draw_landmarks
from .logging_config import configure_logging
from .models.baselines import LogisticBaseline, MajorityBaseline
from .models.checkpoint import (
    load_bundle,
    preprocessing_fingerprint,
    save_bundle,
)
from .models.gru import GRUConfig, LandmarkGRU
from .models.predictor import GRUPredictor
from .streaming import StreamingConfig, StreamingRecognizer
from .training.data import SequenceExample, discover_cache_index, load_cached_examples
from .training.engine import TrainingConfig, evaluate, train_model

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="asl-recognizer")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("data-index", help="build a validated ASL Citizen manifest")
    index.add_argument("--train-csv", type=Path, required=True)
    index.add_argument("--val-csv", type=Path, required=True)
    index.add_argument("--test-csv", type=Path, required=True)
    index.add_argument("--video-root", type=Path, required=True)
    index.add_argument("--output-dir", type=Path, required=True)
    index.add_argument("--vocabulary-size", type=int, default=25)
    index.add_argument("--allow-missing-videos", action="store_true")
    index.add_argument("--skip-official-count-check", action="store_true")
    extract = commands.add_parser("extract", help="extract and cache landmark sequences")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--vocabulary", type=Path, required=True)
    extract.add_argument("--video-root", type=Path, required=True)
    extract.add_argument("--cache-root", type=Path, required=True)
    extract.add_argument("--holistic-model", type=Path, required=True)
    extract.add_argument("--holistic-model-sha256", required=True)
    extract.add_argument("--sequence-length", type=int, default=32)
    inspect = commands.add_parser("inspect", help="write opt-in landmark overlay frames")
    inspect.add_argument("--video", type=Path, required=True)
    inspect.add_argument("--output-dir", type=Path, required=True)
    inspect.add_argument("--holistic-model", type=Path, required=True)
    inspect.add_argument("--holistic-model-sha256", required=True)
    inspect.add_argument("--frames", type=int, default=8)
    train = commands.add_parser("train", help="train baselines and a compact GRU")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--vocabulary", type=Path, required=True)
    train.add_argument("--video-root", type=Path, required=True)
    train.add_argument("--cache-root", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    demo = commands.add_parser("demo", help="run continuous webcam recognition")
    demo.add_argument("--bundle", type=Path, required=True)
    demo.add_argument("--vocabulary", type=Path, required=True)
    demo.add_argument("--holistic-model", type=Path, required=True)
    demo.add_argument("--holistic-model-sha256", required=True)
    demo.add_argument("--camera", type=int, default=0)
    demo.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    try:
        return _run(_parser().parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.command == "data-index":
        records = build_manifest(
            {"train": args.train_csv, "val": args.val_csv, "test": args.test_csv},
            args.video_root,
            require_files=not args.allow_missing_videos,
            expected_signer_counts=(
                None if args.skip_official_count_check else OFFICIAL_SIGNER_COUNTS
            ),
        )
        vocabulary = select_vocabulary(records, size=args.vocabulary_size)
        filtered = assign_class_ids(records, vocabulary)
        split_paths = {
            "train.csv": args.train_csv,
            "val.csv": args.val_csv,
            "test.csv": args.test_csv,
        }
        source_hashes = {name: file_sha256(path) for name, path in split_paths.items()}
        fingerprint = manifest_fingerprint(filtered)
        write_manifest(
            filtered,
            args.output_dir / "manifest.csv",
            video_root=args.video_root,
            source_hashes=source_hashes,
        )
        write_vocabulary(
            vocabulary,
            args.output_dir / "vocabulary.json",
            dataset_version="1.0",
            manifest_fingerprint_value=fingerprint,
            source_hashes=source_hashes,
        )
        print(f"indexed {len(filtered)} videos; fingerprint={manifest_fingerprint(filtered)}")
        return 0
    if args.command == "extract":
        vocabulary = read_vocabulary(args.vocabulary)
        records = read_manifest(args.manifest, args.video_root, vocabulary)
        model_hash = file_sha256(args.holistic_model)
        if model_hash != args.holistic_model_sha256.lower():
            raise ValueError("Holistic model SHA-256 does not match the approved asset")
        cache = FeatureCache(args.cache_root)
        with MediaPipeHolisticBackend(args.holistic_model) as detector:
            summary = extract_dataset(
                records,
                detector,
                lambda path, count: sampled_video_frames(path, sample_count=count),
                cache,
                sequence_length=args.sequence_length,
                extractor_identity={"holistic_model_sha256": model_hash},
            )
        report_path = args.cache_root / "extraction-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "model_asset": {
                        "filename": args.holistic_model.name,
                        "sha256": model_hash,
                    },
                    "summary": asdict(summary),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        LOGGER.info("extracted %d videos; %d failed", summary.succeeded, summary.failed)
        if summary.success_rate >= 0.95:
            return 0
        return 1 if summary.succeeded else 2
    if args.command == "inspect":
        model_hash = file_sha256(args.holistic_model)
        if model_hash != args.holistic_model_sha256.lower():
            raise ValueError("Holistic model SHA-256 does not match the approved asset")
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required for diagnostic output") from error
        sampled = sampled_video_frames(args.video, sample_count=args.frames)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with MediaPipeHolisticBackend(args.holistic_model) as detector:
            for index, (image, timestamp_ms) in enumerate(sampled.frames):
                landmarks = detector.detect(image, timestamp_ms)
                overlay = draw_landmarks(image, landmarks)
                output = args.output_dir / f"frame-{index:03d}.jpg"
                if not cv2.imwrite(str(output), overlay):
                    raise OSError(f"unable to write diagnostic frame {index}")
        LOGGER.info("wrote %d diagnostic frames", len(sampled.frames))
        return 0
    if args.command == "train":
        return _train(args)
    if args.command == "demo":
        return _demo(args)
    raise AssertionError(f"unhandled command: {args.command}")


def _train(args: argparse.Namespace) -> int:
    vocabulary = read_vocabulary(args.vocabulary)
    records = read_manifest(args.manifest, args.video_root, vocabulary)
    cache_keys, preprocessing_identity = discover_cache_index(
        args.cache_root,
        {record.video_id for record in records},
        {record.video_id: record.path for record in records},
    )
    usable_records = [record for record in records if record.video_id in cache_keys]
    examples = load_cached_examples(usable_records, args.cache_root, cache_keys)
    paired = list(zip(usable_records, examples, strict=True))
    split_examples = {
        split: [example for record, example in paired if record.split == split]
        for split in ("train", "val", "test")
    }
    if any(not values for values in split_examples.values()):
        raise ValueError("train, val, and test feature sets must all be non-empty")
    expected_classes = set(range(len(vocabulary.labels)))
    for split, split_values in split_examples.items():
        observed_classes = {example.class_id for example in split_values}
        if observed_classes != expected_classes:
            missing = ", ".join(
                vocabulary.labels[class_id]
                for class_id in sorted(expected_classes - observed_classes)
            )
            raise ValueError(f"{split} feature set is missing vocabulary classes: {missing}")
    train_examples = split_examples["train"]
    validation_examples = split_examples["val"]
    test_examples = split_examples["test"]
    train_labels = np.asarray([example.class_id for example in train_examples], dtype=np.int64)

    majority = MajorityBaseline().fit(train_labels)
    logistic = LogisticBaseline(random_state=args.seed).fit(
        [example.sequence for example in train_examples], train_labels
    )
    majority_validation = majority.predict(len(validation_examples))
    logistic_validation = logistic.predict(
        [example.sequence for example in validation_examples]
    )
    majority_test = majority.predict(len(test_examples))
    logistic_test = logistic.predict([example.sequence for example in test_examples])

    model = LandmarkGRU(GRUConfig(num_classes=len(vocabulary.labels)))
    history = train_model(
        model,
        train_examples,
        validation_examples,
        TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            device=args.device,
        ),
    )
    validation = evaluate(
        model, validation_examples, batch_size=args.batch_size, device=args.device
    )
    test = evaluate(model, test_examples, batch_size=args.batch_size, device=args.device)
    preprocessing_hash = preprocessing_fingerprint(preprocessing_identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.output_dir / "model.pt"
    save_bundle(
        bundle_path,
        model,
        labels=vocabulary.labels,
        preprocessing_hash=preprocessing_hash,
        dataset_hash=vocabulary.manifest_fingerprint,
    )
    report = {
        "seed": args.seed,
        "dataset_hash": vocabulary.manifest_fingerprint,
        "preprocessing_hash": preprocessing_hash,
        "best_epoch": history.best_epoch,
        "training": asdict(history),
        "validation": {
            "majority": _prediction_metrics(
                validation_examples, majority_validation, vocabulary.labels
            ),
            "logistic": _prediction_metrics(
                validation_examples, logistic_validation, vocabulary.labels
            ),
            "gru": {
                "loss": validation.loss,
                **_prediction_metrics(
                    validation_examples,
                    np.asarray(validation.predictions, dtype=np.int64),
                    vocabulary.labels,
                ),
            },
        },
        "test": {
            "majority": _prediction_metrics(
                test_examples, majority_test, vocabulary.labels
            ),
            "logistic": _prediction_metrics(
                test_examples, logistic_test, vocabulary.labels
            ),
            "gru": {
                "loss": test.loss,
                **_prediction_metrics(
                    test_examples,
                    np.asarray(test.predictions, dtype=np.int64),
                    vocabulary.labels,
                ),
            },
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_size_bytes": bundle_path.stat().st_size,
    }
    (args.output_dir / "training-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("trained GRU; validation macro-F1 %.3f", validation.macro_f1)
    return 0


def _prediction_metrics(
    examples: list[SequenceExample], predictions: np.ndarray, labels: tuple[str, ...]
) -> dict[str, object]:
    true_ids = [example.class_id for example in examples]
    report = classification_report(
        [labels[class_id] for class_id in true_ids],
        [labels[int(class_id)] for class_id in predictions],
        labels,
        [example.signer_id for example in examples],
    )
    return {
        "accuracy": report.accuracy,
        "macro_precision": report.macro_precision,
        "macro_recall": report.macro_recall,
        "macro_f1": report.macro_f1,
        "confusion_matrix": report.confusion_matrix.tolist(),
        "per_sign_accuracy": report.per_sign_accuracy,
        "per_signer_accuracy": report.per_signer_accuracy,
    }


def _demo(args: argparse.Namespace) -> int:
    from .app import (
        OpenCVCapture,
        OpenCVDisplay,
        PerformanceSnapshot,
        WebcamRuntime,
        build_benchmark,
    )

    vocabulary = read_vocabulary(args.vocabulary)
    model_hash = file_sha256(args.holistic_model)
    if model_hash != args.holistic_model_sha256.lower():
        raise ValueError("Holistic model SHA-256 does not match the approved asset")
    identity = {
        "holistic_model_sha256": model_hash,
        "sequence_length": 32,
        "preprocessing_version": PREPROCESSING_VERSION,
        "minimum_decode_rate": 0.95,
    }
    model, metadata = load_bundle(
        args.bundle,
        expected_labels=vocabulary.labels,
        expected_preprocessing_hash=preprocessing_fingerprint(identity),
        expected_dataset_hash=vocabulary.manifest_fingerprint,
    )
    predictor = GRUPredictor(model, metadata.labels)
    recognizer = StreamingRecognizer(predictor, StreamingConfig())
    runtime = WebcamRuntime(
        OpenCVCapture(args.camera),
        OpenCVDisplay(),
        MediaPipeHolisticBackend(args.holistic_model),
        recognizer,
    )
    status = runtime.run()
    if args.report is not None:
        performance = PerformanceSnapshot(
            frame_count=status.frame_count,
            fps=status.fps,
            cpu_percent=status.cpu_percent,
            extraction_ms=status.extraction_ms,
            inference_ms=status.inference_ms,
            end_to_end_ms=status.end_to_end_ms,
        )
        benchmark = build_benchmark(performance, args.bundle)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(asdict(benchmark), indent=2) + "\n", encoding="utf-8")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
