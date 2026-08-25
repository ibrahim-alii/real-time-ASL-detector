# Real-Time ASL Recognizer

Laptop-first MVP for signer-independent isolated-sign training and stabilized webcam recognition
using MediaPipe hand/pose landmarks and a compact GRU.

## Architecture

```text
ASL Citizen CSV/video -> validated signer splits -> normalized landmark cache
                     -> baselines + GRU -> validated model bundle
Camera -> MediaPipe -> rolling buffer -> GRU -> activity/confidence gates -> display
```

Raw videos, extracted features, and model artifacts stay local and are ignored by Git.

## Setup

Python 3.11-3.13 is supported.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

Download ASL Citizen v1.0 and its official split CSVs from Microsoft Research. Also acquire a
MediaPipe Holistic Task model and calculate its SHA-256 with `Get-FileHash`. Do not commit either
asset. The CLI requires the expected hash so an accidental model replacement cannot silently alter
features.

## Pipeline

```powershell
asl-recognizer data-index `
  --train-csv data/raw/asl-citizen/train.csv `
  --val-csv data/raw/asl-citizen/val.csv `
  --test-csv data/raw/asl-citizen/test.csv `
  --video-root data/raw/asl-citizen/videos `
  --output-dir artifacts/index

asl-recognizer extract `
  --manifest artifacts/index/manifest.csv `
  --vocabulary artifacts/index/vocabulary.json `
  --video-root data/raw/asl-citizen/videos `
  --cache-root artifacts/features `
  --holistic-model models/holistic_landmarker.task `
  --holistic-model-sha256 <SHA256>

asl-recognizer train `
  --manifest artifacts/index/manifest.csv `
  --vocabulary artifacts/index/vocabulary.json `
  --video-root data/raw/asl-citizen/videos `
  --cache-root artifacts/features `
  --output-dir artifacts/training

asl-recognizer demo `
  --bundle artifacts/training/model.pt `
  --vocabulary artifacts/index/vocabulary.json `
  --holistic-model models/holistic_landmarker.task `
  --holistic-model-sha256 <SHA256> `
  --report artifacts/webcam-benchmark.json
```

Press Space to start or stop recognition and Escape to exit. The demo validates the label,
dataset, feature, and preprocessing contracts before inference.

## Current Limits

- The repository does not include ASL Citizen, a Holistic model asset, or trained weights.
- Continuous thresholds need calibration on separate recorded streams before accuracy claims.
- No learned background class is available from the isolated-sign dataset; the MVP uses hand,
  motion, confidence, hysteresis, and cooldown gates for the no-sign state.
- Transformer, personalization, ONNX, and quantization are deferred until GRU measurements justify
  them. Raspberry Pi deployment should begin with a 64-bit Python environment, CPU PyTorch, a
  supported camera backend, and a benchmark of the unchanged bundle before optimization.

Implementation details and remaining gates are recorded in `docs/stages/`.
