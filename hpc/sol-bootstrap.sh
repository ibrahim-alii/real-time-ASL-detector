#!/bin/bash
# Executed on a compute node; the legacy MediaPipe environments remain untouched.
set -euo pipefail
: "${ASL_CODE_ROOT:?}"; : "${ASL_RUNTIME_ROOT:?}"
if ! command -v python3.11 >/dev/null || ! command -v ffmpeg >/dev/null || ! command -v node >/dev/null; then
    # Sol's supported mamba module supplies conda; keep both prefixes in scratch.
    # Site login scripts reference interactive-only variables such as HISTCONTROL.
    set +u
    source /etc/profile
    module load mamba
    set -u
    if [[ ! -x "$ASL_RUNTIME_ROOT/python/bin/python" ]]; then
        conda create -y -c conda-forge -p "$ASL_RUNTIME_ROOT/python" python=3.11 ffmpeg=7 nodejs=22
    fi
    export PATH="$ASL_RUNTIME_ROOT/python/bin:$PATH"
fi
command -v ffmpeg
command -v ffprobe
command -v node
base_python="$(command -v python3.11 || command -v python)"
mkdir -p "$ASL_RUNTIME_ROOT"
for role in extract train; do
    [[ -x "$ASL_RUNTIME_ROOT/$role/bin/python" ]] || "$base_python" -m venv "$ASL_RUNTIME_ROOT/$role"
done
extract_python="$ASL_RUNTIME_ROOT/extract/bin/python"
train_python="$ASL_RUNTIME_ROOT/train/bin/python"
"$extract_python" -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
"$extract_python" -m pip install --only-binary=:all: mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
"$extract_python" -m pip install setuptools==75.8.0 wheel==0.45.1
"$extract_python" -m pip install --no-build-isolation -r "$ASL_CODE_ROOT/requirements/mmpose-local.txt"
"$train_python" -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
"$train_python" -m pip install numpy==2.2.6 opencv-python-headless==4.11.0.86 scikit-learn==1.6.1 transformers==4.49.0 sentencepiece==0.2.0 sacrebleu==2.5.1 gdown==5.2.0 'yt-dlp[default]==2026.8.19' pytest ruff mypy
"$extract_python" -m pip check
"$train_python" -m pip check
"$extract_python" -m pip freeze > "$ASL_RUNTIME_ROOT/extract-freeze.txt"
"$train_python" -m pip freeze > "$ASL_RUNTIME_ROOT/train-freeze.txt"
