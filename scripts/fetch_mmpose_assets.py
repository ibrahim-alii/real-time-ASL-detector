"""Stage pinned official configs/checkpoints without modifying installed packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "models" / "mmpose"
TAG = "v1.3.2"
POSE = "rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth"
DETECTOR = "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth"


def download(url: str, path: Path) -> None:
    if path.is_file():
        return
    temporary = path.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(path)


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT)
    ROOT = parser.parse_args().output.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / f"mmpose-{TAG}.zip"
    download(f"https://codeload.github.com/open-mmlab/mmpose/zip/refs/tags/{TAG}", archive)
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            target = (ROOT / item.filename).resolve()
            if not target.is_relative_to(ROOT.resolve()):
                raise ValueError("archive member escapes asset directory")
        bundle.extractall(ROOT)
    source = ROOT / "mmpose-1.3.2"
    configs = {
        "pose_config": source / "projects/rtmpose/rtmpose/wholebody_2d_keypoint/"
        "rtmpose-m_8xb64-270e_coco-wholebody-256x192.py",
        "detector_config": source / "projects/rtmpose/rtmdet/person/"
        "rtmdet_nano_320-8xb32_coco-person.py",
    }
    for path in configs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    download("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/" + POSE, ROOT / POSE)
    download(
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/" + DETECTOR, ROOT / DETECTOR
    )
    assets = {**configs, "pose_weights": ROOT / POSE, "detector_weights": ROOT / DETECTOR}
    (ROOT / "assets.json").write_text(
        json.dumps(
            {
                "tag": TAG,
                "assets": {name: str(path) for name, path in assets.items()},
                "sha256": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in assets.items()
                },
                "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(ROOT / "assets.json")


if __name__ == "__main__":
    main()
