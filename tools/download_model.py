"""
Pre-downloads the YOLO weights file so it's cached locally before your
first real run or demo -- otherwise Ultralytics downloads it automatically
the first time main.py runs, which can be slow (or embarrassing) live.

Usage:
    python tools/download_model.py
    python tools/download_model.py --model yolov8s.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    print(f"Downloading/caching {args.model} via Ultralytics...")
    model = YOLO(args.model)  # triggers the download if not already cached

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / args.model

    ckpt_path = Path(getattr(model, "ckpt_path", "")) if getattr(model, "ckpt_path", None) else None
    if ckpt_path and ckpt_path.exists():
        shutil.copy(ckpt_path, dest)
        print(f"Copied weights to {dest}")
    else:
        print("Model downloaded and cached by Ultralytics. "
              "Point src/config.py's MODEL_PATH at it directly if needed.")


if __name__ == "__main__":
    main()
