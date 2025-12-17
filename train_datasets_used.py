"""
Train a YOLO model (Ultralytics) using the prepared datasets_used split.

Workflow:
  1) python prepare_datasets_used.py --clear-out --create-empty-labels
  2) python train_datasets_used.py --epochs 100 --imgsz 640
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

# Avoid extra network calls (e.g., AMP checks) in some environments
os.environ.setdefault("YOLO_OFFLINE", "1")

from ultralytics import YOLO


def train(
    *,
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
    workers: int,
    seed: int,
    amp: bool,
) -> None:
    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        workers=workers,
        seed=seed,
        val=True,
        plots=True,
        exist_ok=True,
        amp=amp,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("datasets_used_yolo/dataset.yaml"))
    # Note: repo root contains a small placeholder yolo11n.pt; prefer the larger one under ./ultralytics/
    p.add_argument("--model", type=str, default="ultralytics/yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0", help="'0' for GPU0, 'cpu' for CPU")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="datasets_used_train")
    p.add_argument("--workers", type=int, default=0, help="Windows often works best with 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Enable AMP (mixed precision)")
    args = p.parse_args(argv)

    if not args.data.is_file():
        raise FileNotFoundError(f"dataset.yaml not found: {args.data} (run prepare_datasets_used.py first)")

    train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        amp=args.amp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
