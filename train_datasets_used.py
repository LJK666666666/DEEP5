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
        # ========== 激进数据增强策略 ==========
        # 颜色空间增强
        hsv_h=0.015,       # 色调变化 (默认 0.015)
        hsv_s=0.7,        # 饱和度变化 (默认 0.7)
        hsv_v=0.4,        # 亮度变化 (默认 0.4)
        # 几何变换增强
        degrees=15.0,     # 旋转角度 ±15° (默认 0.0)
        translate=0.1,    # 平移比例 (默认 0.1)
        scale=0.5,        # 缩放比例 (默认 0.5)
        shear=5.0,        # 剪切角度 ±5° (默认 0.0)
        perspective=0.0,  # 透视变换 (默认 0.0)
        # 翻转增强
        flipud=0.0,       # 上下翻转概率 (默认 0.0)
        fliplr=0.5,       # 左右翻转概率 (默认 0.5)
        # 高级增强
        mosaic=1.0,       # Mosaic 增强 (默认 1.0)
        mixup=0.05,        # MixUp 增强 (默认 0.0)
        copy_paste=0.0,   # 复制粘贴增强 (默认 0.0)
        erasing=0.5,      # 随机擦除概率 (默认 0.4)
        # auto_augment="randaugment",  # 自动增强策略
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("datasets_used_yolo/dataset.yaml"))
    # Note: repo root contains a small placeholder yolo11n.pt; prefer the larger one under ./ultralytics/
    p.add_argument("--model", type=str, default="ultralytics/yolo11m.pt")
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
