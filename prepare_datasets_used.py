"""
Prepare a unified YOLO dataset from ./datasets_used (multiple sub-datasets),
split into train/val, and generate a dataset YAML for Ultralytics YOLO.

Expected input layout (per sub-dataset):
  datasets_used/<subset>/image/*.jpg|png|...
  datasets_used/<subset>/label/*.txt   (YOLO txt labels)

Output layout:
  <out_dir>/
    images/train, images/val
    labels/train, labels/val
    dataset.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    subset: str
    image_path: Path
    label_path: Optional[Path]
    new_stem: str  # used for both image and label filenames (without extension)


def _iter_subsets(datasets_used_dir: Path) -> List[Path]:
    if not datasets_used_dir.exists():
        raise FileNotFoundError(f"Not found: {datasets_used_dir}")
    subsets = [p for p in datasets_used_dir.iterdir() if p.is_dir()]
    subsets.sort(key=lambda p: p.name)
    return subsets


def _collect_samples(datasets_used_dir: Path) -> List[Sample]:
    samples: List[Sample] = []
    for subset_dir in _iter_subsets(datasets_used_dir):
        img_dir = subset_dir / "image"
        lbl_dir = subset_dir / "label"
        if not img_dir.is_dir():
            continue

        subset = subset_dir.name
        images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        images.sort(key=lambda p: p.name)
        for img_path in images:
            stem = img_path.stem
            label_path = (lbl_dir / f"{stem}.txt") if lbl_dir.is_dir() else None
            if label_path is not None and not label_path.is_file():
                label_path = None
            new_stem = f"{subset}__{stem}"
            samples.append(Sample(subset=subset, image_path=img_path, label_path=label_path, new_stem=new_stem))
    return samples


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_or_link(src: Path, dst: Path, mode: str) -> None:
    _safe_mkdir(dst.parent)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            # hardlink can fail across volumes or due to permissions; fall back to copy
            shutil.copy2(src, dst)
        return
    raise ValueError(f"Unknown mode: {mode}")


_NAMES_KV_RE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")


def _load_names_from_yaml(yaml_path: Path) -> Optional[Dict[int, str]]:
    if not yaml_path.is_file():
        return None
    lines = yaml_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_names = False
    names: Dict[int, str] = {}
    for line in lines:
        if not in_names:
            if line.strip() == "names:":
                in_names = True
            continue
        if not line.startswith((" ", "\t")):
            break
        m = _NAMES_KV_RE.match(line)
        if m:
            idx = int(m.group(1))
            val = m.group(2).strip().strip("'").strip('"')
            names[idx] = val
    if not names:
        return None
    return dict(sorted(names.items(), key=lambda kv: kv[0]))


def _load_names_from_classes_txt(classes_txt: Path) -> Optional[Dict[int, str]]:
    if not classes_txt.is_file():
        return None
    names: Dict[int, str] = {}
    lines = classes_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        names[idx] = s
        idx += 1
    return names or None


def _find_first_classes_txt(datasets_used_dir: Path) -> Optional[Path]:
    # typical location: datasets_used/<subset>/label/classes.txt
    for subset_dir in _iter_subsets(datasets_used_dir):
        candidate = subset_dir / "label" / "classes.txt"
        if candidate.is_file():
            return candidate
    return None


def _parse_label_file_for_ids(label_path: Path) -> List[int]:
    ids: List[int] = []
    try:
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            first = s.split()[0]
            try:
                ids.append(int(first))
            except ValueError:
                continue
    except FileNotFoundError:
        return []
    return ids


def _write_dataset_yaml(out_dir: Path, names: Dict[int, str], *, absolute_path: bool) -> Path:
    dataset_yaml = out_dir / "dataset.yaml"
    root = out_dir.resolve() if absolute_path else out_dir
    lines = [
        "# Auto-generated dataset config for Ultralytics YOLO",
        f"path: {root.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, name in sorted(names.items(), key=lambda kv: kv[0]):
        lines.append(f"  {i}: {name}")
    dataset_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_yaml


def _split_indices(n: int, train_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    cut = int(n * train_ratio)
    return idxs[:cut], idxs[cut:]


def _split_samples(
    samples: Sequence[Sample],
    *,
    train_ratio: float,
    seed: int,
    always_train_subsets: Iterable[str],
) -> Tuple[List[Sample], List[Sample]]:
    always_train = set(always_train_subsets)
    forced_train = [s for s in samples if s.subset in always_train]
    remaining = [s for s in samples if s.subset not in always_train]
    train_idxs, val_idxs = _split_indices(len(remaining), train_ratio=train_ratio, seed=seed)
    train = forced_train + [remaining[i] for i in train_idxs]
    val = [remaining[i] for i in val_idxs]
    train.sort(key=lambda s: s.new_stem)
    val.sort(key=lambda s: s.new_stem)
    return train, val


def prepare(
    datasets_used_dir: Path,
    out_dir: Path,
    *,
    train_ratio: float,
    seed: int,
    mode: str,
    create_empty_labels: bool,
    names: Dict[int, str],
    absolute_yaml_path: bool,
    always_train_subsets: Iterable[str],
    dry_run: bool,
) -> Path:
    samples = _collect_samples(datasets_used_dir)
    if not samples:
        raise RuntimeError(f"No images found under: {datasets_used_dir}")

    train_samples, val_samples = _split_samples(
        samples,
        train_ratio=train_ratio,
        seed=seed,
        always_train_subsets=always_train_subsets,
    )

    out_images_train = out_dir / "images" / "train"
    out_images_val = out_dir / "images" / "val"
    out_labels_train = out_dir / "labels" / "train"
    out_labels_val = out_dir / "labels" / "val"

    if dry_run:
        print(f"Found samples: {len(samples)}")
        print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")
        always_train = set(always_train_subsets)
        if always_train:
            forced = sum(1 for s in samples if s.subset in always_train)
            print(f"Always-train subsets: {sorted(always_train)} (forced train samples: {forced})")
        print(f"Output: {out_dir}")
        return out_dir / "dataset.yaml"

    _safe_mkdir(out_images_train)
    _safe_mkdir(out_images_val)
    _safe_mkdir(out_labels_train)
    _safe_mkdir(out_labels_val)

    seen_class_ids: List[int] = []

    def emit(sample: Sample, split: str) -> None:
        if split == "train":
            img_dst_dir = out_images_train
            lbl_dst_dir = out_labels_train
        else:
            img_dst_dir = out_images_val
            lbl_dst_dir = out_labels_val

        img_dst = img_dst_dir / f"{sample.new_stem}{sample.image_path.suffix.lower()}"
        _copy_or_link(sample.image_path, img_dst, mode=mode)

        lbl_dst = lbl_dst_dir / f"{sample.new_stem}.txt"
        if sample.label_path is not None:
            _copy_or_link(sample.label_path, lbl_dst, mode="copy")  # labels are tiny; always copy
            seen_class_ids.extend(_parse_label_file_for_ids(sample.label_path))
        elif create_empty_labels:
            lbl_dst.write_text("", encoding="utf-8")

    for s in train_samples:
        emit(s, "train")
    for s in val_samples:
        emit(s, "val")

    if seen_class_ids:
        max_id = max(seen_class_ids)
        if max_id >= len(names):
            raise ValueError(
                f"Label contains class id {max_id}, but dataset.yaml only defines nc={len(names)} classes. "
                "Pass --class-names or --names-from-yaml to match your dataset."
            )

    dataset_yaml = _write_dataset_yaml(out_dir, names, absolute_path=absolute_yaml_path)
    return dataset_yaml


def _parse_class_names_arg(s: str) -> Dict[int, str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return {i: name for i, name in enumerate(parts)}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets-used", type=Path, default=Path("datasets_used"), help="Input dir (default: datasets_used)")
    p.add_argument("--out", type=Path, default=Path("datasets_used_yolo"), help="Output dir (default: datasets_used_yolo)")
    p.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio (default: 0.9)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument("--mode", choices=["copy", "hardlink"], default="copy", help="How to place images (default: copy)")
    p.add_argument(
        "--create-empty-labels",
        action="store_true",
        help="Create empty .txt labels for images without labels",
    )
    p.add_argument(
        "--classes-txt",
        type=Path,
        default=Path("datasets_used/0last/label/classes.txt"),
        help="Read class names from classes.txt (default: datasets_used/0last/label/classes.txt)",
    )
    p.add_argument(
        "--names-from-yaml",
        type=Path,
        default=None,
        help="Read class names from an existing dataset.yaml (fallback option)",
    )
    p.add_argument(
        "--class-names",
        type=str,
        default="",
        help="Comma-separated class names, e.g. 'book,spoon,bowl,...' (overrides other sources)",
    )
    p.add_argument(
        "--always-train-subsets",
        type=str,
        default="0last",
        help="Comma-separated subset names that must be entirely in train (default: 0last)",
    )
    p.add_argument("--absolute-yaml-path", action="store_true", help="Write absolute 'path:' in dataset.yaml")
    p.add_argument("--clear-out", action="store_true", help="Delete output dir first (DANGEROUS)")
    p.add_argument("--dry-run", action="store_true", help="Only print counts, do not write files")
    args = p.parse_args(argv)

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")

    if args.clear_out and args.out.exists():
        shutil.rmtree(args.out)

    if args.class_names.strip():
        names = _parse_class_names_arg(args.class_names)
    else:
        loaded = _load_names_from_classes_txt(args.classes_txt)
        if loaded is None:
            auto = _find_first_classes_txt(args.datasets_used)
            loaded = _load_names_from_classes_txt(auto) if auto else None
        if loaded is None and args.names_from_yaml is not None:
            loaded = _load_names_from_yaml(args.names_from_yaml)
        if loaded is None:
            raise FileNotFoundError(
                "Could not load class names. Provide --class-names, or ensure a valid "
                "--classes-txt (e.g. datasets_used/<subset>/label/classes.txt), or pass --names-from-yaml."
            )
        names = loaded

    always_train_subsets = [s.strip() for s in args.always_train_subsets.split(",") if s.strip()]

    dataset_yaml = prepare(
        args.datasets_used,
        args.out,
        train_ratio=args.train_ratio,
        seed=args.seed,
        mode=args.mode,
        create_empty_labels=args.create_empty_labels,
        names=names,
        absolute_yaml_path=args.absolute_yaml_path,
        always_train_subsets=always_train_subsets,
        dry_run=args.dry_run,
    )
    print(f"OK: {dataset_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
