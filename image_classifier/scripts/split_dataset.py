#!/usr/bin/env python3
"""Create train/val/test splits from the local Real/Fake face dataset."""

import argparse
import os
import random
import shutil
from pathlib import Path


def collect_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def split_list(items, train_ratio, val_ratio):
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def copy_files(files, dst_dir: Path):
    safe_mkdir(dst_dir)
    for src in files:
        dst = dst_dir / src.name
        # Avoid overwrite collisions by prefixing if needed
        if dst.exists():
            dst = dst_dir / f"{src.stem}_{src.stat().st_ino}{src.suffix}"
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Split real/fake images into train/val/test folders.")
    parser.add_argument("--input", required=True, help="Input dataset root with training_real/training_fake.")
    parser.add_argument("--output", required=True, help="Output root for split dataset.")
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio (default: 0.7)")
    parser.add_argument("--val", type=float, default=0.15, help="Val split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.train + args.val >= 1.0:
        raise SystemExit("Train + val ratios must be < 1.0")

    input_root = Path(args.input)
    out_root = Path(args.output)

    class_map = {
        "real": input_root / "training_real",
        "fake": input_root / "training_fake",
    }

    rng = random.Random(args.seed)

    for label, folder in class_map.items():
        if not folder.exists():
            raise SystemExit(f"Missing folder: {folder}")
        images = collect_images(folder)
        if not images:
            raise SystemExit(f"No images found in {folder}")
        rng.shuffle(images)
        train, val, test = split_list(images, args.train, args.val)

        copy_files(train, out_root / "train" / label)
        copy_files(val, out_root / "val" / label)
        copy_files(test, out_root / "test" / label)

        print(f"{label}: total={len(images)} train={len(train)} val={len(val)} test={len(test)}")

    print(f"Done. Output at: {out_root}")


if __name__ == "__main__":
    main()
