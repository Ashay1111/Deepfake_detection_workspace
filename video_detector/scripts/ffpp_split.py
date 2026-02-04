#!/usr/bin/env python3
"""Create train/val/test splits by video_id from FF++ index CSV."""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_group(df, train_ratio, val_ratio, seed):
    # Split by unique video_id to avoid leakage
    video_ids = df["video_id"].unique()
    train_ids, temp_ids = train_test_split(
        video_ids,
        test_size=1.0 - train_ratio,
        random_state=seed,
        shuffle=True,
    )
    # Now split temp into val/test
    val_size = val_ratio / (1.0 - train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1.0 - val_size,
        random_state=seed,
        shuffle=True,
    )
    return set(train_ids), set(val_ids), set(test_ids)


def main():
    parser = argparse.ArgumentParser(description="Split FF++ index into train/val/test by video_id.")
    parser.add_argument("--index", required=True, help="Path to index.csv")
    parser.add_argument("--out-dir", default="data/ffpp/splits", help="Output directory for splits")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train + args.val >= 1.0:
        raise SystemExit("Train + val must be < 1.0")

    df = pd.read_csv(args.index)

    # Stratify splits by (label, compression) to keep balance
    split_map = {"train": [], "val": [], "test": []}

    for (label, compression), group in df.groupby(["label", "compression"]):
        train_ids, val_ids, test_ids = split_group(group, args.train, args.val, args.seed)
        split_map["train"].append(group[group["video_id"].isin(train_ids)])
        split_map["val"].append(group[group["video_id"].isin(val_ids)])
        split_map["test"].append(group[group["video_id"].isin(test_ids)])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        out_df = pd.concat(split_map[split]).reset_index(drop=True)
        out_df.to_csv(out_dir / f"{split}.csv", index=False)
        print(f"{split}: {len(out_df)} rows")


if __name__ == "__main__":
    main()
