#!/usr/bin/env python3
"""Aggregate frame-level scores to video-level predictions."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Aggregate video scores from CSV.")
    parser.add_argument("--scores", required=True, help="CSV file produced by video_infer.py")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fake probability threshold")
    parser.add_argument("--out", default="artifacts/video_predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(scores_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        mean_fake = float(row["mean_fake"])
        row["prediction"] = "FAKE" if mean_fake >= args.threshold else "REAL"
        row["threshold"] = f"{args.threshold:.2f}"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
