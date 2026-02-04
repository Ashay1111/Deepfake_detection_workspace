#!/usr/bin/env python3
"""Extract frames for a split CSV produced by ffpp_split.py."""

import argparse
from pathlib import Path
import pandas as pd
import cv2


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, out_dir: Path, every_n: int, max_frames: int, resize: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open: {video_path}")
        return 0

    saved = 0
    idx = 0

    safe_mkdir(out_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            if resize:
                frame = cv2.resize(frame, (resize, resize))
            out_path = out_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames for a given split CSV.")
    parser.add_argument("--split", required=True, help="Path to split CSV (train/val/test)")
    parser.add_argument("--frames-root", default="data/ffpp/frames", help="Output frames root")
    parser.add_argument("--every-n", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--resize", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.split)
    frames_root = Path(args.frames_root)
    resize = args.resize if args.resize > 0 else None

    for _, row in df.iterrows():
        video_path = Path(row["path"])
        split_name = Path(args.split).stem
        out_dir = frames_root / split_name / row["compression"] / row["label"] / row["video_id"]
        saved = extract_frames(video_path, out_dir, args.every_n, args.max_frames, resize)
        print(f"{video_path.name}: saved {saved}")


if __name__ == "__main__":
    main()
