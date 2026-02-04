#!/usr/bin/env python3
"""Extract frames from videos into per-video folders."""

import argparse
import os
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def iter_videos(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def extract_frames(video_path: Path, out_dir: Path, every_n: int, max_frames: int, resize: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open: {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    frame_idx = 0

    safe_mkdir(out_dir)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            if resize:
                frame = cv2.resize(frame, (resize, resize))
            out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"{video_path.name}: saved {saved}/{total} frames -> {out_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--input", required=True, help="Input folder with videos.")
    parser.add_argument("--output", required=True, help="Output folder for extracted frames.")
    parser.add_argument("--every-n", type=int, default=5, help="Save every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames per video (0 = no limit).")
    parser.add_argument("--resize", type=int, default=0, help="Resize to square (0 = no resize).")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise SystemExit(f"Input folder not found: {input_root}")

    resize = args.resize if args.resize > 0 else None
    max_frames = args.max_frames if args.max_frames > 0 else 0

    for video in iter_videos(input_root):
        rel = video.relative_to(input_root)
        out_dir = output_root / rel.parent / rel.stem
        extract_frames(video, out_dir, args.every_n, max_frames, resize)


if __name__ == "__main__":
    main()
