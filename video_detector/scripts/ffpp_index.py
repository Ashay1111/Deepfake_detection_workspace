#!/usr/bin/env python3
"""Index FF++ videos into a CSV catalog.

Expected raw structure under data/ffpp/raw:
- original_sequences/.../c23/videos/*.mp4
- original_sequences/.../c40/videos/*.mp4
- manipulated_sequences/<method>/c23/videos/*.mp4
- manipulated_sequences/<method>/c40/videos/*.mp4
"""

import argparse
from pathlib import Path
import csv


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def infer_meta(path: Path):
    parts = [p.lower() for p in path.parts]
    compression = "c23" if "c23" in parts else "c40" if "c40" in parts else "unknown"

    if "original_sequences" in parts:
        label = "real"
        method = "original"
    elif "manipulated_sequences" in parts:
        label = "fake"
        try:
            idx = parts.index("manipulated_sequences")
            method = path.parts[idx + 1]
        except Exception:
            method = "unknown"
    else:
        label = "unknown"
        method = "unknown"

    video_id = path.stem
    return label, method, compression, video_id


def main():
    parser = argparse.ArgumentParser(description="Index FF++ videos into CSV.")
    parser.add_argument("--raw", required=True, help="Path to FF++ raw root.")
    parser.add_argument("--out", default="data/ffpp/index.csv", help="Output CSV path.")
    args = parser.parse_args()

    raw_root = Path(args.raw)
    if not raw_root.exists():
        raise SystemExit(f"Raw root not found: {raw_root}")

    rows = []
    for p in raw_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            label, method, compression, video_id = infer_meta(p)
            rows.append({
                "video_id": video_id,
                "label": label,
                "method": method,
                "compression": compression,
                "path": str(p),
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "label", "method", "compression", "path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Indexed {len(rows)} videos -> {out_path}")


if __name__ == "__main__":
    main()
