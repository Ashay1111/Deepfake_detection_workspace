#!/usr/bin/env python3
"""Detect and crop faces from extracted frames using MediaPipe."""

import argparse
from pathlib import Path
import csv

import cv2
import mediapipe as mp


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def crop_largest_face(image, detections, margin=0.2):
    h, w = image.shape[:2]
    best = None
    best_area = 0

    for det in detections:
        box = det.location_data.relative_bounding_box
        x = int(box.xmin * w)
        y = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        # Expand by margin
        mx = int(bw * margin)
        my = int(bh * margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + bw + mx)
        y2 = min(h, y + bh + my)

        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)

    if best is None:
        return None

    x1, y1, x2, y2 = best
    return image[y1:y2, x1:x2]


def iter_frames(frames_root: Path):
    for p in frames_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def main():
    parser = argparse.ArgumentParser(description="Crop faces from frames using MediaPipe.")
    parser.add_argument("--frames-root", required=True, help="Root folder with extracted frames")
    parser.add_argument("--faces-root", default="data/ffpp/faces", help="Output root for face crops")
    parser.add_argument("--min-conf", type=float, default=0.5, help="Minimum face detection confidence")
    parser.add_argument("--margin", type=float, default=0.2, help="Crop margin as fraction")
    parser.add_argument("--max-faces", type=int, default=1, help="Max faces per frame (1 = largest face)")
    parser.add_argument("--log", default="data/ffpp/faces/face_crop_log.csv", help="CSV log for failures")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    faces_root = Path(args.faces_root)
    safe_mkdir(faces_root)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(min_detection_confidence=args.min_conf, model_selection=1)

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "status", "reason"])
        writer.writeheader()

        for frame_path in iter_frames(frames_root):
            img = cv2.imread(str(frame_path))
            if img is None:
                writer.writerow({"frame": str(frame_path), "status": "fail", "reason": "read_error"})
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if not results.detections:
                writer.writerow({"frame": str(frame_path), "status": "fail", "reason": "no_face"})
                continue

            crop = crop_largest_face(img, results.detections, margin=args.margin)
            if crop is None or crop.size == 0:
                writer.writerow({"frame": str(frame_path), "status": "fail", "reason": "crop_empty"})
                continue

            rel = frame_path.relative_to(frames_root)
            out_path = faces_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop)

            writer.writerow({"frame": str(frame_path), "status": "ok", "reason": ""})

    detector.close()
    print(f"Done. Crops saved to: {faces_root}")


if __name__ == "__main__":
    main()
