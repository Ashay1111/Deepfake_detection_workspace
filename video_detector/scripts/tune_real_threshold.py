#!/usr/bin/env python3
"""Tune a conservative REAL threshold using only real videos from val split."""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


def build_model(name: str):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        input_size = 224
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
        input_size = 224
    else:
        raise ValueError("Unsupported model. Use resnet18 or efficientnet_b0")
    return model, input_size


def sample_frames(video_path: Path, every_n: int, max_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


def crop_largest_face(image, faces, margin=0.2):
    h, w = image.shape[:2]
    best = None
    best_area = 0
    for (x, y, bw, bh) in faces:
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


def video_prob(model, tfms, video_path: Path, every_n: int, max_frames: int, min_face: int, margin: float):
    frames = sample_frames(video_path, every_n=every_n, max_frames=max_frames)
    if not frames:
        return None
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    probs = []
    with torch.no_grad():
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face))
            if len(faces) == 0:
                continue
            crop = crop_largest_face(frame, faces, margin=margin)
            if crop is None or crop.size == 0:
                continue
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            x = tfms(img).unsqueeze(0)
            logits = model(x).squeeze(1)
            prob_fake = torch.sigmoid(logits).item()
            probs.append(prob_fake)
    if not probs:
        return None
    return sum(probs) / len(probs)


def main():
    parser = argparse.ArgumentParser(description="Tune REAL threshold using real-only videos.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--splits", default=str(base_dir / "data" / "ffpp" / "splits" / "val.csv"))
    parser.add_argument("--model-path", default=str(base_dir / "artifacts" / "expert_c23_effnetb0.pth"))
    parser.add_argument("--every-n", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--min-face", type=int, default=40)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--real-recall", type=float, default=0.9, help="Target recall for REAL class.")
    parser.add_argument("--out", default=str(base_dir / "artifacts" / "real_threshold.json"))
    args = parser.parse_args()

    ckpt = torch.load(args.model_path, map_location="cpu")
    model_name = ckpt["model"]

    model, input_size = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(args.splits)
    df = df[df["label"] == "real"]
    probs = []
    used = 0
    for _, row in df.iterrows():
        p = video_prob(model, tfms, Path(row["path"]), args.every_n, args.max_frames, args.min_face, args.margin)
        if p is None:
            continue
        probs.append(p)
        used += 1

    if not probs:
        raise SystemExit("No real video probabilities computed.")

    probs_sorted = sorted(probs)
    idx = max(0, min(len(probs_sorted) - 1, int(len(probs_sorted) * args.real_recall) - 1))
    real_threshold = probs_sorted[idx]

    summary = {
        "real_threshold": real_threshold,
        "real_recall_target": args.real_recall,
        "num_real_videos": used,
        "min_prob": min(probs),
        "max_prob": max(probs),
        "median_prob": probs_sorted[len(probs_sorted) // 2],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Real-only threshold summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
