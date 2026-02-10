#!/usr/bin/env python3
"""Run video-level inference on a single video."""

import argparse
import json
from pathlib import Path

import cv2
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


def main():
    parser = argparse.ArgumentParser(description="Video-level inference with tuned threshold.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model-path", default=str(base_dir / "artifacts" / "expert_c23.pth"))
    parser.add_argument("--threshold-json", default=str(base_dir / "artifacts" / "threshold.json"))
    parser.add_argument("--every-n", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--real-threshold", type=float, default=None, help="Max fake prob to call REAL.")
    parser.add_argument("--fake-threshold", type=float, default=None, help="Min fake prob to call FAKE.")
    parser.add_argument("--margin", type=float, default=0.2, help="Face crop margin.")
    parser.add_argument("--min-face", type=int, default=40, help="Min face size for Haar detector.")
    args = parser.parse_args()

    threshold = 0.5
    try:
        with open(args.threshold_json, "r") as f:
            threshold = float(json.load(f).get("threshold", 0.5))
    except Exception:
        pass
    fake_threshold = threshold if args.fake_threshold is None else args.fake_threshold
    real_threshold = args.real_threshold
    if real_threshold is None:
        try:
            real_path = Path(args.threshold_json).with_name("real_threshold.json")
            with open(real_path, "r") as f:
                real_threshold = float(json.load(f).get("real_threshold", 0.01))
        except Exception:
            real_threshold = min(0.01, fake_threshold * 0.5)
    if args.fake_threshold is None:
        try:
            fake_path = Path(args.threshold_json).with_name("fake_threshold.json")
            with open(fake_path, "r") as f:
                fake_threshold = float(json.load(f).get("fake_threshold", fake_threshold))
        except Exception:
            pass
    if real_threshold >= fake_threshold:
        fake_threshold = min(1.0, max(real_threshold + 0.05, fake_threshold))

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

    video_path = Path(args.video)
    frames = sample_frames(video_path, every_n=args.every_n, max_frames=args.max_frames)
    if not frames:
        raise SystemExit("No frames read from video.")

    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    probs = []
    with torch.no_grad():
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(args.min_face, args.min_face))
            if len(faces) == 0:
                continue
            crop = crop_largest_face(frame, faces, margin=args.margin)
            if crop is None or crop.size == 0:
                continue
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            x = tfms(img).unsqueeze(0)
            logits = model(x).squeeze(1)
            probs.append(torch.sigmoid(logits).item())

    if not probs:
        raise SystemExit("No faces detected in sampled frames.")

    mean_fake = sum(probs) / len(probs)
    if mean_fake <= real_threshold:
        pred = "REAL"
    elif mean_fake >= fake_threshold:
        pred = "FAKE"
    else:
        pred = "UNCERTAIN"

    print("Inference result")
    print(f"video: {video_path}")
    print(f"frames_used: {len(probs)}")
    print(f"mean_fake_prob: {mean_fake:.4f}")
    print(f"real_threshold: {real_threshold:.2f}")
    print(f"fake_threshold: {fake_threshold:.2f}")
    print(f"prediction: {pred}")


if __name__ == "__main__":
    main()
