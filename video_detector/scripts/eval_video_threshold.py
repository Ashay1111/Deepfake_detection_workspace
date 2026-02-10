#!/usr/bin/env python3
"""Evaluate video-level performance on the test split using a tuned threshold."""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
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


def video_prob(model, tfms, video_path: Path, every_n: int, max_frames: int):
    frames = sample_frames(video_path, every_n=every_n, max_frames=max_frames)
    if not frames:
        return None
    probs = []
    with torch.no_grad():
        for frame in frames:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            x = tfms(img).unsqueeze(0)
            logits = model(x).squeeze(1)
            prob_fake = torch.sigmoid(logits).item()
            probs.append(prob_fake)
    return sum(probs) / len(probs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate video-level test metrics with tuned threshold.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--splits", default=str(base_dir / "data" / "ffpp" / "splits" / "test.csv"))
    parser.add_argument("--model-path", default=str(base_dir / "artifacts" / "expert_c23.pth"))
    parser.add_argument("--threshold-json", default=str(base_dir / "artifacts" / "threshold.json"))
    parser.add_argument("--every-n", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--out", default=str(base_dir / "artifacts" / "video_test_metrics.json"))
    args = parser.parse_args()

    with open(args.threshold_json, "r") as f:
        threshold_data = json.load(f)
    threshold = float(threshold_data.get("threshold", 0.5))

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
    probs = []
    targets = []
    for _, row in df.iterrows():
        p = video_prob(model, tfms, Path(row["path"]), args.every_n, args.max_frames)
        if p is None:
            continue
        probs.append(p)
        targets.append(1 if row["label"] == "fake" else 0)

    if not probs:
        raise SystemExit("No video probabilities computed. Check paths.")

    preds = [1 if p >= threshold else 0 for p in probs]
    acc = accuracy_score(targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(targets, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(targets, probs)
    except ValueError:
        auc = 0.0

    metrics = {
        "threshold": threshold,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "num_videos": len(probs),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
