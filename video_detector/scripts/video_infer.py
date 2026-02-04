#!/usr/bin/env python3
"""Run frame-level inference for videos and save per-video scores."""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def load_model(model_path: Path, device):
    ckpt = torch.load(model_path, map_location=device)
    model_name = ckpt["model"]
    input_size = ckpt["input_size"]
    model, _ = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    return model, input_size


def get_transforms(input_size: int):
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def iter_video_dirs(frame_root: Path):
    for p in frame_root.rglob("*"):
        if p.is_dir():
            # Only consider directories that contain frames
            if any(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in p.iterdir() if f.is_file()):
                yield p


def infer_video_dir(model, tfms, video_dir: Path, device):
    frames = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not frames:
        return None

    probs = []
    with torch.no_grad():
        for frame in frames:
            img = Image.open(frame).convert("RGB")
            x = tfms(img).unsqueeze(0).to(device)
            logits = model(x).squeeze(1)
            prob_fake = torch.sigmoid(logits).item()
            probs.append(prob_fake)

    return {
        "video_id": str(video_dir),
        "num_frames": len(frames),
        "mean_fake": sum(probs) / len(probs),
        "max_fake": max(probs),
        "min_fake": min(probs),
    }


def main():
    parser = argparse.ArgumentParser(description="Run frame-level inference on extracted frames.")
    parser.add_argument("--frames", required=True, help="Root folder with extracted frames per video.")
    parser.add_argument("--model-path", required=True, help="Path to trained model .pth")
    parser.add_argument("--out", default="artifacts/video_scores.csv", help="CSV output path")
    args = parser.parse_args()

    frames_root = Path(args.frames)
    if not frames_root.exists():
        raise SystemExit(f"Frames folder not found: {frames_root}")

    device = get_device()
    model, input_size = load_model(Path(args.model_path), device)
    tfms = get_transforms(input_size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "num_frames", "mean_fake", "max_fake", "min_fake"])
        writer.writeheader()
        for video_dir in iter_video_dirs(frames_root):
            row = infer_video_dir(model, tfms, video_dir, device)
            if row:
                writer.writerow(row)
                print(f"Scored: {video_dir}")

    print(f"Saved scores to: {out_path}")


if __name__ == "__main__":
    main()
