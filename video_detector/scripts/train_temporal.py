#!/usr/bin/env python3
"""Train a temporal model (CNN + LSTM) on face crops."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_backbone(name: str, pretrained: bool):
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError("Unsupported backbone. Use resnet18 or efficientnet_b0")
    return backbone, feat_dim


def list_images(root: Path):
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


class FaceSequenceDataset(Dataset):
    def __init__(self, faces_root: Path, split_csv: Path, split_name: str, seq_len: int, transform=None):
        import pandas as pd
        self.faces_root = faces_root
        self.seq_len = seq_len
        self.transform = transform
        self.split_name = split_name
        self.rows = pd.read_csv(split_csv)

    def __len__(self):
        return len(self.rows)

    def _sample_sequence(self, frames):
        if len(frames) == 0:
            return []
        if len(frames) >= self.seq_len:
            idxs = np.linspace(0, len(frames) - 1, self.seq_len).astype(int).tolist()
            return [frames[i] for i in idxs]
        # Pad by repeating last frame
        pad = [frames[-1]] * (self.seq_len - len(frames))
        return frames + pad

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        compression = row["compression"]
        label = row["label"]
        video_id = row["video_id"]

        video_dir = self.faces_root / self.split_name / compression / label / video_id
        frames = list_images(video_dir) if video_dir.exists() else []
        frames = sorted(frames)
        frames = self._sample_sequence(frames)
        if not frames:
            # Return a blank tensor if no faces found
            blank = torch.zeros((self.seq_len, 3, 224, 224), dtype=torch.float32)
            target = 1 if label == "fake" else 0
            return blank, target

        seq = []
        from PIL import Image
        for p in frames:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            seq.append(img)
        seq = torch.stack(seq, dim=0)
        target = 1 if label == "fake" else 0
        return seq, target


class TemporalModel(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, t, -1)
        out, _ = self.lstm(feats)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)


def evaluate(model, loader, device):
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for seqs, targets in loader:
            seqs = seqs.to(device)
            targets = targets.float().to(device)
            logits = model(seqs)
            probs = torch.sigmoid(logits)
            all_targets.extend(targets.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_targets, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Train CNN+LSTM temporal model.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--faces-root", default=str(base_dir / "data" / "ffpp" / "faces"))
    parser.add_argument("--splits-dir", default=str(base_dir / "data" / "ffpp" / "splits"))
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", default=str(base_dir / "artifacts" / "temporal_model.pth"))
    parser.add_argument("--metrics-out", default=str(base_dir / "artifacts" / "temporal_metrics.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    backbone, feat_dim = build_backbone(args.backbone, pretrained=not args.no_pretrained)
    model = TemporalModel(backbone, feat_dim, hidden=args.hidden, dropout=args.dropout).to(device)

    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    faces_root = Path(args.faces_root)
    splits_dir = Path(args.splits_dir)

    train_ds = FaceSequenceDataset(faces_root, splits_dir / "train.csv", "train", seq_len=args.seq_len, transform=tfms)
    val_ds = FaceSequenceDataset(faces_root, splits_dir / "val.csv", "val", seq_len=args.seq_len, transform=tfms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    os.makedirs(Path(args.out).parent, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for seqs, targets in train_loader:
            seqs = seqs.to(device)
            targets = targets.float().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * seqs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={epoch_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['auc']:.4f}"
        )

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save({
                "backbone": args.backbone,
                "state_dict": model.state_dict(),
                "seq_len": args.seq_len,
                "hidden": args.hidden,
            }, args.out)
            with open(args.metrics_out, "w") as f:
                json.dump(metrics, f, indent=2)

    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Saved model to: {args.out}")
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
