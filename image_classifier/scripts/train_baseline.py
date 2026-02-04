#!/usr/bin/env python3
"""Train a baseline real vs fake classifier on image splits."""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from functools import partial


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def map_target(t, pos_index):
    return int(t == pos_index)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(name: str):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
        input_size = 224
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        input_size = 224
    else:
        raise ValueError("Unsupported model. Use resnet18 or efficientnet_b0")
    return model, input_size


def get_loaders(data_root: Path, input_size: int, batch_size: int, num_workers: int):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tfms)

    # Ensure label 1 always means "fake" for consistency across runs
    pos_index = train_ds.class_to_idx["fake"]
    target_fn = partial(map_target, pos_index=pos_index)
    train_ds.target_transform = target_fn
    val_ds.target_transform = target_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, train_ds.class_to_idx


def compute_pos_weight(data_root: Path):
    counts = {"real": 0, "fake": 0}
    for cls_name in ["real", "fake"]:
        cls_dir = data_root / "train" / cls_name
        counts[cls_name] = len([p for p in cls_dir.iterdir() if p.is_file()])
    # Positive class is "fake" -> label 1
    pos = max(counts["fake"], 1)
    neg = max(counts["real"], 1)
    return torch.tensor(neg / pos, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.float().to(device)
            logits = model(images).squeeze(1)
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
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline real vs fake classifier.")
    parser.add_argument("--data", required=True, help="Path to split dataset root.")
    parser.add_argument("--model", default="resnet18", help="resnet18 or efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--out", default=str(base_dir / "artifacts" / "baseline_model.pth"))
    parser.add_argument("--metrics-out", default=str(base_dir / "artifacts" / "metrics.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    model, input_size = build_model(args.model)
    model = model.to(device)

    data_root = Path(args.data)
    train_loader, val_loader, class_to_idx = get_loaders(data_root, input_size, args.batch_size, args.num_workers)

    pos_weight = compute_pos_weight(data_root)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0
    os.makedirs(Path(args.out).parent, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

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
                "model": args.model,
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "pos_class": "fake",
                "input_size": input_size,
            }, args.out)
            with open(args.metrics_out, "w") as f:
                json.dump(metrics, f, indent=2)

    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Saved model to: {args.out}")
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
