#!/usr/bin/env python3
"""Train an expert model for a specific FF++ compression level (C23 or C40)."""

import argparse
import json
import os
import random
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


def build_model(name: str, pretrained: bool):
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        input_size = 224
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        input_size = 224
    else:
        raise ValueError("Unsupported model. Use resnet18 or efficientnet_b0")
    return model, input_size


def freeze_backbone(model, name: str):
    if name == "efficientnet_b0":
        for p in model.features.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif name == "resnet18":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True


def unfreeze_last_blocks(model, name: str, num_blocks: int):
    if num_blocks <= 0:
        return
    if name == "efficientnet_b0":
        blocks = list(model.features.children())
        for block in blocks[-num_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    elif name == "resnet18":
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        for layer in layers[:num_blocks]:
            for p in layer.parameters():
                p.requires_grad = True


def build_optimizer(model, name: str, head_lr: float, backbone_lr: float | None):
    if backbone_lr is None:
        return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=head_lr)
    if name == "efficientnet_b0":
        backbone_params = [p for p in model.features.parameters() if p.requires_grad]
        head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    else:
        backbone_params = [p for p in model.parameters() if p.requires_grad and p not in list(model.fc.parameters())]
        head_params = [p for p in model.fc.parameters() if p.requires_grad]
    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    return optim.AdamW(param_groups)


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def map_target(t, pos_index):
    return 1 if t == pos_index else 0


class FaceCropDataset(Dataset):
    def __init__(self, faces_root: Path, split: str, compression: str, transform=None):
        self.transform = transform
        self.samples = []

        base = faces_root / split / compression
        for label in ["real", "fake"]:
            label_dir = base / label
            if not label_dir.exists():
                continue
            for img in list_images(label_dir):
                self.samples.append((img, label))

        if not self.samples:
            raise SystemExit(f"No images found under {base}")

        self.class_to_idx = {"real": 0, "fake": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = self.class_to_idx[label]
        return img, target


def compute_pos_weight(dataset: FaceCropDataset):
    counts = {"real": 0, "fake": 0}
    for _, label in dataset.samples:
        counts[label] += 1
    pos = max(counts["fake"], 1)
    neg = max(counts["real"], 1)
    return torch.tensor(neg / pos, dtype=torch.float32)


def build_sampler(dataset: FaceCropDataset):
    counts = {"real": 0, "fake": 0}
    for _, label in dataset.samples:
        counts[label] += 1
    weights = []
    for _, label in dataset.samples:
        if counts[label] == 0:
            weights.append(1.0)
        else:
            weights.append(1.0 / counts[label])
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


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
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Train expert model for C23 or C40.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--faces-root", default=str(base_dir / "data" / "ffpp" / "faces"))
    parser.add_argument("--compression", required=True, choices=["c23", "c40"])
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-sampling", action="store_true", help="Use balanced sampling for classes.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Override positive class weight.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and train head only.")
    parser.add_argument("--unfreeze-last", type=int, default=0, help="Unfreeze last N blocks/layers for fine-tuning.")
    parser.add_argument("--head-lr", type=float, default=None, help="LR for classifier head (default: --lr).")
    parser.add_argument("--backbone-lr", type=float, default=None, help="LR for backbone during fine-tuning.")
    parser.add_argument("--stage2-epochs", type=int, default=0, help="Extra epochs after unfreezing last blocks.")
    parser.add_argument("--out", default=str(base_dir / "artifacts" / "expert_model.pth"))
    parser.add_argument("--metrics-out", default=str(base_dir / "artifacts" / "expert_metrics.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    model, input_size = build_model(args.model, pretrained=not args.no_pretrained)
    model = model.to(device)

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

    faces_root = Path(args.faces_root)
    train_ds = FaceCropDataset(faces_root, "train", args.compression, transform=train_tfms)
    val_ds = FaceCropDataset(faces_root, "val", args.compression, transform=eval_tfms)

    if args.balanced_sampling:
        sampler = build_sampler(train_ds)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.pos_weight is None:
        pos_weight = compute_pos_weight(train_ds).to(device)
    else:
        pos_weight = torch.tensor(args.pos_weight, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    head_lr = args.head_lr if args.head_lr is not None else args.lr
    optimizer = None

    best_auc = -1.0
    os.makedirs(Path(args.out).parent, exist_ok=True)

    def train_epochs(num_epochs: int, start_epoch: int, total_epochs: int):
        nonlocal best_auc
        for e in range(num_epochs):
            epoch = start_epoch + e
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
                f"Epoch {epoch}/{total_epochs} | "
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
                    "input_size": input_size,
                    "pos_class": "fake",
                    "compression": args.compression,
                }, args.out)
                with open(args.metrics_out, "w") as f:
                    json.dump(metrics, f, indent=2)

    total_epochs = args.epochs + args.stage2_epochs
    if args.freeze_backbone:
        freeze_backbone(model, args.model)

    optimizer = build_optimizer(model, args.model, head_lr=head_lr, backbone_lr=args.backbone_lr)
    train_epochs(args.epochs, start_epoch=1, total_epochs=total_epochs)

    if args.stage2_epochs > 0 and args.unfreeze_last > 0:
        unfreeze_last_blocks(model, args.model, args.unfreeze_last)
        optimizer = build_optimizer(model, args.model, head_lr=head_lr, backbone_lr=args.backbone_lr)
        train_epochs(args.stage2_epochs, start_epoch=args.epochs + 1, total_epochs=total_epochs)

    print(f"Best val AUC: {best_auc:.4f}")
    print(f"Saved model to: {args.out}")
    print(f"Saved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()
