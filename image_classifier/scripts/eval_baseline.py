#!/usr/bin/env python3
"""Evaluate a saved baseline model on the test split."""

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from functools import partial


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def map_target(t, pos_index):
    return 1 if t == pos_index else 0


def build_model(name: str, input_size: int):
    name = name.lower()
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError("Unsupported model. Use resnet18 or efficientnet_b0")
    return model


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
    parser = argparse.ArgumentParser(description="Evaluate baseline model on test split.")
    parser.add_argument("--data", required=True, help="Path to split dataset root.")
    parser.add_argument("--model-path", required=True, help="Path to saved model .pth")
    args = parser.parse_args()

    device = get_device()

    ckpt = torch.load(args.model_path, map_location=device)
    model_name = ckpt["model"]
    input_size = ckpt["input_size"]
    pos_class = ckpt.get("pos_class", "fake")

    model = build_model(model_name, input_size)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    eval_tfms = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageFolder(Path(args.data) / "test", transform=eval_tfms)
    pos_index = test_ds.class_to_idx[pos_class]
    test_ds.target_transform = partial(map_target, pos_index=pos_index)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    metrics = evaluate(model, test_loader, device)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()