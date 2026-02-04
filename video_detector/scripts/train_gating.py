#!/usr/bin/env python3
"""Train a gating model to choose between C23 and C40 experts.

Uses simple per-video features computed from face crops.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def iter_frames(video_dir: Path):
    return [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]


def frame_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_intensity = gray.mean()
    std_intensity = gray.std()
    return np.array([lap_var, mean_intensity, std_intensity], dtype=np.float32)


def video_features(video_dir: Path):
    frames = iter_frames(video_dir)
    if not frames:
        return None

    feats = []
    for frame in frames:
        img = cv2.imread(str(frame))
        if img is None:
            continue
        feats.append(frame_features(img))

    if not feats:
        return None

    feats = np.stack(feats, axis=0)
    # Aggregate per-video
    mean_feat = feats.mean(axis=0)
    std_feat = feats.std(axis=0)
    return np.concatenate([mean_feat, std_feat], axis=0)


def build_dataset(faces_root: Path, split_df: pd.DataFrame):
    X = []
    y = []
    for _, row in split_df.iterrows():
        split_name = row.get("split", "train")
        compression = row["compression"]
        label = row["label"]
        video_id = row["video_id"]

        video_dir = faces_root / split_name / compression / label / video_id
        feats = video_features(video_dir)
        if feats is None:
            continue
        X.append(feats)
        # Target is compression class: c23=0, c40=1
        y.append(0 if compression == "c23" else 1)

    if not X:
        raise SystemExit("No features built. Check face crop paths.")

    return np.stack(X, axis=0), np.array(y, dtype=np.int64)


def evaluate(model, X, y):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def main():
    parser = argparse.ArgumentParser(description="Train gating model for C23 vs C40.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--faces-root", default=str(base_dir / "data" / "ffpp" / "faces"))
    parser.add_argument("--splits-dir", default=str(base_dir / "data" / "ffpp" / "splits"))
    parser.add_argument("--out", default=str(base_dir / "artifacts" / "gating_model.joblib"))
    args = parser.parse_args()

    faces_root = Path(args.faces_root)
    splits_dir = Path(args.splits_dir)

    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    train_df["split"] = "train"
    val_df["split"] = "val"

    X_train, y_train = build_dataset(faces_root, train_df)
    X_val, y_val = build_dataset(faces_root, val_df)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200)),
    ])

    model.fit(X_train, y_train)
    metrics = evaluate(model, X_val, y_val)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": [
        "lap_var_mean", "intensity_mean", "intensity_std",
        "lap_var_std", "intensity_mean_std", "intensity_std_std"
    ]}, out_path)

    print("Gating metrics (val):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Saved gating model to: {out_path}")


if __name__ == "__main__":
    main()
