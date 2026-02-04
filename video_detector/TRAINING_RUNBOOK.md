# Training Runbook (GPU)

This runbook is for training FF++ experts and the gating model on a GPU machine.

## 0) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Place FF++ data
Copy the official FaceForensics++ dataset into:
```
video_detector/data/ffpp/raw/
```
Keep the official folder structure (see `DATA_LAYOUT.md`).

## 2) Index FF++ videos
```bash
python3 scripts/ffpp_index.py \
  --raw data/ffpp/raw \
  --out data/ffpp/index.csv
```

## 3) Create splits (by video_id)
```bash
python3 scripts/ffpp_split.py \
  --index data/ffpp/index.csv \
  --out-dir data/ffpp/splits
```

## 4) Extract frames
Run for train/val/test (adjust sampling to taste):
```bash
python3 scripts/ffpp_extract_frames.py \
  --split data/ffpp/splits/train.csv \
  --frames-root data/ffpp/frames \
  --every-n 5 \
  --max-frames 32 \
  --resize 224

python3 scripts/ffpp_extract_frames.py \
  --split data/ffpp/splits/val.csv \
  --frames-root data/ffpp/frames \
  --every-n 5 \
  --max-frames 32 \
  --resize 224

python3 scripts/ffpp_extract_frames.py \
  --split data/ffpp/splits/test.csv \
  --frames-root data/ffpp/frames \
  --every-n 5 \
  --max-frames 32 \
  --resize 224
```

## 5) Face crops (MediaPipe)
```bash
python3 scripts/ffpp_face_crop.py \
  --frames-root data/ffpp/frames/train \
  --faces-root data/ffpp/faces/train

python3 scripts/ffpp_face_crop.py \
  --frames-root data/ffpp/frames/val \
  --faces-root data/ffpp/faces/val

python3 scripts/ffpp_face_crop.py \
  --frames-root data/ffpp/frames/test \
  --faces-root data/ffpp/faces/test
```

## 6) Train Expert A (C23)
```bash
python3 scripts/train_expert.py \
  --faces-root data/ffpp/faces \
  --compression c23 \
  --model resnet18 \
  --epochs 10 \
  --batch-size 32 \
  --out artifacts/expert_c23.pth \
  --metrics-out artifacts/expert_c23_metrics.json
```

## 7) Train Expert B (C40)
```bash
python3 scripts/train_expert.py \
  --faces-root data/ffpp/faces \
  --compression c40 \
  --model resnet18 \
  --epochs 10 \
  --batch-size 32 \
  --out artifacts/expert_c40.pth \
  --metrics-out artifacts/expert_c40_metrics.json
```

## 8) Train Gating Model (C23 vs C40)
```bash
python3 scripts/train_gating.py \
  --faces-root data/ffpp/faces \
  --splits-dir data/ffpp/splits \
  --out artifacts/gating_model.joblib
```

## 9) Artifacts to share back
Copy these back to the main repo:
- `artifacts/expert_c23.pth`
- `artifacts/expert_c40.pth`
- `artifacts/gating_model.joblib`

## Notes
- All splits are done by `video_id` to avoid leakage.
- You can increase `--epochs` or reduce `--every-n` to improve accuracy (at the cost of time).
