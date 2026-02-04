# Video Deepfake Detector

This project builds a video-level deepfake detector using FaceForensics++ (FF++), with:
- Face-cropped frames
- Two expert models (C23 and C40)
- A gating model to route videos to the right expert

## Data layout
See `DATA_LAYOUT.md`.

## Pipeline (high level)
1) Index FF++ videos
2) Split by video_id into train/val/test
3) Extract frames
4) Crop faces with MediaPipe
5) Train expert models (C23, C40)
6) Train gating model

## Scripts
- `scripts/ffpp_index.py`
- `scripts/ffpp_split.py`
- `scripts/ffpp_extract_frames.py`
- `scripts/ffpp_face_crop.py`

## Quick start
```bash
# 1) Index FF++ videos
python3 scripts/ffpp_index.py \
  --raw data/ffpp/raw \
  --out data/ffpp/index.csv

# 2) Create splits
python3 scripts/ffpp_split.py \
  --index data/ffpp/index.csv \
  --out-dir data/ffpp/splits

# 3) Extract frames (train split shown)
python3 scripts/ffpp_extract_frames.py \
  --split data/ffpp/splits/train.csv \
  --frames-root data/ffpp/frames \
  --every-n 5 \
  --max-frames 32 \
  --resize 224

# 4) Crop faces
python3 scripts/ffpp_face_crop.py \
  --frames-root data/ffpp/frames/train \
  --faces-root data/ffpp/faces/train
```
