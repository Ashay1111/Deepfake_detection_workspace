# Video Pipeline Scripts

Run these commands from the `video_detector` folder.

## 1) Extract frames
```bash
python3 scripts/extract_frames.py \
  --input /path/to/videos \
  --output data/frames \
  --every-n 5 \
  --max-frames 32 \
  --resize 224
```

## 2) Run frame-level inference
```bash
python3 scripts/video_infer.py \
  --frames data/frames \
  --model-path ../image_classifier/artifacts/baseline_model.pth \
  --out artifacts/video_scores.csv
```

## 3) Aggregate to video-level predictions
```bash
python3 scripts/aggregate.py \
  --scores artifacts/video_scores.csv \
  --threshold 0.5 \
  --out artifacts/video_predictions.csv
```
