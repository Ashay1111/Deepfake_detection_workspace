# FF++ Data Layout (Local)

We will store the FaceForensics++ dataset inside the repo at:

```
video_detector/data/ffpp/
  raw/          # FF++ original folder structure goes here
  frames/       # Extracted frames (per split)
  faces/        # Face crops (per split)
  splits/       # train/val/test CSVs
```

## Expected FF++ raw structure
Place the official FF++ dataset inside `video_detector/data/ffpp/raw/` without changing its folder structure.
The indexer will look for the official folders:

- `original_sequences/.../c23/videos/*.mp4`
- `original_sequences/.../c40/videos/*.mp4`
- `manipulated_sequences/<method>/c23/videos/*.mp4`
- `manipulated_sequences/<method>/c40/videos/*.mp4`

Where `<method>` is typically one of:
- `Deepfakes`
- `Face2Face`
- `FaceSwap`
- `NeuralTextures`

## Generated files
- `data/ffpp/index.csv` — catalog of all videos
- `data/ffpp/splits/train.csv`
- `data/ffpp/splits/val.csv`
- `data/ffpp/splits/test.csv`

## Notes
- Splits are done **by video_id** to prevent leakage.
- We create separate experts for **C23** (high quality) and **C40** (low quality).
