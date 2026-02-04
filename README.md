# Deepfake Detection Workspace

This repository contains two related projects:

- `image_classifier/` — image-level real vs fake classifier (starter pipeline)
- `video_detector/` — video-level deepfake detector with FF++ and expert routing

## Quick start

### Image classifier
```bash
cd image_classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Video detector
```bash
cd video_detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For FF++ pipeline and training, see `video_detector/README.md` and `video_detector/TRAINING_RUNBOOK.md`.

## Repo hygiene
- Datasets, weights, and large artifacts are ignored by `.gitignore`.
- Share trained weights separately (e.g., Google Drive).

## Licensing
- Please follow FaceForensics++ data usage terms for any dataset distribution.
