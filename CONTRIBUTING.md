# Contributing

Thanks for collaborating on this project!

## Setup
1. Create a venv in the project you are working on:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow
- Use the scripts under each project (`image_classifier/scripts`, `video_detector/scripts`).
- Do not commit datasets, frames, or model weights.
- Keep changes focused; update docs if behavior changes.

## GPU training
If you are training on GPU, follow:
- `video_detector/TRAINING_RUNBOOK.md`

## PR checklist
- Code runs locally
- Docs updated
- No large files added
