# Experiment Log

This log tracks what we tried, what changed, and outcomes.

## 2026-02-10
Change: Installed `video_detector` dependencies and added `tqdm`.
Outcome: Success. Environment ready.

Change: Downloaded FF++ subset (C23) using official script from EU2 server.
Details: `original` 50, `Deepfakes` 50, `FaceSwap` 50.
Outcome: Success. 150 videos indexed.

Change: Ran indexing and split by video_id.
Details: Train 102, Val 21, Test 27.
Outcome: Success.

Change: Extracted frames.
Details: `every-n 10`, `max-frames 16`, `resize 160`.
Outcome: Success.

Change: Face cropping.
Details: MediaPipe missing `solutions`, added OpenCV Haar cascade fallback in `video_detector/scripts/ffpp_face_crop.py`.
Outcome: Success. Crops generated for train/val/test.

Change: Trained expert model (C23).
Details: `resnet18`, `epochs 3`, `batch-size 16`, `lr 1e-4`.
Outcome: Low performance. Best val AUC 0.3126.

Change: Expanded FF++ subset (C23) and added Face2Face.
Details: `original` 100, `Deepfakes` 100, `FaceSwap` 100, `Face2Face` 100.
Outcome: Success. 400 videos indexed.

Change: Rebuilt splits.
Details: Train 276, Val 60, Test 64.
Outcome: Success.

Change: Re-extracted frames with more coverage.
Details: `every-n 10`, `max-frames 32`, `resize 160`.
Outcome: Success.

Change: Re-cropped faces using Haar fallback.
Outcome: Success.

Change: Retrained expert model (C23).
Details: `resnet18`, `epochs 5`, `batch-size 16`, `lr 1e-4`.
Outcome: Improved. Best val AUC 0.9123.

Change: Added threshold tuning for video-level inference.
Files: `video_detector/scripts/tune_threshold.py`
Outcome: Best threshold 0.01 on val. Precision 0.7458, Recall 0.9778, F1 0.8462.

Change: Added test evaluation with tuned threshold.
Files: `video_detector/scripts/eval_video_threshold.py`
Outcome: Test accuracy 0.7344, Precision 0.7541, Recall 0.9583, F1 0.8440, AUC 0.5729 (64 videos).

Change: Added CLI inference script.
Files: `video_detector/scripts/infer_video.py`
Outcome: Success.

Change: Generated PR curve.
Files: `video_detector/scripts/plot_pr_curve.py`, `video_detector/artifacts/pr_curve.png`
Outcome: Success.

Change: App now auto-loads tuned threshold and prefers `expert_c23.pth`.
File: `video_detector/app/app.py`
Outcome: Success.

Change: Added balanced sampling and pos_weight override to training.
File: `video_detector/scripts/train_expert.py`
Outcome: Success.

Change: Added recall-constrained threshold tuning option with fallback to best F1.
File: `video_detector/scripts/tune_threshold.py`
Outcome: Success.

Change: Precision-favoring retrain.
Details: `resnet18`, `epochs 5`, `batch-size 16`, `lr 1e-4`, `balanced sampling`, `pos_weight=1.0`.
Outcome: Best val AUC 0.8760.

Change: Threshold tuned with min recall 0.90 (fallback to best F1).
Outcome: Selected threshold 0.02. Test accuracy 0.6406, Precision 0.7660, Recall 0.7500, F1 0.7579, AUC 0.6589.

Change: Tried EfficientNet-B0 precision-favoring training.
Details: `efficientnet_b0`, `epochs 5`, `batch-size 16`, `lr 1e-4`, `balanced sampling`, `pos_weight=0.75`.
Issue: Pretrained weights download blocked; trained with `--no-pretrained` fallback.
Outcome: Poor validation metrics (acc 0.3813, prec 0.3300, rec 0.5765, f1 0.4197, auc 0.3483).

Change: Downloaded EfficientNet-B0 pretrained weights to workspace cache and retried training.
Cache: `/Users/ashaypatel/Documents/Deepfake_detection_workspace/.torch_cache/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth`
Outcome: Validation metrics improved (acc 0.6279, prec 0.5146, rec 0.7235, f1 0.6015, auc 0.7034).

Change: Tuned threshold for EfficientNet-B0 (min recall 0.90).
Outcome: Selected threshold 0.01. Test accuracy 0.7500, Precision 0.7500, Recall 1.0000, F1 0.8571, AUC 0.5430.

Change: Added staged fine-tuning (freeze backbone then unfreeze last blocks).
File: `video_detector/scripts/train_expert.py`
Outcome: Success.

Change: EfficientNet-B0 staged fine-tuning run.
Details: `epochs 2` (head-only), `stage2-epochs 3`, `freeze-backbone`, `unfreeze-last 2`, `head-lr 1e-4`, `backbone-lr 1e-5`.
Outcome: Validation metrics (acc 0.6347, prec 0.5373, rec 0.4235, f1 0.4737, auc 0.6295). Test with tuned threshold: acc 0.7500, prec 0.7500, rec 1.0000, f1 0.8571, auc 0.4792.

Change: Added temporal CNN+LSTM training script.
File: `video_detector/scripts/train_temporal.py`
Outcome: Success.

Change: Trained temporal model (ResNet18 + LSTM).
Details: `seq_len 16`, `epochs 3`, `batch-size 4`, `lr 1e-4`.
Outcome: Metrics saved (acc 0.7000, prec 0.7368, rec 0.9333, f1 0.8235, auc 0.3067). Training run timed out but metrics written.
