# Models

This directory stores trained model weights. Files are not tracked by git due to size — download them separately or retrain using the notebooks in `src/`.

---

## Files

| File | Description | Size |
|---|---|:---:|
| `best.pt` | YOLOv8s trained weights | ~22.5 MB |
| `lstm_congestion.pt` | LSTM congestion classifier weights | ~1.2 MB |

---

## YOLOv8s — Vehicle Detection

**Architecture:** YOLOv8s (small) · 11.1M parameters · 28.4 GFLOPs  
**Training:** 80 epochs · batch 16 · imgsz 640 · AdamW optimizer  
**Platform:** Kaggle Tesla T4 GPU · ~2.5 hours

**Performance:**

| Metric | Value |
|---|:---:|
| Overall mAP50 | 0.503 |
| Overall mAP50-95 | 0.297 |
| Car mAP50 | 0.800 |
| Inference speed | 7.6ms/frame |

Trained on BDD100K + VisDrone combined — covering dashcam and aerial perspectives across varied weather, lighting, and road conditions.

---

## LSTM — Congestion Prediction

**Architecture:** 2-layer LSTM · hidden size 128 · dropout 0.3 · 4-class output  
**Input:** Sliding window of 20 frames (vehicle count + congestion level)  
**Training:** 100 epochs · Adam optimizer · CrossEntropyLoss  
**Platform:** Kaggle Tesla T4 GPU · ~5 minutes

**Performance:**

| Metric | Value |
|---|:---:|
| Overall Accuracy | 57.4% |
| Weighted F1-Score | 45.7% |
| Best Val Loss | 1.087 |

> Accuracy is constrained by a small validation set (129 samples). The app uses a rule-based fallback when fewer than 20 frames of history are available, which ensures reliable output from the first frame of any video.

---

## Using the Models

Place both files in the same directory as `app.py`. They are auto-detected — no path configuration needed.

To retrain from scratch, run the notebooks in `src/` in order:
```
src/build_dataset.ipynb  →  src/train_yolo.ipynb  →  src/train_lstm.ipynb
```
