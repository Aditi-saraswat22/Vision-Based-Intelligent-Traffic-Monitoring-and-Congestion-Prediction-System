<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B6B?style=for-the-badge"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>

# TrafficVision AI
### Vision-Based Intelligent Traffic Monitoring & Congestion Prediction

*Real-time vehicle detection · Density estimation · Proactive congestion prediction — using just a camera feed.*

</div>

---

## Overview

VisionFlow is a deep learning pipeline that analyzes traffic video in real time, counts vehicles, estimates road density, and predicts congestion level — without any physical sensors. It factors in environmental context (weather, time of day, road type) to calibrate its thresholds intelligently.

```
Video Input → YOLOv8 Detection → Vehicle Count → LSTM Prediction → Congestion Level
                                                                  (Low / Medium / High / Critical)
```

---

## Model Performance

**YOLOv8s** — trained on BDD100K + VisDrone (5,987 images, two camera perspectives)

| Class | mAP50 |
|---|:---:|
| Car | 0.800 |
| Bus | 0.498 |
| Person | 0.461 |
| Motorcycle | 0.394 |
| Truck | 0.363 |
| **Overall** | **0.503** |

Inference speed: **7.6ms/frame** on Tesla T4 · Parameters: 11.1M

**LSTM Congestion Classifier** — overall accuracy 57.4%, weighted F1 45.7%

> Accuracy is limited by a small validation set (129 samples). Real-time performance is strong due to a rule-based fallback when LSTM history is insufficient. A Random Forest upgrade is planned as future work.

---

## Tech Stack

`Python` · `YOLOv8s (Ultralytics)` · `PyTorch` · `OpenCV` · `Streamlit` · `Plotly` · `Pandas`

---

## Getting Started

```bash
git clone https://github.com/Aditi-saraswat22/Vision-Based-Intelligent-Traffic-Monitoring-and-Congestion-Prediction-System.git
cd Vision-Based-Intelligent-Traffic-Monitoring-and-Congestion-Prediction-System
pip install -r requirements.txt
streamlit run app.py
```

Place model weights in the same folder as `app.py` — they are auto-detected:
```
best.pt               ← YOLOv8 weights
lstm_congestion.pt    ← LSTM weights
```

**App flow:** Splash → set weather/time/road conditions → upload video → live analysis → download CSV

---

## Project Structure

```
├── app.py                        # Streamlit dashboard
├── requirements.txt
├── data/                         # Dataset instructions
├── models/                       # Trained weights (not tracked by git)
├── src/                          # Training notebooks
│   ├── build_dataset.ipynb
│   ├── train_yolo.ipynb
│   └── train_lstm.ipynb
└── Minor Project Synopsis Report1.pdf
```

---

## Dataset

Trained on a unified pipeline combining **BDD100K** (dashcam, varied weather) and **VisDrone** (aerial drone views). All labels remapped to 5 unified classes. Dataset not included — see [`data/README.md`](data/README.md) for download links.

---

## Challenges

- **Generalization** — initial UA-DETRAC-only training failed on new footage; solved by multi-dataset training across two different camera perspectives
- **Label inconsistency** — built a custom converter to unify BDD100K JSON + VisDrone class IDs into one format
- **LSTM data scarcity** — only 747 frames available for temporal modelling; mitigated with rule-based fallback

---

## Future Work

- Live RTSP/CCTV stream support
- Random Forest congestion classifier for better accuracy on limited data
- Lane-level density analysis
- Traffic signal integration

---

<div align="center">
<sub>Minor Project · Deep Learning · Computer Vision</sub>
</div>
