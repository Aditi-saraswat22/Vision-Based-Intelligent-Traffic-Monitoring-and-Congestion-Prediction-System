# Data

This directory is a placeholder. Dataset files are not tracked by git due to size.

## Datasets Used

### BDD100K
- **Source:** [Kaggle — solesensei/solesensei_bdd100k](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
- **What it contains:** 100K dashcam driving videos across varied weather conditions, times of day, and road types
- **Why we used it:** Diverse real-world dashcam perspective — rain, fog, night, highway, city streets

### VisDrone
- **Source:** [Kaggle — banuprasadb/visdrone-dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
- **What it contains:** Images captured from drone/aerial viewpoints across dense urban traffic scenes
- **Why we used it:** Aerial perspective — completely different camera angle from BDD100K, forces the model to generalize

---

## How We Combined Them

All labels were remapped to a unified 5-class schema before training:

| Class ID | Class Name |
|:---:|---|
| 0 | car |
| 1 | truck |
| 2 | bus |
| 3 | motorcycle |
| 4 | person |

BDD100K labels (originally JSON) were converted to YOLO `.txt` format. VisDrone class IDs were remapped from their original 10-class schema. The combined training set contains **5,987 images** and the validation set **747 images**.

---

## Directory Structure (after download)

```
data/
├── README.md
├── combined/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
└── density_data.csv       ← generated after running YOLO on val set
```

---

## Reproducing the Dataset

Run the notebooks in `src/` in this order:

```
1. src/build_dataset.ipynb    ← downloads, converts, and merges both datasets
2. src/train_yolo.ipynb       ← trains YOLOv8s on the combined set
3. src/train_lstm.ipynb       ← generates density_data.csv and trains LSTM
```
