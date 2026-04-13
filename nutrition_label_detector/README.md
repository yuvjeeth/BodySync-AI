# Nutrition Label Detector

This folder contains the YOLOv8 pipeline used to detect nutrition labels in images. It includes scripts for downloading and converting the dataset, validating labels, training a baseline model, and running small hyperparameter experiments.

## Overview

- Dataset: `openfoodfacts/nutrition-table-detection`
- Task: single-class object detection
- Class name: `nutrition_label`
- Model family: YOLOv8 (`yolov8n.pt`)

## Folder Layout

```text
nutrition_label_detector/
├─ data.yaml
├─ download_dataset_convert.py
├─ validate_dataset.py
├─ train_baseline.py
├─ experiments.py
├─ best.pt
├─ dataset/
│  ├─ images/
│  │  ├─ train/
│  │  └─ val/
│  └─ labels/
│     ├─ train/
│     └─ val/
└─ runs/
   └─ detect/
```

## Requirements

Install the Python dependencies listed in `requirements.txt` from the project root. The detector scripts rely on:

- `ultralytics`
- `torch`
- `datasets`
- `opencv-python`
- `matplotlib`

## Dataset Preparation

Download the dataset from Hugging Face and convert the labels into YOLO format:

```powershell
python .\download_dataset_convert.py
```

This creates:

- `dataset/images/train`
- `dataset/images/val`
- `dataset/labels/train`
- `dataset/labels/val`

## Dataset Validation

Run a quick sanity check before training:

```powershell
python .\validate_dataset.py
```

The validator checks:

- image/label file pairing
- label line formatting
- normalized coordinate ranges
- tiny or suspiciously small boxes

## Baseline Training

Train the baseline model with the tuned report-friendly settings:

```powershell
python .\train_baseline.py
```

Current baseline settings in `train_baseline.py`:

- `epochs = 25`
- `imgsz = 640`
- `batch = 8`
- `workers = 0`
- `optimizer = SGD`
- `lr0 = 0.01`

Training outputs are saved under:

- `runs/detect/baseline_default`

The best checkpoint is stored at:

- `runs/detect/baseline_default/weights/best.pt`

A copy of the final model is also kept at:

- `best.pt`

## Hyperparameter Experiments

Run a small sweep to compare parameter trends:

```powershell
python .\experiments.py
```

This script generates:

- `hyperparameter_report.csv`
- `hyperparameter_graphs/trend_epochs.png`
- `hyperparameter_graphs/trend_lr0.png`
- `hyperparameter_graphs/trend_optimizer.png`
- `hyperparameter_graphs/trend_batch.png`
- `hyperparameter_graphs/trend_imgsz.png`

The sweep compares one parameter at a time against a stable baseline.

## Output Locations

Training and validation artifacts are written to `runs/detect/`.

Typical outputs include:

- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `PR_curve.png`
- `weights/best.pt`
- `weights/last.pt`

## Notes

- The dataset uses one class: `nutrition_label`.
- All scripts assume they are run from the `nutrition_label_detector` folder unless noted otherwise.
- If you want to use the trained model in another script, load `best.pt` with Ultralytics YOLO.

## Troubleshooting

- If training fails, run `validate_dataset.py` first.
- If results are all zero, re-run `download_dataset_convert.py` to regenerate labels.
- If `best.pt` is missing, check the most recent folder in `runs/detect/`.
