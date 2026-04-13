from pathlib import Path

import torch
from ultralytics import YOLO

EVAL_ONLY = False
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs" / "detect"

FOLDER_NAME = "baseline_default"
CHECKPOINT_PATH = str(RUNS_DIR / FOLDER_NAME / "weights" / "best.pt")

# Hyperparameters to modify for project report experiments.
REPORT_PARAMS = {
    "lr0": 0.01,
    "optimizer": "SGD",
    "batch": 8,
    "imgsz": 640,
    "epochs": 25,
    "workers": 0,
}

if __name__ == '__main__':
    # GPU check
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if EVAL_ONLY:
        best_path = CHECKPOINT_PATH
    else:
        model = YOLO("yolov8n.pt")
        model.train(
            data="data.yaml",
            epochs=REPORT_PARAMS["epochs"],
            imgsz=REPORT_PARAMS["imgsz"],
            batch=REPORT_PARAMS["batch"],
            workers=REPORT_PARAMS["workers"],
            cache=True,
            optimizer=REPORT_PARAMS["optimizer"],
            lr0=REPORT_PARAMS["lr0"],
            project=str(RUNS_DIR),
            name=FOLDER_NAME,
            exist_ok=True,
        )
        
        best_path = str(model.trainer.best)

    print("Using best checkpoint:", best_path)
    model = YOLO(best_path)
    metrics = model.val(
        data="data.yaml",
        project=str(RUNS_DIR),
        name=FOLDER_NAME,
        exist_ok=True,
    )

    print(metrics)

    # Save prediction image instead of displaying
    results = model("0.jpg")
    results[0].save(filename="prediction_output.jpg")
    print("Saved prediction to prediction_output.jpg")