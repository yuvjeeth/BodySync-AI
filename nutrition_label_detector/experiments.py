import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs" / "detect"


# Keep a stable baseline and vary one hyperparameter at a time for clean trends.
BASE_CONFIG = {
    "lr0": 0.01,
    "optimizer": "SGD",
    "batch": 8,
    "imgsz": 640,
    "epochs": 25,
    "workers": 0,
}

# Each key below generates one separate trend graph.
SWEEPS = {
    "epochs": [15, 25, 35],
    "lr0": [0.01, 0.005, 0.001],
    "optimizer": ["SGD", "Adam"],
    "batch": [6, 8, 10],
    "imgsz": [512, 640, 768],
}


def metric_value(metrics_dict, *keys):
    for key in keys:
        if key in metrics_dict:
            return metrics_dict[key]
    return None


def build_experiments():
    experiments = []
    for param_name, values in SWEEPS.items():
        for value in values:
            exp = dict(BASE_CONFIG)
            exp[param_name] = value
            exp["varied_param"] = param_name
            exp["varied_value"] = value
            exp["name"] = f"{param_name}_{str(value).replace('.', 'p')}"
            experiments.append(exp)
    return experiments


def create_param_trend_graph(rows, param_name, output_path):
    subset = [row for row in rows if row["varied_param"] == param_name]
    if not subset:
        return

    labels = [str(row["varied_value"]) for row in subset]
    metrics_to_plot = ["precision", "recall", "map50", "map50_95"]
    x_positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for metric_name in metrics_to_plot:
        values = [float(row[metric_name]) if row[metric_name] is not None else 0.0 for row in subset]
        ax.plot(x_positions, values, marker="o", linewidth=2, label=metric_name)

    ax.set_title(f"Trend: {param_name}")
    ax.set_ylabel("Score")
    ax.set_xlabel(param_name)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    experiments = build_experiments()
    rows = []

    for exp in experiments:
        run_name = f"report_{exp['name']}"
        print(f"\nStarting experiment: {run_name}")

        model = YOLO("yolov8n.pt")
        model.train(
            data="data.yaml",
            epochs=exp["epochs"],
            imgsz=exp["imgsz"],
            batch=exp["batch"],
            workers=exp["workers"],
            lr0=exp["lr0"],
            optimizer=exp["optimizer"],
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
        )

        best_path = str(model.trainer.best)
        print("Using best checkpoint:", best_path)
        best_model = YOLO(best_path)
        val_metrics = best_model.val(data="data.yaml")
        results_dict = getattr(val_metrics, "results_dict", {})

        row = {
            "run_name": run_name,
            "optimizer": exp["optimizer"],
            "lr0": exp["lr0"],
            "batch": exp["batch"],
            "imgsz": exp["imgsz"],
            "epochs": exp["epochs"],
            "workers": exp["workers"],
            "varied_param": exp["varied_param"],
            "varied_value": exp["varied_value"],
            "precision": metric_value(results_dict, "metrics/precision(B)", "metrics/precision"),
            "recall": metric_value(results_dict, "metrics/recall(B)", "metrics/recall"),
            "map50": metric_value(results_dict, "metrics/mAP50(B)", "metrics/mAP50"),
            "map50_95": metric_value(results_dict, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
            "fitness": results_dict.get("fitness"),
        }
        rows.append(row)
        print("Completed:", row)

    output_path = BASE_DIR / "hyperparameter_report.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    graphs_dir = BASE_DIR / "hyperparameter_graphs"
    graphs_dir.mkdir(exist_ok=True)
    for param_name in SWEEPS:
        graph_path = graphs_dir / f"trend_{param_name}.png"
        create_param_trend_graph(rows, param_name, graph_path)
        print(f"Saved graph to: {graph_path.resolve()}")

    print(f"\nSaved report to: {output_path.resolve()}")
