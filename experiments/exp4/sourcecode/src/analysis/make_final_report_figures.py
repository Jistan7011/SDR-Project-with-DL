from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_condition_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    **row,
                    "accuracy": float(row["accuracy"]),
                    "count": int(row["count"]),
                    "errors": int(row["errors"]),
                }
            )
    return rows


def plot_condition(rows: list[dict[str, object]], condition: str, output: Path, title: str, sort: bool = False) -> None:
    data = [row for row in rows if row["condition"] == condition]
    if sort:
        data = sorted(data, key=lambda row: float(row["accuracy"]))
    labels = [str(row["value"]) for row in data]
    values = [float(row["accuracy"]) for row in data]
    colors = ["#c44e52" if value < 0.75 else "#4c8c6b" for value in values]
    plt.figure(figsize=(8, 4.6))
    plt.bar(labels, values, color=colors)
    plt.axhline(0.75, color="#333333", linestyle="--", linewidth=1, alpha=0.65)
    plt.ylim(0.60, 0.92)
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    base = Path(__file__).resolve().parents[3]
    output_dir = base / "results" / "figures" / "final_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("CNN1D", 0.758287037037037, 0.63125),
        ("CNN1D+Aug", 0.7578703703703703, 0.6470833333333333),
        ("VGG1D", 0.758287037037037, 0.6325),
        ("ResNet1D", 0.7578703703703704, 0.6558333333333334),
        ("ResNet1D+ifreq", 0.7552777777777778, 0.70375),
    ]
    labels = [item[0] for item in models]
    accuracy = [item[1] for item in models]
    recall = [item[2] for item in models]
    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width / 2, accuracy, width=width, label="Mean accuracy", color="#3b6ea8")
    plt.bar(x + width / 2, recall, width=width, label="Worst class recall", color="#d1843f")
    plt.axhline(0.75, color="#3b6ea8", linestyle="--", linewidth=1, alpha=0.6)
    plt.axhline(0.65, color="#d1843f", linestyle="--", linewidth=1, alpha=0.6)
    plt.ylim(0.58, 0.80)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Experiment 2 Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()

    rows = read_condition_rows(base / "results" / "reports" / "resnet1d_condition_accuracy.csv")
    plot_condition(rows, "session_id", output_dir / "session_accuracy.png", "ResNet1D Ensemble Accuracy by Test Session")
    plot_condition(rows, "payload", output_dir / "payload_accuracy.png", "ResNet1D Ensemble Accuracy by Payload", sort=True)
    plot_condition(rows, "snr_bin", output_dir / "snr_bin_accuracy.png", "ResNet1D Ensemble Accuracy by Estimated SNR Bin")
    plot_condition(rows, "baseband_offset_hz", output_dir / "offset_accuracy.png", "ResNet1D Ensemble Accuracy by Baseband Offset")
    plot_condition(rows, "rx_gain", output_dir / "rx_gain_accuracy.png", "ResNet1D Ensemble Accuracy by RX Gain")

    flows: list[dict[str, str]] = []
    with (base / "results" / "reports" / "resnet1d_top_error_flows.csv").open(encoding="utf-8") as f:
        flows = list(csv.DictReader(f))[:12]
    labels = [f"{row['true']}->{row['predicted']}\n{row['session_id']} {row['payload']}" for row in flows]
    counts = [int(row["count"]) for row in flows]
    y = np.arange(len(flows))
    plt.figure(figsize=(10, 5.8))
    plt.barh(y, counts, color="#8a5a9e")
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Error count")
    plt.title("Top ResNet1D Ensemble Error Flows")
    plt.tight_layout()
    plt.savefig(output_dir / "top_error_flows.png", dpi=160)
    plt.close()

    metrics = json.loads((base / "results" / "exp02_15session_resnet" / "resnet1d_seed42" / "logs" / "eval_metrics.json").read_text(encoding="utf-8"))
    matrix = np.array(metrics["confusion_matrix"])
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(matrix, cmap="Blues")
    plt.title("ResNet1D Seed 42 Confusion Matrix")
    plt.xticks(range(3), ["BASK", "BFSK", "BPSK"])
    plt.yticks(range(3), ["BASK", "BFSK", "BPSK"])
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "resnet_confusion_matrix_seed42.png", dpi=160)
    plt.close()

    print(f"Final report figures written to {output_dir}")


if __name__ == "__main__":
    main()
