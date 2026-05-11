from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.common import CLASS_NAMES, ensure_dir, write_json


def analyze_exp05_bfsk_bpsk_errors(
    metrics_path: str,
    output_dir: str,
    baseline_metrics_path: str | None = None,
) -> dict[str, Any]:
    metrics = read_json(Path(metrics_path))
    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    summary = summarize_confusion(cm)
    if baseline_metrics_path:
        baseline = summarize_confusion(np.asarray(read_json(Path(baseline_metrics_path))["confusion_matrix"], dtype=np.int64))
        summary["baseline"] = baseline
        summary["bfsk_bpsk_to_bask_error_reduction"] = reduction(
            float(baseline["bfsk_bpsk_to_bask_rate"]),
            float(summary["bfsk_bpsk_to_bask_rate"]),
        )
    out = ensure_dir(output_dir)
    write_json(out / "exp05_bfsk_bpsk_error_summary.json", summary)
    write_summary_md(out / "exp05_bfsk_bpsk_error_summary.md", summary)
    write_class_recall_csv(out / "exp05_class_recall.csv", summary)
    plot_focus_confusion(out / "exp05_focus_confusion.png", cm)
    return summary


def summarize_confusion(cm: np.ndarray) -> dict[str, Any]:
    recalls: dict[str, float] = {}
    for idx, name in enumerate(CLASS_NAMES):
        total = int(cm[idx].sum())
        recalls[name] = float(cm[idx, idx] / total) if total else 0.0
    bfsk_to_bask = int(cm[CLASS_NAMES.index("BFSK"), CLASS_NAMES.index("BASK")])
    bpsk_to_bask = int(cm[CLASS_NAMES.index("BPSK"), CLASS_NAMES.index("BASK")])
    bfsk_total = int(cm[CLASS_NAMES.index("BFSK")].sum())
    bpsk_total = int(cm[CLASS_NAMES.index("BPSK")].sum())
    focused_total = max(bfsk_total + bpsk_total, 1)
    return {
        "accuracy_from_confusion": float(np.trace(cm) / max(cm.sum(), 1)),
        "class_recall": recalls,
        "worst_recall": float(min(recalls.values())),
        "bfsk_to_bask_count": bfsk_to_bask,
        "bpsk_to_bask_count": bpsk_to_bask,
        "bfsk_bpsk_to_bask_count": bfsk_to_bask + bpsk_to_bask,
        "bfsk_bpsk_to_bask_rate": float((bfsk_to_bask + bpsk_to_bask) / focused_total),
        "confusion_matrix": cm.tolist(),
    }


def reduction(baseline_rate: float, current_rate: float) -> float:
    if baseline_rate <= 0:
        return 0.0
    return float((baseline_rate - current_rate) / baseline_rate)


def write_class_recall_csv(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "recall"])
        writer.writeheader()
        for name, recall in summary["class_recall"].items():
            writer.writerow({"class": name, "recall": recall})


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Experiment 5 BFSK/BPSK Error Analysis",
        "",
        f"- Accuracy: `{summary['accuracy_from_confusion']:.4f}`",
        f"- Worst recall: `{summary['worst_recall']:.4f}`",
        f"- BFSK/BPSK -> BASK count: `{summary['bfsk_bpsk_to_bask_count']}`",
        f"- BFSK/BPSK -> BASK rate: `{summary['bfsk_bpsk_to_bask_rate']:.4f}`",
    ]
    if "bfsk_bpsk_to_bask_error_reduction" in summary:
        lines.append(f"- Reduction vs baseline: `{summary['bfsk_bpsk_to_bask_error_reduction']:.4f}`")
    lines.extend(["", "## Class Recall", "", "| class | recall |", "| --- | ---: |"])
    for name, recall in summary["class_recall"].items():
        lines.append(f"| {name} | {recall:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_focus_confusion(path: Path, cm: np.ndarray) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title("Exp5 Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", default="../results/analysis")
    parser.add_argument("--baseline-metrics", default=None)
    args = parser.parse_args()
    analyze_exp05_bfsk_bpsk_errors(args.metrics, args.output_dir, args.baseline_metrics)


if __name__ == "__main__":
    main()
