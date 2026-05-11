from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_figures(report_dir: str = "../results/reports", output_dir: str = "../results/figures/final_report") -> Path:
    base = Path(__file__).resolve().parents[3]
    report_arg = Path(report_dir)
    output_arg = Path(output_dir)
    report_dir_path = (report_arg if report_arg.is_absolute() else Path.cwd() / report_arg).resolve()
    output_dir_path = (output_arg if output_arg.is_absolute() else Path.cwd() / output_arg).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    analysis_path = report_dir_path / "exp03_error_analysis.json"
    if not analysis_path.exists():
        raise SystemExit(f"Missing analysis file: {analysis_path}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    splits = list(analysis["splits"].keys())
    accuracies = [analysis["splits"][split]["accuracy"] for split in splits]
    plt.figure(figsize=(6, 4))
    plt.bar(splits, accuracies, color=["#4c78a8", "#f58518"][: len(splits)])
    plt.axhline(0.7594, color="#333333", linestyle="--", linewidth=1, label="exp2 ensemble 0.7594")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Experiment 3 Accuracy by Evaluation Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir_path / "exp03_split_accuracy.png", dpi=160)
    plt.close()

    condition_csv = report_dir_path / "exp03_condition_accuracy.csv"
    if condition_csv.exists():
        rows = list(csv.DictReader(condition_csv.open(encoding="utf-8")))
        session_rows = [row for row in rows if row["condition"] == "session_id"]
        labels = [f"{row['split']}\n{row['value']}" for row in session_rows]
        values = [float(row["accuracy"]) for row in session_rows]
        plt.figure(figsize=(10, 4.8))
        plt.bar(labels, values, color="#54a24b")
        plt.axhline(0.70, color="#333333", linestyle="--", linewidth=1)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Accuracy")
        plt.title("Experiment 3 Session Accuracy")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir_path / "exp03_session_accuracy.png", dpi=160)
        plt.close()

    for split in splits:
        matrix = np.array(analysis["splits"][split]["confusion_matrix"])
        plt.figure(figsize=(4.8, 4.2))
        plt.imshow(matrix, cmap="Blues")
        plt.title(f"Confusion Matrix: {split}")
        plt.xticks(range(3), ["BASK", "BFSK", "BPSK"])
        plt.yticks(range(3), ["BASK", "BFSK", "BPSK"])
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(matrix[i, j]), ha="center", va="center")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(output_dir_path / f"exp03_confusion_{split}.png", dpi=160)
        plt.close()

    print(f"Experiment 3 figures written to {output_dir_path}")
    return output_dir_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="../results/reports")
    parser.add_argument("--output-dir", default="../results/figures/final_report")
    args = parser.parse_args()
    make_figures(report_dir=args.report_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
