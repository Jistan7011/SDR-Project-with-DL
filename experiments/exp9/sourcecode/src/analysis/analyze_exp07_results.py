from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.common import ensure_dir, load_config, write_json


def analyze_exp07_results(
    summary_path: str = "../results/unknown_analysis/exp07_unknown_analysis_summary.json",
    output_dir: str = "../results/analysis",
    config_path: str = "../config/config.exp07.yaml",
) -> dict[str, Any]:
    cfg = load_config(config_path)
    summary = load_json(summary_path)
    test = summary["splits"].get("test", {})
    minimum = cfg["evaluation"]["success_minimum"]
    result = {
        "test": test,
        "exp5_baseline": {
            "accuracy": cfg["experiment7"]["exp5_accuracy"],
            "worst_recall": cfg["experiment7"]["exp5_worst_recall"],
            "bfsk_bpsk_to_bask_rate": cfg["experiment7"]["exp5_bfsk_bpsk_to_bask_rate"],
        },
        "minimum_pass": {
            "forced_accuracy": float(test.get("forced_3class_accuracy", 0.0)) >= float(minimum["forced_accuracy"]),
            "worst_recall": float(test.get("worst_recall", 0.0)) >= float(minimum["worst_recall"]),
            "bfsk_bpsk_hard_bask_rate": float(test.get("bfsk_bpsk_hard_bask_final_rate", 1.0)) <= float(minimum["bfsk_bpsk_hard_bask_rate"]),
            "candidate_retention": float(test.get("bfsk_bpsk_candidate_retention_rate", 0.0)) >= float(minimum["bfsk_bpsk_candidate_retention"]),
            "true_bask_precision": float(test.get("true_bask_hard_bask_precision", 0.0)) >= float(minimum["true_bask_hard_bask_precision"]),
            "ambiguous_rate": float(test.get("ambiguous_decision_rate", 1.0)) <= float(minimum["ambiguous_rate"]),
        },
    }
    out = ensure_dir(output_dir)
    write_json(out / "exp07_analysis.json", result)
    write_summary(out / "exp07_analysis.md", result)
    write_decision_counts(out / "exp07_final_decision_counts.csv", test.get("final_decision_counts", {}))
    plot_decision_counts(out / "exp07_final_decision_counts.png", test.get("final_decision_counts", {}))
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(path: Path, result: dict[str, Any]) -> None:
    test = result["test"]
    base = result["exp5_baseline"]
    passes = result["minimum_pass"]
    lines = [
        "# Experiment 7 Analysis",
        "",
        f"- forced_3class_accuracy: `{float(test.get('forced_3class_accuracy', 0.0)):.4f}`",
        f"- worst_recall: `{float(test.get('worst_recall', 0.0)):.4f}`",
        f"- BFSK/BPSK hard-BASK final rate: `{float(test.get('bfsk_bpsk_hard_bask_final_rate', 0.0)):.4f}`",
        f"- BFSK/BPSK candidate retention: `{float(test.get('bfsk_bpsk_candidate_retention_rate', 0.0)):.4f}`",
        f"- true BASK hard-BASK precision: `{float(test.get('true_bask_hard_bask_precision', 0.0)):.4f}`",
        f"- ambiguous decision rate: `{float(test.get('ambiguous_decision_rate', 0.0)):.4f}`",
        "",
        "## Exp5 Baseline",
        "",
        f"- accuracy: `{float(base['accuracy']):.4f}`",
        f"- worst recall: `{float(base['worst_recall']):.4f}`",
        f"- BFSK/BPSK -> BASK rate: `{float(base['bfsk_bpsk_to_bask_rate']):.4f}`",
        "",
        "## Minimum Criteria",
        "",
        "| criterion | pass |",
        "| --- | --- |",
    ]
    for key, value in passes.items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_counts(path: Path, counts: dict[str, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["final_decision", "count"])
        for key, value in sorted(counts.items()):
            writer.writerow([key, value])


def plot_decision_counts(path: Path, counts: dict[str, int]) -> None:
    if not counts:
        return
    names = list(counts.keys())
    values = [counts[name] for name in names]
    plt.figure(figsize=(9, 4))
    plt.bar(range(len(names)), values)
    plt.xticks(range(len(names)), names, rotation=35, ha="right")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="../results/unknown_analysis/exp07_unknown_analysis_summary.json")
    parser.add_argument("--output-dir", default="../results/analysis")
    parser.add_argument("--config", default="../config/config.exp07.yaml")
    args = parser.parse_args()
    analyze_exp07_results(args.summary, args.output_dir, args.config)


if __name__ == "__main__":
    main()
