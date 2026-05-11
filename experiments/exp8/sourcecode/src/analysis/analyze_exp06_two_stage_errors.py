from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.common import CLASS_NAMES, ensure_dir, write_json


def analyze_exp06_two_stage_errors(summary_path: str, output_dir: str, exp5_rate: float = 0.1894) -> dict[str, object]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    test = summary["test"]
    rate = float(test["bfsk_bpsk_to_bask_rate"])
    reduction = (exp5_rate - rate) / exp5_rate if exp5_rate > 0 else 0.0
    recalls = test["class_recall"]
    result = {
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "class_recall": recalls,
        "worst_recall": min(float(v) for v in recalls.values()),
        "bfsk_bpsk_to_bask_rate": rate,
        "exp5_bfsk_bpsk_to_bask_rate": exp5_rate,
        "error_reduction": reduction,
        "confusion_matrix": test["confusion_matrix"],
        "success_minimum": {
            "accuracy": float(test["accuracy"]) >= 0.76,
            "bfsk_recall": float(recalls["BFSK"]) >= 0.76,
            "bpsk_recall": float(recalls["BPSK"]) >= 0.74,
            "worst_recall": min(float(v) for v in recalls.values()) >= 0.72,
            "bask_recall": float(recalls["BASK"]) >= 0.72,
            "error_reduction": reduction >= 0.25,
        },
    }
    out = ensure_dir(output_dir)
    write_json(out / "exp06_two_stage_error_analysis.json", result)
    write_md(out / "exp06_two_stage_error_analysis.md", result)
    return result


def write_md(path: Path, result: dict[str, object]) -> None:
    recalls = result["class_recall"]
    lines = [
        "# Experiment 6 Two-Stage Error Analysis",
        "",
        f"- Accuracy: `{result['accuracy']:.4f}`",
        f"- Macro F1: `{result['macro_f1']:.4f}`",
        f"- Worst recall: `{result['worst_recall']:.4f}`",
        f"- BFSK/BPSK -> BASK rate: `{result['bfsk_bpsk_to_bask_rate']:.4f}`",
        f"- Reduction vs exp5: `{result['error_reduction']:.4f}`",
        "",
        "## Class Recall",
        "",
        "| class | recall |",
        "| --- | ---: |",
    ]
    for name in CLASS_NAMES:
        lines.append(f"| {name} | {float(recalls[name]):.4f} |")
    lines.extend(["", "## Minimum Criteria", "", "| criterion | pass |", "| --- | --- |"])
    for key, ok in result["success_minimum"].items():
        lines.append(f"| {key} | {ok} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="../results/two_stage_eval/two_stage_summary.json")
    parser.add_argument("--output-dir", default="../results/analysis")
    parser.add_argument("--exp5-rate", type=float, default=0.1894)
    args = parser.parse_args()
    analyze_exp06_two_stage_errors(args.summary, args.output_dir, args.exp5_rate)


if __name__ == "__main__":
    main()
