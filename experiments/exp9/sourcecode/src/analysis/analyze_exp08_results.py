from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.common import ensure_dir, write_json


def analyze_exp08_results(results_root: str, output_dir: str) -> dict[str, Any]:
    root = Path(results_root)
    summaries = sorted(root.glob("**/eval_summary.json"))
    rows = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        run_name = path.parent.name
        rows.append(
            {
                "run": run_name,
                "path": str(path.parent),
                "accuracy": float(data.get("accuracy", 0.0)),
                "macro_f1": float(data.get("macro_f1", 0.0)),
                "worst_recall": float(data.get("worst_recall", 0.0)),
                "bask_recall": float(data.get("class_recall", {}).get("BASK", 0.0)),
                "bfsk_recall": float(data.get("class_recall", {}).get("BFSK", 0.0)),
                "bpsk_recall": float(data.get("class_recall", {}).get("BPSK", 0.0)),
                "bfsk_bpsk_to_bask_rate": float(data.get("bfsk_bpsk_to_bask_rate", 0.0)),
                "bask_to_nonbask_rate": float(data.get("bask_to_nonbask_rate", 0.0)),
            }
        )
    rows.sort(key=lambda row: (row["worst_recall"], row["accuracy"]), reverse=True)
    out = ensure_dir(output_dir)
    write_json(out / "exp08_comparison.json", {"runs": rows, "best": rows[0] if rows else None})
    write_csv(out / "exp08_comparison.csv", rows)
    write_markdown(out / "exp08_summary.md", rows)
    return {"runs": rows, "best": rows[0] if rows else None}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(csv_escape(row.get(key, "")) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Experiment 8 Result Summary", ""]
    if not rows:
        lines.append("No evaluation summaries found.")
    else:
        best = rows[0]
        lines.extend(
            [
                "## Best Run",
                "",
                f"- run: `{best['run']}`",
                f"- accuracy: `{best['accuracy']:.4f}`",
                f"- worst recall: `{best['worst_recall']:.4f}`",
                f"- BFSK/BPSK -> BASK rate: `{best['bfsk_bpsk_to_bask_rate']:.4f}`",
                "",
                "## All Runs",
                "",
                "| run | accuracy | macro F1 | worst recall | BASK recall | BFSK recall | BPSK recall | BFSK/BPSK -> BASK |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['run']} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['worst_recall']:.4f} | "
                f"{row['bask_recall']:.4f} | {row['bfsk_recall']:.4f} | {row['bpsk_recall']:.4f} | {row['bfsk_bpsk_to_bask_rate']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def csv_escape(value: object) -> str:
    text = str(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="../results")
    parser.add_argument("--results", default=None, help="Compatibility alias; parent directory is used when a file is passed.")
    parser.add_argument("--output-dir", default="../results/analysis")
    args = parser.parse_args()
    root = Path(args.results).parent if args.results else Path(args.results_root)
    analyze_exp08_results(str(root), args.output_dir)


if __name__ == "__main__":
    main()
