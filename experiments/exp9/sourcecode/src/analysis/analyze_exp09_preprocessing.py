from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.common import CLASS_NAMES, ensure_dir, write_json
from src.dataset.iq_dataset import sample_metadata


NUMERIC_KEYS = [
    "exp09_dc_magnitude",
    "exp09_estimated_cfo_hz",
    "exp09_phase_slope_before",
    "exp09_phase_slope_after",
    "exp09_phase_slope_abs_reduction",
    "exp09_amplitude_bimodality_score",
    "exp09_instantaneous_frequency_bimodality_score",
    "exp09_spectral_peak_separation_score",
    "exp09_differential_phase_transition_score",
]


def analyze_exp09_preprocessing(data_root: str, output_dir: str) -> dict[str, Any]:
    root = Path(data_root)
    rows = []
    for split in ["train", "val", "test"]:
        for path in sorted((root / split).glob("*.npz")):
            meta = sample_metadata(path)
            row = {"split": split, "file": path.name, "modulation": str(meta.get("modulation", ""))}
            for key in NUMERIC_KEYS:
                if key in meta:
                    row[key] = float(meta[key])
            rows.append(row)
    summary = {
        "sample_count": len(rows),
        "overall": summarize_rows(rows),
        "by_modulation": {mod: summarize_rows([row for row in rows if row["modulation"] == mod]) for mod in CLASS_NAMES},
    }
    out = ensure_dir(output_dir)
    write_json(out / "exp09_preprocessing_summary.json", summary)
    write_csv(out / "exp09_preprocessing_by_sample.csv", rows)
    write_markdown(out / "exp09_preprocessing_summary.md", summary)
    write_histograms(out, rows)
    return summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result = {}
    for key in NUMERIC_KEYS:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        if len(values):
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p10": float(np.percentile(values, 10)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
            }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = ["split", "file", "modulation", *NUMERIC_KEYS]
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Experiment 9 Preprocessing Analysis", "", f"- sample count: `{summary['sample_count']}`", ""]
    lines.extend(["## By Modulation", "", "| modulation | CFO mean Hz | DC mean | freq evidence mean | phase evidence mean |", "| --- | ---: | ---: | ---: | ---: |"])
    for mod, stats in summary["by_modulation"].items():
        lines.append(
            f"| {mod} | {mean_of(stats, 'exp09_estimated_cfo_hz'):.4f} | {mean_of(stats, 'exp09_dc_magnitude'):.6f} | "
            f"{mean_of(stats, 'exp09_instantaneous_frequency_bimodality_score'):.4f} | {mean_of(stats, 'exp09_differential_phase_transition_score'):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_histograms(out: Path, rows: list[dict[str, Any]]) -> None:
    plot_dir = ensure_dir(out / "figures")
    for key in ["exp09_estimated_cfo_hz", "exp09_dc_magnitude", "exp09_instantaneous_frequency_bimodality_score", "exp09_differential_phase_transition_score"]:
        plt.figure(figsize=(6, 4))
        for mod in CLASS_NAMES:
            values = [float(row[key]) for row in rows if row.get("modulation") == mod and key in row]
            if values:
                plt.hist(values, bins=40, alpha=0.45, label=mod)
        plt.title(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{key}.png", dpi=150)
        plt.close()


def mean_of(stats: dict[str, dict[str, float]], key: str) -> float:
    return float(stats.get(key, {}).get("mean", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument("--output-dir", default="../results/preprocessing_analysis")
    args = parser.parse_args()
    analyze_exp09_preprocessing(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
