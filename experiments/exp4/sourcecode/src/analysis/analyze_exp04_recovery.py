from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import CLASS_NAMES, ensure_dir, write_json


def analyze_exp04_recovery(results: str, output_dir: str = "../results/reports") -> dict[str, object]:
    with Path(results).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = list(payload.get("samples", []))
    if not rows:
        raise SystemExit(f"No samples found in {results}")
    out = ensure_dir(output_dir)
    metrics = {
        "overall": aggregate(rows),
        "by_modulation": group_aggregate(rows, "expected_modulation"),
        "by_payload": group_aggregate(rows, "expected_payload"),
        "by_session": group_aggregate(rows, "session_id"),
        "by_failure_stage": count_values(rows, "failure_stage"),
        "confusion_matrix": confusion_matrix(rows),
    }
    write_json(out / "recovery_metrics.json", metrics)
    write_group_csv(out / "recovery_by_modulation.csv", metrics["by_modulation"])
    write_group_csv(out / "recovery_by_payload.csv", metrics["by_payload"])
    write_group_csv(out / "recovery_by_session.csv", metrics["by_session"])
    write_summary(out / "recovery_summary.md", metrics)
    plot_confusion_matrix(metrics["confusion_matrix"], out / "confusion_matrix.png")
    plot_success_bars(metrics, out / "crc_packet_success.png")
    print(f"Recovery analysis written to {out}")
    return metrics


def aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    return {
        "count": len(rows),
        "classification_accuracy": mean([row["expected_modulation"] == row["predicted_modulation"] for row in rows]),
        "crc_pass_rate": mean([bool(row["crc_ok"]) for row in rows]),
        "packet_success_rate": mean([bool(row["packet_success"]) for row in rows]),
        "oracle_crc_pass_rate": mean([bool(row["oracle_demod_crc_ok"]) for row in rows]),
        "oracle_packet_success_rate": mean([bool(row["oracle_packet_success"]) for row in rows]),
        "mean_ber": mean([float(row["ber"]) for row in rows]),
        "mean_cer": mean([float(row["cer"]) for row in rows]),
        "mean_confidence": mean([float(row["classifier_confidence"]) for row in rows]),
    }


def group_aggregate(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return {name: aggregate(items) for name, items in sorted(groups.items())}


def count_values(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return counts


def confusion_matrix(rows: list[dict[str, object]]) -> list[list[int]]:
    index = {name: i for i, name in enumerate(CLASS_NAMES)}
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for row in rows:
        true = index.get(str(row["expected_modulation"]))
        pred = index.get(str(row["predicted_modulation"]))
        if true is not None and pred is not None:
            matrix[true, pred] += 1
    return matrix.tolist()


def mean(values: list[object]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def write_group_csv(path: Path, groups: dict[str, dict[str, float | int]]) -> None:
    fields = ["group", "count", "classification_accuracy", "crc_pass_rate", "packet_success_rate", "oracle_crc_pass_rate", "mean_ber", "mean_cer", "mean_confidence"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, metrics in groups.items():
            writer.writerow({"group": group, **{key: metrics.get(key, "") for key in fields if key != "group"}})


def write_summary(path: Path, metrics: dict[str, object]) -> None:
    overall = dict(metrics["overall"])
    lines = [
        "# Experiment 4 Recovery Analysis",
        "",
        "## Overall",
        f"- samples: {overall['count']}",
        f"- classification_accuracy: {overall['classification_accuracy']:.4f}",
        f"- crc_pass_rate: {overall['crc_pass_rate']:.4f}",
        f"- packet_success_rate: {overall['packet_success_rate']:.4f}",
        f"- oracle_crc_pass_rate: {overall['oracle_crc_pass_rate']:.4f}",
        f"- mean_ber: {overall['mean_ber']:.4f}",
        f"- mean_cer: {overall['mean_cer']:.4f}",
        "",
        "## Failure Stages",
    ]
    for stage, count in dict(metrics["by_failure_stage"]).items():
        lines.append(f"- {stage}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion_matrix(matrix: list[list[int]], path: Path) -> None:
    arr = np.asarray(matrix, dtype=int)
    plt.figure(figsize=(5, 4))
    plt.imshow(arr, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Expected")
    plt.title("Exp4 Modulation Confusion Matrix")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_success_bars(metrics: dict[str, object], path: Path) -> None:
    by_mod = dict(metrics["by_modulation"])
    labels = list(by_mod)
    crc = [float(by_mod[label]["crc_pass_rate"]) for label in labels]
    packet = [float(by_mod[label]["packet_success_rate"]) for label in labels]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, crc, width, label="CRC pass")
    plt.bar(x + width / 2, packet, width, label="Packet success")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Rate")
    plt.title("Exp4 Recovery Success by Modulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", default="../results/reports")
    args = parser.parse_args()
    analyze_exp04_recovery(args.results, args.output_dir)


if __name__ == "__main__":
    main()
