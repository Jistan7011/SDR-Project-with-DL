from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common import ensure_dir, write_json


DEFAULT_RUNS = {
    "oshea2018_raw_iq_resnet": "../results/ota_resnet/logs/eval_test.json",
    "oshea2018_raw_iq_vgg": "../results/ota_vgg/logs/eval_test.json",
    "exp5_style_resnet_5ch": "../results/ota_resnet_5ch/logs/eval_test.json",
    "exp8_style_multitask_5ch": "../results/ota_multitask_5ch/logs/eval_test.json",
    "exp8_style_multitask_margin_5ch": "../results/ota_multitask_margin_5ch/logs/eval_test.json",
    "exp9_style_rf_preprocessed_resnet": "../results/ota_rf_preprocessed_resnet/logs/eval_test.json",
}

FIXED_RUNS = {
    "fixed_oshea2018_raw_iq_resnet": "../results/fixed_full_ota_resnet/logs/eval_test.json",
    "fixed_oshea2018_raw_iq_vgg": "../results/fixed_full_ota_vgg/logs/eval_test.json",
    "fixed_exp5_style_resnet_5ch": "../results/fixed_full_ota_resnet_5ch/logs/eval_test.json",
    "fixed_exp8_style_multitask_5ch": "../results/fixed_full_ota_multitask_5ch/logs/eval_test.json",
    "fixed_exp8_style_multitask_margin_5ch": "../results/fixed_full_ota_multitask_margin_5ch/logs/eval_test.json",
    "fixed_exp9_style_rf_preprocessed_resnet": "../results/fixed_full_ota_rf_preprocessed_resnet/logs/eval_test.json",
}


def summarize(results_root: str, output_dir: str, runs: dict[str, str] | None = None) -> dict[str, Any]:
    base = Path(results_root)
    rows = []
    for name, rel in (runs or DEFAULT_RUNS).items():
        path = (base / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            rows.append({"run": name, "status": "missing", "path": str(path)})
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        report = data.get("classification_report", {})
        rows.append(
            {
                "run": name,
                "status": "ok",
                "path": str(path),
                "accuracy": data.get("accuracy"),
                "macro_f1": data.get("macro_f1"),
                "worst_recall": data.get("worst_recall") or min_recall_from_report(report),
                "bask_recall": recall_from(data, report, "BASK"),
                "bfsk_recall": recall_from(data, report, "BFSK"),
                "bpsk_recall": recall_from(data, report, "BPSK"),
                "bfsk_bpsk_to_bask_rate": data.get("bfsk_bpsk_to_bask_rate"),
                "model_type": data.get("model_type"),
                "feature_mode": data.get("feature_mode"),
            }
        )
    out = ensure_dir(output_dir)
    result = {"rows": rows}
    write_json(out / "oshea2018_extension_summary.json", result)
    (out / "oshea2018_extension_summary.md").write_text(render_markdown(rows), encoding="utf-8")
    return result


def recall_from(data: dict[str, Any], report: dict[str, Any], name: str) -> float | None:
    if "class_recall" in data and name in data["class_recall"]:
        return data["class_recall"][name]
    if name in report and "recall" in report[name]:
        return report[name]["recall"]
    return None


def min_recall_from_report(report: dict[str, Any]) -> float | None:
    recalls = [report.get(name, {}).get("recall") for name in ["BASK", "BFSK", "BPSK"]]
    recalls = [float(value) for value in recalls if value is not None]
    return min(recalls) if recalls else None


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = ["run", "accuracy", "macro_f1", "worst_recall", "BASK", "BFSK", "BPSK", "absorption", "feature_mode"]
    lines = ["# Oshea2018 Extension Summary", "", "|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        if row.get("status") != "ok":
            lines.append(f"|{row['run']}|missing||||||| |")
            continue
        lines.append(
            "|"
            + "|".join(
                [
                    str(row.get("run", "")),
                    fmt(row.get("accuracy")),
                    fmt(row.get("macro_f1")),
                    fmt(row.get("worst_recall")),
                    fmt(row.get("bask_recall")),
                    fmt(row.get("bfsk_recall")),
                    fmt(row.get("bpsk_recall")),
                    fmt(row.get("bfsk_bpsk_to_bask_rate")),
                    str(row.get("feature_mode") or ""),
                ]
            )
            + "|"
        )
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default=".")
    parser.add_argument("--output-dir", default="../results/oshea2018_extension_summary")
    parser.add_argument("--profile", choices=["clean", "fixed"], default="clean")
    args = parser.parse_args()
    runs = FIXED_RUNS if args.profile == "fixed" else DEFAULT_RUNS
    summarize(args.results_root, args.output_dir, runs)


if __name__ == "__main__":
    main()
