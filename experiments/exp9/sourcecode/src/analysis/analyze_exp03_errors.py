from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def load_model(checkpoint_path: Path, cfg: dict[str, Any]) -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "fusion_resnet1d"))),
        input_channels=int(model_cfg.get("input_channels", cfg["model"]["input_channels"])),
        num_classes=int(model_cfg.get("num_classes", cfg["model"]["num_classes"])),
        dropout=float(model_cfg.get("dropout", cfg["model"]["dropout"])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def predict(model: torch.nn.Module, dataset: IQDataset, batch_size: int) -> list[int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds: list[int] = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.extend(model(xb).argmax(dim=1).numpy().tolist())
    return preds


def analyze_split(cfg: dict[str, Any], data_root: str, split: str, checkpoints: list[Path]) -> dict[str, Any]:
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq"))
    dataset = IQDataset(data_root, split, feature_mode=feature_mode)
    batch_size = int(cfg["train"]["batch_size"])
    seed_names = [path.parent.parent.name for path in checkpoints]
    all_preds = {seed_name: predict(load_model(path, cfg), dataset, batch_size) for seed_name, path in zip(seed_names, checkpoints)}

    rows: list[dict[str, Any]] = []
    for index, path in enumerate(dataset.files):
        meta = sample_metadata(path)
        label = CLASS_NAMES.index(str(meta["modulation"]))
        seed_preds = [all_preds[name][index] for name in seed_names]
        vote_counts = np.bincount(seed_preds, minlength=len(CLASS_NAMES))
        ensemble = int(np.argmax(vote_counts))
        row = {
            "file": str(path),
            "split": split,
            "label": label,
            "true": CLASS_NAMES[label],
            "ensemble": ensemble,
            "ensemble_name": CLASS_NAMES[ensemble],
            "all_seeds_agree": len(set(seed_preds)) == 1,
            "session_id": str(meta.get("session_id", "")),
            "payload": str(meta.get("payload", "")),
            "tx_vga_gain": str(meta.get("tx_vga_gain", "")),
            "rx_gain": str(meta.get("rx_gain", "")),
            "baseband_offset_hz": str(meta.get("baseband_offset_hz", "")),
            "snr_db": float(meta.get("snr_db", np.nan)),
            "snr_bin": snr_bin(float(meta.get("snr_db", np.nan))),
            "tx_rx_distance_m": str(meta.get("tx_rx_distance_m", "")),
            "antenna_layout": str(meta.get("antenna_layout", "")),
            "rf_path": str(meta.get("rf_path", "")),
            "rf_cable_between_sdr": str(meta.get("rf_cable_between_sdr", "")),
        }
        for seed_name, preds in all_preds.items():
            pred = int(preds[index])
            row[f"{seed_name}_pred"] = pred
            row[f"{seed_name}_pred_name"] = CLASS_NAMES[pred]
        rows.append(row)

    return summarize_rows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    condition_keys = ["session_id", "payload", "tx_vga_gain", "rx_gain", "baseband_offset_hz", "snr_bin", "tx_rx_distance_m", "antenna_layout"]
    agreement_rows = [row for row in rows if row["all_seeds_agree"]]
    disagreement_rows = [row for row in rows if not row["all_seeds_agree"]]
    return {
        "sample_count": len(rows),
        "accuracy": accuracy([row["label"] == row["ensemble"] for row in rows]),
        "class_recall": class_recall(rows),
        "confusion_matrix": confusion(rows),
        "agreement_summary": {
            "all_seeds_agree_fraction": len(agreement_rows) / max(len(rows), 1),
            "all_seeds_agree_accuracy": accuracy([row["label"] == row["ensemble"] for row in agreement_rows]),
            "seed_disagreement_fraction": len(disagreement_rows) / max(len(rows), 1),
            "seed_disagreement_accuracy": accuracy([row["label"] == row["ensemble"] for row in disagreement_rows]),
        },
        "condition_summary": {key: group_accuracy(rows, key) for key in condition_keys},
        "top_error_flows": error_flows(rows, limit=30),
        "rows": rows,
    }


def analyze(config_path: str, data_root: str, checkpoints: list[str], output_dir: str, splits: list[str]) -> dict[str, Any]:
    cfg = load_config(config_path)
    checkpoint_paths = [Path(path) for path in checkpoints]
    result = {
        "model_type": str(cfg["model"].get("type", "fusion_resnet1d")),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "splits": {},
    }
    out = ensure_dir(output_dir)
    all_condition_rows: list[dict[str, Any]] = []
    all_error_rows: list[dict[str, Any]] = []

    for split in splits:
        split_result = analyze_split(cfg, data_root, split, checkpoint_paths)
        rows = split_result.pop("rows")
        result["splits"][split] = split_result
        write_csv(out / f"exp03_{split}_predictions.csv", rows)
        for condition, values in split_result["condition_summary"].items():
            for value, metrics in values.items():
                all_condition_rows.append({"split": split, "condition": condition, "value": value, **metrics})
        for item in split_result["top_error_flows"]:
            all_error_rows.append({"split": split, **item})

    result["error_reduction_targets"] = error_reduction_targets(result)
    write_json(out / "exp03_error_analysis.json", result)
    write_csv(out / "exp03_condition_accuracy.csv", all_condition_rows)
    write_csv(out / "exp03_top_error_flows.csv", all_error_rows)
    write_markdown(out / "exp03_error_analysis.md", result)
    return result


def error_reduction_targets(result: dict[str, Any]) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for split, summary in result["splits"].items():
        flows = summary["top_error_flows"]
        bask_confusions = sum(item["count"] for item in flows if item["predicted"] == "BASK" and item["true"] in {"BFSK", "BPSK"})
        targets[split] = {"bfsk_bpsk_to_bask_top_flow_count": bask_confusions}
    return targets


def accuracy(items: list[bool]) -> float:
    return float(np.mean(items)) if items else float("nan")


def class_recall(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {name: accuracy([row["ensemble"] == index for row in rows if row["label"] == index]) for index, name in enumerate(CLASS_NAMES)}


def confusion(rows: list[dict[str, Any]]) -> list[list[int]]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for row in rows:
        matrix[int(row["label"])][int(row["ensemble"])] += 1
    return matrix


def group_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key, ""))].append(row["label"] == row["ensemble"])
    return {value: {"accuracy": accuracy(values), "count": len(values), "errors": int(len(values) - sum(values))} for value, values in sorted(buckets.items())}


def error_flows(rows: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in rows:
        if row["label"] == row["ensemble"]:
            continue
        counts[(row["true"], row["ensemble_name"], str(row["session_id"]), str(row["payload"]), str(row["snr_bin"]))] += 1
    return [{"true": t, "predicted": p, "session_id": s, "payload": payload, "snr_bin": snr, "count": count} for (t, p, s, payload, snr), count in counts.most_common(limit)]


def snr_bin(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value < -2.0:
        return "< -2 dB"
    if value < 0.0:
        return "-2 to 0 dB"
    if value < 2.0:
        return "0 to 2 dB"
    if value < 4.0:
        return "2 to 4 dB"
    if value < 6.0:
        return "4 to 6 dB"
    return ">= 6 dB"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    rows = []
    for split, summary in result["splits"].items():
        recall = summary["class_recall"]
        rows.append([split, f"{summary['accuracy']:.4f}", f"{recall['BASK']:.4f}", f"{recall['BFSK']:.4f}", f"{recall['BPSK']:.4f}", summary["sample_count"]])
    text = f"""# Experiment 3 Error Analysis

## Summary

{markdown_table(['split', 'accuracy', 'BASK recall', 'BFSK recall', 'BPSK recall', 'samples'], rows)}

## Interpretation

Use `test_a` for the 10 cm OTA reference comparison and `test_b` for the 1 m OTA distance-generalization result.
The key target is reducing BFSK/BPSK to BASK error flow while preserving session-held-out accuracy.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp03.yaml")
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "../results/exp03_fusion_resnet/fusion_resnet1d_seed42/checkpoints/best.pt",
            "../results/exp03_fusion_resnet/fusion_resnet1d_seed43/checkpoints/best.pt",
            "../results/exp03_fusion_resnet/fusion_resnet1d_seed44/checkpoints/best.pt",
        ],
    )
    parser.add_argument("--splits", nargs="+", default=["test_a", "test_b"])
    parser.add_argument("--output-dir", default="../results/reports")
    args = parser.parse_args()
    analyze(args.config, args.data_root, args.checkpoints, args.output_dir, args.splits)


if __name__ == "__main__":
    main()
