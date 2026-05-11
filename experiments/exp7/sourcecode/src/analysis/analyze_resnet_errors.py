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
        str(ckpt.get("model_type", model_cfg.get("type", "resnet1d"))),
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


def accuracy(items: list[bool]) -> float:
    return float(np.mean(items)) if items else float("nan")


def group_accuracy(rows: list[dict[str, Any]], key: str, pred_key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[str(row[key])].append(row["label"] == row[pred_key])
    return {
        value: {"accuracy": accuracy(values), "count": len(values), "errors": int(len(values) - sum(values))}
        for value, values in sorted(buckets.items(), key=lambda item: item[0])
    }


def confusion(rows: list[dict[str, Any]], pred_key: str) -> list[list[int]]:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for row in rows:
        matrix[int(row["label"])][int(row[pred_key])] += 1
    return matrix


def error_flows(rows: list[dict[str, Any]], pred_key: str, limit: int = 20) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for row in rows:
        if row["label"] == row[pred_key]:
            continue
        counts[
            (
                CLASS_NAMES[int(row["label"])],
                CLASS_NAMES[int(row[pred_key])],
                str(row["session_id"]),
                str(row["payload"]),
                str(row["snr_bin"]),
            )
        ] += 1
    return [
        {
            "true": true,
            "predicted": pred,
            "session_id": session_id,
            "payload": payload,
            "snr_bin": bucket,
            "count": count,
        }
        for (true, pred, session_id, payload, bucket), count in counts.most_common(limit)
    ]


def class_recall(rows: list[dict[str, Any]], pred_key: str) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for idx, name in enumerate(CLASS_NAMES):
        selected = [row for row in rows if row["label"] == idx]
        recalls[name] = accuracy([row[pred_key] == idx for row in selected])
    return recalls


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


def analyze(
    config_path: str,
    data_root: str,
    checkpoints: list[str],
    output_dir: str,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq"))
    dataset = IQDataset(data_root, "test", feature_mode=feature_mode)
    batch_size = int(cfg["train"]["batch_size"])
    checkpoint_paths = [Path(path) for path in checkpoints]
    seed_names = [path.parent.parent.name for path in checkpoint_paths]
    all_preds: dict[str, list[int]] = {}

    for seed_name, checkpoint_path in zip(seed_names, checkpoint_paths):
        model = load_model(checkpoint_path, cfg)
        all_preds[seed_name] = predict(model, dataset, batch_size)

    rows: list[dict[str, Any]] = []
    for idx, path in enumerate(dataset.files):
        meta = sample_metadata(path)
        label = CLASS_NAMES.index(str(meta["modulation"]))
        seed_preds = [all_preds[name][idx] for name in seed_names]
        vote_counts = np.bincount(seed_preds, minlength=len(CLASS_NAMES))
        ensemble_pred = int(np.argmax(vote_counts))
        row = {
            "file": str(path),
            "label": label,
            "true": CLASS_NAMES[label],
            "ensemble_pred": ensemble_pred,
            "ensemble_pred_name": CLASS_NAMES[ensemble_pred],
            "seed_agreement_count": int(vote_counts[ensemble_pred]),
            "all_seeds_agree": len(set(seed_preds)) == 1,
            "session_id": str(meta.get("session_id", "")),
            "payload": str(meta.get("payload", "")),
            "tx_vga_gain": str(meta.get("tx_vga_gain", "")),
            "rx_gain": str(meta.get("rx_gain", "")),
            "baseband_offset_hz": str(meta.get("baseband_offset_hz", "")),
            "snr_db": float(meta.get("snr_db", np.nan)),
            "snr_bin": snr_bin(float(meta.get("snr_db", np.nan))),
        }
        for seed_name, preds in all_preds.items():
            pred = int(preds[idx])
            row[f"{seed_name}_pred"] = pred
            row[f"{seed_name}_pred_name"] = CLASS_NAMES[pred]
        rows.append(row)

    pred_keys = [f"{name}_pred" for name in all_preds] + ["ensemble"]
    for row in rows:
        row["ensemble"] = row["ensemble_pred"]

    seed_summary = {
        pred_key: {
            "accuracy": accuracy([row["label"] == row[pred_key] for row in rows]),
            "class_recall": class_recall(rows, pred_key),
            "confusion_matrix": confusion(rows, pred_key),
        }
        for pred_key in pred_keys
    }
    agreement_rows = [row for row in rows if row["all_seeds_agree"]]
    disagreement_rows = [row for row in rows if not row["all_seeds_agree"]]
    agreement_summary = {
        "all_seeds_agree_fraction": len(agreement_rows) / len(rows),
        "all_seeds_agree_accuracy": accuracy([row["label"] == row["ensemble"] for row in agreement_rows]),
        "seed_disagreement_fraction": len(disagreement_rows) / len(rows),
        "seed_disagreement_accuracy": accuracy([row["label"] == row["ensemble"] for row in disagreement_rows]),
    }

    group_keys = ["session_id", "payload", "tx_vga_gain", "rx_gain", "baseband_offset_hz", "snr_bin"]
    condition_summary = {key: group_accuracy(rows, key, "ensemble") for key in group_keys}
    top_errors = error_flows(rows, "ensemble", limit=30)

    result = {
        "model": "resnet1d",
        "checkpoint_names": seed_names,
        "sample_count": len(rows),
        "seed_summary": seed_summary,
        "agreement_summary": agreement_summary,
        "condition_summary": condition_summary,
        "top_ensemble_error_flows": top_errors,
    }

    out = ensure_dir(output_dir)
    write_json(out / "resnet1d_error_analysis.json", result)
    write_csv(out / "resnet1d_top_error_flows.csv", top_errors)
    write_csv(
        out / "resnet1d_condition_accuracy.csv",
        [
            {"condition": key, "value": value, **metrics}
            for key, values in condition_summary.items()
            for value, metrics in values.items()
        ],
    )
    write_markdown(out / "resnet1d_error_analysis.md", result)
    return result


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    ensemble = result["seed_summary"]["ensemble"]
    recall = ensemble["class_recall"]
    agreement = result["agreement_summary"]
    conditions = result["condition_summary"]
    top_errors = result["top_ensemble_error_flows"][:12]

    session_rows = [
        [value, f"{metrics['accuracy']:.4f}", metrics["count"], metrics["errors"]]
        for value, metrics in conditions["session_id"].items()
    ]
    payload_rows = [
        [value, f"{metrics['accuracy']:.4f}", metrics["count"], metrics["errors"]]
        for value, metrics in sorted(conditions["payload"].items(), key=lambda item: item[1]["accuracy"])
    ]
    snr_rows = [
        [value, f"{metrics['accuracy']:.4f}", metrics["count"], metrics["errors"]]
        for value, metrics in conditions["snr_bin"].items()
    ]
    error_rows = [
        [item["true"], item["predicted"], item["session_id"], item["payload"], item["snr_bin"], item["count"]]
        for item in top_errors
    ]

    text = f"""# ResNet1D Error Analysis

## Summary

This analysis uses the three ResNet1D checkpoints from the 15-session Experiment 2 run and reports both per-seed behavior and majority-vote ensemble behavior.

```text
test samples: {result['sample_count']}
ensemble accuracy: {ensemble['accuracy']:.4f}
ensemble recall BASK: {recall['BASK']:.4f}
ensemble recall BFSK: {recall['BFSK']:.4f}
ensemble recall BPSK: {recall['BPSK']:.4f}
all-seed agreement fraction: {agreement['all_seeds_agree_fraction']:.4f}
all-seed agreement accuracy: {agreement['all_seeds_agree_accuracy']:.4f}
seed-disagreement fraction: {agreement['seed_disagreement_fraction']:.4f}
seed-disagreement accuracy: {agreement['seed_disagreement_accuracy']:.4f}
```

## Session Accuracy

{markdown_table(['session', 'accuracy', 'count', 'errors'], session_rows)}

## Payload Accuracy

{markdown_table(['payload', 'accuracy', 'count', 'errors'], payload_rows)}

## SNR Bin Accuracy

{markdown_table(['snr_bin', 'accuracy', 'count', 'errors'], snr_rows)}

## Top Ensemble Error Flows

{markdown_table(['true', 'predicted', 'session', 'payload', 'snr_bin', 'count'], error_rows)}

## Interpretation

ResNet1D improves class-balance stability compared with CNN1D and VGG1D, but the remaining errors are still condition-dependent. The most useful next improvement is not another blind architecture jump; it is targeted data and preprocessing work around the weakest conditions identified above.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/config.exp02.yaml")
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "../results/exp02_15session_resnet/resnet1d_seed42/checkpoints/best.pt",
            "../results/exp02_15session_resnet/resnet1d_seed43/checkpoints/best.pt",
            "../results/exp02_15session_resnet/resnet1d_seed44/checkpoints/best.pt",
        ],
    )
    parser.add_argument("--output-dir", default="../results/reports")
    args = parser.parse_args()
    analyze(args.config, args.data_root, args.checkpoints, args.output_dir)


if __name__ == "__main__":
    main()
