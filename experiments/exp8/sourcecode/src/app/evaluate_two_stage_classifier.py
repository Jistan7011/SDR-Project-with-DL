from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def evaluate_two_stage_classifier(
    stage1_checkpoint: str,
    stage2_checkpoint: str,
    config_path: str = "../config/config.exp06.yaml",
    data_root: str | None = None,
    output_dir: str = "../results/two_stage_eval",
    threshold: float | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = data_root or cfg["dataset"]["source_root"]
    result_root = ensure_dir(output_dir)
    stage1 = load_binary_model(stage1_checkpoint, cfg)
    stage2 = load_binary_model(stage2_checkpoint, cfg)
    if threshold is None:
        sweep = threshold_sweep(stage1, stage2, cfg, root, "val")
        threshold = float(sweep["best_threshold"])
        write_json(result_root / "threshold_sweep.json", sweep)
    else:
        sweep = {"best_threshold": threshold}
    val_result = evaluate_split(stage1, stage2, cfg, root, "val", threshold, result_root)
    test_result = evaluate_split(stage1, stage2, cfg, root, "test", threshold, result_root)
    summary = {
        "stage1_checkpoint": stage1_checkpoint,
        "stage2_checkpoint": stage2_checkpoint,
        "threshold": threshold,
        "threshold_sweep": sweep,
        "val": compact_metrics(val_result),
        "test": compact_metrics(test_result),
    }
    write_json(result_root / "two_stage_summary.json", summary)
    write_summary_md(result_root / "two_stage_summary.md", summary)
    return summary


def load_binary_model(checkpoint: str, cfg: dict[str, Any]) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    class_names = list(ckpt["class_names"])
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "resnet1d"))),
        input_channels=int(model_cfg.get("input_channels", cfg["model"]["input_channels"])),
        num_classes=len(class_names),
        dropout=float(model_cfg.get("dropout", cfg["model"]["dropout"])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def threshold_sweep(stage1: torch.nn.Module, stage2: torch.nn.Module, cfg: dict[str, Any], data_root: str, split: str) -> dict[str, Any]:
    sweep_cfg = cfg["experiment6"]["threshold_sweep"]
    min_t = float(sweep_cfg["min"])
    max_t = float(sweep_cfg["max"])
    step = float(sweep_cfg["step"])
    min_bask_recall = float(sweep_cfg["min_bask_recall"])
    rows = []
    best_row: dict[str, Any] | None = None
    for threshold in np.arange(min_t, max_t + 0.5 * step, step):
        metrics = evaluate_split(stage1, stage2, cfg, data_root, split, float(threshold), None)
        row = compact_metrics(metrics)
        row["threshold"] = float(threshold)
        rows.append(row)
        if row["class_recall"]["BASK"] < min_bask_recall:
            continue
        if best_row is None or threshold_objective(row) > threshold_objective(best_row):
            best_row = row
    if best_row is None:
        best_row = max(rows, key=threshold_objective)
    return {"best_threshold": best_row["threshold"], "rows": rows, "objective": "minimize_bfsk_bpsk_to_bask_with_bask_recall_floor"}


def threshold_objective(row: dict[str, Any]) -> tuple[float, float, float]:
    return (-float(row["bfsk_bpsk_to_bask_rate"]), float(row["worst_recall"]), float(row["accuracy"]))


def evaluate_split(
    stage1: torch.nn.Module,
    stage2: torch.nn.Module,
    cfg: dict[str, Any],
    data_root: str,
    split: str,
    threshold: float,
    output_root: Path | None,
) -> dict[str, Any]:
    feature_mode = str(cfg["dataset"].get("feature_mode", "iq"))
    dataset = IQDataset(data_root, split, feature_mode=feature_mode, preload=bool(cfg["dataset"].get("preload", False)))
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))
    y_true: list[int] = []
    y_pred: list[int] = []
    rows: list[dict[str, Any]] = []
    file_offset = 0
    with torch.no_grad():
        for xb, yb in loader:
            stage1_prob = torch.softmax(stage1(xb), dim=1).numpy()
            stage2_prob = torch.softmax(stage2(xb), dim=1).numpy()
            for i in range(len(xb)):
                true_label = int(yb[i])
                non_bask_prob = float(stage1_prob[i, 1])
                stage1_pred_nonbask = non_bask_prob >= threshold
                if not stage1_pred_nonbask:
                    pred_label = CLASS_NAMES.index("BASK")
                    stage2_name = ""
                    stage2_conf = float(np.max(stage2_prob[i]))
                else:
                    stage2_idx = int(np.argmax(stage2_prob[i]))
                    pred_label = CLASS_NAMES.index("BFSK" if stage2_idx == 0 else "BPSK")
                    stage2_name = "BFSK" if stage2_idx == 0 else "BPSK"
                    stage2_conf = float(stage2_prob[i, stage2_idx])
                y_true.append(true_label)
                y_pred.append(pred_label)
                meta = sample_metadata(dataset.files[file_offset + i])
                rows.append(
                    {
                        "file": str(dataset.files[file_offset + i]),
                        "true": CLASS_NAMES[true_label],
                        "predicted": CLASS_NAMES[pred_label],
                        "ok": true_label == pred_label,
                        "stage1_non_bask_prob": non_bask_prob,
                        "stage1_threshold": threshold,
                        "stage1_decision": "NON_BASK" if stage1_pred_nonbask else "BASK",
                        "stage2_decision": stage2_name,
                        "stage2_confidence": stage2_conf,
                        "session_id": meta.get("session_id", ""),
                        "payload": meta.get("payload", ""),
                        "tx_vga_gain": meta.get("tx_vga_gain", ""),
                        "rx_gain": meta.get("rx_gain", ""),
                        "baseband_offset_hz": meta.get("baseband_offset_hz", ""),
                    }
                )
            file_offset += len(xb)
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(true, pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(true, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics = {
        "split": split,
        "threshold": threshold,
        "accuracy": float(np.mean(true == pred)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "bfsk_bpsk_to_bask_rate": bfsk_bpsk_to_bask_rate(cm),
        "condition_accuracy": condition_accuracy(rows),
        "rows": rows,
    }
    if output_root is not None:
        write_json(output_root / f"{split}_metrics.json", {k: v for k, v in metrics.items() if k != "rows"})
        write_predictions(output_root / f"{split}_stage_predictions.csv", rows)
        plot_confusion(output_root / f"{split}_confusion_matrix.png", cm)
    return metrics


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    recalls = {name: float(metrics["classification_report"][name]["recall"]) for name in CLASS_NAMES}
    return {
        "accuracy": float(metrics["accuracy"]),
        "class_recall": recalls,
        "macro_f1": float(metrics["classification_report"]["macro avg"]["f1-score"]),
        "worst_recall": float(min(recalls.values())),
        "bfsk_bpsk_to_bask_rate": float(metrics["bfsk_bpsk_to_bask_rate"]),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def bfsk_bpsk_to_bask_rate(cm: np.ndarray) -> float:
    bask = CLASS_NAMES.index("BASK")
    bfsk = CLASS_NAMES.index("BFSK")
    bpsk = CLASS_NAMES.index("BPSK")
    return float((cm[bfsk, bask] + cm[bpsk, bask]) / max(cm[bfsk].sum() + cm[bpsk].sum(), 1))


def condition_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ["session_id", "payload", "tx_vga_gain", "rx_gain", "baseband_offset_hz"]
    buckets: dict[str, dict[str, list[bool]]] = {key: {} for key in keys}
    for row in rows:
        for key in keys:
            value = str(row.get(key, ""))
            buckets[key].setdefault(value, []).append(bool(row["ok"]))
    return {key: {value: float(np.mean(items)) for value, items in values.items()} for key, values in buckets.items()}


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(path: Path, cm: np.ndarray) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Experiment 6 Two-Stage Evaluation",
        "",
        f"Threshold: `{summary['threshold']:.3f}`",
        "",
        "| split | accuracy | macro F1 | BASK recall | BFSK recall | BPSK recall | worst recall | BFSK/BPSK -> BASK rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ["val", "test"]:
        m = summary[split]
        r = m["class_recall"]
        lines.append(
            f"| {split} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {r['BASK']:.4f} | "
            f"{r['BFSK']:.4f} | {r['BPSK']:.4f} | {m['worst_recall']:.4f} | {m['bfsk_bpsk_to_bask_rate']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--config", default="../config/config.exp06.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="../results/two_stage_eval")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    evaluate_two_stage_classifier(args.stage1_checkpoint, args.stage2_checkpoint, args.config, args.data_root, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
