from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def load_model(checkpoint: str, config_path: str) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    cfg = load_config(config_path)
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "cnn1d"))),
        input_channels=int(model_cfg.get("input_channels", cfg["model"]["input_channels"])),
        num_classes=int(model_cfg.get("num_classes", cfg["model"]["num_classes"])),
        dropout=float(model_cfg.get("dropout", cfg["model"]["dropout"])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt_cfg


def collect_logits(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    ckpt_cfg: dict[str, Any],
    data_root: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    dataset_cfg = ckpt_cfg.get("dataset", cfg["dataset"])
    feature_mode = str(dataset_cfg.get("feature_mode", cfg["dataset"].get("feature_mode", "iq")))
    preload = bool(dataset_cfg.get("preload", cfg["dataset"].get("preload", False)))
    dataset = IQDataset(data_root, split, feature_mode=feature_mode, preload=preload)
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits.append(model(xb).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0), dataset.files


def metrics_from_logits(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(logits + bias[None, :], axis=1)
    accuracy = float(np.mean(pred == labels))
    recalls: dict[str, float] = {}
    recall_values: list[float] = []
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for true, guessed in zip(labels, pred):
        cm[int(true), int(guessed)] += 1
    for idx, name in enumerate(CLASS_NAMES):
        mask = labels == idx
        recall = float(np.mean(pred[mask] == idx)) if np.any(mask) else 0.0
        recalls[name] = recall
        recall_values.append(recall)
    return {
        "accuracy": accuracy,
        "class_recall": recalls,
        "mean_recall": float(np.mean(recall_values)),
        "worst_recall": float(np.min(recall_values)),
        "confusion_matrix": cm.tolist(),
        "predictions": pred,
    }


def tune_bias(
    logits: np.ndarray,
    labels: np.ndarray,
    search_min: float,
    search_max: float,
    step: float,
    objective: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.arange(search_min, search_max + 0.5 * step, step, dtype=np.float32)
    best_bias = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    best_metrics = metrics_from_logits(logits, labels, best_bias)
    best_key = objective_key(best_metrics, objective)
    for bfs_k_bias in values:
        for bps_k_bias in values:
            bias = np.array([0.0, bfs_k_bias, bps_k_bias], dtype=np.float32)
            metrics = metrics_from_logits(logits, labels, bias)
            key = objective_key(metrics, objective)
            if key > best_key:
                best_key = key
                best_bias = bias
                best_metrics = metrics
    return best_bias, best_metrics


def objective_key(metrics: dict[str, Any], objective: str) -> tuple[float, float, float]:
    key = objective.lower().replace("-", "_")
    accuracy = float(metrics["accuracy"])
    mean_recall = float(metrics["mean_recall"])
    worst_recall = float(metrics["worst_recall"])
    if key == "accuracy":
        return (accuracy, worst_recall, mean_recall)
    if key == "mean_recall":
        return (mean_recall, worst_recall, accuracy)
    if key == "worst_recall":
        return (worst_recall, accuracy, mean_recall)
    if key == "balanced_score":
        return (0.5 * accuracy + 0.5 * worst_recall, worst_recall, accuracy)
    raise ValueError(f"Unsupported objective: {objective}")


def condition_accuracy(files: list[Path], pred: np.ndarray) -> dict[str, dict[str, float]]:
    keys = ["session_id", "payload", "tx_vga_gain", "rx_gain", "baseband_offset_hz"]
    buckets: dict[str, dict[str, list[bool]]] = {key: {} for key in keys}
    for path, guessed in zip(files, pred):
        meta = sample_metadata(path)
        label = CLASS_NAMES.index(str(meta["modulation"]))
        ok = label == int(guessed)
        for key in keys:
            if key in meta:
                value = str(meta[key])
                buckets[key].setdefault(value, []).append(ok)
    return {
        key: {value: float(np.mean(items)) for value, items in values.items()}
        for key, values in buckets.items()
        if values
    }


def write_predictions(path: Path, files: list[Path], labels: np.ndarray, pred: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "true", "predicted", "ok"])
        writer.writeheader()
        for sample_path, true, guessed in zip(files, labels, pred):
            writer.writerow(
                {
                    "file": str(sample_path),
                    "true": CLASS_NAMES[int(true)],
                    "predicted": CLASS_NAMES[int(guessed)],
                    "ok": int(true) == int(guessed),
                }
            )


def calibrate(
    checkpoint: str,
    config_path: str,
    data_root: str,
    output_dir: str,
    tune_split: str,
    eval_splits: list[str],
    search_min: float,
    search_max: float,
    step: float,
    objective: str,
) -> dict[str, Any]:
    model, cfg, ckpt_cfg = load_model(checkpoint, config_path)
    tune_logits, tune_labels, tune_files = collect_logits(model, cfg, ckpt_cfg, data_root, tune_split)
    bias, tune_metrics = tune_bias(tune_logits, tune_labels, search_min, search_max, step, objective)
    results: dict[str, Any] = {
        "checkpoint": checkpoint,
        "tune_split": tune_split,
        "objective": objective,
        "bias": {name: float(value) for name, value in zip(CLASS_NAMES, bias)},
        "splits": {},
    }
    tune_metrics["condition_accuracy"] = condition_accuracy(tune_files, tune_metrics["predictions"])
    tune_metrics.pop("predictions")
    results["splits"][tune_split] = tune_metrics

    result_root = ensure_dir(output_dir)
    for split in eval_splits:
        logits, labels, files = collect_logits(model, cfg, ckpt_cfg, data_root, split)
        metrics = metrics_from_logits(logits, labels, bias)
        write_predictions(result_root / f"{split}_predictions.csv", files, labels, metrics["predictions"])
        metrics["condition_accuracy"] = condition_accuracy(files, metrics["predictions"])
        metrics.pop("predictions")
        results["splits"][split] = metrics

    write_json(result_root / "exp03_class_bias_calibration.json", results)
    write_markdown(result_root / "exp03_class_bias_calibration.md", results)
    return results


def write_markdown(path: Path, results: dict[str, Any]) -> None:
    lines = [
        "# Experiment 3 Class Bias Calibration",
        "",
        f"Checkpoint: `{results['checkpoint']}`",
        f"Tune split: `{results['tune_split']}`",
        f"Objective: `{results['objective']}`",
        "",
        "## Bias",
        "",
        "| class | logit bias |",
        "| --- | ---: |",
    ]
    for name, value in results["bias"].items():
        lines.append(f"| {name} | {value:.3f} |")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| split | accuracy | BASK recall | BFSK recall | BPSK recall | worst recall |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split, metrics in results["splits"].items():
        recalls = metrics["class_recall"]
        lines.append(
            f"| {split} | {metrics['accuracy']:.4f} | {recalls['BASK']:.4f} | "
            f"{recalls['BFSK']:.4f} | {recalls['BPSK']:.4f} | {metrics['worst_recall']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--eval-splits", nargs="+", default=["test"])
    parser.add_argument("--search-min", type=float, default=-1.0)
    parser.add_argument("--search-max", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--objective", default="balanced_score", choices=["accuracy", "mean_recall", "worst_recall", "balanced_score"])
    args = parser.parse_args()
    calibrate(
        checkpoint=args.checkpoint,
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        tune_split=args.tune_split,
        eval_splits=args.eval_splits,
        search_min=args.search_min,
        search_max=args.search_max,
        step=args.step,
        objective=args.objective,
    )


if __name__ == "__main__":
    main()
