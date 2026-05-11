from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model
from src.train.train_oshea2018_extension import compute_metrics


def evaluate_oshea2018_extension(
    checkpoint: str,
    config_path: str = "../config/config.oshea2018.yaml",
    data_root: str | None = None,
    output_dir: str = "../results/oshea2018_extension_eval",
    split: str = "test",
) -> dict[str, Any]:
    cfg = load_config(config_path)
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    dataset_cfg = ckpt_cfg.get("dataset", cfg["dataset"])
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    root = data_root or dataset_cfg.get("root", cfg["dataset"].get("root", "../data/ota_processed"))
    feature_mode = str(ckpt.get("feature_mode", dataset_cfg.get("feature_mode", "iq")))
    input_channels = int(ckpt.get("input_channels", model_cfg.get("input_channels", 2)))
    model_type = str(ckpt.get("model_type", model_cfg.get("type", "resnet1d")))
    device = torch.device("cuda" if str(cfg["train"].get("device", "cpu")) == "cuda" and torch.cuda.is_available() else "cpu")
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_memory = bool(cfg["train"].get("pin_memory", device.type == "cuda"))
    loader_kwargs = {"batch_size": int(cfg["train"]["batch_size"]), "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))

    dataset = IQDataset(root, split, feature_mode=feature_mode, preload=bool(dataset_cfg.get("preload", False)))
    loader = DataLoader(dataset, **loader_kwargs)
    model = build_model(
        model_type,
        input_channels=input_channels,
        num_classes=int(model_cfg.get("num_classes", len(CLASS_NAMES))),
        dropout=float(model_cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    margins: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb.to(device, non_blocking=pin_memory))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(axis=1)
            sorted_probs = np.sort(probs, axis=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.tolist())
            confidences.extend(sorted_probs[:, -1].tolist())
            margins.extend((sorted_probs[:, -1] - sorted_probs[:, -2]).tolist())

    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    metrics = compute_metrics(true, pred)
    metrics["classification_report"] = classification_report(true, pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics["session_accuracy"] = accuracy_by_metadata(dataset.files, y_pred, "session_id")
    metrics["confidence_summary"] = summarize_numeric(confidences)
    metrics["margin_summary"] = summarize_numeric(margins)
    metrics["checkpoint"] = checkpoint
    metrics["data_root"] = str(root)
    metrics["feature_mode"] = feature_mode
    metrics["model_type"] = model_type
    metrics["input_channels"] = input_channels
    metrics["split"] = split

    out = ensure_dir(output_dir)
    write_json(out / "eval_summary.json", metrics)
    write_json(out / "logs" / f"eval_{split}.json", metrics)
    write_prediction_csv(out / "predictions.csv", dataset.files, true.tolist(), pred.tolist(), confidences, margins)
    print(
        f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"worst_recall={metrics['worst_recall']:.4f} bfsk_bpsk_to_bask_rate={metrics['bfsk_bpsk_to_bask_rate']:.4f}"
    )
    return metrics


def accuracy_by_metadata(files: list[Path], y_pred: list[int], key: str) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for path, pred in zip(files, y_pred):
        meta = sample_metadata(path)
        value = str(meta.get(key, "unknown"))
        label = CLASS_NAMES.index(str(meta["modulation"]))
        buckets.setdefault(value, []).append(label == pred)
    return {value: float(np.mean(items)) for value, items in buckets.items()}


def summarize_numeric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
        "p10": float(np.percentile(arr, 10)) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
    }


def write_prediction_csv(path: Path, files: list[Path], y_true: list[int], y_pred: list[int], confidences: list[float], margins: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["file,expected,predicted,confidence,margin,session_id"]
    for file, true, pred, confidence, margin in zip(files, y_true, y_pred, confidences, margins):
        meta = sample_metadata(file)
        lines.append(
            ",".join(
                [
                    file.name,
                    CLASS_NAMES[int(true)],
                    CLASS_NAMES[int(pred)],
                    f"{confidence:.8f}",
                    f"{margin:.8f}",
                    str(meta.get("session_id", "")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="../results/oshea2018_extension_eval")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    evaluate_oshea2018_extension(args.checkpoint, args.config, args.data_root, args.output_dir, args.split)


if __name__ == "__main__":
    main()
