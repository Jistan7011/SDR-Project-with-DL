from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.exp09_fusion_dataset import Exp09FusionDataset
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model
from src.train.train_exp08_multitask_classifier import compute_exp08_metrics


def evaluate_exp09_blind_classifier(
    checkpoint: str,
    config_path: str,
    data_root: str,
    output_dir: str,
    split: str = "test",
) -> dict[str, Any]:
    cfg = load_config(config_path)
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    model_type = str(ckpt.get("model_type", model_cfg.get("type", "resnet1d")))
    model = build_model(
        model_type,
        input_channels=int(model_cfg.get("input_channels", cfg["model"]["input_channels"])),
        num_classes=int(model_cfg.get("num_classes", cfg["model"]["num_classes"])),
        dropout=float(model_cfg.get("dropout", cfg["model"]["dropout"])),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    is_fusion = model_type.lower().replace("-", "_") in {"fusion_resnet1d_exp9", "exp9_fusion_resnet1d", "fusion_resnet1d_exp9_margin"}
    dataset = Exp09FusionDataset(data_root, split, preload=bool(cfg["dataset"].get("preload", False))) if is_fusion else IQDataset(data_root, split, feature_mode="precomputed")
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    margins: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            outputs = model(xb)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            sorted_probs = np.sort(probs, axis=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(probs.argmax(axis=1).tolist())
            confidences.extend(sorted_probs[:, -1].tolist())
            margins.extend((sorted_probs[:, -1] - sorted_probs[:, -2]).tolist())
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    metrics = compute_exp08_metrics(true, pred)
    metrics["classification_report"] = classification_report(true, pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    metrics["condition_accuracy"] = accuracy_by_conditions(dataset.files, pred.tolist(), cfg["evaluation"].get("condition_keys", []))
    metrics["confidence_summary"] = summarize_numeric(confidences)
    metrics["margin_summary"] = summarize_numeric(margins)
    metrics["checkpoint"] = checkpoint
    metrics["model_type"] = model_type
    metrics["split"] = split
    result_root = ensure_dir(output_dir)
    write_json(result_root / "eval_summary.json", metrics)
    write_json(result_root / "logs" / "eval_metrics.json", metrics)
    write_confusion_matrix(result_root / "confusion_matrices" / "confusion_matrix.png", np.asarray(metrics["confusion_matrix"], dtype=np.int64))
    write_prediction_csv(result_root / "predictions.csv", dataset.files, true.tolist(), pred.tolist(), confidences, margins)
    print(f"accuracy={metrics['accuracy']:.4f} worst_recall={metrics['worst_recall']:.4f} bfsk_bpsk_to_bask_rate={metrics['bfsk_bpsk_to_bask_rate']:.4f}")
    return metrics


def accuracy_by_conditions(files: list[Path], y_pred: list[int], keys: list[str]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[bool]]] = {key: {} for key in keys}
    for path, pred in zip(files, y_pred):
        meta = sample_metadata(path)
        label = CLASS_NAMES.index(str(meta["modulation"]))
        ok = label == pred
        for key in keys:
            if key in meta:
                buckets[key].setdefault(str(meta[key]), []).append(ok)
    return {key: {value: float(np.mean(items)) for value, items in values.items()} for key, values in buckets.items() if values}


def summarize_numeric(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
        "p10": float(np.percentile(arr, 10)) if len(arr) else 0.0,
        "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
    }


def write_confusion_matrix(path: Path, cm: np.ndarray) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_prediction_csv(path: Path, files: list[Path], y_true: list[int], y_pred: list[int], confidences: list[float], margins: list[float]) -> None:
    ensure_dir(path.parent)
    lines = ["file,expected,predicted,confidence,margin,session_id,payload,snr_db,estimated_cfo_hz,dc_magnitude"]
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
                    str(meta.get("payload", "")),
                    str(meta.get("snr_db", "")),
                    str(meta.get("exp09_estimated_cfo_hz", "")),
                    str(meta.get("exp09_dc_magnitude", "")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="../config/config.exp09.yaml")
    parser.add_argument("--data-root", default="../data/processed")
    parser.add_argument("--output-dir", default="../results/eval")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    evaluate_exp09_blind_classifier(args.checkpoint, args.config, args.data_root, args.output_dir, args.split)


if __name__ == "__main__":
    main()
