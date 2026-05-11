from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def evaluate(
    checkpoint: str,
    config_path: str = "config.yaml",
    split: str = "test",
    data_root: str | None = None,
    output_dir: str = "results",
) -> dict[str, object]:
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
    feature_mode = str(ckpt_cfg.get("dataset", cfg["dataset"]).get("feature_mode", cfg["dataset"].get("feature_mode", "iq")))
    preload = bool(ckpt_cfg.get("dataset", cfg["dataset"]).get("preload", cfg["dataset"].get("preload", False)))
    dataset = IQDataset(data_root or cfg["dataset"]["root"], split, feature_mode=feature_mode, preload=preload)
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]))
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(dim=1)
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(pred.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    result_root = Path(output_dir)
    ensure_dir(result_root / "confusion_matrices")
    ensure_dir(result_root / "figures")
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(result_root / "confusion_matrices" / "confusion_matrix.png", dpi=150)
    plt.close()

    snr_acc = accuracy_by_snr(dataset.files, y_pred)
    if snr_acc:
        plt.figure(figsize=(6, 4))
        xs = sorted(snr_acc)
        plt.plot(xs, [snr_acc[x] for x in xs], "o-")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(result_root / "figures" / "snr_accuracy.png", dpi=150)
        plt.close()

    condition_metrics = accuracy_by_conditions(dataset.files, y_pred)
    result = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "snr_accuracy": snr_acc,
        "condition_accuracy": condition_metrics,
    }
    write_json(result_root / "logs" / "eval_metrics.json", result)
    print(f"accuracy={acc:.4f}")
    return result


def accuracy_by_snr(files: list[Path], y_pred: list[int]) -> dict[float, float]:
    buckets: dict[float, list[bool]] = {}
    for path, pred in zip(files, y_pred):
        data = np.load(path, allow_pickle=False)
        snr = float(data["snr_db"])
        if np.isnan(snr):
            continue
        label = CLASS_NAMES.index(str(data["modulation"]))
        buckets.setdefault(snr, []).append(label == pred)
    return {snr: float(np.mean(values)) for snr, values in buckets.items()}


def accuracy_by_conditions(files: list[Path], y_pred: list[int]) -> dict[str, dict[str, float]]:
    keys = ["session_id", "payload", "tx_vga_gain", "rx_gain", "baseband_offset_hz"]
    buckets: dict[str, dict[str, list[bool]]] = {key: {} for key in keys}
    for path, pred in zip(files, y_pred):
        meta = sample_metadata(path)
        label = CLASS_NAMES.index(str(meta["modulation"]))
        ok = label == pred
        for key in keys:
            if key in meta:
                value = str(meta[key])
                buckets[key].setdefault(value, []).append(ok)
    return {key: {value: float(np.mean(items)) for value, items in values.items()} for key, values in buckets.items() if values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.split, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
