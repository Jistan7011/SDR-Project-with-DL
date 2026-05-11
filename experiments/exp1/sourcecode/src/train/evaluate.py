from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset
from src.models.cnn1d import CNN1DClassifier


def evaluate(checkpoint: str, config_path: str = "config.yaml", split: str = "test", data_root: str | None = None) -> dict[str, object]:
    cfg = load_config(config_path)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = CNN1DClassifier(
        input_channels=int(cfg["model"]["input_channels"]),
        num_classes=int(cfg["model"]["num_classes"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dataset = IQDataset(data_root or cfg["dataset"]["root"], split)
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
    ensure_dir("../results/confusion_matrices")
    ensure_dir("../results/figures")
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
    plt.savefig("../results/confusion_matrices/confusion_matrix.png", dpi=150)
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
        plt.savefig("../results/figures/snr_accuracy.png", dpi=150)
        plt.close()

    result = {"accuracy": acc, "classification_report": report, "confusion_matrix": cm.tolist(), "snr_accuracy": snr_acc}
    write_json("../results/logs/eval_metrics.json", result)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.split, args.data_root)


if __name__ == "__main__":
    main()
