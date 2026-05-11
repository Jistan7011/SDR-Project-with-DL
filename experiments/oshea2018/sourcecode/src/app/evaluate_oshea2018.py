from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.common import CLASS_NAMES, ensure_dir, load_config, write_json
from src.dataset.iq_dataset import IQDataset, sample_metadata
from src.models.cnn1d import build_model


def evaluate_oshea2018(
    checkpoint: str,
    config_path: str = "../config/config.oshea2018.yaml",
    split: str = "test",
    data_root: str | None = None,
    output_dir: str = "../results/oshea2018_eval",
) -> dict[str, object]:
    cfg = load_config(config_path)
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_cfg = ckpt.get("config", cfg)
    dataset_root = data_root or ckpt_cfg.get("dataset", cfg["dataset"]).get("root", cfg["dataset"]["root"])
    device = torch.device("cuda" if str(cfg["train"].get("device", "cpu")) == "cuda" and torch.cuda.is_available() else "cpu")
    num_workers = int(cfg["train"].get("num_workers", 0))
    pin_memory = bool(cfg["train"].get("pin_memory", device.type == "cuda"))
    loader_kwargs = {"batch_size": int(cfg["train"]["batch_size"]), "num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    dataset = IQDataset(dataset_root, split, feature_mode="iq", preload=bool(cfg["dataset"].get("preload", False)))
    loader = DataLoader(dataset, **loader_kwargs)
    model_cfg = ckpt_cfg.get("model", cfg["model"])
    model = build_model(
        str(ckpt.get("model_type", model_cfg.get("type", "oshea2018_resnet1d"))),
        input_channels=int(model_cfg.get("input_channels", 2)),
        num_classes=int(model_cfg.get("num_classes", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device, non_blocking=pin_memory))
            y_true.extend(yb.numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    absorption = bfsk_bpsk_to_bask_rate(cm)
    session_accuracy = accuracy_by_metadata(dataset.files, y_pred, "session_id")
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": cm.tolist(),
        "bfsk_bpsk_to_bask_rate": absorption,
        "session_accuracy": session_accuracy,
    }
    out = ensure_dir(output_dir)
    write_json(out / "logs" / f"eval_{split}.json", result)
    print(
        f"accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"bfsk_bpsk_to_bask_rate={absorption:.4f}"
    )
    return result


def bfsk_bpsk_to_bask_rate(cm: np.ndarray) -> float:
    bask = CLASS_NAMES.index("BASK")
    non_bask = [CLASS_NAMES.index("BFSK"), CLASS_NAMES.index("BPSK")]
    denom = float(cm[non_bask, :].sum())
    if denom <= 0:
        return 0.0
    return float(cm[non_bask, bask].sum() / denom)


def accuracy_by_metadata(files: list[Path], y_pred: list[int], key: str) -> dict[str, float]:
    buckets: dict[str, list[bool]] = {}
    for path, pred in zip(files, y_pred):
        meta = sample_metadata(path)
        value = str(meta.get(key, "unknown"))
        label = CLASS_NAMES.index(str(meta["modulation"]))
        buckets.setdefault(value, []).append(label == pred)
    return {value: float(np.mean(items)) for value, items in buckets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="../config/config.oshea2018.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="../results/oshea2018_eval")
    args = parser.parse_args()
    evaluate_oshea2018(args.checkpoint, args.config, args.split, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
